# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenSQLAgenticNode implementation for SQL generation with enhanced configuration.

This module provides a specialized implementation of AgenticNode focused on
SQL generation with support for limited context, enhanced template variables,
and flexible configuration through agent.yml.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.agent_models import SubAgentConfig
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput, GenSQLNodeResult
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, FilesystemFuncTool, PlatformDocSearchTool
from datus.tools.func_tool.date_parsing_tools import DateParsingTools
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenSQLAgenticNode(AgenticNode):
    """
    SQL generation agentic node with enhanced configuration and limited context support.

    This node provides specialized SQL generation capabilities with:
    - Enhanced system prompt with template variables
    - Limited context support (tables, metrics, reference_sql)
    - Tool detection and dynamic template preparation
    - Configurable tool sets and MCP server integration
    - Session-based conversation management
    """

    result_class = GenSQLNodeResult

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[GenSQLNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the GenSQLAgenticNode as a workflow-compatible node.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (should be 'gensql')
            input_data: SQL generation input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
            node_name: Name of the node configuration in agent.yml (e.g., "gensql", "gen_sql")
            execution_mode: Execution mode - "interactive" (default) or "workflow"
            is_subagent: When True, skip SubAgentTaskTool setup (2-level depth enforcement)
        """
        self.execution_mode = execution_mode
        # Determine node name from node_type if not provided
        self.configured_node_name = node_name

        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        # Initialize tool attributes BEFORE calling parent constructor
        # This is required because parent's __init__ calls _get_system_prompt()
        # which may reference these attributes
        self.db_func_tool: Optional[DBFuncTool] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.date_parsing_tools: Optional[DateParsingTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self._platform_doc_tool: Optional[PlatformDocSearchTool] = None
        self.reference_template_tools: Optional[ReferenceTemplateTools] = None
        self.semantic_tools: Optional[SemanticTools] = None
        # PlanTool instance when `plan_tools` is declared in the sub-agent's
        # `tools:` list. Distinct from `plan_hooks` below, which is only
        # active in full plan_mode workflows.
        self.plan_tool = None

        # Plan mode state lives on AgenticNode base class; nothing to declare here.

        # Call parent constructor with all required Node parameters
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},  # Initialize empty, will setup after parent init
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Initialize MCP servers based on configuration (after node_config is available)
        self.mcp_servers = self._setup_mcp_servers()

        # Debug: Log final MCP servers assignment
        logger.debug(
            f"GenSQLAgenticNode final mcp_servers: {len(self.mcp_servers)} servers - {list(self.mcp_servers.keys())}"
        )

        # Setup tools based on configuration (includes subagent task tool wiring)
        self.setup_tools()

        # Setup ask_user tool for clarification questions (interactive mode only)
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(f"GenSQLAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this SQL generation agentic node.

        Returns:
            The configured node name from agent.yml (e.g., "gensql", "gen_sql")
        """
        return self.configured_node_name

    def setup_input(self, workflow: Workflow) -> dict:
        """
        Setup GenSQL input from workflow context.

        Creates GenSQLNodeInput with user message from task and context data.

        Args:
            workflow: Workflow instance containing context and task

        Returns:
            Dictionary with success status and message
        """
        # Update database connection if task specifies a different database
        task_database = workflow.task.database_name
        if task_database and self.db_func_tool and task_database != self.db_func_tool.connector.database_name:
            logger.debug(
                f"Updating database connection from '{self.db_func_tool.connector.database_name}' "
                f"to '{task_database}' based on workflow task"
            )
            self._update_database_connection(task_database)

        # Read plan mode flags from workflow metadata
        plan_mode = workflow.metadata.get("plan_mode", False)
        auto_execute_plan = workflow.metadata.get("auto_execute_plan", False)

        logger.debug(f"context: {workflow.context}")
        # Create GenSQLNodeInput if not already set
        if not self.input or not isinstance(self.input, GenSQLNodeInput):
            logger.debug(f"creating GenSQLNodeInput: {self.input}")
            self.input = GenSQLNodeInput(
                user_message=workflow.task.task,
                external_knowledge=workflow.task.external_knowledge,
                catalog=workflow.task.catalog_name,
                database=workflow.task.database_name,
                db_schema=workflow.task.schema_name,
                schemas=workflow.context.table_schemas,
                metrics=workflow.context.metrics,
                reference_date=workflow.task.current_date,
                plan_mode=plan_mode,
                auto_execute_plan=auto_execute_plan,
            )
        else:
            # Update existing input with workflow data
            self.input.user_message = workflow.task.task
            self.input.external_knowledge = workflow.task.external_knowledge
            self.input.catalog = workflow.task.catalog_name
            self.input.database = workflow.task.database_name
            self.input.db_schema = workflow.task.schema_name
            self.input.schemas = workflow.context.table_schemas
            self.input.metrics = workflow.context.metrics
            self.input.reference_date = workflow.task.current_date
            self.input.plan_mode = plan_mode
            self.input.auto_execute_plan = auto_execute_plan

        # Set reference date for date parsing tools if configured
        # Always call set_reference_date to clear previous state even when current_date is None
        if self.date_parsing_tools:
            self.date_parsing_tools.set_reference_date(workflow.task.current_date)

        return {"success": True, "message": "GenSQL input prepared from workflow"}

    def _update_database_connection(self, database_name: str):
        """
        Update database connection to a different database.

        Args:
            database_name: The name of the database to connect to
        """
        self.db_func_tool = DBFuncTool(
            agent_config=self.agent_config,
            default_datasource=database_name,
            sub_agent_name=self.node_config.get("system_prompt"),
        )
        self._rebuild_tools()

    def _rebuild_tools(self):
        """Rebuild the tools list with current tool instances."""
        self.tools = []
        if self.db_func_tool:
            self.tools.extend(self.db_func_tool.available_tools())
        if self.context_search_tools:
            self.tools.extend(self.context_search_tools.available_tools())
        if self.semantic_tools:
            self.tools.extend(self.semantic_tools.available_tools())
        if self.reference_template_tools:
            self.tools.extend(self.reference_template_tools.available_tools())
        if self.date_parsing_tools:
            self.tools.extend(self.date_parsing_tools.available_tools())
        if self.filesystem_func_tool:
            self.tools.extend(self.filesystem_func_tool.available_tools())
        if self._platform_doc_tool:
            self.tools.extend(self._platform_doc_tool.available_tools())
        if self.ask_user_tool:
            self.tools.extend(self.ask_user_tool.available_tools())
        if self.sub_agent_task_tool:
            self.tools.extend(self.sub_agent_task_tool.available_tools())
        # Plan-mode tools (confirm_plan + todo_*) must survive every rebuild,
        # otherwise a mid-session datasource switch (which routes through
        # ``_update_database_connection`` → ``_rebuild_tools``) silently
        # strips plan tooling for the rest of the session.
        self._register_plan_mode_tools()

    # Default tools when not configured in agent.yml
    DEFAULT_TOOLS = "db_tools.*, semantic_tools.*, context_search_tools.*"

    def setup_tools(self):
        """Setup tools based on configuration, falling back to DEFAULT_TOOLS."""
        if not self.agent_config:
            return

        self.tools = []
        tools_str = self.node_config.get("tools")
        if not tools_str:
            tools_str = self.DEFAULT_TOOLS

        tool_patterns = [p.strip() for p in tools_str.split(",") if p.strip()]
        for pattern in tool_patterns:
            self._setup_tool_pattern(pattern)

        # Ensure filesystem tools are always available (required for memory and file operations)
        if not self.filesystem_func_tool:
            self._setup_filesystem_tools()

        # Rebuild subagent task tool so repeated setup_tools() calls (e.g. via
        # ChatCommands.update_chat_node_tools after a datasource switch) keep the
        # "task" tool available for delegation.
        self._setup_sub_agent_task_tool()
        if self.sub_agent_task_tool:
            self.tools.extend(self.sub_agent_task_tool.available_tools())

        # Plan-mode tools (confirm_plan + todo_*) for main agents; no-op for sub-agents.
        self._register_plan_mode_tools()

        logger.debug(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

    def _setup_platform_doc_tools(self):
        """Setup tools based on configuration."""
        try:
            self._platform_doc_tool = PlatformDocSearchTool(self.agent_config)
            self.tools.extend(self._platform_doc_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup platform_doc_search tools: {e}")

    def _setup_db_tools(self):
        """Setup database tools."""
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
            )
            self.tools.extend(self.db_func_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_context_search_tools(self):
        """Setup context search tools."""
        try:
            self.context_search_tools = ContextSearchTools(self.agent_config, sub_agent_name=self.get_node_name())
            self.tools.extend(self.context_search_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup context search tools: {e}")

    def _setup_semantic_tools(self):
        """Setup semantic tools for metric/dimension exploration."""
        try:
            adapter_type = self.node_config.get("adapter_type", "metricflow")
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
                adapter_type=adapter_type,
            )
            self.tools.extend(self.semantic_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup semantic tools: {e}")

    def _setup_reference_template_tools(self):
        """Setup reference template tools.

        If db_tools are not configured but reference_template_tools are requested,
        create an internal-only db_func_tool for execute_reference_template
        without exposing db_tools (read_query, list_tables, etc.) to the LLM.
        """
        try:
            db_tool = self.db_func_tool
            if not db_tool:
                db_tool = DBFuncTool(
                    agent_config=self.agent_config,
                    sub_agent_name=self.node_config.get("system_prompt"),
                )
            self.reference_template_tools = ReferenceTemplateTools(
                self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
                db_func_tool=db_tool,
            )
            self.tools.extend(self.reference_template_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup reference template tools: {e}")

    def _setup_date_parsing_tools(self):
        """Setup date parsing tools."""
        try:
            self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            self.tools.extend(self.date_parsing_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup date parsing tools: {e}")

    def _setup_plan_tools(self):
        """Setup plan/todo tools so the agent can track multi-step work.

        PlanTool exposes `todo_list`, `todo_read`, `todo_write`, and
        `todo_update`, backed by the agent's conversation session. Unlike
        plan_mode — which is a full interactive replan workflow — this just
        makes the todo surface available as regular function tools, so a
        long-horizon task (e.g. generating a complex marts table with 30+
        output columns) can write a plan up front and check off items as it
        goes, avoiding MaxTurnsExceeded and drift.
        """
        try:
            from datus.tools.func_tool.plan_tools import PlanTool

            session, _ = self._get_or_create_session()
            # Lazy resolver mirrors AgenticNode._get_plan_mode_tools — even
            # though we've already called _get_or_create_session here, future
            # session swaps (rewind / switch) should be picked up too.
            self.plan_tool = PlanTool(session, session_id=lambda: self.session_id)
            self.tools.extend(self.plan_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup plan tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools (all available tools)."""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug(f"Setup filesystem tools with root path: {self.filesystem_func_tool.root_path}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_tool_pattern(self, pattern: str):
        """Setup tools based on pattern."""
        try:
            # Handle wildcard patterns (e.g., "db_tools.*")
            if pattern.endswith(".*"):
                base_type = pattern[:-2]  # Remove ".*"
                if base_type == "db_tools":
                    self._setup_db_tools()
                elif base_type == "context_search_tools":
                    self._setup_context_search_tools()
                elif base_type == "semantic_tools":
                    self._setup_semantic_tools()
                elif base_type == "reference_template_tools":
                    self._setup_reference_template_tools()
                elif base_type == "date_parsing_tools":
                    self._setup_date_parsing_tools()
                elif base_type == "plan_tools":
                    self._setup_plan_tools()
                elif base_type == "filesystem_tools":
                    self._setup_filesystem_tools()
                elif base_type == "platform_doc_tools":
                    self._setup_platform_doc_tools()
                else:
                    logger.warning(f"Unknown tool type: {base_type}")

            # Handle exact type patterns (e.g., "db_tools")
            elif pattern == "db_tools":
                self._setup_db_tools()
            elif pattern == "context_search_tools":
                self._setup_context_search_tools()
            elif pattern == "semantic_tools":
                self._setup_semantic_tools()
            elif pattern == "reference_template_tools":
                self._setup_reference_template_tools()
            elif pattern == "date_parsing_tools":
                self._setup_date_parsing_tools()
            elif pattern == "plan_tools":
                self._setup_plan_tools()
            elif pattern == "filesystem_tools":
                self._setup_filesystem_tools()
            elif pattern == "platform_doc_tools":
                self._setup_platform_doc_tools()

            # Handle specific method patterns (e.g., "db_tools.list_tables")
            elif "." in pattern:
                tool_type, method_name = pattern.split(".", 1)
                self._setup_specific_tool_method(tool_type, method_name)

            else:
                logger.warning(f"Unknown tool pattern: {pattern}")

        except Exception as e:
            logger.error(f"Failed to setup tool pattern '{pattern}': {e}")

    def _setup_specific_tool_method(self, tool_type: str, method_name: str):
        """Setup a specific tool method."""
        try:
            if tool_type == "context_search_tools":
                if not self.context_search_tools:
                    self.context_search_tools = ContextSearchTools(self.agent_config, self.node_config["system_prompt"])
                tool_instance = self.context_search_tools
            elif tool_type == "semantic_tools":
                if not self.semantic_tools:
                    adapter_type = self.node_config.get("adapter_type", "metricflow")
                    self.semantic_tools = SemanticTools(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                        adapter_type=adapter_type,
                    )
                tool_instance = self.semantic_tools
            elif tool_type == "db_tools":
                if not self.db_func_tool:
                    self.db_func_tool = DBFuncTool(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                    )
                tool_instance = self.db_func_tool
            elif tool_type == "date_parsing_tools":
                if not self.date_parsing_tools:
                    self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
                tool_instance = self.date_parsing_tools
            elif tool_type == "filesystem_tools":
                if not self.filesystem_func_tool:
                    self.filesystem_func_tool = self._make_filesystem_tool()
                tool_instance = self.filesystem_func_tool
            elif tool_type == "platform_doc_tools":
                if not self._platform_doc_tool:
                    self._platform_doc_tool = PlatformDocSearchTool(self.agent_config)
                tool_instance = self._platform_doc_tool
            elif tool_type == "reference_template_tools":
                if not self.reference_template_tools:
                    db_tool = self.db_func_tool
                    if not db_tool:
                        db_tool = DBFuncTool(
                            agent_config=self.agent_config,
                            sub_agent_name=self.node_config.get("system_prompt"),
                        )
                    self.reference_template_tools = ReferenceTemplateTools(
                        self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                        db_func_tool=db_tool,
                    )
                tool_instance = self.reference_template_tools
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                return

            if hasattr(tool_instance, method_name):
                method = getattr(tool_instance, method_name)
                from datus.tools.func_tool import trans_to_function_tool

                if isinstance(tool_instance, DBFuncTool):
                    self.tools.append(tool_instance.to_function_tool(method))
                else:
                    self.tools.append(trans_to_function_tool(method))
                logger.debug(f"Added specific tool method: {tool_type}.{method_name}")
            else:
                logger.warning(f"Method '{method_name}' not found in {tool_type}")
        except Exception as e:
            logger.error(f"Failed to setup {tool_type}.{method_name}: {e}")

    def _setup_mcp_server_from_config(self, server_name: str) -> Optional[Any]:
        """Setup MCP server from {agent.home}/conf/.mcp.json using mcp_manager."""
        try:
            from datus.tools.mcp_tools.mcp_manager import MCPManager

            # Use MCPManager to get server config
            mcp_manager = MCPManager(agent_config=self.agent_config)
            server_config = mcp_manager.get_server_config(server_name)

            if not server_config:
                logger.warning(f"MCP server '{server_name}' not found in configuration")
                return None

            # Create server instance using the manager
            server_instance, details = mcp_manager._create_server_instance(server_config)

            if server_instance:
                logger.debug(f"Added MCP server '{server_name}' from configuration: {details}")
                return server_instance
            else:
                error_msg = details.get("error", "Unknown error")
                logger.warning(f"Failed to create MCP server '{server_name}': {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Failed to setup MCP server '{server_name}' from config: {e}")
            return None

    def _setup_mcp_servers(self) -> Dict[str, Any]:
        """Set up MCP servers based on configuration."""
        mcp_servers = {}

        config_value = self.node_config.get("mcp", "")
        if not config_value:
            return mcp_servers

        mcp_server_names = [p.strip() for p in config_value.split(",") if p.strip()]

        for server_name in mcp_server_names:
            try:
                server = self._setup_mcp_server_from_config(server_name)
                if server:
                    mcp_servers[server_name] = server

            except Exception as e:
                logger.error(f"Failed to setup MCP server '{server_name}': {e}")

        logger.debug(f"Setup {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())}")

        # Debug: Log detailed info about each server
        for name, server in mcp_servers.items():
            logger.debug(f"MCP server '{name}': type={type(server)}, instance={server}")

        return mcp_servers

    def _get_available_tool_names(self) -> list[str]:
        """Return native tool names plus discoverable MCP tool names."""
        tool_names = {getattr(tool, "name", "") for tool in (self.tools or []) if getattr(tool, "name", "")}
        tool_names.update(self._get_mcp_tool_names_for_prompt())
        return sorted(tool_names)

    def _get_mcp_tool_names_for_prompt(self) -> set[str]:
        """Collect MCP tool names that can be known without starting new MCP connections."""
        active_server_names = [name for name in (self.mcp_servers or {}).keys() if name]
        if not active_server_names or not self.agent_config:
            return set()

        try:
            from datus.tools.mcp_tools.mcp_manager import MCPManager

            mcp_manager = MCPManager(agent_config=self.agent_config)
        except Exception as e:
            logger.debug(f"Unable to inspect MCP tool filters for prompt context: {e}")
            return set()

        tool_names: set[str] = set()
        for server_name in active_server_names:
            server_config = mcp_manager.get_server_config(server_name)
            tool_filter = getattr(server_config, "tool_filter", None) if server_config else None

            allowed_tool_names = getattr(tool_filter, "allowed_tool_names", None)
            if allowed_tool_names:
                tool_names.update(name for name in allowed_tool_names if tool_filter.is_tool_allowed(name))

            cached_tool_names = self._get_cached_mcp_tool_names(self.mcp_servers[server_name])
            if tool_filter:
                cached_tool_names = {name for name in cached_tool_names if tool_filter.is_tool_allowed(name)}
            tool_names.update(cached_tool_names)

        return tool_names

    @staticmethod
    def _get_cached_mcp_tool_names(server: Any) -> set[str]:
        """Read SDK-cached MCP tool names when the server already discovered them."""
        cached_tools = getattr(server, "_tools_list", None)
        if not isinstance(cached_tools, (list, tuple, set)):
            return set()

        return {getattr(tool, "name", "") for tool in cached_tools if getattr(tool, "name", "")}

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Get the system prompt for this SQL generation node using enhanced template context.

        Args:
            prompt_version: Optional prompt version to use, overrides agent config version

        Returns:
            System prompt string loaded from the template
        """
        context = prepare_template_context(
            node_config=self.node_config,
            has_db_tools=bool(self.db_func_tool),
            has_filesystem_tools=bool(self.filesystem_func_tool),
            has_mf_tools=False,
            has_context_search_tools=bool(self.context_search_tools),
            has_reference_template_tools=bool(
                self.reference_template_tools and self.reference_template_tools.has_reference_templates
            ),
            has_parsing_tools=bool(self.date_parsing_tools),
            has_platform_doc_tools=bool(self._platform_doc_tool),
            agent_config=self.agent_config,
            workspace_root=self._resolve_workspace_root(),
        )
        context["has_task_tool"] = bool(self.sub_agent_task_tool)
        available_tool_names = self._get_available_tool_names()
        context["available_tool_names"] = available_tool_names
        context["has_read_query_tool"] = "read_query" in available_tool_names
        context["has_describe_table_tool"] = "describe_table" in available_tool_names
        context["has_list_metrics_tool"] = "list_metrics" in available_tool_names
        context["has_query_metrics_tool"] = "query_metrics" in available_tool_names
        context["has_ask_user_tool"] = "ask_user" in available_tool_names
        context["has_reference_template_tools"] = any(
            name in available_tool_names
            for name in (
                "search_reference_template",
                "get_reference_template",
                "render_reference_template",
                "execute_reference_template",
            )
        )
        from datus.utils.time_utils import get_default_current_date

        ref = self.date_parsing_tools.reference_date if self.date_parsing_tools else None
        context["current_date"] = get_default_current_date(ref)
        prompt_version = prompt_version or self.node_config.get("prompt_version")
        # Construct template name: {system_prompt}_system or fallback to {node_name}_system
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        # Use prompt manager to render the template
        from datus.prompts.prompt_manager import get_prompt_manager

        pm = get_prompt_manager(agent_config=self.agent_config)
        try:
            base_prompt = pm.render_template(template_name=template_name, version=prompt_version, **context)
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError:
            # Template not found - throw DatusException
            logger.warning(f"Failed to render system prompt '{system_prompt_name}', using the default template instead")
            base_prompt = pm.render_template(template_name="sql_system", version=None, **context)
            return self._finalize_system_prompt(base_prompt)
        except Exception as e:
            # Other template errors - wrap in DatusException
            logger.error(f"Template loading error for '{template_name}': {e}")

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    # ── SQL file storage helpers ─────────────────────────────────────

    def _get_sql_preview_lines(self) -> int:
        """Get the number of preview lines from node_config, default 5."""
        return int(self.node_config.get("sql_preview_lines", 5))

    @staticmethod
    def _get_sql_preview(sql: str, max_lines: int = 5) -> str:
        """Return the first N lines of SQL as a preview."""
        lines = sql.splitlines()
        preview_lines = lines[:max_lines]
        preview = "\n".join(preview_lines)
        if len(lines) > max_lines:
            preview += f"\n-- ... ({len(lines) - max_lines} more lines)"
        return preview

    def _read_existing_sql_file(self, file_path: str) -> Optional[str]:
        """Read SQL file content via FilesystemFuncTool.

        Returns:
            File content string on success, None if file doesn't exist or read fails.
        """
        if not self.filesystem_func_tool or not file_path:
            return None
        result = self.filesystem_func_tool.read_file(path=file_path)
        if result.success and result.result:
            return result.result if isinstance(result.result, str) else None
        return None

    def _build_success_result(self, ctx: StreamRunContext) -> GenSQLNodeResult:
        # GenSQL's parsing scans the full action history (not just the last
        # assistant action) so it can recover SQL stashed in summary_report
        # actions when the assistant returned a markdown overview instead.
        response_content, sql_content = self._collect_final_response(ctx.action_history_manager)

        all_actions = ctx.action_history_manager.get_actions()
        tokens_used = self._extract_total_tokens(all_actions)
        tool_calls = [
            action for action in all_actions if action.role == ActionRole.TOOL and action.status == ActionStatus.SUCCESS
        ]
        execution_stats = {
            "total_actions": len(all_actions),
            "tool_calls_count": len(tool_calls),
            "tools_used": list({a.action_type for a in tool_calls}),
            "total_tokens": int(tokens_used),
        }

        # Detect LLM-saved SQL file vs inline SQL — read the file when the
        # response looks like a path so callers receive the full statement.
        sql_file_path = None
        sql_preview = None
        result_sql = sql_content
        if sql_content and sql_content.strip().endswith(".sql"):
            candidate_path = sql_content.strip()
            full_sql = self._read_existing_sql_file(candidate_path)
            if full_sql:
                sql_file_path = candidate_path
                sql_preview = self._get_sql_preview(full_sql, self._get_sql_preview_lines())
                result_sql = full_sql

        return GenSQLNodeResult(
            success=True,
            response=response_content,
            sql=result_sql,
            sql_file_path=sql_file_path,
            sql_preview=sql_preview,
            tokens_used=int(tokens_used),
            action_history=[a.model_dump() for a in all_actions],
            execution_stats=execution_stats,
        )

    def update_context(self, workflow: Workflow) -> dict:
        """
        Update workflow context with SQL generation results.

        Stores SQL query, explanation, and execution results to workflow context.
        """
        if not self.result:
            return {"success": False, "message": "No result to update context"}

        result = self.result

        try:
            if hasattr(result, "sql") and result.sql:
                from datus.schemas.node_models import SQLContext

                # Extract SQL result from response if available
                sql_result = ""
                if hasattr(result, "response") and result.response:
                    _, sql_result = self._extract_sql_and_output_from_response({"content": result.response})
                    sql_result = sql_result or ""

                # Create complete SQLContext record
                new_record = SQLContext(
                    sql_query=result.sql,
                    explanation=result.response if hasattr(result, "response") else "",
                    sql_return=sql_result,
                )
                workflow.context.sql_contexts.append(new_record)

            return {"success": True, "message": "Updated SQL generation context"}
        except Exception as e:
            logger.error(f"Failed to update SQL generation context: {e}")
            return {"success": False, "message": str(e)}

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Register tools under their canonical categories so permission rules fire.

        Without this override the bound tools fall through to the ``tools.*``
        catch-all (default ASK on normal/auto profiles), which would block at
        ``InteractionBroker.request`` whenever a caller wires permission hooks
        but does not also run an interactive broker listener.
        """
        mapping = super()._tool_category_map()
        if getattr(self, "db_func_tool", None):
            mapping["db_tools"] = list(self.db_func_tool.available_tools())
        if getattr(self, "context_search_tools", None):
            mapping["context_search_tools"] = list(self.context_search_tools.available_tools())
        if getattr(self, "semantic_tools", None):
            mapping["semantic_tools"] = list(self.semantic_tools.available_tools())
        if getattr(self, "reference_template_tools", None):
            mapping["semantic_tools"] = mapping.get("semantic_tools", []) + list(
                self.reference_template_tools.available_tools()
            )
        if getattr(self, "date_parsing_tools", None):
            mapping["date_parsing_tools"] = list(self.date_parsing_tools.available_tools())
        if getattr(self, "filesystem_func_tool", None):
            mapping["filesystem_tools"] = list(self.filesystem_func_tool.available_tools())
        return mapping

    def _extract_sql_and_output_from_response(self, output: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extract SQL content and formatted output from model response.

        Uses the existing llm_result2json utility for robust JSON parsing.
        Handles the current template format: {"sql": "...", "output": "..."}.
        Older {"sql": "...", "tables": [...], "explanation": "..."} responses are
        still accepted as a compatibility fallback.

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (sql_string, output_string) - both can be None if not found
        """
        try:
            from datus.utils.json_utils import llm_result2json

            content = output.get("content", "")
            logger.debug(
                f"extract_sql_and_output_from_final_resp: {content[:200] if isinstance(content, str) else content}"
            )

            if not isinstance(content, str) or not content.strip():
                return None, None

            # Parse the JSON content
            parsed = llm_result2json(content, expected_type=dict)

            if parsed and isinstance(parsed, dict):
                # Extract SQL
                sql = parsed.get("sql")

                # New sql_system protocol uses `output` as the single user-facing field.
                output_text = parsed.get("output")
                explanation = parsed.get("explanation", "")
                tables = parsed.get("tables", [])

                # Backward compatibility for older prompts that returned explanation/tables.
                if not output_text and (explanation or tables):
                    output_parts = []
                    if explanation:
                        output_parts.append(f"Explanation: {explanation}")
                    if tables:
                        tables_str = ", ".join(tables) if isinstance(tables, list) else str(tables)
                        output_parts.append(f"Tables used: {tables_str}")
                    output_text = "\n".join(output_parts)

                # Unescape output content if present
                if output_text and isinstance(output_text, str):
                    output_text = output_text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

                return sql, output_text

            return None, None
        except Exception as e:
            logger.warning(f"Failed to extract SQL and output from response: {e}")
            return None, None

    def _collect_final_response(
        self,
        action_history_manager: ActionHistoryManager,
    ) -> tuple[str, Optional[str]]:
        """Collect final response text and SQL from accumulated action history.

        Only assistant-role actions are considered when accumulating
        ``response_content``; tool results may carry their own ``raw_output``
        / ``content`` fields and would otherwise overwrite the model's reply.
        """

        response_content = ""
        last_successful_output = None
        for stream_action in action_history_manager.get_actions():
            if (
                stream_action.role == ActionRole.ASSISTANT
                and stream_action.status == ActionStatus.SUCCESS
                and stream_action.output
            ):
                if isinstance(stream_action.output, dict):
                    last_successful_output = stream_action.output
                    candidate = (
                        stream_action.output.get("content", "")
                        or stream_action.output.get("response", "")
                        or stream_action.output.get("raw_output", "")
                    )
                    if isinstance(candidate, str) and candidate:
                        response_content = candidate
                    elif candidate and not isinstance(candidate, str):
                        response_content = str(candidate)

        if not response_content and last_successful_output:
            logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
            response_content = (
                last_successful_output.get("content", "")
                or last_successful_output.get("text", "")
                or last_successful_output.get("response", "")
                or last_successful_output.get("raw_output", "")
            )
            if not response_content:
                logger.warning(
                    f"Falling back to str() for response content: keys={list(last_successful_output.keys())}"
                )
                response_content = str(last_successful_output)

        sql_content = None
        for stream_action in reversed(action_history_manager.get_actions()):
            if stream_action.action_type == "summary_report" and stream_action.output:
                if isinstance(stream_action.output, dict):
                    sql_content = stream_action.output.get("sql")
                    if not response_content:
                        response_content = (
                            stream_action.output.get("markdown", "")
                            or stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                        )
                    if sql_content:
                        logger.debug(f"Extracted SQL from summary_report action: {sql_content[:100]}...")
                        break

        if not sql_content:
            extracted_sql, extracted_output = self._extract_sql_and_output_from_response({"content": response_content})
            if extracted_sql:
                sql_content = extracted_sql
            if extracted_output:
                response_content = extracted_output

        return response_content, sql_content


def prepare_template_context(
    node_config: Union[Dict[str, Any], SubAgentConfig],
    has_db_tools: bool = True,
    has_filesystem_tools: bool = True,
    has_mf_tools: bool = False,
    has_context_search_tools: bool = True,
    has_reference_template_tools: bool = False,
    has_parsing_tools: bool = True,
    has_platform_doc_tools: bool = False,
    has_semantic_tools: bool = False,
    agent_config: Optional[AgentConfig] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """
    Prepare template context variables for the gen_sql_system template.

    Args:
        node_config: Node configuration
        has_db_tools: Whether database tools are available
        has_filesystem_tools: Whether filesystem tools are available
        has_mf_tools: Legacy MetricFlow prompt flag
        has_context_search_tools: Whether context search tools are available
        has_reference_template_tools: Whether reference template tools are available
        has_parsing_tools: Whether date parsing tools are available
        has_platform_doc_tools: Whether platform documentation search tools are available
        has_semantic_tools: Whether semantic / metric tools are available
        agent_config: Agent configuration
        workspace_root: Workspace root path

    Returns:
        Dictionary of template variables
    """
    context: Dict[str, Any] = {
        "has_db_tools": has_db_tools,
        "has_filesystem_tools": has_filesystem_tools,
        "has_mf_tools": has_mf_tools,
        "has_context_search_tools": has_context_search_tools,
        "has_reference_template_tools": has_reference_template_tools,
        "has_parsing_tools": has_parsing_tools,
        "has_platform_doc_tools": has_platform_doc_tools,
        "has_semantic_tools": has_semantic_tools,
    }
    if not isinstance(node_config, SubAgentConfig):
        node_config = SubAgentConfig.model_validate(node_config)

    # Tool name lists for template display
    context["native_tools"] = node_config.tool_list
    context["mcp_tools"] = node_config.mcp
    # Limited context support
    has_scoped_context = False

    scoped_context = node_config.scoped_context
    if scoped_context:
        has_scoped_context = bool(
            scoped_context.tables or scoped_context.metrics or scoped_context.sqls or scoped_context.ext_knowledge
        )

    context["scoped_context"] = has_scoped_context

    # Add rules from configuration
    context["rules"] = node_config.rules or []

    # Add agent description from configuration or input
    context["agent_description"] = node_config.agent_description

    # Add datasource and workspace info
    if agent_config:
        context["agent_config"] = agent_config
        from datus.utils.node_utils import build_datasource_prompt_context

        context.update(build_datasource_prompt_context(agent_config))
        context["db_name"] = context.get("datasource")
        context["workspace_root"] = workspace_root or getattr(agent_config, "project_root", None)
    logger.debug(f"Prepared template context: {context}")
    return context
