# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ChatAgenticNode implementation for flexible CLI chat interactions.

This module provides a concrete implementation of AgenticNode for general-purpose
chat interactions with markdown output, database/filesystem tool support,
skills, and permissions. This node is fully independent from GenSQLAgenticNode.
"""

from typing import Any, AsyncGenerator, Dict, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.gen_sql_agentic_node import build_enhanced_message, prepare_template_context
from datus.agent.workflow import Workflow
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, FilesystemFuncTool, PlatformDocSearchTool
from datus.tools.func_tool.date_parsing_tools import DateParsingTools
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools
from datus.tools.mcp_tools import MCPServer
from datus.tools.permission.permission_hooks import CompositeHooks, PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.message_utils import (
    MessagePart,
    build_structured_content,
    extract_enhanced_context,
    extract_user_input,
    is_structured_content,
)

logger = get_logger(__name__)


class ChatAgenticNode(AgenticNode):
    """
    General-purpose chat agentic node with markdown output.

    This node provides flexible chat capabilities with:
    - Direct markdown response output (no JSON/SQL extraction)
    - Full tool support: database, filesystem, context search, date parsing
    - Skill discovery and execution with permission control
    - Permission hooks for tool access control
    - Plan mode support with recursive replanning
    - Session-based conversation management with MCP server integration
    """

    DEFAULT_SUBAGENTS = "*"

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ChatNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        scope: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        is_subagent: bool = False,
    ):
        """
        Initialize the ChatAgenticNode.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (should be 'chat')
            input_data: Chat input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
            execution_mode: Execution mode - "interactive" (default) or "workflow"
        """
        self.execution_mode = execution_mode

        # Node name for config lookup and template resolution
        self.configured_node_name = "chat"

        # Max turns from config
        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and "chat" in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes["chat"]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        # Initialize tool attributes BEFORE calling parent constructor
        self.db_func_tool: Optional[DBFuncTool] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.date_parsing_tools: Optional[DateParsingTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self._platform_doc_tool: Optional[PlatformDocSearchTool] = None
        self.reference_template_tools: Optional[ReferenceTemplateTools] = None

        # Plan mode attributes
        self.plan_mode_active = False
        self.plan_hooks = None

        # Chat-specific: permission hooks
        self.permission_hooks: Optional[PermissionHooks] = None

        # Call parent constructor
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
        )

        # Execution mode: "interactive" enables ask_user tool; "workflow"
        # disables it so the agent never pauses for user input.
        self.execution_mode = execution_mode

        # Initialize MCP servers based on configuration
        self.mcp_servers = self._setup_mcp_servers()

        # Setup tools
        self.setup_tools()
        logger.debug(f"ChatAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")
        logger.debug(f"ChatAgenticNode initialized: {self.agent_config.current_database}")

    def get_node_name(self) -> str:
        """Get the configured node name."""
        return self.configured_node_name

    # ── Tool Setup ──────────────────────────────────────────────────────

    def setup_tools(self):
        """Initialize all tools with default database connection."""
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_database, self.agent_config.current_database)
        self.db_func_tool = DBFuncTool(conn, agent_config=self.agent_config)
        self.context_search_tools = ContextSearchTools(self.agent_config)
        self.reference_template_tools = ReferenceTemplateTools(self.agent_config, db_func_tool=self.db_func_tool)
        self._setup_date_parsing_tools()
        self._setup_filesystem_tools()
        self._setup_skill_tools()
        self._setup_sub_agent_task_tool()
        # Setup ask_user tool for clarification questions (interactive mode only)
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        self._rebuild_tools()
        self._setup_platform_doc_tools()

        # Setup permission hooks after all tools are initialized
        self._setup_permission_hooks()

    def _setup_date_parsing_tools(self):
        """Setup date parsing tools."""
        try:
            self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            self.tools.extend(self.date_parsing_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup date parsing tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug(f"Setup filesystem tools with root path: {self.filesystem_func_tool.root_path}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_platform_doc_tools(self):
        """Setup platform documentation search tools."""
        try:
            self._platform_doc_tool = PlatformDocSearchTool(self.agent_config)
            self.tools.extend(self._platform_doc_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup platform_doc_search tools: {e}")

    def _setup_skill_tools(self):
        """Setup skill discovery and loading tools with permission control."""
        try:
            from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule

            base_config = self.agent_config.permissions_config
            if base_config is not None:
                base_config = base_config.model_copy(deep=True)
            else:
                base_config = PermissionConfig()

            # Add default ASK for skill_execute_command (bash) at position 0 (lowest priority)
            has_bash_rule = any(r.tool == "skills" and r.pattern == "skill_execute_command" for r in base_config.rules)
            if not has_bash_rule:
                base_config.rules.insert(
                    0,
                    PermissionRule(
                        tool="skills",
                        pattern="skill_execute_command",
                        permission=PermissionLevel.ASK,
                    ),
                )

            self.permission_manager = PermissionManager(
                global_config=base_config,
                node_overrides=self._get_node_permission_overrides(),
            )
            self.permission_manager.set_permission_callback(self._handle_permission_ask)

            skills_config = getattr(self.agent_config, "skills_config", None) if self.agent_config else None
            self.skill_manager = SkillManager(
                config=skills_config,
                permission_manager=self.permission_manager,
            )
            self.skill_func_tool = SkillFuncTool(
                manager=self.skill_manager,
                node_name="chat",
            )
            logger.debug(f"Setup skill tools: {self.skill_manager.get_skill_count()} skills discovered")
        except Exception as e:
            logger.error(f"Failed to setup skill tools: {e}")

    def _setup_permission_hooks(self):
        """Setup permission hooks and register all tool categories."""
        if not self.permission_manager:
            logger.debug("No permission manager available, skipping permission hooks setup")
            return

        try:
            # 1. Populate the node-level tool_registry
            if self.db_func_tool:
                self.tool_registry.register_tools("db_tools", self.db_func_tool.available_tools())
            if self.context_search_tools:
                self.tool_registry.register_tools("context_search_tools", self.context_search_tools.available_tools())
            if self.reference_template_tools:
                self.tool_registry.register_tools(
                    "reference_template_tools", self.reference_template_tools.available_tools()
                )
            if self.date_parsing_tools:
                self.tool_registry.register_tools("date_parsing_tools", self.date_parsing_tools.available_tools())
            if self.filesystem_func_tool:
                self.tool_registry.register_tools("filesystem_tools", self.filesystem_func_tool.available_tools())
            if self.skill_func_tool:
                self.tool_registry.register_tools("skills", self.skill_func_tool.available_tools())
            if self.sub_agent_task_tool:
                self.tool_registry.register_tools("sub_agent_tools", self.sub_agent_task_tool.available_tools())
            if self.ask_user_tool:
                self.tool_registry.register_tools("tools", self.ask_user_tool.available_tools())
            if self._platform_doc_tool:
                self.tool_registry.register_tools("tools", self._platform_doc_tool.available_tools())
            # 2. Create PermissionHooks sharing the same tool_registry instance
            broker = self._get_or_create_broker()
            self.permission_hooks = PermissionHooks(
                broker=broker,
                permission_manager=self.permission_manager,
                node_name=self.get_node_name(),
                tool_registry=self.tool_registry,
                fs_policy=self._make_filesystem_policy(),
            )

            logger.debug(f"Permission hooks setup with {len(self.tool_registry)} registered tools")
        except Exception as e:
            logger.error(f"Failed to setup permission hooks: {e}")
            self.permission_hooks = None

    def _rebuild_tools(self):
        """Rebuild the tools list with current tool instances including skills."""
        self.tools = []
        if self.db_func_tool:
            self.tools.extend(self.db_func_tool.available_tools())
        if self.context_search_tools:
            self.tools.extend(self.context_search_tools.available_tools())
        if self.reference_template_tools:
            self.tools.extend(self.reference_template_tools.available_tools())
        if self.date_parsing_tools:
            self.tools.extend(self.date_parsing_tools.available_tools())
        if self.filesystem_func_tool:
            self.tools.extend(self.filesystem_func_tool.available_tools())
        if self.skill_func_tool:
            self.tools.extend(self.skill_func_tool.available_tools())
        if self.sub_agent_task_tool:
            self.tools.extend(self.sub_agent_task_tool.available_tools())
        if self.ask_user_tool:
            self.tools.extend(self.ask_user_tool.available_tools())

    def _update_database_connection(self, database_name: str):
        """Update database connection to a different database."""
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_database, database_name)
        self.db_func_tool = DBFuncTool(conn, agent_config=self.agent_config)
        self._rebuild_tools()

    # ── Permission Helpers ──────────────────────────────────────────────

    async def _handle_permission_ask(
        self,
        tool_category: str,
        tool_name: str,
        context: dict,
    ) -> bool:
        """Handle ASK permission by prompting user for confirmation."""
        try:
            from rich.console import Console
            from rich.prompt import Confirm

            console = Console()
            console.print(f"\n[yellow]Permission required:[/yellow] {tool_category}.{tool_name}")
            if context:
                console.print(f"[dim]Context: {context}[/dim]")

            approved = Confirm.ask(f"Allow {tool_name}?", default=False)

            if approved:
                always = Confirm.ask("Always allow this session?", default=False)
                if always and self.permission_manager:
                    self.permission_manager.approve_for_session(tool_category, tool_name)

            return approved
        except Exception as e:
            logger.error(f"Permission prompt failed: {e}")
            return False

    def _get_node_permission_overrides(self) -> dict:
        """Get node-specific permission overrides from agent config."""
        if not self.agent_config:
            return {}

        chat_config = self.agent_config.agentic_nodes.get("chat", {})
        if isinstance(chat_config, dict) and "permissions" in chat_config:
            return {"chat": chat_config["permissions"]}

        return {}

    # ── MCP Servers ─────────────────────────────────────────────────────

    def _setup_mcp_servers(self) -> Dict[str, Any]:
        """Set up MCP servers based on configuration."""
        mcp_servers = {}

        config_value = self.node_config.get("mcp", "")
        if not config_value:
            return mcp_servers

        mcp_server_names = [p.strip() for p in config_value.split(",") if p.strip()]

        for server_name in mcp_server_names:
            try:
                if server_name == "metricflow_mcp":
                    server = self._setup_metricflow_mcp()
                    if server:
                        mcp_servers["metricflow_mcp"] = server
                        logger.info(
                            f"Setup metricflow_mcp MCP server for database: {self.agent_config.current_database}"
                        )
                    continue

                server = self._setup_mcp_server_from_config(server_name)
                if server:
                    mcp_servers[server_name] = server

            except Exception as e:
                logger.error(f"Failed to setup MCP server '{server_name}': {e}")

        logger.debug(f"Setup {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())}")
        return mcp_servers

    def _setup_metricflow_mcp(self) -> Optional[Any]:
        """Setup MetricFlow MCP server."""
        try:
            if not self.agent_config:
                return None

            db_config = self.agent_config.current_db_config()
            if not db_config:
                return None

            metricflow_server = MCPServer.get_metricflow_mcp_server(namespace=self.agent_config.current_database)
            if metricflow_server:
                logger.info(f"Added metricflow_mcp MCP server for database: {db_config.database}")
                return metricflow_server
        except Exception as e:
            logger.error(f"Failed to setup metricflow_mcp: {e}")
        return None

    def _setup_mcp_server_from_config(self, server_name: str) -> Optional[Any]:
        """Setup MCP server from {agent.home}/conf/.mcp.json using mcp_manager."""
        try:
            from datus.tools.mcp_tools.mcp_manager import MCPManager

            mcp_manager = MCPManager(agent_config=self.agent_config)
            server_config = mcp_manager.get_server_config(server_name)

            if not server_config:
                logger.warning(f"MCP server '{server_name}' not found in configuration")
                return None

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

    # ── System Prompt ───────────────────────────────────────────────────

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        """Get the system prompt using enhanced template context."""
        context = prepare_template_context(
            node_config=self.node_config,
            has_db_tools=bool(self.db_func_tool),
            has_filesystem_tools=bool(self.filesystem_func_tool),
            has_mf_tools=any("metricflow" in k for k in self.mcp_servers.keys()),
            has_context_search_tools=bool(self.context_search_tools),
            has_reference_template_tools=bool(
                self.reference_template_tools and self.reference_template_tools.has_reference_templates
            ),
            has_parsing_tools=bool(self.date_parsing_tools),
            has_platform_doc_tools=bool(self._platform_doc_tool),
            agent_config=self.agent_config,
            workspace_root=self._resolve_workspace_root(),
        )
        context["conversation_summary"] = conversation_summary
        context["has_task_tool"] = bool(self.sub_agent_task_tool)
        from datus.utils.time_utils import get_default_current_date

        context["current_date"] = get_default_current_date(None)
        prompt_version = prompt_version or self.node_config.get("prompt_version")

        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        from datus.prompts.prompt_manager import get_prompt_manager

        pm = get_prompt_manager(agent_config=self.agent_config)
        try:
            base_prompt = pm.render_template(template_name=template_name, version=prompt_version, **context)
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError:
            logger.warning(f"Failed to render system prompt '{system_prompt_name}', using the default template instead")
            base_prompt = pm.render_template(template_name="chat_system", version=None, **context)
            return self._finalize_system_prompt(base_prompt)
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    def _get_system_instruction(self, original_input: ChatNodeInput) -> str:
        """Get system instruction for normal mode."""
        _, conversation_summary = self._get_or_create_session()
        return self._get_system_prompt(conversation_summary, original_input.prompt_version)

    # ── Execution ───────────────────────────────────────────────────────

    def _get_execution_config(self, execution_mode: str, original_input) -> dict:
        """Get execution configuration with permission hooks."""
        if execution_mode == "normal":
            config = {
                "tools": self.tools,
                "instruction": self._get_system_instruction(original_input),
                "hooks": None,
            }
        elif execution_mode == "plan":
            plan_tools = self.plan_hooks.get_plan_tools() if self.plan_hooks else []
            base_instruction = self._get_system_instruction(original_input)
            config = {
                "tools": self.tools + plan_tools,
                "instruction": base_instruction,
                "hooks": self.plan_hooks,
            }
        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

        # Add permission hooks if available
        if self.permission_hooks:
            existing_hooks = config.get("hooks")
            if existing_hooks:
                config["hooks"] = CompositeHooks([existing_hooks, self.permission_hooks])
            else:
                config["hooks"] = self.permission_hooks

        return config

    def _build_plan_prompt(self, original_prompt: str) -> str:
        """Build enhanced prompt for plan mode based on current phase."""
        from datus.prompts.prompt_manager import get_prompt_manager

        current_phase = getattr(self.plan_hooks, "plan_phase", "generating") if self.plan_hooks else "generating"
        replan_feedback = getattr(self.plan_hooks, "replan_feedback", "") if self.plan_hooks else ""

        try:
            plan_prompt_addition = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name="plan_mode_system",
                version=None,
                current_phase=current_phase,
                replan_feedback=replan_feedback,
            )
        except FileNotFoundError:
            logger.warning("plan_mode_system template not found, using inline prompt")
            plan_prompt_addition = "\n\nPLAN MODE\nCheck todo_read to see current plan status and proceed accordingly."

        if is_structured_content(original_prompt):
            user_input = extract_user_input(original_prompt)
            enhanced = extract_enhanced_context(original_prompt)
            new_enhanced = (enhanced or "") + "\n\n" + plan_prompt_addition
            return build_structured_content(
                [
                    MessagePart(type="enhanced", content=new_enhanced),
                    MessagePart(type="user", content=user_input),
                ]
            )

        return original_prompt + "\n\n" + plan_prompt_addition

    async def _execute_with_recursive_replan(
        self,
        prompt: str,
        execution_mode: str,
        original_input,
        action_history_manager: ActionHistoryManager,
        session,
    ):
        """Unified recursive execution function that handles all execution modes."""
        logger.info(f"Executing mode: {execution_mode}")

        config = self._get_execution_config(execution_mode, original_input)

        if execution_mode == "plan" and self.plan_hooks:
            self.plan_hooks.plan_phase = "generating"

        try:
            final_prompt = prompt
            if execution_mode == "plan":
                final_prompt = self._build_plan_prompt(prompt)

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=final_prompt,
                tools=config["tools"],
                mcp_servers=self.mcp_servers,
                instruction=config["instruction"],
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=config.get("hooks"),
                agent_name=self.get_node_name(),
                interrupt_controller=self.interrupt_controller,
            ):
                yield stream_action

        except Exception as e:
            if "REPLAN_REQUIRED" in str(e):
                logger.info("Replan requested, recursing...")
                async for action in self._execute_with_recursive_replan(
                    prompt=prompt,
                    execution_mode=execution_mode,
                    original_input=original_input,
                    action_history_manager=action_history_manager,
                    session=session,
                ):
                    yield action
            else:
                raise

    # ── Workflow Integration ────────────────────────────────────────────

    def setup_input(self, workflow: Workflow) -> dict:
        """Setup chat input from workflow context."""
        task_database = workflow.task.database_name
        if task_database and self.db_func_tool and task_database != self.db_func_tool.connector.database_name:
            logger.debug(
                f"Updating database connection from '{self.db_func_tool.connector.database_name}' "
                f"to '{task_database}' based on workflow task"
            )
            self._update_database_connection(task_database)

        plan_mode = workflow.metadata.get("plan_mode", False)
        auto_execute_plan = workflow.metadata.get("auto_execute_plan", False)

        if not self.input:
            self.input = ChatNodeInput(
                user_message=workflow.task.task,
                external_knowledge=workflow.task.external_knowledge,
                catalog=workflow.task.catalog_name,
                database=workflow.task.database_name,
                db_schema=workflow.task.schema_name,
                schemas=workflow.context.table_schemas,
                metrics=workflow.context.metrics,
                reference_sql=None,
                plan_mode=plan_mode,
                auto_execute_plan=auto_execute_plan,
                prompt_version=self.node_config.get("prompt_version"),
            )
        else:
            self.input.user_message = workflow.task.task
            self.input.external_knowledge = workflow.task.external_knowledge
            self.input.catalog = workflow.task.catalog_name
            self.input.database = workflow.task.database_name
            self.input.db_schema = workflow.task.schema_name
            self.input.schemas = workflow.context.table_schemas
            self.input.metrics = workflow.context.metrics

        return {"success": True, "message": "Chat input prepared from workflow"}

    def update_context(self, workflow: Workflow) -> dict:
        """Update workflow context with chat results. Chat node produces markdown, no SQL."""
        if not self.result:
            return {"success": False, "message": "No result to update context"}

        return {"success": True, "message": "Updated chat context"}

    # ── execute_stream ──────────────────────────────────────────────────

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the chat interaction with streaming support.

        Produces markdown output directly — no SQL extraction or JSON parsing.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        if not self.input:
            raise ValueError("Chat input not set. Call setup_input() first or set self.input directly.")

        user_input = self.input

        is_plan_mode = getattr(user_input, "plan_mode", False)
        if is_plan_mode:
            self.plan_mode_active = True

            from datus.cli.plan_hooks import PlanModeHooks

            broker = self._get_or_create_broker()
            session = self._get_or_create_session()[0]

            auto_mode = getattr(user_input, "auto_execute_plan", False)
            logger.info(f"Plan mode auto_mode: {auto_mode} (from input)")

            self.plan_hooks = PlanModeHooks(broker=broker, session=session, auto_mode=auto_mode)

        # Create initial action
        action_type = "plan_mode_interaction" if is_plan_mode else "chat_interaction"
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=action_type,
            messages=f"User: {user_input.user_message}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        yield action

        try:
            await self._auto_compact()

            session, conversation_summary = self._get_or_create_session()

            # Build enhanced message with database context
            enhanced_message = build_enhanced_message(
                user_message=user_input.user_message,
                db_type=self.agent_config.db_type,
                catalog=user_input.catalog,
                database=user_input.database,
                db_schema=user_input.db_schema,
                external_knowledge=user_input.external_knowledge,
                schemas=user_input.schemas,
                metrics=user_input.metrics,
                reference_sql=user_input.reference_sql,
            )

            response_content = ""
            tokens_used = 0
            last_successful_output = None

            execution_mode = "plan" if is_plan_mode and self.plan_hooks else "normal"

            async for stream_action in self._execute_with_recursive_replan(
                prompt=enhanced_message,
                execution_mode=execution_mode,
                original_input=user_input,
                action_history_manager=action_history_manager,
                session=session,
            ):
                yield stream_action

                # Collect response content from successful actions
                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        raw_output_value = ""
                        if stream_action.action_type == "message" and "raw_output" in stream_action.output:
                            raw_output_value = stream_action.output.get("raw_output", "")

                        # Extract string content only; dict values from tool results must not leak through
                        candidate = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or raw_output_value
                        )
                        if isinstance(candidate, str) and candidate:
                            response_content = candidate
                        elif candidate and not isinstance(candidate, str):
                            response_content = str(candidate)

            # Fallback: extract from last successful output
            if not response_content and last_successful_output:
                logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
                candidate = (
                    last_successful_output.get("content", "")
                    or last_successful_output.get("text", "")
                    or last_successful_output.get("response", "")
                    or last_successful_output.get("raw_output", "")
                )
                if isinstance(candidate, str) and candidate:
                    response_content = candidate
                elif candidate and not isinstance(candidate, str):
                    response_content = str(candidate)

            # Check summary_report actions for content if still empty
            if not response_content:
                for stream_action in reversed(action_history_manager.get_actions()):
                    if stream_action.action_type == "summary_report" and stream_action.output:
                        if isinstance(stream_action.output, dict):
                            candidate = (
                                stream_action.output.get("markdown", "")
                                or stream_action.output.get("content", "")
                                or stream_action.output.get("response", "")
                            )
                            if candidate:
                                response_content = str(candidate) if not isinstance(candidate, str) else candidate
                                break

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")

            # Extract token usage
            final_actions = action_history_manager.get_actions()
            tokens_used = 0

            for action in reversed(final_actions):
                if action.role == "assistant":
                    if action.output and isinstance(action.output, dict):
                        usage_info = action.output.get("usage", {})
                        if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                            tokens_used = usage_info.get("total_tokens", 0)
                            if tokens_used > 0:
                                break
                            else:
                                logger.warning(f"no usage token found in this action {action.messages}")

            # Collect execution stats
            all_actions = action_history_manager.get_actions()
            tool_calls = [
                action
                for action in all_actions
                if action.role == ActionRole.TOOL and action.status == ActionStatus.SUCCESS
            ]

            execution_stats = {
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": list(set([a.action_type for a in tool_calls])),
                "total_tokens": int(tokens_used),
            }

            # Create final result — markdown output, no SQL
            result = ChatNodeResult(
                success=True,
                response=response_content,
                tokens_used=int(tokens_used),
                action_history=[action.model_dump() for action in all_actions],
                execution_stats=execution_stats,
            )

            self.actions.extend(action_history_manager.get_actions())

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="chat_response",
                messages="Chat interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise

        except Exception as e:
            if "User cancelled" in str(e) or "UserCancelledException" in str(type(e).__name__):
                logger.info("User cancelled execution, stopping gracefully...")

                result = ChatNodeResult(
                    success=True,
                    response="Execution cancelled by user.",
                    tokens_used=0,
                )

                action_history_manager.update_current_action(
                    status=ActionStatus.SUCCESS,
                    output=result.model_dump(),
                    messages="Execution cancelled by user",
                )

                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="user_cancellation",
                    messages="Execution cancelled by user",
                    input_data=user_input.model_dump(),
                    output_data=result.model_dump(),
                    status=ActionStatus.SUCCESS,
                )
            else:
                logger.error(f"Chat execution error: {e}")

                result = ChatNodeResult(
                    success=False,
                    error=str(e),
                    response="Sorry, I encountered an error while processing your request.",
                    tokens_used=0,
                )

                action_history_manager.update_current_action(
                    status=ActionStatus.FAILED,
                    output=result.model_dump(),
                    messages=f"Error: {str(e)}",
                )

                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="error",
                    messages=f"Chat interaction failed: {str(e)}",
                    input_data=user_input.model_dump(),
                    output_data=result.model_dump(),
                    status=ActionStatus.FAILED,
                )

            action_history_manager.add_action(action)
            yield action

        finally:
            if is_plan_mode:
                self.plan_mode_active = False
                self.plan_hooks = None
