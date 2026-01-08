# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenSQLAgenticNode implementation for SQL generation with enhanced configuration.

This module provides a specialized implementation of AgenticNode focused on
SQL generation with support for limited context, enhanced template variables,
and flexible configuration through agent.yml.
"""

from typing import Any, AsyncGenerator, Dict, Optional, Union

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.agent_models import SubAgentConfig
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput, GenSQLNodeResult
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, FilesystemFuncTool
from datus.tools.func_tool.date_parsing_tools import DateParsingTools
from datus.tools.mcp_tools import MCPServer
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import to_str
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

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[GenSQLNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
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
        """
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

        # Initialize plan mode attributes
        self.plan_mode_active = False
        self.plan_hooks = None

        # Call parent constructor with all required Node parameters
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},  # Initialize empty, will setup after parent init
        )

        # Initialize MCP servers based on configuration (after node_config is available)
        self.mcp_servers = self._setup_mcp_servers()

        # Debug: Log final MCP servers assignment
        logger.debug(
            f"GenSQLAgenticNode final mcp_servers: {len(self.mcp_servers)} servers - {list(self.mcp_servers.keys())}"
        )

        # Setup tools based on configuration
        self.setup_tools()
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
            self.input.plan_mode = plan_mode
            self.input.auto_execute_plan = auto_execute_plan

        return {"success": True, "message": "GenSQL input prepared from workflow"}

    def _update_database_connection(self, database_name: str):
        """
        Update database connection to a different database.

        Args:
            database_name: The name of the database to connect to
        """
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_namespace, database_name)
        self.db_func_tool = DBFuncTool(
            conn,
            agent_config=self.agent_config,
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
        if self.date_parsing_tools:
            self.tools.extend(self.date_parsing_tools.available_tools())
        if self.filesystem_func_tool:
            self.tools.extend(self.filesystem_func_tool.available_tools())

    def setup_tools(self):
        """Setup tools based on configuration."""
        if not self.agent_config:
            return

        self.tools = []
        config_value = self.node_config.get("tools", "")
        if not config_value:
            return

        tool_patterns = [p.strip() for p in config_value.split(",") if p.strip()]
        for pattern in tool_patterns:
            self._setup_tool_pattern(pattern)

        logger.debug(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

    def _setup_db_tools(self):
        """Setup database tools."""
        try:
            db_manager = db_manager_instance(self.agent_config.namespaces)
            conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
            self.db_func_tool = DBFuncTool(
                conn,
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
            )
            self.tools.extend(self.db_func_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_context_search_tools(self):
        """Setup context search tools."""
        try:
            self.context_search_tools = ContextSearchTools(
                self.agent_config, sub_agent_name=self.node_config["system_prompt"]
            )
            self.tools.extend(self.context_search_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup context search tools: {e}")

    def _setup_date_parsing_tools(self):
        """Setup date parsing tools."""
        try:
            self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            self.tools.extend(self.date_parsing_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup date parsing tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools (all available tools)."""
        try:
            root_path = self._resolve_workspace_root()
            self.filesystem_func_tool = FilesystemFuncTool(root_path=root_path)
            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug(f"Setup filesystem tools with root path: {root_path}")
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
                elif base_type == "date_parsing_tools":
                    self._setup_date_parsing_tools()
                elif base_type == "filesystem_tools":
                    self._setup_filesystem_tools()
                else:
                    logger.warning(f"Unknown tool type: {base_type}")

            # Handle exact type patterns (e.g., "db_tools")
            elif pattern == "db_tools":
                self._setup_db_tools()
            elif pattern == "context_search_tools":
                self._setup_context_search_tools()
            elif pattern == "date_parsing_tools":
                self._setup_date_parsing_tools()
            elif pattern == "filesystem_tools":
                self._setup_filesystem_tools()

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
            elif tool_type == "db_tools":
                if not self.db_func_tool:
                    db_manager = db_manager_instance(self.agent_config.namespaces)
                    conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
                    self.db_func_tool = DBFuncTool(
                        conn,
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
                    root_path = self._resolve_workspace_root()
                    self.filesystem_func_tool = FilesystemFuncTool(root_path=root_path)
                tool_instance = self.filesystem_func_tool
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                return

            if hasattr(tool_instance, method_name):
                method = getattr(tool_instance, method_name)
                from datus.tools.func_tool import trans_to_function_tool

                self.tools.append(trans_to_function_tool(method))
                logger.debug(f"Added specific tool method: {tool_type}.{method_name}")
            else:
                logger.warning(f"Method '{method_name}' not found in {tool_type}")
        except Exception as e:
            logger.error(f"Failed to setup {tool_type}.{method_name}: {e}")

    def _setup_metricflow_mcp(self) -> Optional[Any]:
        """Setup MetricFlow MCP server."""
        try:
            if not self.agent_config:
                logger.warning("Agent config not available for metricflow MCP setup")
                return None

            # Get current database config
            db_config = self.agent_config.current_db_config()
            if not db_config:
                logger.warning("Database config not found")
                return None

            metricflow_server = MCPServer.get_metricflow_mcp_server(namespace=self.agent_config.current_namespace)
            if metricflow_server:
                logger.info(f"Added metricflow_mcp MCP server for database: {db_config.database}")
                return metricflow_server
            else:
                logger.warning(f"Failed to create metricflow MCP server for db_config: {db_config}")
        except Exception as e:
            logger.error(f"Failed to setup metricflow_mcp: {e}")
        return None

    def _setup_mcp_server_from_config(self, server_name: str) -> Optional[Any]:
        """Setup MCP server from {agent.home}/conf/.mcp.json using mcp_manager."""
        try:
            from datus.tools.mcp_tools.mcp_manager import MCPManager

            # Use MCPManager to get server config
            mcp_manager = MCPManager()
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
                # Handle metricflow_mcp
                if server_name == "metricflow_mcp":
                    server = self._setup_metricflow_mcp()
                    if server:
                        mcp_servers["metricflow_mcp"] = server
                        logger.info(
                            f"Setup metricflow_mcp MCP server for database: {self.agent_config.current_database}"
                        )

                # Handle MCP servers from {agent.home}/conf/.mcp.json using mcp_manager
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

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Get the system prompt for this SQL generation node using enhanced template context.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use, overrides agent config version

        Returns:
            System prompt string loaded from the template
        """
        context = prepare_template_context(
            node_config=self.node_config,
            has_db_tools=bool(self.db_func_tool),
            has_filesystem_tools=bool(self.filesystem_func_tool),
            has_mf_tools=any("metricflow" in k for k in self.mcp_servers.keys()),
            has_context_search_tools=bool(self.context_search_tools),
            has_parsing_tools=bool(self.date_parsing_tools),
            agent_config=self.agent_config,
            workspace_root=self._resolve_workspace_root(),
        )
        context["conversation_summary"] = conversation_summary

        raw_version = prompt_version if prompt_version is not None else self.node_config.get("prompt_version")
        version = None if raw_version in (None, "") else str(raw_version)
        # Construct template name: {system_prompt}_system or fallback to {node_name}_system
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        # Use prompt manager to render the template
        from datus.prompts.prompt_manager import prompt_manager

        try:
            return prompt_manager.render_template(template_name=template_name, version=version, **context)

        except FileNotFoundError:
            # Template not found - throw DatusException
            logger.warning(f"Failed to render system prompt '{system_prompt_name}', using the default template instead")
            return prompt_manager.render_template(template_name="sql_system", version=version, **context)
        except Exception as e:
            # Other template errors - wrap in DatusException
            logger.error(f"Template loading error for '{template_name}': {e}")

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the customized node interaction with streaming support.

        Input is accessed from self.input instead of parameters.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Get input from self.input (set by setup_input or directly)
        if not self.input:
            raise ValueError("GenSQL input not set. Call setup_input() first or set self.input directly.")

        user_input = self.input

        # Create initial action
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=self.get_node_name(),
            messages=f"User: {user_input.user_message}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        yield action

        # Track plan mode state for cleanup
        is_plan_mode = False

        try:
            # Check for auto-compact before session creation to ensure fresh context
            await self._auto_compact()

            # Get or create session and any available summary
            session, conversation_summary = self._get_or_create_session()

            # Check for plan mode activation
            is_plan_mode = getattr(user_input, "plan_mode", False)
            if is_plan_mode:
                self.plan_mode_active = True
                from rich.console import Console

                from datus.cli.plan_hooks import PlanModeHooks

                auto_mode = getattr(user_input, "auto_execute_plan", False)
                self.plan_hooks = PlanModeHooks(console=Console(), session=session, auto_mode=auto_mode)
                logger.info(f"Plan mode activated (auto_mode={auto_mode})")

            system_instruction = self._get_system_prompt(conversation_summary, user_input.prompt_version)

            # Add context to user message if provided
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

            # Execute with streaming
            response_content = ""
            sql_content = None
            tokens_used = 0
            last_successful_output = None

            # Create assistant action for processing
            assistant_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="llm_generation",
                messages="Generating response with tools...",
                input_data={"prompt": enhanced_message, "system": system_instruction},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(assistant_action)
            yield assistant_action

            logger.debug(f"Tools available : {len(self.tools)} tools - {[tool.name for tool in self.tools]}")
            logger.debug(f"MCP servers available : {len(self.mcp_servers)} servers - {list(self.mcp_servers.keys())}")

            # Choose execution mode based on plan_mode flag
            execution_mode = "plan" if is_plan_mode else "normal"

            # Stream response using unified execution (supports plan mode and normal mode)
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
                        # Look for content in various possible fields
                        response_content = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or stream_action.output.get("raw_output", "")
                            or response_content
                        )

            # If we still don't have response_content, check the last successful output
            if not response_content and last_successful_output:
                logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
                # Try different fields that might contain the response
                response_content = (
                    last_successful_output.get("content", "")
                    or last_successful_output.get("text", "")
                    or last_successful_output.get("response", "")
                    or last_successful_output.get("raw_output", "")
                    or str(last_successful_output)  # Fallback to string representation
                )

            # Extract SQL directly from summary_report action if available
            sql_content = None
            for stream_action in reversed(action_history_manager.get_actions()):
                if stream_action.action_type == "summary_report" and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        sql_content = stream_action.output.get("sql")
                        # Also get the markdown/content if response_content is still empty
                        if not response_content:
                            response_content = (
                                stream_action.output.get("markdown", "")
                                or stream_action.output.get("content", "")
                                or stream_action.output.get("response", "")
                            )
                        if sql_content:  # Found SQL, stop searching
                            logger.debug(f"Extracted SQL from summary_report action: {sql_content[:100]}...")
                            break

            # Fallback: try to extract SQL and output from response_content if not found
            if not sql_content:
                extracted_sql, extracted_output = self._extract_sql_and_output_from_response(
                    {"content": response_content}
                )
                if extracted_sql:
                    sql_content = extracted_sql
                if extracted_output:
                    response_content = extracted_output

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")
            logger.debug(f"Final sql_content: {sql_content[:100] if sql_content else 'None'}...")

            # Extract token usage from final actions
            final_actions = action_history_manager.get_actions()
            tokens_used = 0

            # Find the final assistant action with token usage
            for action in reversed(final_actions):
                if action.role == "assistant":
                    if action.output and isinstance(action.output, dict):
                        usage_info = action.output.get("usage", {})
                        if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                            conversation_tokens = usage_info.get("total_tokens", 0)
                            if conversation_tokens > 0:
                                # Add this conversation's tokens to the session
                                self._add_session_tokens(conversation_tokens)
                                tokens_used = conversation_tokens
                                logger.info(f"Added {conversation_tokens} tokens to session")
                                break
                            else:
                                logger.warning(f"no usage token found in this action {action.messages}")

            # Collect action history and calculate execution stats
            all_actions = action_history_manager.get_actions()
            tool_calls = [action for action in all_actions if action.role == ActionRole.TOOL]

            execution_stats = {
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": list(set([a.action_type for a in tool_calls])),
                "total_tokens": int(tokens_used),
            }

            # Create final result with action history
            result = GenSQLNodeResult(
                success=True,
                response=response_content,
                sql=sql_content,
                tokens_used=int(tokens_used),
                action_history=[action.model_dump() for action in all_actions],
                execution_stats=execution_stats,
            )

            # Add to internal actions list
            self.actions.extend(all_actions)

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=f"{self.get_node_name()}_response",
                messages=f"{self.get_node_name()} interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except Exception as e:
            logger.error(f"{self.get_node_name()} execution error: {e}")

            # Create error result
            error_result = GenSQLNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while processing your request.",
                tokens_used=0,
            )

            # Update action with error
            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {str(e)}",
            )

            # Create error action
            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="error",
                messages=f"{self.get_node_name()} interaction failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

        finally:
            # Clean up plan mode state
            if is_plan_mode:
                self.plan_mode_active = False
                self.plan_hooks = None
                logger.info("Plan mode deactivated")

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

    async def _execute_with_recursive_replan(
        self,
        prompt: str,
        execution_mode: str,
        original_input: GenSQLNodeInput,
        action_history_manager: ActionHistoryManager,
        session,
    ):
        """
        Unified recursive execution function that handles all execution modes.

        Args:
            prompt: The prompt to send to LLM
            execution_mode: "normal", "plan", or "replan"
            original_input: Original SQL generation input for context
            action_history_manager: Action history manager
            session: SQL generation session
        """
        logger.info(f"Executing mode: {execution_mode}")

        # Get execution configuration for this mode
        config = self._get_execution_config(execution_mode, original_input)

        # Reset state for replan mode
        if execution_mode == "plan" and self.plan_hooks:
            self.plan_hooks.plan_phase = "generating"

        try:
            # Build enhanced prompt for plan mode
            final_prompt = prompt
            if execution_mode == "plan":
                final_prompt = self._build_plan_prompt(prompt)

            # Unified execution using configuration
            async for stream_action in self.model.generate_with_tools_stream(
                prompt=final_prompt,
                tools=config["tools"],
                mcp_servers=self.mcp_servers,
                instruction=config["instruction"],
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=config.get("hooks"),
            ):
                yield stream_action

        except Exception as e:
            if "REPLAN_REQUIRED" in str(e):
                logger.info("Replan requested, recursing...")

                # Recursive call - enter replan mode with original user prompt
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

    def _get_execution_config(self, execution_mode: str, original_input: GenSQLNodeInput) -> dict:
        """
        Get execution configuration based on mode.

        Args:
            execution_mode: "normal", "plan"
            original_input: Original SQL generation input for context

        Returns:
            Configuration dict with tools, instruction, and hooks
        """
        if execution_mode == "normal":
            return {"tools": self.tools, "instruction": self._get_system_instruction(original_input), "hooks": None}
        elif execution_mode == "plan":
            # Plan mode: standard tools + plan tools
            plan_tools = self.plan_hooks.get_plan_tools() if self.plan_hooks else []

            # Use base instruction (chat_system.j2) which contains tool usage rules
            base_instruction = self._get_system_instruction(original_input)

            return {
                "tools": self.tools + plan_tools,
                "instruction": base_instruction,
                "hooks": self.plan_hooks,
            }
        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

    def _get_system_instruction(self, original_input: GenSQLNodeInput) -> str:
        """Get system instruction for normal mode."""
        _, conversation_summary = self._get_or_create_session()
        return self._get_system_prompt(conversation_summary, original_input.prompt_version)

    def _build_plan_prompt(self, original_prompt: str) -> str:
        """Build enhanced prompt for plan mode based on current phase."""
        from datus.prompts.prompt_manager import prompt_manager

        # Check current phase and replan feedback
        current_phase = getattr(self.plan_hooks, "plan_phase", "generating") if self.plan_hooks else "generating"
        replan_feedback = getattr(self.plan_hooks, "replan_feedback", "") if self.plan_hooks else ""

        # Load plan mode prompt from template
        try:
            plan_prompt_addition = prompt_manager.render_template(
                template_name="plan_mode_system",
                version=None,  # Use latest version
                current_phase=current_phase,
                replan_feedback=replan_feedback,
            )
        except FileNotFoundError:
            # Fallback to inline prompt if template not found
            logger.warning("plan_mode_system template not found, using inline prompt")
            plan_prompt_addition = "\n\nPLAN MODE\nCheck todo_read to see current plan status and proceed accordingly."

        return original_prompt + "\n\n" + plan_prompt_addition

    def _extract_sql_and_output_from_response(self, output: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extract SQL content and formatted output from model response.

        Uses the existing llm_result2json utility for robust JSON parsing.
        Handles the expected template format: {"sql": "...", "tables": [...], "explanation": "..."}

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

                # Build output from explanation and tables if available
                output_text = None
                explanation = parsed.get("explanation", "")
                tables = parsed.get("tables", [])

                # If we have explanation or tables, format them as output
                if explanation or tables:
                    output_parts = []
                    if explanation:
                        output_parts.append(f"Explanation: {explanation}")
                    if tables:
                        tables_str = ", ".join(tables) if isinstance(tables, list) else str(tables)
                        output_parts.append(f"Tables used: {tables_str}")
                    output_text = "\n".join(output_parts)

                # Fallback to direct output field if no explanation/tables
                if not output_text:
                    output_text = parsed.get("output")

                # Unescape output content if present
                if output_text and isinstance(output_text, str):
                    output_text = output_text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

                return sql, output_text

            return None, None
        except Exception as e:
            logger.warning(f"Failed to extract SQL and output from response: {e}")
            return None, None


def prepare_template_context(
    node_config: Union[Dict[str, Any], SubAgentConfig],
    has_db_tools: bool = True,
    has_filesystem_tools: bool = True,
    has_mf_tools: bool = True,
    has_context_search_tools: bool = True,
    has_parsing_tools: bool = True,
    agent_config: Optional[AgentConfig] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """
    Prepare template context variables for the gen_sql_system template.

    Args:
        node_config: Node configuration
        has_db_tools: Whether database tools are available
        has_filesystem_tools: Whether filesystem tools are available
        has_mf_tools: Whether MetricFlow MCP tools are available
        has_context_search_tools: Whether context search tools are available
        has_parsing_tools: Whether date parsing tools are available
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
        "has_parsing_tools": has_parsing_tools,
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
        has_scoped_context = bool(scoped_context.tables or scoped_context.metrics or scoped_context.sqls)

    context["scoped_context"] = has_scoped_context

    # Add rules from configuration
    context["rules"] = node_config.rules or []

    # Add agent description from configuration or input
    context["agent_description"] = node_config.agent_description

    # Add namespace and workspace info
    if agent_config:
        context["agent_config"] = agent_config
        context["namespace"] = getattr(agent_config, "current_namespace", None)
        context["db_name"] = getattr(agent_config, "current_database", None)
        context["workspace_root"] = workspace_root or agent_config.workspace_root
    logger.debug(f"Prepared template context: {context}")
    return context


def build_enhanced_message(
    user_message: str,
    db_type: str,
    catalog: str = "",
    database: str = "",
    db_schema: str = "",
    external_knowledge: str = "",
    schemas: Optional[list[TableSchema]] = None,
    metrics: Optional[list[Metric]] = None,
    reference_sql: Optional[list[ReferenceSql]] = None,
) -> str:
    enhanced_message = user_message
    enhanced_parts = []
    if external_knowledge:
        enhanced_parts.append(f"MUST use these business logic:\n{external_knowledge}")

    context_parts = [f"**Dialect**: {db_type}"]
    if catalog:
        context_parts.append(f"**Catalog**: {catalog}")
    if database:
        context_parts.append(f"**Database**: {database}")
    if db_schema:
        context_parts.append(f"**Schema**: {db_schema}")
    context_part_str = f'Context: \n{", ".join(context_parts)}'
    enhanced_parts.append(context_part_str)

    if schemas:
        table_names_str = TableSchema.table_names_to_prompt(schemas)
        enhanced_parts.append(
            f"Available tables (MUST use these tables and ONLY use these table names in FROM/JOIN clauses):"
            f" \n{table_names_str}"
        )
    if metrics:
        enhanced_parts.append(f"Metrics: \n{to_str([item.model_dump() for item in metrics])}")

    if reference_sql:
        enhanced_parts.append(f"Reference SQL: \n{to_str([item.model_dump() for item in reference_sql])}")

    if enhanced_parts:
        enhanced_message = (
            f"{'\n\n'.join(enhanced_parts)}\n\nNow based on the rules above, answer the user question: {user_message}"
        )

    return enhanced_message
