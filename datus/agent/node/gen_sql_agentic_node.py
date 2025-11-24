# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenSQLAgenticNode implementation for SQL generation with enhanced configuration.

This module provides a specialized implementation of AgenticNode focused on
SQL generation with support for limited context, enhanced template variables,
and flexible configuration through agent.yml.
"""

import os
from typing import Any, AsyncGenerator, Dict, Optional, Union

from agents.mcp import MCPServerStdio

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.agent_models import SubAgentConfig
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput, GenSQLNodeResult
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import ContextSearchTools, DBFuncTool
from datus.tools.func_tool.date_parsing_tools import DateParsingTools
from datus.tools.mcp_tools.mcp_server import MCPServer
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
        max_turns: int = 30,
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
            max_turns: Maximum conversation turns per interaction
        """
        # Determine node name from node_type if not provided
        self.configured_node_name = node_name
        self.max_turns = max_turns

        # Initialize tool attributes BEFORE calling parent constructor
        # This is required because parent's __init__ calls _get_system_prompt()
        # which may reference these attributes
        self.db_func_tool: Optional[DBFuncTool] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.date_parsing_tools: Optional[DateParsingTools] = None

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
            logger.info(
                f"Updating database connection from '{self.db_func_tool.connector.database_name}' "
                f"to '{task_database}' based on workflow task"
            )
            self._update_database_connection(task_database)

        # Create GenSQLNodeInput if not already set
        if not self.input or not isinstance(self.input, GenSQLNodeInput):
            self.input = GenSQLNodeInput(
                user_message=workflow.task.task,
                external_knowledge=workflow.task.external_knowledge,
                catalog=workflow.task.catalog_name,
                database=workflow.task.database_name,
                db_schema=workflow.task.schema_name,
                schemas=workflow.context.table_schemas,
                metrics=workflow.context.metrics,
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

        logger.info(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

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
                else:
                    logger.warning(f"Unknown tool type: {base_type}")

            # Handle exact type patterns (e.g., "db_tools")
            elif pattern == "db_tools":
                self._setup_db_tools()
            elif pattern == "context_search_tools":
                self._setup_context_search_tools()
            elif pattern == "date_parsing_tools":
                self._setup_date_parsing_tools()

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

    def _setup_filesystem_mcp(self) -> Optional[MCPServerStdio]:
        """Setup filesystem MCP server."""
        try:
            root_path = self._resolve_workspace_root()
            # Handle relative vs absolute paths
            if root_path and os.path.isabs(root_path):
                filesystem_path = root_path
            else:
                filesystem_path = os.path.join(os.getcwd(), root_path)

            filesystem_server = MCPServer.get_filesystem_mcp_server(path=filesystem_path)
            if filesystem_server:
                logger.info(f"Added filesystem MCP server at path: {filesystem_path}")
                return filesystem_server
            else:
                logger.warning(f"Failed to create filesystem MCP server for path: {filesystem_path}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem MCP server: {e}")
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
                logger.info(f"Added MCP server '{server_name}' from configuration: {details}")
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
                # Handle filesystem_mcp
                if server_name == "filesystem_mcp":
                    server = self._setup_filesystem_mcp()
                    if server:
                        mcp_servers["filesystem_mcp"] = server

                # Handle MCP servers from {agent.home}/conf/.mcp.json using mcp_manager
                else:
                    server = self._setup_mcp_server_from_config(server_name)
                    if server:
                        mcp_servers[server_name] = server

            except Exception as e:
                logger.error(f"Failed to setup MCP server '{server_name}': {e}")

        logger.info(f"Setup {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())}")

        # Debug: Log detailed info about each server
        for name, server in mcp_servers.items():
            logger.debug(f"MCP server '{name}': type={type(server)}, instance={server}")

        return mcp_servers

    def _get_system_prompt(
        self, conversation_summary: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> str:
        """
        Get the system prompt for this SQL generation node using enhanced template context.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use, overrides agent config version
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """
        context = prepare_template_context(
            node_config=self.node_config,
            has_db_tools=bool(self.db_func_tool),
            has_mcp_filesystem="filesystem_mcp" in self.mcp_servers,
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

        try:
            # Check for auto-compact before session creation to ensure fresh context
            await self._auto_compact()

            # Get or create session and any available summary
            session, conversation_summary = self._get_or_create_session()

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

            # Stream response using the model's generate_with_tools_stream
            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
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
            logger.info(
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
    has_mcp_filesystem: bool = True,
    has_mf_tools: bool = True,
    has_context_search_tools: bool = True,
    has_parsing_tools: bool = True,
    agent_config: Optional[AgentConfig] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """
    Prepare template context variables for the gen_sql_system template.

    Args:
        user_input: User input containing limited context settings

    Returns:
        Dictionary of template variables
    """
    context: Dict[str, Any] = {
        "has_db_tools": has_db_tools,
        "has_mcp_filesystem": has_mcp_filesystem,
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
        enhanced_parts.append(f"### External Knowledge (AUTHORITATIVE)\n{external_knowledge}")

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
        table_schemas_str = TableSchema.list_to_prompt(schemas, dialect=db_type)
        enhanced_parts.append(f"Table Schemas: \n{table_schemas_str}")
    if metrics:
        enhanced_parts.append(f"Metrics: \n{to_str([item.model_dump() for item in metrics])}")

    if reference_sql:
        enhanced_parts.append(f"Reference SQL: \n{to_str([item.model_dump() for item in reference_sql])}")

    if enhanced_parts:
        enhanced_message = f"{'\n\n'.join(enhanced_parts)}\n\nUser question: {user_message}"

    return enhanced_message
