# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ChatAgenticNode implementation for flexible CLI chat interactions.

This module provides a concrete implementation of AgenticNode specifically
designed for chat interactions with database and filesystem tool support.
"""
from typing import AsyncGenerator, Dict, Optional

from agents.mcp import MCPServerStdio

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import ContextSearchTools, DateParsingTools, DBFuncTool, FilesystemFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ChatAgenticNode(AgenticNode):
    """
    Chat-focused agentic node with database and filesystem tool support.

    This node provides flexible chat capabilities with:
    - Namespace-based database MCP server selection
    - Default filesystem MCP server
    - Streaming response generation
    - Session-based conversation management
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ChatNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        """
        Initialize the ChatAgenticNode as a workflow-compatible node.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (should be 'chat')
            input_data: Chat input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
        """
        # Extract namespace from agent_config
        namespace = agent_config.current_namespace if agent_config else None
        self.namespace = namespace

        # Get max_turns from nodes configuration, default to 30
        self.max_turns = 30
        if agent_config and hasattr(agent_config, "nodes") and "chat" in agent_config.nodes:
            chat_node_config = agent_config.nodes["chat"]
            if (
                chat_node_config.input
                and hasattr(chat_node_config.input, "max_turns")
                and chat_node_config.input.max_turns is not None
            ):
                self.max_turns = chat_node_config.input.max_turns

        # Initialize MCP servers based on namespace
        mcp_servers = self._setup_mcp_servers(agent_config)

        # Call parent constructor with all required Node parameters
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers=mcp_servers,
        )

        # ChatAgenticNode-specific attributes
        self.db_func_tool: DBFuncTool
        self.context_search_tools: ContextSearchTools
        self.date_parsing_tools: Optional[DateParsingTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.plan_mode_active = False
        self.plan_hooks = None

        # Setup tools after initialization
        self.setup_tools()

    def setup_input(self, workflow: Workflow) -> dict:
        """
        Setup chat input from workflow context.

        Creates ChatNodeInput with user message from task and context data.

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

        # Read plan_mode from workflow metadata
        plan_mode = workflow.metadata.get("plan_mode", False)
        auto_execute_plan = workflow.metadata.get("auto_execute_plan", False)

        # Create ChatNodeInput if not already set
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

        return {"success": True, "message": "Chat input prepared from workflow"}

    def update_context(self, workflow: Workflow) -> dict:
        """
        Update workflow context with chat results.

        Stores SQL to workflow context if present in result.

        Args:
            workflow: Workflow instance to update

        Returns:
            Dictionary with success status and message
        """
        if not self.result:
            return {"success": False, "message": "No result to update context"}

        result = self.result

        try:
            if hasattr(result, "sql") and result.sql:
                from datus.schemas.node_models import SQLContext

                # Extract SQL result from the response if available
                sql_result = ""
                if hasattr(result, "response") and result.response:
                    # Try to extract SQL result from the response
                    _, sql_result = self._extract_sql_and_output_from_response({"content": result.response})
                    sql_result = sql_result or ""

                new_record = SQLContext(
                    sql_query=result.sql,
                    explanation=result.response if hasattr(result, "response") else "",
                    sql_return=sql_result,
                )
                workflow.context.sql_contexts.append(new_record)

            return {"success": True, "message": "Updated chat context"}
        except Exception as e:
            logger.error(f"Failed to update chat context: {e}")
            return {"success": False, "message": str(e)}

    def _update_database_connection(self, database_name: str):
        """
        Update database connection to a different database.

        Args:
            database_name: The name of the database to connect to
        """
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_namespace, database_name)
        self.db_func_tool = DBFuncTool(conn, agent_config=self.agent_config)
        self._rebuild_tools()

    def _rebuild_tools(self):
        """Rebuild the tools list with current tool instances."""
        self.tools = (
            self.db_func_tool.available_tools()
            + self.context_search_tools.available_tools()
            + (self.date_parsing_tools.available_tools() if self.date_parsing_tools else [])
            + (self.filesystem_func_tool.available_tools() if self.filesystem_func_tool else [])
        )

    def setup_tools(self):
        """Initialize all tools with default database connection."""
        # Only a single database connection is now supported
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
        self.db_func_tool = DBFuncTool(conn, agent_config=self.agent_config)
        self.context_search_tools = ContextSearchTools(self.agent_config)
        self._setup_date_parsing_tools()
        self._setup_filesystem_tools()
        self._rebuild_tools()

    def _setup_date_parsing_tools(self):
        """Setup date parsing tools."""
        try:
            self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            logger.info("Setup date parsing tools")
        except Exception as e:
            logger.error(f"Failed to setup date parsing tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools (all available tools)."""
        try:
            root_path = self._resolve_workspace_root()
            self.filesystem_func_tool = FilesystemFuncTool(root_path=root_path)
            logger.info(f"Setup filesystem tools with root path: {root_path}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _resolve_workspace_root(self) -> str:
        """
        Resolve workspace_root from chat node configuration.
        Expands ~ to user home directory.

        Returns:
            Resolved workspace_root path with ~ expanded
        """
        import os

        # Default workspace_root
        workspace_root = "."

        # Read from chat node configuration if available
        if self.agent_config and hasattr(self.agent_config, "workspace_root"):
            configured_root = self.agent_config.workspace_root
            if configured_root is not None:
                workspace_root = configured_root

        # Expand ~ to user home directory
        if workspace_root.startswith("~"):
            workspace_root = os.path.expanduser(workspace_root)
            logger.debug(f"Expanded workspace_root: {workspace_root}")

        # Handle relative vs absolute paths
        if os.path.isabs(workspace_root):
            return workspace_root
        else:
            return os.path.join(os.getcwd(), workspace_root)

    def _setup_mcp_servers(self, agent_config: Optional[AgentConfig] = None) -> Dict[str, MCPServerStdio]:
        """
        Set up MCP servers based on namespace and configuration.

        Args:
            agent_config: Agent configuration

        Returns:
            Dictionary of MCP servers
        """
        # No MCP servers for chat node currently
        # (Previously had filesystem MCP server, now using native filesystem tools)
        return {}

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the chat interaction with streaming support.

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
            raise ValueError("Chat input not set. Call setup_input() first or set self.input directly.")

        user_input = self.input

        is_plan_mode = getattr(user_input, "plan_mode", False)
        if is_plan_mode:
            self.plan_mode_active = True

            # Create plan mode hooks
            from rich.console import Console

            from datus.cli.plan_hooks import PlanModeHooks

            console = Console()
            session = self._get_or_create_session()[0]

            # Workflow sets 'auto_execute_plan' in metadata, CLI REPL does not
            auto_mode = getattr(user_input, "auto_execute_plan", False)
            logger.info(f"Plan mode auto_mode: {auto_mode} (from input)")

            self.plan_hooks = PlanModeHooks(console=console, session=session, auto_mode=auto_mode)

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
            # Check for auto-compact before session creation to ensure fresh context
            await self._auto_compact()

            # Get or create session and any available summary
            session, conversation_summary = self._get_or_create_session()

            # Get system instruction from template, passing summary and prompt version if available
            system_instruction = self._get_system_prompt(conversation_summary, user_input.prompt_version)

            # Add database context to user message if provided
            from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

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

            # Determine execution mode and start unified recursive execution
            execution_mode = "plan" if is_plan_mode and self.plan_hooks else "normal"

            # Start unified recursive execution
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
                        # Only collect raw_output if it's from a "message" type action (Thinking messages)
                        raw_output_value = ""
                        if stream_action.action_type == "message" and "raw_output" in stream_action.output:
                            raw_output_value = stream_action.output.get("raw_output", "")

                        response_content = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or raw_output_value
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
                    or last_successful_output.get("raw_output", "")  # Try raw_output from any action type
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

            # Extract token usage from final actions using our new approach
            # With our streaming token fix, only the final assistant action will have accurate usage
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
            result = ChatNodeResult(
                success=True,
                response=response_content,
                sql=sql_content,
                tokens_used=int(tokens_used),
                action_history=[action.model_dump() for action in all_actions],
                execution_stats=execution_stats,
            )

            # # Update assistant action with success
            # action_history_manager.update_action_by_id(
            #     assistant_action.action_id,
            #     status=ActionStatus.SUCCESS,
            #     output=result.model_dump(),
            #     messages=(
            #         f"Generated response: {response_content[:100]}..."
            #         if len(response_content) > 100
            #         else response_content
            #     ),
            # )

            # Add to internal actions list
            self.actions.extend(action_history_manager.get_actions())

            # Create final action
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

        except Exception as e:
            # Handle user cancellation as success, not error
            if "User cancelled" in str(e) or "UserCancelledException" in str(type(e).__name__):
                logger.info("User cancelled execution, stopping gracefully...")

                # Create cancellation result (success=True)
                result = ChatNodeResult(
                    success=True,
                    response="Execution cancelled by user.",
                    tokens_used=0,
                )

                # Update action with cancellation
                action_history_manager.update_current_action(
                    status=ActionStatus.SUCCESS,
                    output=result.model_dump(),
                    messages="Execution cancelled by user",
                )

                # Create cancellation action
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

                # Create error result for all other exceptions
                result = ChatNodeResult(
                    success=False,
                    error=str(e),
                    response="Sorry, I encountered an error while processing your request.",
                    tokens_used=0,
                )

                # Update action with error
                action_history_manager.update_current_action(
                    status=ActionStatus.FAILED,
                    output=result.model_dump(),
                    messages=f"Error: {str(e)}",
                )

                # Create error action
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
            # Clean up plan mode state
            if is_plan_mode:
                self.plan_mode_active = False
                self.plan_hooks = None

    async def _execute_with_recursive_replan(
        self,
        prompt: str,
        execution_mode: str,
        original_input: ChatNodeInput,
        action_history_manager: ActionHistoryManager,
        session,
    ):
        """
        Unified recursive execution function that handles all execution modes.

        Args:
            prompt: The prompt to send to LLM
            execution_mode: "normal", "plan", or "replan"
            original_input: Original chat input for context
            action_history_manager: Action history manager
            session: Chat session
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

    def _get_execution_config(self, execution_mode: str, original_input: ChatNodeInput) -> dict:
        """
        Get execution configuration based on mode.

        Args:
            execution_mode: "normal", "plan"
            original_input: Original chat input for context

        Returns:
            Configuration dict with tools, instruction, and hooks
        """
        if execution_mode == "normal":
            return {"tools": self.tools, "instruction": self._get_system_instruction(original_input), "hooks": None}
        elif execution_mode == "plan":
            # Plan mode: standard tools + plan tools
            plan_tools = self.plan_hooks.get_plan_tools() if self.plan_hooks else []

            # Add execution steps to instruction for consistency
            base_instruction = self._get_system_instruction(original_input)
            current_phase = getattr(self.plan_hooks, "plan_phase", "generating") if self.plan_hooks else "generating"

            if current_phase in ["executing", "confirming"]:
                plan_instruction = (
                    base_instruction
                    + "\n\nEXECUTION steps:\n"
                    + "For each todo step: todo_update(id, 'pending') → execute task → todo_update(id, 'completed')\n"
                    + "Always follow this exact sequence for every step."
                )
            else:
                plan_instruction = base_instruction

            return {
                "tools": self.tools + plan_tools,
                "instruction": plan_instruction,
                "hooks": self.plan_hooks,
            }
        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

    def _get_system_instruction(self, original_input: ChatNodeInput) -> str:
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
                sql = parsed.get("sql")
                output_text = parsed.get("output")

                # Unescape output content if present
                if output_text and isinstance(output_text, str):
                    output_text = output_text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

                return sql, output_text

            return None, None

        except Exception as e:
            logger.warning(f"Failed to extract SQL and output from response: {e}")
            return None, None
