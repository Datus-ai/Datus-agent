# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ChatAgenticNode implementation for flexible CLI chat interactions.

This module provides a concrete implementation of AgenticNode specifically
designed for chat interactions with database and filesystem tool support.
"""
import json
from typing import AsyncGenerator, Dict, Optional

from agents.mcp import MCPServerStdio

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.schemas.node_models import TableSchema
from datus.tools.context_search import ContextSearchTools
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.mcp_server import MCPServer
from datus.tools.tools import DBFuncTool
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
        namespace: Optional[str] = None,
        agent_config: Optional[AgentConfig] = None,
        max_turns: int = 30,
    ):
        """
        Initialize the ChatAgenticNode.

        Args:
            namespace: Database namespace for MCP server selection
            agent_config: Agent configuration
            max_turns: Maximum conversation turns per interaction
        """
        self.namespace = namespace
        # Get max_turns from node configuration if available
        node_max_turns = None
        if agent_config and hasattr(agent_config, "nodes") and "chat" in agent_config.nodes:
            chat_node_config = agent_config.nodes["chat"]
            if chat_node_config.input and hasattr(chat_node_config.input, "max_turns"):
                node_max_turns = chat_node_config.input.max_turns

        # Priority: provided value > node config > default 30
        self.max_turns = max_turns if max_turns != 30 else (node_max_turns or 30)

        # Initialize MCP servers based on namespace
        self.mcp_servers = self._setup_mcp_servers(agent_config)

        super().__init__(
            tools=[],
            mcp_servers=self.mcp_servers,
            agent_config=agent_config,
        )
        self.db_func_tool: DBFuncTool
        self.context_search_tools: ContextSearchTools
        self.plan_mode_active = False
        self.setup_tools()

    def setup_tools(self):
        # Only a single database connection is now supported
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
        self.db_func_tool = DBFuncTool(conn, agent_config=self.agent_config)
        self.context_search_tools = ContextSearchTools(self.agent_config)
        self.tools = self.db_func_tool.available_tools() + self.context_search_tools.available_tools()

    def _setup_mcp_servers(self, agent_config: Optional[AgentConfig] = None) -> Dict[str, MCPServerStdio]:
        """
        Set up MCP servers based on namespace and configuration.


        Returns:
            Dictionary of MCP servers
        """
        mcp_servers = {}

        try:
            # Add filesystem MCP server with configurable root path
            import os

            root_path = "."
            if agent_config and hasattr(agent_config, "workspace_root"):
                workspace_root = agent_config.workspace_root
                if workspace_root is not None:
                    root_path = workspace_root

            # Handle relative vs absolute paths
            if root_path and os.path.isabs(root_path):
                filesystem_path = root_path
            else:
                filesystem_path = os.path.join(os.getcwd(), root_path)

            filesystem_server = MCPServer.get_filesystem_mcp_server(path=filesystem_path)
            if filesystem_server:
                # Add filesystem server first
                mcp_servers["filesystem"] = filesystem_server
                logger.info(f"Added filesystem MCP server at path: {filesystem_path}")
            else:
                logger.warning(f"Failed to create filesystem MCP server for path: {filesystem_path}")

        except Exception as e:
            logger.error(f"Error setting up MCP servers: {e}")

        return mcp_servers

    async def execute_stream(
        self, user_input: ChatNodeInput, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the chat interaction with streaming support.

        Args:
            user_input: Chat input containing user message and context
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        is_plan_mode = getattr(user_input, "plan_mode", False)
        if is_plan_mode:
            self.plan_mode_active = True

            # Create plan mode hooks
            from rich.console import Console

            from datus.cli.plan_hooks import PlanModeHooks

            console = Console()
            session = self._get_or_create_session()[0]
            self.plan_hooks = PlanModeHooks(console=console, session=session)

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
            enhanced_message = user_input.user_message
            enhanced_parts = []
            if user_input.catalog or user_input.database or user_input.db_schema:
                context_parts = [f"dialect: {self.agent_config.db_type}"]
                if user_input.catalog:
                    context_parts.append(f"catalog: {user_input.catalog}")
                if user_input.database:
                    context_parts.append(f"database: {user_input.database}")
                if user_input.db_schema:
                    context_parts.append(f"schema: {user_input.db_schema}")
                context_part_str = f'Context: {", ".join(context_parts)}'
                enhanced_parts.append(context_part_str)
            if user_input.schemas:
                table_schemas_str = TableSchema.list_to_prompt(user_input.schemas, dialect=self.agent_config.db_type)
                enhanced_parts.append(f"Table Schemas: \n{table_schemas_str}")
            if user_input.metrics:
                enhanced_parts.append(f"Metrics: \n{json.dumps([item.model_dump() for item in user_input.metrics])}")

            if user_input.reference_sql:
                enhanced_parts.append(
                    f"Reference SQL: \n{json.dumps([item.model_dump() for item in user_input.reference_sql])}"
                )

            if enhanced_parts:
                enhanced_message = f"{'\n\n'.join(enhanced_parts)}\n\nUser question: {user_input.user_message}"

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

            # Create final result
            result = ChatNodeResult(
                success=True,
                response=response_content,
                sql=sql_content,
                tokens_used=int(tokens_used),
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
        original_input: "ChatNodeInput",
        action_history_manager: "ActionHistoryManager",
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

    def _get_execution_config(self, execution_mode: str, original_input: "ChatNodeInput") -> dict:
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

    def _get_system_instruction(self, original_input: "ChatNodeInput") -> str:
        """Get system instruction for normal mode."""
        _, conversation_summary = self._get_or_create_session()
        return self._get_system_prompt(conversation_summary, original_input.prompt_version)

    def _extract_sql_and_output_from_response(self, output: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extract SQL content and formatted output from model response.

        Uses the existing llm_result2json utility for robust JSON parsing.

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

            # First, try direct parsing of the content
            parsed = llm_result2json(content, expected_type=dict)

            if parsed and isinstance(parsed, dict):
                # Check if it has sql/output fields directly
                if "sql" in parsed or "output" in parsed:
                    sql = parsed.get("sql")
                    output_text = parsed.get("output")

                    # Unescape output content if present
                    if output_text and isinstance(output_text, str):
                        output_text = output_text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

                    return sql, output_text

            # If direct parsing failed, try to handle nested raw_output structure
            # Format: {'raw_output': '{"sql": "...", "output": "..."}'}
            import ast

            if content.strip().startswith("{'"):
                try:
                    parsed_dict = ast.literal_eval(content.strip())
                    if isinstance(parsed_dict, dict) and "raw_output" in parsed_dict:
                        # Parse the nested JSON in raw_output
                        nested_parsed = llm_result2json(parsed_dict["raw_output"], expected_type=dict)

                        if nested_parsed and isinstance(nested_parsed, dict):
                            sql = nested_parsed.get("sql")
                            output_text = nested_parsed.get("output")

                            # Unescape output content
                            if output_text and isinstance(output_text, str):
                                output_text = output_text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

                            return sql, output_text

                except (ValueError, SyntaxError) as e:
                    logger.debug(f"Failed to parse Python dict format: {e}")

            return None, None

        except Exception as e:
            logger.warning(f"Failed to extract SQL and output from response: {e}")
            return None, None

    def _extract_sql_from_response(self, output: dict) -> Optional[str]:
        """
        Extract SQL content from model response (backward compatibility).

        Args:
            output: Output dictionary from model generation

        Returns:
            SQL string if found, None otherwise
        """
        sql_content, _ = self._extract_sql_and_output_from_response(output)
        return sql_content
