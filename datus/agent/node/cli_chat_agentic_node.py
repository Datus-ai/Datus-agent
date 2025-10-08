"""
CliChatAgenticNode implementation for CLI chat interactions with plan mode support.

This module provides a concrete implementation of GenSQLAgenticNode specifically
designed for CLI chat interactions with intelligent tool loading and plan mode support.
"""

from typing import AsyncGenerator, Optional

from rich.console import Console

from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.tools.context_search import ContextSearchTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CliChatAgenticNode(GenSQLAgenticNode):
    """
    CLI chat-focused agentic node with intelligent tool loading and plan mode support.

    This node provides flexible chat capabilities with:
    - Intelligent tool filtering based on storage availability
    - Internal filesystem tools (not MCP-based)
    - Plan mode support with PlanModeHooks
    - Enhanced context with domain/layer support
    """

    def __init__(
        self,
        node_name: str,
        agent_config: Optional[AgentConfig] = None,
        max_turns: int = 30,
    ):
        """
        Initialize the CliChatAgenticNode.

        Args:
            node_name: Name of the node configuration in agent.yml (e.g., "sql")
            agent_config: Agent configuration
            max_turns: Maximum conversation turns per interaction
        """
        logger.info(f"🎯 Initializing CliChatAgenticNode with node_name='{node_name}'")

        # Initialize plan mode state
        self.plan_mode_active = False
        self.plan_hooks = None
        self.filesystem_tools = None

        # Call parent constructor
        super().__init__(node_name, agent_config, max_turns)

        logger.info(f"✅ CliChatAgenticNode initialized successfully")

    def setup_tools(self):
        """Override to add intelligent tool filtering + filesystem tools."""
        self.tools = []

        # 1. DB tools (always included)
        self._setup_db_tools()

        # 2. Context search tools (conditional based on storage)
        has_metrics, has_sql_history = self._check_storage_availability()
        if has_metrics or has_sql_history:
            self._setup_context_search_tools_filtered(has_metrics, has_sql_history)

        # 3. Date parsing tools (optional from config)
        if "date_parsing_tools" in self.node_config.get("tools", ""):
            self._setup_date_parsing_tools()

        # 4. Filesystem tools (CliChatAgenticNode only)
        self._setup_filesystem_tools()

        logger.info(f"Setup {len(self.tools)} tools for CliChatAgenticNode: {[tool.name for tool in self.tools]}")

    def _check_storage_availability(self) -> tuple[bool, bool]:
        """
        Check if metrics and sql_history tables exist and have data.

        Returns:
            Tuple of (has_metrics, has_sql_history) booleans
        """
        has_metrics = False
        has_sql_history = False

        # Check metrics storage
        try:
            from datus.storage.metric.store import metrics_rag_by_configuration

            metric_rag = metrics_rag_by_configuration(self.agent_config)

            if "metrics" in metric_rag.metric_storage.db.table_names():
                has_metrics = metric_rag.metric_storage.table_size() > 0
                logger.debug(f"Metrics storage: exists=True, has_data={has_metrics}")
        except Exception as e:
            logger.debug(f"Metrics storage check failed: {e}")

        # Check SQL history storage
        try:
            from datus.storage.sql_history.store import sql_history_rag_by_configuration

            sql_history_store = sql_history_rag_by_configuration(self.agent_config)

            if "sql_history" in sql_history_store.sql_history_storage.db.table_names():
                has_sql_history = sql_history_store.sql_history_storage.table_size() > 0
                logger.debug(f"SQL history storage: exists=True, has_data={has_sql_history}")
        except Exception as e:
            logger.debug(f"SQL history storage check failed: {e}")

        return has_metrics, has_sql_history

    def _setup_context_search_tools_filtered(self, has_metrics: bool, has_sql_history: bool):
        """
        Setup only available context search tools based on storage availability.

        Args:
            has_metrics: Whether metrics storage has data
            has_sql_history: Whether SQL history storage has data
        """
        try:
            self.context_search_tools = ContextSearchTools(self.agent_config)
            all_tools = self.context_search_tools.available_tools()

            # Define tool-to-storage mapping
            metrics_tools = {"search_metrics", "list_domains", "list_layers_by_domain", "get_metrics"}
            sql_history_tools = {"search_historical_sql", "get_sql_history"}
            shared_tools = {"list_items"}  # Works for both

            # Filter tools based on availability
            for tool in all_tools:
                if (
                    (tool.name in metrics_tools and has_metrics)
                    or (tool.name in sql_history_tools and has_sql_history)
                    or (tool.name in shared_tools and (has_metrics or has_sql_history))
                ):
                    self.tools.append(tool)

            logger.info(
                f"Added {len([t for t in self.tools if t.name in (metrics_tools | sql_history_tools | shared_tools)])} "
                f"context search tools (metrics={has_metrics}, sql_history={has_sql_history})"
            )
        except Exception as e:
            logger.error(f"Failed to setup context search tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup internal filesystem tools (read_file, read_multiple_files, list_directory)."""
        try:
            from datus.tools.filesystem_tools.filesystem_tool import FilesystemFuncTool

            root_path = self._resolve_workspace_root()
            self.filesystem_tools = FilesystemFuncTool(root_path=root_path)

            # Get all tools
            all_fs_tools = self.filesystem_tools.available_tools()

            # Filter to only allowed tools
            allowed_tools = {"read_file", "read_multiple_files", "list_directory"}
            filtered_tools = [t for t in all_fs_tools if t.name in allowed_tools]

            self.tools.extend(filtered_tools)
            logger.info(f"Added {len(filtered_tools)} filesystem tools: {[t.name for t in filtered_tools]}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _get_system_prompt(
        self, conversation_summary: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> str:
        """
        Override to set search_localfile=True for CLI chat.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use

        Returns:
            System prompt string loaded from the template
        """
        # Simplified context matching sql_system_1.0.j2 with search_localfile=True
        context = {
            "conversation_summary": conversation_summary,
            "agent_config": self.agent_config,
            "namespace": self.agent_config.current_namespace if self.agent_config else None,
            "workspace_root": self._resolve_workspace_root(),
            "rules": self.node_config.get("rules", []),
            "agent_description": self.node_config.get("agent_description", ""),
            "context_search_tools": bool(self.context_search_tools),
            "search_localfile": True,  # Enable filesystem tools guidance for CLI chat
        }

        version = prompt_version or self.node_config.get("prompt_version", "")
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        try:
            from datus.prompts.prompt_manager import prompt_manager

            return prompt_manager.render_template(template_name=template_name, version=version, **context)
        except FileNotFoundError as e:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": version or "latest"},
            ) from e
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    async def execute_stream(
        self, user_input: ChatNodeInput, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the chat interaction with streaming support and plan mode.

        Args:
            user_input: Chat input containing user message and context
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        is_plan_mode = getattr(user_input, "plan_mode", False)

        # Initialize plan hooks if needed
        if is_plan_mode:
            self.plan_mode_active = True
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
            # Check for auto-compact before session creation
            await self._auto_compact()

            # Get or create session and any available summary
            session, conversation_summary = self._get_or_create_session()

            # Get system instruction from template
            system_instruction = self._get_system_prompt(conversation_summary, user_input.prompt_version)

            # Build enhanced message with domain/layer support using base class method
            enhanced_message = self._build_enhanced_message(
                user_message=user_input.user_message,
                catalog=user_input.catalog,
                database=user_input.database,
                db_schema=user_input.db_schema,
                domain=getattr(user_input, "domain", None),  # NEW: domain support
                layer1=getattr(user_input, "layer1", None),  # NEW: layer1 support
                layer2=getattr(user_input, "layer2", None),  # NEW: layer2 support
                schemas=getattr(user_input, "schemas", None),
                metrics=getattr(user_input, "metrics", None),
                historical_sql=getattr(user_input, "historical_sql", None),
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
                        response_content = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or response_content
                        )

            # If we still don't have response_content, check the last successful output
            if not response_content and last_successful_output:
                logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
                response_content = (
                    last_successful_output.get("content", "")
                    or last_successful_output.get("text", "")
                    or last_successful_output.get("response", "")
                    or str(last_successful_output)
                )

            # Extract SQL and output from the final response_content
            sql_content, extracted_output = self._extract_sql_and_output_from_response({"content": response_content})
            if extracted_output:
                response_content = extracted_output

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")

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
                                self._add_session_tokens(conversation_tokens)
                                tokens_used = conversation_tokens
                                logger.info(f"Added {conversation_tokens} tokens to session")
                                break

            # Create final result
            result = ChatNodeResult(
                success=True,
                response=response_content,
                sql=sql_content,
                tokens_used=int(tokens_used),
            )

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
            execution_mode: "normal" or "plan"
            original_input: Original chat input for context
            action_history_manager: Action history manager
            session: Chat session
        """
        logger.info(f"Executing mode: {execution_mode}")

        # Get execution configuration for this mode
        config = self._get_execution_config(execution_mode, original_input)

        # Reset state for plan mode
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
            execution_mode: "normal" or "plan"
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

    def _build_plan_prompt(self, original_prompt: str) -> str:
        """Build enhanced prompt for plan mode based on current phase."""
        # Check current phase and replan feedback
        current_phase = getattr(self.plan_hooks, "plan_phase", "generating") if self.plan_hooks else "generating"
        replan_feedback = getattr(self.plan_hooks, "replan_feedback", "") if self.plan_hooks else ""

        execution_prompt = (
            "After the plan has been confirmed, execute the pending steps.\n\n"
            + "Execution steps for each pending step:\n"
            + "1. FIRST: call todo_update(todo_id, 'pending') to mark step as pending (triggers user confirmation)\n"
            + "2. then execute the actual task (SQL queries, data processing, etc.)\n"
            + "3. then call todo_update(todo_id, 'completed') to mark step as completed\n\n"
            + "Start with the first pending step in the plan."
        )

        # Only enter replan mode if we have feedback AND we're still in generating phase
        if replan_feedback and current_phase == "generating":
            # REPLAN MODE: Generate revised plan
            plan_prompt_addition = (
                "\n\nREPLAN MODE\n"
                + f"Revise the current plan based on USER FEEDBACK: {replan_feedback}\n\n"
                + "STEPS:\n"
                + "1. FIRST: call todo_read to review the current plan, the completed and pending steps\n"
                + "2. then call todo_write to generate revised plan following these rules:\n"
                + "   - COMPLETED steps(if there are any): keep items as 'completed'\n"
                + "   - PENDING steps that are no longer needed: DISCARD (don't include in new plan)\n"
                + "   - PENDING steps that are still needed: keep as 'pending' or revise content\n"
                + "   - NEW steps(if there are any): add as 'pending'\n"
                + "3. Only include steps that are actually needed in the revised plan\n"
                + execution_prompt
            )
        elif current_phase == "generating":
            # INITIAL PLANNING PHASE
            plan_prompt_addition = (
                "\n\nPLAN MODE - PLANNING PHASE\n"
                + "Task: Break down user request into 3-8 steps.\n\n"
                + "call todo_write to generate complete todo list (3-8 steps)\n"
                + 'Example: todo_write(\'[{"content": "Connect to database", "status": "pending"}, '
                + '{"content": "Query data", "status": "pending"}]\')'
                + execution_prompt
            )
        else:
            # Default fallback
            plan_prompt_addition = (
                "\n\nPLAN MODE\n" + "Check todo_read to see current plan status and proceed accordingly."
            )

        return original_prompt + plan_prompt_addition
