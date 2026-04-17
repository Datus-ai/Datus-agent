# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
FeedbackAgenticNode implementation for conversation feedback analysis.

This module provides an agentic node that takes over an existing session
(copies messages, swaps system prompt) to analyze conversation history
and archive reusable knowledge, SQL patterns, metrics, and skills
via existing sub-agents.
"""

from typing import AsyncGenerator, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.feedback_agentic_node_models import FeedbackNodeResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FeedbackAgenticNode(AgenticNode):
    """
    Conversation feedback analysis agentic node.

    This node copies an existing chat session, replaces the system prompt
    with a feedback-specific prompt, and uses the LLM to analyze the full
    conversation history. It delegates archival to gen_* sub-agents via
    the task() tool and updates MEMORY.md via filesystem tools.
    """

    # Default subagents feedback delegates archival to. Users can override via
    # agent.yml (agentic_nodes.feedback.subagents = "...") — the base-class
    # _setup_sub_agent_task_tool reads node_config first and falls back here.
    DEFAULT_SUBAGENTS = "gen_ext_knowledge, gen_sql_summary, gen_metrics, gen_skill"

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
    ):
        self.execution_mode = execution_mode
        self.configured_node_name = "feedback"

        # Get max_turns from agentic_nodes configuration
        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and "feedback" in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes["feedback"]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        # Tool holders BEFORE super().__init__()
        self.sub_agent_task_tool = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.ask_user_tool = None
        self.hooks = None

        from datus.configuration.node_type import NodeType

        super().__init__(
            node_id="feedback_node",
            description="Conversation feedback analysis node",
            node_type=NodeType.TYPE_FEEDBACK,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            scope=scope,
        )

        self.setup_tools()

    def get_node_name(self) -> str:
        return self.configured_node_name

    def setup_tools(self):
        """Setup tools: sub-agent task delegation + filesystem for MEMORY.md."""
        if not self.agent_config:
            return

        self.tools = []
        self._setup_sub_agent_task_tool()
        self._setup_filesystem_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        self._rebuild_tools()

    def _setup_filesystem_tools(self):
        """Setup filesystem tools for writing MEMORY.md and other files."""
        try:
            root_path = self._resolve_workspace_root()
            self.filesystem_func_tool = FilesystemFuncTool(root_path=root_path)
            logger.debug(f"Setup filesystem tools with root path: {root_path}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _rebuild_tools(self):
        """Rebuild the tools list with current tool instances."""
        self.tools = []
        if self.filesystem_func_tool:
            self.tools.extend(self.filesystem_func_tool.available_tools())
        if self.sub_agent_task_tool:
            self.tools.extend(self.sub_agent_task_tool.available_tools())
        if self.ask_user_tool:
            self.tools.extend(self.ask_user_tool.available_tools())

    def _resolve_caller_node_name(self) -> str:
        """Return the caller node whose memory this feedback run should update.

        The CLI (or other callers) set ``self.caller_node_name`` on the node
        at switch time. Falls back to ``"chat"`` — Datus's default top-level
        node — when no explicit caller was set.
        """
        return self.caller_node_name or "chat"

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """Get the feedback system prompt."""
        template_name = f"{self.configured_node_name}_system"

        try:
            template_vars = {
                "agent_config": self.agent_config,
                "conversation_summary": conversation_summary,
                "native_tools": ", ".join([tool.name for tool in self.tools]) if self.tools else "None",
                "has_task_tool": bool(self.sub_agent_task_tool),
                "has_ask_user_tool": self.ask_user_tool is not None,
                "knowledge_base_dir": str(self.agent_config.path_manager.knowledge_base_home),
                "current_database": self.agent_config.current_database,
                "workspace_root": self._resolve_workspace_root(),
            }

            if template_context:
                template_vars.update(template_context)

            from datus.prompts.prompt_manager import get_prompt_manager

            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, **template_vars
            )
            # Feedback has no memory of its own — inject the caller's memory via the
            # standard memory_context template by overriding the node name.
            return self._finalize_system_prompt(
                base_prompt,
                memory_node_name_override=self._resolve_caller_node_name(),
            )

        except FileNotFoundError as e:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": prompt_version},
            ) from e
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the feedback node interaction with streaming support.

        If source_session_id is provided and no session exists yet,
        copies the source session first so the LLM sees full conversation history.
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        if self.input is None:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message_args={"field_name": "self.input (FeedbackNodeInput)"},
            )

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
            # Copy source session if provided and no session exists yet
            if user_input.source_session_id and not self.session_id:
                from datus.models.session_manager import SessionManager

                sm = SessionManager(session_dir=self.model.session_dir, scope=self.scope)
                self.session_id = sm.copy_session(user_input.source_session_id, target_node_name="feedback")
                logger.info(f"Copied source session {user_input.source_session_id} -> {self.session_id}")

            # Session management
            session = None
            conversation_summary = None
            if self.execution_mode == "interactive":
                await self._auto_compact()

            session, conversation_summary = self._get_or_create_session()

            # Get system prompt
            system_instruction = self._get_system_prompt(conversation_summary)

            response_content = ""
            tokens_used = 0
            last_successful_output = None

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=user_input.user_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=self.hooks if self.execution_mode == "interactive" else None,
                agent_name=self.get_node_name(),
                interrupt_controller=self.interrupt_controller,
            ):
                yield stream_action

                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        raw_output = stream_action.output.get("raw_output", "")
                        if isinstance(raw_output, dict):
                            response_content = raw_output
                        elif raw_output:
                            response_content = raw_output

            if not response_content and last_successful_output:
                candidate = (
                    last_successful_output.get("content", "")
                    or last_successful_output.get("response", "")
                    or last_successful_output.get("raw_output", "")
                )
                if isinstance(candidate, str) and candidate:
                    response_content = candidate
                elif candidate and not isinstance(candidate, str):
                    response_content = str(candidate)

            # Extract token usage
            tokens_used = 0
            if self.execution_mode == "interactive":
                final_actions = action_history_manager.get_actions()
                for act in reversed(final_actions):
                    if act.role == "assistant":
                        if act.output and isinstance(act.output, dict):
                            usage_info = act.output.get("usage", {})
                            if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                                tokens_used = usage_info.get("total_tokens", 0)
                                if tokens_used > 0:
                                    break

            # Count storage from the current run's actions so reused node
            # instances don't leak counts from previous runs.
            current_actions = action_history_manager.get_actions()
            items_saved, storage_summary = self._extract_storage_info(current_actions)
            self.actions.extend(current_actions)

            result = FeedbackNodeResult(
                success=True,
                response=response_content if isinstance(response_content, str) else str(response_content),
                items_saved=items_saved,
                storage_summary=storage_summary,
                tokens_used=int(tokens_used),
            )
            self.result = result

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="feedback_response",
                messages=f"{self.get_node_name()} interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise

        except Exception as e:
            logger.error(f"{self.get_node_name()} execution error: {e}")

            error_result = FeedbackNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while processing your feedback request.",
                tokens_used=0,
            )
            self.result = error_result

            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {str(e)}",
            )

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

    def _extract_storage_info(self, actions: list) -> tuple[int, Optional[dict]]:
        """Extract items_saved count and storage_summary from action history.

        Counts successful task() tool calls to estimate items archived.
        """
        items_saved = 0
        storage_summary: dict = {}

        for act in actions:
            if act.role == ActionRole.TOOL and act.status == ActionStatus.SUCCESS:
                action_type = act.action_type or ""
                if action_type == "task" and act.input and isinstance(act.input, dict):
                    task_type = act.input.get("arguments", "")
                    try:
                        import json

                        args = json.loads(task_type) if isinstance(task_type, str) else task_type
                        sub_type = args.get("type", "other")
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        sub_type = "other"

                    category = sub_type.replace("gen_", "")
                    storage_summary[category] = storage_summary.get(category, 0) + 1
                    items_saved += 1

        return items_saved, storage_summary if storage_summary else None
