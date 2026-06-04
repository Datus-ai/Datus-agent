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

from typing import Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionRole, ActionStatus
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
    DEFAULT_SUBAGENTS = "gen_sql_summary, gen_metrics, gen_skill"
    result_class = FeedbackNodeResult

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        session_id: Optional[str] = None,
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
            session_id=session_id,
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
        """Setup filesystem tools for writing MEMORY.md and other files.

        The tool is rooted with ``current_node=self._resolve_caller_node_name()``
        so ``.datus/memory/{caller}/**`` lands in the WHITELIST zone — the
        whole point of this node is to update the caller's memory, so the
        policy must permit writes under the caller's memory subtree rather
        than feedback's own.
        """
        try:
            self.filesystem_func_tool = self._make_filesystem_tool(current_node=self._resolve_caller_node_name())
            logger.debug(f"Setup filesystem tools with root path: {self.filesystem_func_tool.root_path}")
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

    @property
    def caller_node_name(self) -> Optional[str]:
        """The node whose memory this feedback run should update.

        Exposed as a property so assignment by the CLI (``node.caller_node_name
        = "gen_sql"``) rebuilds the filesystem tool with the new caller —
        otherwise the whitelist baked in at construction time (defaulting to
        ``"chat"``) would keep the caller's memory path classified as HIDDEN.
        """
        return getattr(self, "_caller_node_name", None)

    @caller_node_name.setter
    def caller_node_name(self, value: Optional[str]) -> None:
        self._caller_node_name = value
        # Guard for the base class's __init__-time assignment: at that point
        # ``filesystem_func_tool`` is still ``None`` and we must not rebuild.
        if getattr(self, "filesystem_func_tool", None) is not None:
            try:
                self.filesystem_func_tool = self._make_filesystem_tool(current_node=self._resolve_caller_node_name())
                self._rebuild_tools()
            except Exception as e:
                logger.debug(f"Could not rebuild filesystem tool on caller change: {e}")

    def _resolve_caller_node_name(self) -> str:
        """Return the caller node whose memory this feedback run should update.

        The CLI (or other callers) set ``self.caller_node_name`` on the node
        at switch time. Falls back to ``"chat"`` — Datus's default top-level
        node — when no explicit caller was set.
        """
        return self.caller_node_name or "chat"

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """Get the feedback system prompt."""
        template_name = f"{self.configured_node_name}_system"

        try:
            template_vars = {
                "agent_config": self.agent_config,
                "native_tools": ", ".join([tool.name for tool in self.tools]) if self.tools else "None",
                "has_task_tool": bool(self.sub_agent_task_tool),
                "has_ask_user_tool": self.ask_user_tool is not None,
                "knowledge_base_dir": str(self.agent_config.path_manager.subject_dir),
                "current_datasource": self.agent_config.current_datasource,
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

    def _build_success_result(self, ctx: StreamRunContext) -> FeedbackNodeResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            candidate = (
                ctx.last_successful_output.get("content", "")
                or ctx.last_successful_output.get("response", "")
                or ctx.last_successful_output.get("raw_output", "")
            )
            if isinstance(candidate, str) and candidate:
                response_content = candidate
            elif candidate and not isinstance(candidate, str):
                response_content = str(candidate)
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())

        current_actions = ctx.action_history_manager.get_actions()
        items_saved, storage_summary = self._extract_storage_info(current_actions)

        return FeedbackNodeResult(
            success=True,
            response=response_content,
            items_saved=items_saved,
            storage_summary=storage_summary,
            tokens_used=int(tokens_used),
        )

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
