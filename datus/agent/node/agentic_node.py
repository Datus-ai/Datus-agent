# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Agentic Node Architecture for Datus-agent.

This module provides a new agentic node system that supports session-based,
streaming interactions with tool integration and action history management.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set

from agents import Tool
from agents.extensions.memory import AdvancedSQLiteSession
from agents.mcp import MCPServerStdio

from datus.agent.node.node import Node
from datus.cli.execution_state import ExecutionInterrupted, InteractionBroker, InterruptController, PendingInputQueue
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import get_prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import to_str
from datus.utils.loggings import get_logger
from datus.utils.message_utils import build_structured_content
from datus.utils.node_utils import build_database_context

if TYPE_CHECKING:
    from datus.agent.node.stream_run_context import StreamRunContext
    from datus.agent.workflow import Workflow
    from datus.schemas.token_usage import TokenUsage
    from datus.tools.permission.permission_manager import PermissionManager
    from datus.tools.skill_tools.skill_manager import SkillManager

logger = get_logger(__name__)


_LANGUAGE_NAME_MAP: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}


def _resolve_language_name(code: str) -> str:
    """Map a language code (e.g. ``"zh"``) to a human-readable name.

    Unknown codes are returned as-is so operators can plug in custom values
    without a code change.
    """
    if not code:
        return "English"
    return _LANGUAGE_NAME_MAP.get(code.strip().lower(), code)


class AgenticNode(Node):
    """
    Base agentic node that provides session-based, streaming interactions
    with tool integration and automatic context management.
    """

    DEFAULT_SUBAGENTS = "explore"

    # Default skill patterns injected into ``<available_skills>`` when the user's
    # ``agent.yml`` does not override ``skills:`` for this node. Subclasses declare
    # the skills their workflow expects so every built-in subagent works out of
    # the box without forcing users to wire each skill manually. Set to an explicit
    # empty string in yml to opt out of the defaults.
    DEFAULT_SKILLS: Optional[str] = None

    # When True, this node's ``SkillFuncTool`` loads skills in *authoring* mode:
    # ``allowed_agents`` scoping on ``load_skill`` is bypassed so the agent can
    # read any skill by name (used by ``gen_skill`` for edit/optimize flows).
    # Visibility in ``<available_skills>`` is still filtered normally.
    SKILL_AUTHORING_MODE: bool = False

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: BaseInput = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        scope: Optional[str] = None,
        is_subagent: bool = False,
        memory_enabled: Optional[bool] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the agentic node.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (e.g., 'chat', 'gensql')
            input_data: Input data for the node
            agent_config: Agent configuration
            tools: List of function tools available to this node
            mcp_servers: Dictionary of MCP servers available to this node
            scope: Optional session scope for directory isolation
            is_subagent: When True, skip SubAgentTaskTool setup (2-level depth enforcement)
            memory_enabled: Whether this node should get the Auto Memory section injected
                into its system prompt. When ``None`` (default), resolved from
                ``has_memory(self.get_node_name())`` — built-in subagents (gen_sql,
                gen_report, feedback, etc.) default to ``False``; only ``chat`` and
                custom/user-defined subagents default to ``True``. Pass an explicit
                bool to override.
            session_id: Optional resume target. When provided, the node opens
                this session id and persisted plan-mode state on disk is
                restored automatically. When ``None``, a fresh id is generated
                eagerly here so ``session_id`` is guaranteed non-empty after
                construction and never changes for the lifetime of the node.
        """
        # Initialize Node base class
        super().__init__(node_id, description, node_type, input_data, agent_config, tools)

        # AgenticNode-specific attributes
        self.scope = scope
        self.mcp_servers = mcp_servers or {}
        self.actions: List[ActionHistory] = []
        # Resume target (or freshly generated id when caller passes ``None``).
        # ``session_id`` is set once below — after ``get_node_name()`` is wired
        # up — and is treated as immutable for the node's lifetime: resume /
        # rewind / agent-switch flows allocate a NEW node with the desired id
        # rather than rewriting this attribute.
        self.session_id: str = session_id or ""
        self._session: Optional[AdvancedSQLiteSession] = None
        # Optional extra path layer between {sessions_dir}/{user_scope}/ and the .db
        # file. Set by SubAgentTaskTool to the parent's session_id so subagent dbs
        # nest under their launching main session — enabling later resume.
        self.session_subdir: Optional[str] = None
        self._session_manager: Optional["SessionManager"] = None  # noqa: F821 — lazy
        # Populated lazily via the ``model`` property so ``/model`` switches
        # take effect on the next access without rebuilding the node.
        # ``_pinned_model`` exists because the parent :class:`Node` writes
        # ``self.model = None`` / ``self.model = llm_model`` directly; the
        # property setter routes those writes here.
        self._agent_config_ref: Optional[AgentConfig] = None
        self._node_model_name: Optional[str] = None
        self._pinned_model: Optional[LLMBaseModel] = None

        # Name of the previous node (set externally by the caller, e.g. the CLI
        # on agent switch). Nodes that need caller context — like feedback,
        # which injects the caller's MEMORY.md — read this instead of inferring
        # it from the session id prefix. ``None`` when no switch occurred.
        self.caller_node_name: Optional[str] = None

        # Whether memory context is injected into this node's system prompt.
        # Resolves from has_memory() when not explicitly set by the caller.
        from datus.utils.memory_loader import has_memory

        self.memory_enabled: bool = memory_enabled if memory_enabled is not None else has_memory(self.get_node_name())

        # Permission and skill management
        self.permission_manager: Optional["PermissionManager"] = None
        # PermissionHooks is attached lazily once tools are set up — see
        # ``_ensure_permission_hooks``. Every subclass that runs the LLM
        # loop must pass ``self._compose_hooks()`` into
        # ``generate_with_tools_stream`` so rules actually fire.
        self.permission_hooks: Optional[Any] = None
        self.skill_manager: Optional["SkillManager"] = None
        self.skill_func_tool = None
        self.ask_user_tool = None
        self.sub_agent_task_tool = None
        self.bash_tool = None
        self._is_subagent = is_subagent
        self._permission_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[bool]]] = None

        # ActionBus - merges tool sub-actions into the main action stream
        from datus.schemas.action_bus import ActionBus

        self.action_bus = ActionBus()

        # Proxy tool channel - used in print mode with --proxy_tools
        from datus.tools.proxy.tool_result_channel import ToolResultChannel

        self.tool_channel = ToolResultChannel()

        # Proxy tool patterns - stored when apply_proxy_tools() is called, inherited by sub-agents
        self.proxy_tool_patterns: Optional[List[str]] = None

        # Names of tools that ``apply_proxy_tools`` actually replaced with proxy
        # wrappers. ``PermissionHooks`` consults this set to short-circuit the
        # permission check for proxied tools — the external caller (e.g.
        # ``print_mode`` stdin protocol) is responsible for any secondary
        # confirmation, so the agent must not double-prompt. Held as a stable
        # reference so PermissionHooks can be built before the first proxy call
        # and still observe later updates without rebuilding.
        self.proxied_tool_names: Set[str] = set()

        # Shared tool_name -> category registry (used by PermissionHooks & proxy_tool)
        from datus.tools.registry.tool_registry import ToolRegistry

        self.tool_registry = ToolRegistry()

        # Parse node configuration from agent.yml (available to all agentic nodes)
        self.node_config = self._parse_node_config(agent_config, self.get_node_name())

        # Setup permission manager (after node_config is available)
        self._setup_permission_manager()

        # Setup skill manager (after permission_manager is available)
        self._setup_skill_manager()

        # Setup skill func tools for non-chat nodes when explicitly configured
        self._setup_skill_func_tools()

        # General-purpose BashTool — available to every agentic node. The
        # actual injection into ``self.tools`` happens lazily via
        # ``_ensure_bash_tool_in_tools`` because subclass ``setup_tools``
        # tends to reset ``self.tools`` after base ``__init__``.
        self._setup_bash_tool()

        # Resolve model lazily so ``/model`` can flip the active target at
        # runtime without rebuilding every node. The node-specific override
        # (``agent.agentic_nodes.<name>.model``) — when present — still wins
        # because ``_resolve_model_name()`` forwards it to
        # :meth:`LLMBaseModel.create_model`; otherwise the resolver falls
        # back to ``agent_config.active_model()`` each call.
        self._agent_config_ref = agent_config
        self._node_model_name = self.node_config.get("model") if agent_config else None

        self.interaction_broker = InteractionBroker()
        self.interrupt_controller = InterruptController()

        # Per-chat-session queue of free-text user messages staged during
        # an agent run. Set to a real ``PendingInputQueue`` by interactive
        # callers (CLI/TUI and the API ``/insert`` route) before
        # ``execute_stream``; ``None`` for non-interactive callers so they
        # keep current behavior. Lifecycle is owned by the caller — the
        # node treats it as a borrowed reference and never replaces it.
        self.pending_input_queue: Optional[PendingInputQueue] = None

        # Plan mode state (managed at base class, shared by all subclasses).
        # Activated manually via REPL/CLI for the current primary agent; sub-agents
        # spawned during execution do NOT inherit these flags.
        self.plan_mode_active: bool = False
        self.workflow_prompt_sent: bool = False
        self.plan_file_path: Optional[str] = None
        # One-shot flag: set by ``confirm_plan`` so the next user prompt
        # carries an "execute the confirmed plan" reminder. Cleared by
        # ``_build_enhanced_message`` after the reminder is injected.
        self._plan_just_confirmed: bool = False

        # Finalize session_id: caller-supplied id wins; otherwise generate
        # eagerly so ``session_id`` is non-empty and stable from here on. We
        # then re-hydrate any persisted plan-mode state — for fresh sessions
        # the state file does not exist and ``restore_plan_mode_state`` is a
        # no-op, preserving the defaults set above.
        if not self.session_id:
            self.session_id = self._generate_session_id()
        try:
            self.restore_plan_mode_state()
        except Exception as exc:  # noqa: BLE001 — restore must never crash construction
            logger.warning("Failed to restore plan-mode state for %s: %s", self.session_id, exc)

    @property
    def model(self) -> Optional[LLMBaseModel]:
        """Return the currently active :class:`LLMBaseModel` for this node.

        Reads :meth:`AgentConfig.active_model` on every access so a runtime
        ``/model`` switch is picked up without recreating the node. The
        heavy lifting is absorbed by :meth:`LLMBaseModel.create_model`'s
        process-wide LRU cache — calls for the same config are O(1).

        An explicit ``self.model = ...`` assignment (used by the parent
        :class:`Node` initializer and by tests that inject a mock) pins
        the instance via the setter below; pinned values win over lazy
        resolution until explicitly cleared with ``self.model = None``.
        """
        if self._pinned_model is not None:
            return self._pinned_model
        if self._agent_config_ref is None:
            return None
        return LLMBaseModel.create_model(
            agent_config=self._agent_config_ref,
            model_name=self._node_model_name,
            scope=self.scope,
        )

    @model.setter
    def model(self, value: Optional[LLMBaseModel]) -> None:
        """Pin (or clear) the model instance used by this node.

        The parent :class:`Node` class writes ``self.model = None`` during
        its own ``__init__`` and ``self.model = llm_model`` inside
        ``_initialize``. Without a setter those assignments would raise
        because ``model`` is declared as a property here. Storing the
        value in ``_pinned_model`` preserves the existing contract for
        legacy callers while still letting ``/model`` switches take effect
        whenever callers clear the pin.
        """
        self._pinned_model = value

    @property
    def session_manager(self):
        """Lazy node-owned SessionManager.

        Path layout (each layer optional except the leaf):
            {agent_config.session_dir}
              / {self.scope}                  — user/tenant isolation
              / {self.session_subdir}         — main_session_id when this node is a subagent
              / {self.session_id}.db
        """
        # Use getattr so tests that bypass __init__ still get the lazy default.
        if getattr(self, "_session_manager", None) is None:
            import os

            from datus.models.session_manager import SessionManager

            cfg = getattr(self, "agent_config", None)
            base_dir = getattr(cfg, "session_dir", None) if cfg is not None else None
            if not base_dir:
                from datus.utils.path_manager import get_path_manager

                base_dir = str(get_path_manager(agent_config=cfg).sessions_dir)
            user_scope = getattr(self, "scope", None)
            session_subdir = getattr(self, "session_subdir", None)

            if session_subdir:
                scoped_dir = SessionManager(session_dir=base_dir, scope=user_scope).session_dir
                nested_dir = os.path.join(scoped_dir, session_subdir)
                self._session_manager = SessionManager(session_dir=nested_dir, scope=None)
            else:
                self._session_manager = SessionManager(session_dir=base_dir, scope=user_scope)
        return self._session_manager

    @property
    def context_length(self) -> Optional[int]:
        """Context window of the current model, refreshed per access.

        Used by ``/compact`` / auto-compaction heuristics that divide
        current token usage by the model's context budget. Falling back
        to ``None`` (rather than 0) keeps those heuristics inert when the
        active model doesn't publish a window.
        """
        current = self.model
        if current is None:
            return None
        try:
            return current.context_length()
        except Exception:
            return None

    # ── Plan mode lifecycle ─────────────────────────────────────────────

    def activate_plan_mode(self) -> str:
        """Turn plan mode on.

        Reuses an existing ``plan_file_path`` when present (e.g. left over
        after ``confirm_plan`` exited plan mode without a full reset) so the
        next plan session continues on the same markdown file. Allocates a
        fresh ``./.datus/plans/{short_uuid}.md`` only when no path is set.

        Always resets ``workflow_prompt_sent=False`` so the next user prompt
        carries the full workflow description.

        Returns:
            The plan file path (absolute or project-relative).
        """
        if self.plan_mode_active and self.plan_file_path:
            return self.plan_file_path

        self.plan_mode_active = True
        self.workflow_prompt_sent = False

        # Reuse a previously-allocated path (e.g. confirm_plan exit) when
        # available — this lets the user re-enter the same plan session.
        if self.plan_file_path:
            logger.info(f"Plan mode reactivated: plan_file_path={self.plan_file_path}")
            self._persist_plan_mode_state()
            return self.plan_file_path

        plan_dir = os.path.join(self._resolve_workspace_root(), ".datus", "plans")
        os.makedirs(plan_dir, exist_ok=True)
        # Short id keeps the path human-friendly; uuid4 has 122 bits of
        # entropy, so 8 hex chars (32 bits) is still ample for collision
        # avoidance within a project's plans directory.
        self.plan_file_path = os.path.join(plan_dir, f"{uuid.uuid4().hex[:8]}.md")
        # Pre-create an empty plan file so the LLM can read/edit it on the
        # first turn (it commonly probes with ``read_file`` before writing).
        try:
            with open(self.plan_file_path, "w", encoding="utf-8") as _f:
                pass
        except OSError as exc:
            logger.warning(f"Failed to pre-create plan file {self.plan_file_path}: {exc}")
        logger.info(f"Plan mode activated: plan_file_path={self.plan_file_path}")
        self._persist_plan_mode_state()
        return self.plan_file_path

    def deactivate_plan_mode(self) -> None:
        """Turn plan mode off for this turn while preserving the plan file.

        ``plan_file_path`` is allocated exactly once per session (per node
        lifetime) and **never** cleared — toggling plan mode off and back on
        always returns to the same markdown file. This keeps the user's
        narrative continuous across Shift+Tab toggles and ``confirm_plan``
        exits within a single session.

        Only ``plan_mode_active`` and ``workflow_prompt_sent`` flip back; the
        on-disk file and its in-memory path remain.
        """
        if self.plan_mode_active:
            logger.info(f"Plan mode paused: plan_file_path={self.plan_file_path}")
        self.plan_mode_active = False
        self.workflow_prompt_sent = False
        self._persist_plan_mode_state()

    def is_in_plan_mode(self) -> bool:
        """Return True when plan mode is currently active for this node."""
        return self.plan_mode_active

    def build_plan_mode_enhanced_prompt(self) -> str:
        """Render the plan_mode_system template based on current plan-mode state.

        The template branches on ``workflow_prompt_sent``: when False, the full
        workflow description is rendered; otherwise a short reminder. After the
        full version is rendered, ``workflow_prompt_sent`` is flipped to True
        so subsequent prompts only carry the reminder.
        """
        if not self.plan_mode_active or not self.plan_file_path:
            return ""

        try:
            rendered = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name="plan_mode_system",
                version=None,
                plan_file_path=self.plan_file_path,
                workflow_prompt_sent=self.workflow_prompt_sent,
            )
        except FileNotFoundError:
            logger.warning("plan_mode_system template not found, using inline fallback")
            if self.workflow_prompt_sent:
                rendered = (
                    f"Plan mode is active. Refer to the workflow already described. "
                    f"Plan file: {self.plan_file_path}. Continue iterating or call confirm_plan."
                )
            else:
                rendered = (
                    f"Plan mode is active. Plan file path: {self.plan_file_path}. "
                    "Use read-only tools and only edit the plan file; call confirm_plan when ready."
                )
        # Flip the flag so future prompts only carry the short reminder.
        self.workflow_prompt_sent = True
        self._persist_plan_mode_state()
        return rendered

    def _agent_state_file(self) -> Optional[Path]:
        """Return the session-bound state file path, or ``None`` when unavailable.

        Returns ``None`` when ``session_id`` has not been allocated yet, or
        when the path manager fails (e.g. empty ``project_name``). Callers
        treat ``None`` as "skip persistence this turn".
        """
        if not self.session_id:
            return None
        try:
            from datus.utils.path_manager import get_path_manager

            return get_path_manager(agent_config=self.agent_config).agent_state_path(self.session_id)
        except Exception as exc:  # noqa: BLE001 — persistence must never crash node logic
            logger.debug("agent_state_path unavailable: %s", exc)
            return None

    def _persist_plan_mode_state(self) -> None:
        """Flush current plan-mode fields to disk. No-op without session_id."""
        state_path = self._agent_state_file()
        if state_path is None:
            return
        from datus.storage.session_state import PlanModeState

        PlanModeState(
            plan_mode_active=self.plan_mode_active,
            plan_file_path=self.plan_file_path,
            workflow_prompt_sent=self.workflow_prompt_sent,
        ).save(state_path)

    def restore_plan_mode_state(self) -> None:
        """Re-hydrate plan-mode fields from disk into this node.

        Idempotent; safe to call multiple times. Invoked automatically by:
          1. ``__init__`` when caller passes ``session_id``
          2. The ``session_id`` setter when value transitions to non-empty

        When no on-disk state file exists yet (fresh session), this is a
        no-op — the in-memory defaults / values already on the node are
        preserved. This matters because ``_get_or_create_session`` may
        allocate a new ``session_id`` *after* the user already activated
        plan mode, and we must not wipe their in-flight plan_file_path.

        ``_plan_just_confirmed`` is intentionally NOT restored — it is a
        turn-local one-shot flag and should always start False on resume.
        """
        state_path = self._agent_state_file()
        if state_path is None or not state_path.exists():
            return
        from datus.storage.session_state import PlanModeState

        loaded = PlanModeState.load(state_path)
        self.plan_mode_active = loaded.plan_mode_active
        self.plan_file_path = loaded.plan_file_path
        self.workflow_prompt_sent = loaded.workflow_prompt_sent
        self._plan_just_confirmed = False
        logger.info(
            "Plan mode state restored for session %s: active=%s path=%s prompt_sent=%s",
            self.session_id,
            self.plan_mode_active,
            self.plan_file_path,
            self.workflow_prompt_sent,
        )

    def _get_plan_mode_tools(self) -> List[Tool]:
        """Build the plan-mode func tools (``confirm_plan`` + ``todo_*``).

        - **Sub-agent** (``self._is_subagent is True``): returns ``[]``.
          Sub-agents are invoked by a parent agent that already owns planning,
          so they must not expose their own planning surface.
        - **Main agent**: always returns the tools, regardless of
          ``plan_mode_active``. The user can pre-activate plan mode to *force*
          the LLM through the file-backed workflow, but the tools themselves
          are visible from the start so the LLM can judge whether to use them.
        """
        if getattr(self, "_is_subagent", False):
            return []

        from datus.tools.func_tool.plan_tools import ConfirmPlanTool, PlanTool

        # PlanTool keeps a reference to the agents-SDK session for backward-
        # compat but does not actually read it, so passing the current value
        # (which may still be None at setup time) is safe.
        # Lambda resolves session_id lazily — at setup time it's still None,
        # ``_get_or_create_session`` allocates it on the first turn. Snapshot
        # would leave the storage permanently unbound and never persist.
        tools: List[Tool] = list(PlanTool(self._session, session_id=lambda: self.session_id).available_tools())
        tools.extend(ConfirmPlanTool(self).available_tools())
        return tools

    def _register_plan_mode_tools(self) -> None:
        """Append plan-mode tools to ``self.tools`` at node setup time.

        Subclasses call this at the end of any code path that (re)builds
        ``self.tools`` from scratch — e.g. inside ``_rebuild_tools()`` /
        ``setup_tools()`` and after datasource swaps. The call is a no-op
        for sub-agents (``_is_subagent=True``).
        """
        if getattr(self, "_is_subagent", False):
            return
        plan_tools = self._get_plan_mode_tools()
        if not plan_tools:
            return
        if self.tools is None:
            self.tools = []
        self.tools.extend(plan_tools)

    def _sync_plan_mode_state(self, user_input: Any) -> None:
        """Reconcile ``self.plan_mode_active`` with the input's ``plan_mode`` flag.

        - ``user_input.plan_mode == True`` → idempotently activate (reuse the
          existing plan file across turns).
        - Flag absent / False but currently active → deactivate (user toggled
          plan mode off mid-session).
        - Sub-agent invocations never carry the flag, so this is a no-op.
        """
        if getattr(user_input, "plan_mode", False):
            self.activate_plan_mode()
        elif self.is_in_plan_mode():
            self.deactivate_plan_mode()

    def _build_enhanced_message(
        self,
        user_input: Any,
        extra_enhanced_parts: Optional[List[str]] = None,
    ) -> str:
        """Assemble the per-turn user prompt for the LLM.

        Composes (in order):
        1. Plan-mode state transition (activate / deactivate) based on
           ``user_input.plan_mode``.
        2. Shared context parts read from *user_input* via ``getattr``
           (so subclasses with sparser inputs still work):
           - ``external_knowledge`` → "MUST use these business logic" block
           - DB-context block (dialect + catalog/database/db_schema)
           - ``schemas`` (list of :class:`TableSchema`) → "Available tables"
           - ``metrics`` → "Metrics:" block
           - ``reference_sql`` → "Reference SQL:" block
        3. Subclass-supplied ``extra_enhanced_parts`` (already formatted).
        4. Plan-mode workflow prompt when plan mode is active.

        Args:
            user_input: The node's input model. Attributes are read via
                ``getattr`` so unrelated fields are simply skipped. The raw
                user text comes from ``user_input.user_message``.
            extra_enhanced_parts: Already-formatted strings to splice into
                the enhanced section (after the standard parts, before the
                plan-mode workflow prompt). Use this for subclass-specific
                context (e.g. compare's pair-of-SQL block) so the user-side
                of the structured envelope remains the raw user message.

        Returns the final user-facing string. When no enhanced parts apply,
        returns the raw user message unchanged; otherwise wraps both in a
        structured JSON ``[enhanced, user]`` envelope.
        """
        self._sync_plan_mode_state(user_input)

        enhanced_parts: List[str] = []

        ext_know = getattr(user_input, "external_knowledge", "") or ""
        if ext_know:
            enhanced_parts.append(f"MUST use these business logic:\n{ext_know}")

        db_type = getattr(self.agent_config, "db_type", "") if self.agent_config else ""
        if db_type:
            # Always resolve empty database via the connector default — the
            # helper is a no-op when no connector is wired or value is set.
            from datus.utils.node_utils import resolve_database_name_for_prompt

            connector = None
            db_func_tool = getattr(self, "db_func_tool", None)
            if db_func_tool is not None:
                connector = getattr(db_func_tool, "connector", None)
            effective_database = resolve_database_name_for_prompt(
                connector,
                getattr(user_input, "database", "") or "",
            )
            ctx = build_database_context(
                db_type,
                catalog=getattr(user_input, "catalog", "") or "",
                database=effective_database or "",
                schema=getattr(user_input, "db_schema", "") or "",
            )
            if ctx:
                enhanced_parts.append(ctx)

        schemas = getattr(user_input, "schemas", None)
        if schemas:
            from datus.schemas.node_models import TableSchema

            table_names_str = TableSchema.table_names_to_prompt(schemas)
            enhanced_parts.append(
                "Available tables (MUST use these tables and ONLY use these "
                f"table names in FROM/JOIN clauses): \n{table_names_str}"
            )

        metrics = getattr(user_input, "metrics", None)
        if metrics:
            enhanced_parts.append(f"Metrics: \n{to_str([item.model_dump() for item in metrics])}")

        reference_sql = getattr(user_input, "reference_sql", None)
        if reference_sql:
            enhanced_parts.append(f"Reference SQL: \n{to_str([item.model_dump() for item in reference_sql])}")

        if extra_enhanced_parts:
            enhanced_parts.extend(p for p in extra_enhanced_parts if p)

        if self.is_in_plan_mode():
            plan_prompt = self.build_plan_mode_enhanced_prompt()
            if plan_prompt:
                enhanced_parts.append(plan_prompt)
        elif getattr(self, "_plan_just_confirmed", False) and self.plan_file_path:
            # One-shot reminder on the turn immediately following a successful
            # ``confirm_plan``. Tells the LLM the plan is approved and it
            # should execute the steps instead of asking for further input.
            enhanced_parts.append(
                "## Post-Plan Execution\n"
                f"You just confirmed the plan at {self.plan_file_path}. Plan mode is "
                "exited. The user's next message is a continuation cue — do NOT ask "
                "what to do next; instead read the plan file, materialise its actionable "
                "steps via todo_write (one item per step, each with a `title` ≤ 8 words "
                "and full `content`), then call todo_update(id, 'in_progress') before "
                "starting each step and todo_update(id, 'completed') when done. Use "
                "ask_user only when a step genuinely requires user input that cannot "
                "be inferred."
            )
            self._plan_just_confirmed = False

        user_message = getattr(user_input, "user_message", "") or ""
        if not enhanced_parts:
            return user_message

        enhanced_context = "\n\n".join(enhanced_parts)
        return build_structured_content(enhanced_context, user_message)

    def get_node_name(self) -> str:
        """
        Get the template name for this agentic node. Overwrite this method if you need a special name

        Default implementation extracts from class name:
        - ChatAgenticNode -> "chat"
        - GenerateAgenticNode -> "generate"

        Returns:
            Node name that will be used to construct the full template filename and use in agent.yml
        """
        class_name = self.__class__.__name__
        # Remove "AgenticNode" suffix and convert to lowercase
        if class_name.endswith("AgenticNode"):
            template_name = class_name[:-11]  # Remove "AgenticNode" (11 characters)
        else:
            template_name = class_name

        return template_name.lower()

    def get_node_class_name(self) -> str:
        """Canonical identifier for this node's underlying class.

        ``get_node_name()`` may return a per-instance alias when a subagent is
        registered under a custom id (e.g. ``my_dashboard`` backed by
        ``GenDashboardAgenticNode``). Scoping mechanisms like
        ``SkillMetadata.allowed_agents`` need a stable class-level identifier so
        a whitelist written against the canonical class (``gen_dashboard``)
        still applies to all aliases of that class.

        Resolution order:
        1. ``type(self).NODE_NAME`` if the subclass declares it — the
           recommended form, used by ``gen_dashboard``, ``gen_table``,
           ``scheduler``, ``gen_skill`` etc.
        2. Otherwise derive from the class name via the *base*
           ``AgenticNode.get_node_name`` (e.g. ``ExploreAgenticNode`` →
           ``explore``). This is the safety net for alias-capable subclasses
           that haven't added ``NODE_NAME``: we must NOT fall back to
           ``self.get_node_name()``, since overrides there return the alias.

        Returns:
            A stable class-level identifier independent of any alias.
        """
        node_class = getattr(type(self), "NODE_NAME", None)
        if node_class:
            return node_class
        return AgenticNode.get_node_name(self)

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
        template_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get the system prompt for this agentic node using PromptManager.

        The template name follows the pattern: {get_node_name()}_system_{version}

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use, overrides agent config version
            template_context: Optional extra keyword arguments forwarded into the
                template render call. Subclasses populate this via
                ``_prepare_template_context`` when their templates need extra
                variables beyond the common ones.

        Returns:
            System prompt string loaded from the template
        """
        # Get prompt version from parameter, fallback to agent config, then use default
        version = prompt_version
        if version is None and self.agent_config and hasattr(self.agent_config, "prompt_version"):
            version = self.agent_config.prompt_version

        root_path = self._resolve_workspace_root()

        # Construct template name: {template_name}_system_{version}
        template_name = f"{self.get_node_name()}_system"

        render_kwargs: Dict[str, Any] = {
            "agent_config": self.agent_config,
            "datasource": getattr(self.agent_config, "current_datasource", None) if self.agent_config else None,
            "workspace_root": root_path,  # DEPRECATED: Use semantic_model_dir or sql_summary_dir instead
            "conversation_summary": conversation_summary,
        }
        if template_context:
            render_kwargs.update(template_context)

        try:
            # Use prompt manager to render the template
            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name,
                version=version,
                **render_kwargs,
            )

        except FileNotFoundError as e:
            # Template not found - throw DatusException
            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": version or "latest"},
            ) from e
        except Exception as e:
            # Other template errors - wrap in DatusException
            logger.error(f"Template loading error for '{template_name}': {e}")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

        return self._finalize_system_prompt(base_prompt)

    def _finalize_system_prompt(self, base_prompt: str, memory_node_name_override: Optional[str] = None) -> str:
        """
        Finalize system prompt by injecting skill context, memory context, and ensuring skill tools.

        All subclasses should call this at the end of their _get_system_prompt() override
        to ensure skills and memory are properly injected regardless of how the template is rendered.

        Args:
            base_prompt: The rendered template prompt
            memory_node_name_override: When provided, inject memory for this node name instead of
                ``self.get_node_name()``. Used by FeedbackAgenticNode to inject the caller's memory
                (the feedback node has no memory of its own).

        Returns:
            Prompt with skills XML and memory context appended
        """
        # Inject AGENTS.md project context if present in cwd
        agents_md = self._load_agents_md()
        if agents_md:
            base_prompt = base_prompt + "\n\n" + agents_md

        # Ensure skill tools are in self.tools (lazy injection after subclass setup_tools()).
        self._ensure_skill_tools_in_tools()

        # Same lazy-injection trick for the general-purpose BashTool.
        self._ensure_bash_tool_in_tools()

        # Inject available skills XML into system prompt when skill_func_tool is active.
        if self.skill_func_tool:
            skills_xml = self._get_available_skills_context()
            if skills_xml:
                base_prompt = base_prompt + "\n\n" + skills_xml

        # Inject memory context for eligible nodes.
        base_prompt = self._inject_memory_context(base_prompt, override_node_name=memory_node_name_override)

        # Inject response language policy so every agentic node — including
        # sub-agents invoked via ``task`` — honors the configured output language.
        base_prompt = self._inject_response_language(base_prompt)

        return base_prompt

    def _inject_response_language(self, base_prompt: str) -> str:
        """Append a language directive driven by ``agent_config.language``.

        When ``language`` is unset (``None`` or empty), this is a no-op so the
        model decides the response language on its own. Setting a code (e.g.
        ``"en"``/``"zh"``) in yaml or via the Chat API pins every AgenticNode
        to that output language through a single append hook.
        """
        language_raw = getattr(self.agent_config, "language", None)
        if not language_raw or not str(language_raw).strip():
            return base_prompt
        language_code = str(language_raw).strip()
        try:
            language_section = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name="response_language",
                version=None,
                language_code=language_code,
                language_name=_resolve_language_name(language_code),
            )
        except Exception as e:
            logger.warning(f"Failed to render response_language template: {e}")
            return base_prompt
        if language_section and language_section.strip():
            base_prompt = base_prompt + "\n\n" + language_section
        return base_prompt

    def _inject_memory_context(self, base_prompt: str, override_node_name: Optional[str] = None) -> str:
        """Inject memory context into the system prompt.

        Injection rules (resolved in order):
        1. ``override_node_name`` provided (feedback path) → unconditional
           injection of that node's memory, writable.
        2. ``self.memory_enabled`` True (chat / custom subagents) → own memory,
           writable.
        3. ``inherited_memory`` contextvar set by ``SubAgentTaskTool`` for this
           node name → render the parent's memory in **read-only** mode. Built-in
           subagents launched via ``task`` see their originating parent's memory
           for context but cannot modify it.
        4. None of the above → no memory section is appended.

        Args:
            base_prompt: The prompt to append memory context to.
            override_node_name: When provided, look up memory for this node name instead of
                ``self.get_node_name()``. Enables injecting another node's memory (e.g. the
                feedback node injects its caller's memory).
        """
        from datus.configuration.inherited_memory_overrides import get_inherited_memory
        from datus.utils.memory_loader import get_memory_dir, load_memory_context

        read_only = False
        if override_node_name:
            node_name = override_node_name
        elif self.memory_enabled:
            node_name = self.get_node_name()
        else:
            inherited = get_inherited_memory(self.get_node_name())
            if not inherited:
                return base_prompt
            node_name = inherited
            read_only = True

        try:
            workspace_root = self._resolve_workspace_root()

            memory_content = load_memory_context(workspace_root, node_name)
            if read_only and not memory_content.strip():
                # Parent has no memory worth inheriting; skip the read-only block
                # entirely so we do not waste prompt budget on an empty notice.
                return base_prompt
            memory_dir = get_memory_dir(workspace_root, node_name)

            memory_section = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name="memory_context",
                version=None,
                has_memory=True,
                memory_content=memory_content,
                memory_dir=memory_dir,
                read_only=read_only,
                originating_agent=node_name,
            )

            if memory_section.strip():
                base_prompt = base_prompt + "\n\n" + memory_section
        except Exception as e:
            logger.warning(f"Failed to inject memory context for node '{node_name}': {e}")
        return base_prompt

    def _load_agents_md(self) -> str:
        """Load AGENTS.md from current working directory as project context.

        Returns first 200 lines wrapped in <project_context> tags.
        Returns empty string if file doesn't exist — all features work without it.
        """
        import os

        agents_md_path = os.path.join(os.getcwd(), "AGENTS.md")
        if not os.path.exists(agents_md_path):
            return ""

        try:
            with open(agents_md_path, encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return ""
            # Keep first 200 lines to stay within reasonable context budget
            max_lines = 200
            content = "".join(lines[:max_lines])
            if len(lines) > max_lines:
                content += f"\n... ({len(lines) - max_lines} more lines, see AGENTS.md for full content)"
            return f"<project_context>\n{content}\n</project_context>"
        except Exception as e:
            logger.debug(f"Failed to load AGENTS.md: {e}")
            return ""

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"{self.get_node_name()}_session_{str(uuid.uuid4())[:8]}"

    def _get_or_create_session(self) -> tuple[AdvancedSQLiteSession, Optional[str]]:
        """
        Get or create the session for this node.

        Returns:
            Tuple of (session, summary). The summary slot is always None now
            that compaction persists the summary directly into the session
            history; it is kept in the return type for backward compatibility
            with existing call sites that unpack two values.
        """
        if self._session is None:
            self._session = self.session_manager.create_session(self.session_id)
            logger.debug(f"Created session: {self.session_id}")

        return self._session, None

    async def _count_session_tokens(self) -> int:
        """
        Estimate current context window usage in tokens.

        Returns the last API call's input_tokens from the most recent execute,
        which represents the actual conversation size in the context window.
        Falls back to the last turn's total_tokens from turn_usage table.

        Returns:
            Estimated context window token usage
        """
        # Primary: get last_call_input_tokens from the most recent root assistant action.
        # Scope to root-level actions (depth == 0) so child/tool usage from sub-agents
        # doesn't leak into the parent session's context estimate.
        for action in reversed(self.actions):
            # Stop at the last root-level user message to scope to the current turn
            if action.role == ActionRole.USER and action.depth == 0:
                break
            if (
                action.role == ActionRole.ASSISTANT
                and action.depth == 0
                and isinstance(action.output, dict)
                and isinstance(action.output.get("usage"), dict)
            ):
                usage = action.output["usage"]
                last_call = usage.get("last_call_input_tokens", 0)
                if last_call > 0:
                    return last_call
                # Fallback within action: use input_tokens (still per-turn, not cumulative sum)
                input_tokens = usage.get("input_tokens", 0)
                if input_tokens > 0:
                    return input_tokens
                break

        # Fallback: get the latest turn's total_tokens from turn_usage table
        if self._session and hasattr(self._session, "get_turn_usage"):
            try:
                turn_usage = await self._session.get_turn_usage()
                if turn_usage:
                    # turn_usage is a list of per-turn records; use the last one
                    last_turn = turn_usage[-1] if isinstance(turn_usage, list) else turn_usage
                    if isinstance(last_turn, dict):
                        return last_turn.get("total_tokens", 0)
            except Exception as e:
                logger.debug(f"Failed to get turn usage for token counting: {e}")

        return 0

    async def _manual_compact(self) -> dict:
        """
        Manually compact the session by summarizing conversation history.

        Generates an LLM summary of the current session, clears the session's
        history, then writes a `user marker + assistant summary` pair back
        into the SAME session so subsequent LLM requests and UI history reads
        see the summary as the new visible turn. The session_id / .db file
        are preserved; no new session is created.

        Returns:
            Dict with success, summary, and summary_token count
        """
        try:
            model = self.model
        except Exception as exc:
            logger.warning("Cannot compact: model resolution failed: %s", exc)
            return {"success": False, "summary": "", "summary_token": 0}
        if not model:
            logger.warning("Cannot compact: no model available")
            return {"success": False, "summary": "", "summary_token": 0}

        # Lazily materialize the SQLite session when only session_id is known.
        # This happens after .resume, which sets self.session_id but leaves
        # self._session as None until the first execute call.
        if self._session is None and self.session_id:
            self._get_or_create_session()

        if not self._session:
            logger.warning("Cannot compact: no session available")
            return {"success": False, "summary": "", "summary_token": 0}

        try:
            logger.info(f"Starting manual compacting for session {self.session_id}")

            # 1. Generate summary using LLM with existing session
            summarization_prompt = (
                "Summarize our conversation up to this point. The summary should be a concise yet comprehensive "
                "overview of all key topics, questions, answers, and important details discussed. This summary "
                "will replace the current chat history to conserve tokens, so it must capture everything "
                "essential to understand the context and continue our conversation effectively as if no "
                "information was lost."
            )

            try:
                result = await self.model.generate_with_tools(
                    prompt=summarization_prompt,
                    session=self._session,
                    max_turns=1,
                    temperature=0.3,
                    max_tokens=2000,
                    agent_name=self.get_node_name(),
                )
                summary = result.get("content", "")
                summary_token = result.get("usage", {}).get("output_tokens", 0)
                logger.debug(f"Generated summary: {len(summary)} characters, {summary_token} tokens")
            except Exception as e:
                logger.error(f"Failed to generate summary with LLM: {e}")
                return {"success": False, "summary": "", "summary_token": 0}

            # 2. Persist summary back into the session: clear the existing history
            #    and append a user/assistant pair so the summary becomes the new
            #    visible turn. Subsequent LLM requests will pick it up as context
            #    via the OpenAI Agents SDK session, and UI history reads will
            #    surface the summary instead of an empty session.
            try:
                await self._session.clear_session()
                await self._session.add_items(
                    [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "[Previous conversation was compacted to save context. Summary below.]",
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": summary}],
                        },
                    ]
                )
            except Exception as persist_err:
                logger.error(f"Failed to persist compact summary: {persist_err}")
                return {"success": False, "summary": "", "summary_token": 0}

            logger.info(
                f"Manual compacting completed. Session {self.session_id} cleared and "
                f"summary persisted ({len(summary)} chars, {summary_token} output tokens)"
            )
            return {"success": True, "summary": summary, "summary_token": summary_token}

        except Exception as e:
            logger.error(f"Manual compacting failed: {e}")
            return {"success": False, "summary": "", "summary_token": 0}

    async def _auto_compact(self) -> bool:
        """
        Automatically compact when session approaches token limit (~90%).

        Returns:
            True if compacting was triggered and successful, False otherwise
        """
        try:
            model = self.model
        except Exception:
            return False
        if not model or not self.context_length:
            return False

        try:
            current_tokens = await self._count_session_tokens()

            if current_tokens > (self.context_length * 0.9):
                logger.info(f"Auto-compacting triggered: {current_tokens}/{self.context_length} tokens")
                try:
                    result = await self._manual_compact()
                    return result.get("success", False)
                except Exception as e:
                    logger.error(f"Auto-compact manual compaction failed: {e}")
                    return False

            return False

        except Exception as e:
            logger.error(f"Auto-compact check failed: {e}")
            return False

    def _parse_node_config(self, agent_config: Optional[AgentConfig], node_name: str) -> dict:
        """
        Parse node configuration from agent.yml.

        Args:
            agent_config: Agent configuration
            node_name: Name of the node configuration

        Returns:
            Dictionary containing node configuration
        """
        if not agent_config or not hasattr(agent_config, "agentic_nodes"):
            return {}

        nodes_config = agent_config.agentic_nodes
        from datus.configuration.scoped_context_overrides import get_override

        override = get_override(node_name)
        if node_name not in nodes_config and override is None:
            logger.debug(f"Node configuration '{node_name}' not found in agent.yml, using default configuration")
            return {}

        if override is not None:
            # Layer the runtime override (with parent-merged scoped_context) on top
            # of any yaml entry, so child node __init__ sees the effective context.
            base = nodes_config.get(node_name) if node_name in nodes_config else {}
            base_dict = (
                dict(base) if isinstance(base, dict) else (base.model_dump() if hasattr(base, "model_dump") else {})
            )
            node_config = {**base_dict, **override.model_dump(exclude_unset=True)}
        else:
            node_config = nodes_config[node_name]

        # Extract configuration attributes
        config = {}

        # Basic node config attributes
        if isinstance(node_config, dict):
            config["model"] = node_config.get("model")
        elif hasattr(node_config, "model"):
            config["model"] = node_config.model

        # Check direct attributes on node_config
        direct_attributes = [
            "system_prompt",
            "agent_description",
            "prompt_version",
            "prompt_language",
            "tools",
            "mcp",
            "skills",  # AgentSkills pattern filter (e.g., "sql-*, data-*")
            "permissions",  # Node-specific permission overrides
            "hooks",
            "rules",
            "max_turns",
            "workspace_root",
            "scoped_context",
            "scoped_kb_path",
            "adapter_type",
            "semantic_adapter",
            "sql_file_threshold",
            "sql_preview_lines",
            "bi_platform",
            "scheduler_service",
            "subagents",
            # Read by ask_report / ask_dashboard nodes to resolve the
            # bound artifact directory at startup.
            "artifact_slug",
        ]
        for attr in direct_attributes:
            # Handle both dict and object access patterns
            if attr not in config:
                value = None
                if isinstance(node_config, dict):
                    value = node_config.get(attr)
                elif hasattr(node_config, attr):
                    value = getattr(node_config, attr)

                if value is not None:
                    config[attr] = value

        # Normalize rules: convert dict items to strings (YAML parsing issue workaround)
        if "rules" in config and isinstance(config["rules"], list):
            normalized_rules = []
            for rule in config["rules"]:
                if isinstance(rule, dict):
                    # Convert dict to string format "key: value"
                    rule_str = ", ".join(f"{k}: {v}" for k, v in rule.items())
                    normalized_rules.append(rule_str)
                else:
                    normalized_rules.append(str(rule))
            config["rules"] = normalized_rules

        logger.info(f"Parsed node configuration for '{node_name}': {config}")
        return config

    def _setup_permission_manager(self) -> None:
        """
        Initialize unified permission manager for tools, MCP, and skills.

        The permission manager uses global config from agent.yml and node-specific
        overrides to control access to tools/MCP/skills with allow/deny/ask levels.

        ``execution_mode="workflow"`` nodes (``/bootstrap``, scheduler subagents,
        ``auto_create``, etc.) ignore the user's profile and run under a fresh
        ``dangerous`` profile. Combined with the non-interactive ``PermissionHooks``
        gate (see :meth:`_ensure_permission_hooks`), this means workflow flows
        execute exactly the operations ``dangerous`` allows and fail loudly on
        anything else — no broker prompts, no auto-approval, no drift when the
        user happens to be on ``normal`` or ``auto``.
        """
        if not self.agent_config or not hasattr(self.agent_config, "permissions_config"):
            return

        try:
            from datus.tools.permission.permission_manager import PermissionManager

            is_workflow = getattr(self, "execution_mode", None) == "workflow"
            if is_workflow:
                from datus.tools.permission.profiles import get_profile

                permissions_config = get_profile("dangerous")
                active_profile = "dangerous"
            else:
                permissions_config = self.agent_config.permissions_config
                active_profile = getattr(self.agent_config, "active_profile_name", None) or "normal"

            if not permissions_config:
                return

            # Get node-specific permission overrides from node_config
            node_permissions = self.node_config.get("permissions", {})

            self.permission_manager = PermissionManager(
                global_config=permissions_config,
                node_overrides={self.get_node_name(): node_permissions} if node_permissions else {},
                active_profile=active_profile,
            )
            # Forward existing callback to permission manager
            if self._permission_callback:
                self.permission_manager.set_permission_callback(self._permission_callback)
            logger.debug(
                f"Permission manager initialized for node '{self.get_node_name()}' "
                f"(profile={active_profile}, workflow={is_workflow})"
            )

        except Exception as e:
            logger.exception("Failed to setup permission manager")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Permission manager init failed: {e}"},
            ) from e

    def _setup_skill_manager(self) -> None:
        """
        Initialize skill manager from agent config.

        The skill manager coordinates skill discovery, permission checking,
        and content loading for the AgentSkills integration.
        """
        if not self.agent_config or not hasattr(self.agent_config, "skills_config"):
            return

        skills_config = self.agent_config.skills_config
        if not skills_config:
            return

        try:
            from datus.tools.skill_tools.skill_manager import SkillManager

            self.skill_manager = SkillManager(
                config=skills_config,
                permission_manager=self.permission_manager,
            )
            logger.debug(
                f"Skill manager initialized for node '{self.get_node_name()}' "
                f"with {self.skill_manager.get_skill_count()} skills"
            )

        except Exception as e:
            logger.error(f"Failed to setup skill manager: {e}")

    def _setup_skill_func_tools(self) -> None:
        """
        Setup skill function tools when explicitly configured in agentic_nodes.

        Only activates if 'skills' is explicitly set in node_config.
        ChatAgenticNode overrides skill setup in its own setup_tools(), so this primarily
        serves other AgenticNode subclasses (GenReport, GenMetrics, etc.).

        If skill_manager was not created (e.g. no global 'skills:' section in agent.yml),
        creates one with default SkillConfig (same behavior as ChatAgenticNode).

        NOTE: This only creates the SkillFuncTool instance (self.skill_func_tool).
        The actual tools are injected into self.tools lazily via _ensure_skill_tools_in_tools(),
        which is called from _get_system_prompt(). This avoids a timing issue where subclass
        setup_tools() resets self.tools = [] after __init__ completes.
        """
        skill_patterns_str = self.node_config.get("skills")
        if skill_patterns_str is None:
            # Fall back to the subclass-declared defaults so built-in subagents
            # work out of the box. An explicit empty string in yml opts out.
            skill_patterns_str = type(self).DEFAULT_SKILLS
            if skill_patterns_str:
                # Persist the resolved pattern so <available_skills> filtering
                # and any downstream reader sees the same value.
                self.node_config["skills"] = skill_patterns_str
        if not skill_patterns_str:
            return

        try:
            # Create skill_manager with defaults if not already initialized
            # (e.g. when agent.yml has no global 'skills:' section)
            if not self.skill_manager:
                from datus.tools.skill_tools.skill_manager import SkillManager

                self.skill_manager = SkillManager(
                    permission_manager=self.permission_manager,
                )
                logger.info(
                    f"Created default SkillManager for node '{self.get_node_name()}' "
                    f"with {self.skill_manager.get_skill_count()} skills"
                )

            from datus.tools.skill_tools.skill_func_tool import SkillFuncTool

            self.skill_func_tool = SkillFuncTool(
                manager=self.skill_manager,
                node_name=self.get_node_name(),
                node_class=self.get_node_class_name(),
                authoring_mode=self.SKILL_AUTHORING_MODE,
            )
            logger.info(
                f"Skill func tools activated for node '{self.get_node_name()}' with pattern '{skill_patterns_str}'"
            )
        except Exception as e:
            logger.error(f"Failed to setup skill func tools: {e}")

    @staticmethod
    def _merge_skill_patterns(existing_skills: Any, injected_skills: List[str]) -> str:
        """Merge runtime-injected skill patterns into the user's configured list.

        Platform-aware subagents (``scheduler``, ``gen_dashboard``, etc.) need to
        append a ``{platform}-*`` skill based on config without overriding the
        user's ``skills:`` yml entry. This helper merges the two sources,
        deduplicates, and returns the canonical comma-separated string that
        ``_setup_skill_func_tools`` expects.

        Args:
            existing_skills: Value of ``node_config["skills"]`` — either a
                comma-separated string, a list of patterns, or ``None``.
            injected_skills: Skill names the subclass wants to guarantee.

        Returns:
            Comma-separated pattern string with injected skills appended after
            the user's patterns and duplicates removed (first occurrence wins).
        """
        merged_patterns: List[str] = []

        if isinstance(existing_skills, str):
            merged_patterns.extend([pattern.strip() for pattern in existing_skills.split(",") if pattern.strip()])
        elif isinstance(existing_skills, list):
            merged_patterns.extend(
                [pattern.strip() for pattern in existing_skills if isinstance(pattern, str) and pattern.strip()]
            )

        for skill in injected_skills:
            if skill not in merged_patterns:
                merged_patterns.append(skill)

        return ", ".join(merged_patterns)

    def _setup_ask_user_tool(self):
        """Setup ask-user tool so the agent can ask clarifying questions.

        Creates an AskUserTool backed by this node's InteractionBroker.
        Subclasses call this from their ``setup_tools()``; tools are
        automatically appended to ``self.tools``.
        """
        try:
            from datus.tools.func_tool.ask_user_tools import AskUserTool

            broker = self._get_or_create_broker()
            self.ask_user_tool = AskUserTool(broker=broker)
            if self.tools is not None:
                self.tools.extend(self.ask_user_tool.available_tools())
            logger.debug("Setup ask_user tool")
        except Exception as e:
            logger.error(f"Failed to setup ask_user tool: {e}")
            self.ask_user_tool = None

    def _setup_sub_agent_task_tool(self):
        """Setup SubAgentTaskTool based on subagents config or node default.

        Skipped when ``is_subagent`` is True (nodes created by SubAgentTaskTool)
        to enforce strict 2-level depth — subagent nodes never get their own task tool.
        """
        if self._is_subagent:
            return

        from datus.schemas.agent_models import SubAgentConfig

        subagents_str = self.node_config.get("subagents")
        if subagents_str is None:
            subagents_str = self.DEFAULT_SUBAGENTS

        parsed = SubAgentConfig(subagents=subagents_str).subagent_list
        if not parsed:
            return  # Empty = no task tool

        if parsed == ["*"]:
            allowed = None  # None = SubAgentTaskTool discovers all types
        else:
            allowed = parsed

        try:
            from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

            self.sub_agent_task_tool = SubAgentTaskTool(
                agent_config=self.agent_config,
                allowed_subagents=allowed,
                parent_node_name=self.get_node_name(),
            )
            self.sub_agent_task_tool.set_action_bus(self.action_bus)
            self.sub_agent_task_tool.set_interaction_broker(self.interaction_broker)
            self.sub_agent_task_tool.set_parent_node(self)
        except Exception as e:
            logger.error(f"Failed to setup SubAgent task tool: {e}")
            self.sub_agent_task_tool = None

    def _ensure_skill_tools_in_tools(self) -> None:
        """
        Ensure skill function tools are present in self.tools.

        Called lazily (from _get_system_prompt) to avoid the timing issue where
        subclass setup_tools() resets self.tools = [] after base __init__ runs.
        Idempotent — safe to call multiple times.
        """
        if not self.skill_func_tool:
            return

        skill_tool_names = {t.name for t in self.skill_func_tool.available_tools()}
        existing_names = {t.name for t in (self.tools or [])}

        if skill_tool_names.issubset(existing_names):
            return  # Already added

        if self.tools is None:
            self.tools = []
        self.tools.extend(self.skill_func_tool.available_tools())
        logger.info(
            f"Skill tools injected into node '{self.get_node_name()}': "
            f"{[t.name for t in self.skill_func_tool.available_tools()]}"
        )

    def _setup_bash_tool(self) -> None:
        """Create the node's general-purpose :class:`BashTool` instance.

        Available to every agentic node when ``agent.bash.enabled`` is
        ``True`` (the default). ``allowed_patterns=["*"]`` means the tool
        exposes ``execute_command`` for any shell command; per-call gating
        is the responsibility of the ``bash_tools`` ASK rule in the
        permission profile, not a static pattern whitelist.

        Only creates the instance — the tool enters ``self.tools`` via
        :meth:`_ensure_bash_tool_in_tools` so subclass ``setup_tools()``
        resets don't drop it.
        """
        if not getattr(self.agent_config, "bash_tool_enabled", True):
            logger.debug("Bash tool disabled via agent.bash.enabled=false")
            self.bash_tool = None
            return
        # Fail closed: ``allowed_patterns=["*"]`` relies on the ``bash_tools``
        # ASK/DENY rule wired by ``_ensure_permission_hooks``. When no
        # ``permission_manager`` exists those hooks are a no-op, so creating
        # the tool would expose unrestricted shell execution to the model.
        if self.permission_manager is None:
            logger.warning("Skipping bash tool because permission enforcement is unavailable")
            self.bash_tool = None
            return
        try:
            from datus.tools.func_tool.bash_tool import BashTool

            self.bash_tool = BashTool(
                workspace_root=self._resolve_workspace_root(),
                allowed_patterns=["*"],
            )
            logger.debug(f"Setup bash tool with workspace: {self.bash_tool.workspace_root}")
        except Exception as e:
            logger.error(f"Failed to setup bash tool: {e}")
            self.bash_tool = None

    def _ensure_bash_tool_in_tools(self) -> None:
        """Ensure the BashTool's ``execute_command`` is in ``self.tools``.

        Mirrors :meth:`_ensure_skill_tools_in_tools` — called lazily so the
        late ``setup_tools()`` reset in subclasses doesn't strip the tool.
        Idempotent.
        """
        if not self.bash_tool:
            return

        bash_tool_names = {t.name for t in self.bash_tool.available_tools()}
        if not bash_tool_names:
            return

        existing_names = {t.name for t in (self.tools or [])}
        if bash_tool_names.issubset(existing_names):
            return

        if self.tools is None:
            self.tools = []
        self.tools.extend(self.bash_tool.available_tools())
        logger.info(
            f"Bash tool injected into node '{self.get_node_name()}': "
            f"{[t.name for t in self.bash_tool.available_tools()]}"
        )

    def set_permission_callback(self, callback: Callable[[str, str, Dict[str, Any]], Awaitable[bool]]) -> None:
        """
        Set callback for ASK permission prompts.

        This callback is invoked when a tool/skill requires user confirmation
        before execution (ASK permission level).

        Args:
            callback: Async function(tool_category, tool_name, context) -> bool
                      Returns True if user approves, False otherwise
        """
        self._permission_callback = callback
        # Forward to permission manager if it exists
        if self.permission_manager:
            self.permission_manager.set_permission_callback(callback)
        logger.debug(f"Permission callback set for node '{self.get_node_name()}'")

    def _get_available_skills_context(self) -> str:
        """
        Generate <available_skills> XML context for system prompt injection.

        Returns the XML block listing skills the LLM can use via load_skill tool.
        Skills with DENY permission are filtered out.

        Returns:
            XML string for system prompt injection, empty string if no skills
        """
        if not self.skill_manager:
            return ""

        # Get skill patterns from node config (e.g., "sql-*, data-*")
        skill_patterns_str = self.node_config.get("skills", "")
        skill_patterns = None
        if skill_patterns_str:
            skill_patterns = self.skill_manager.parse_skill_patterns(skill_patterns_str)

        return self.skill_manager.generate_available_skills_xml(
            node_name=self.get_node_name(),
            patterns=skill_patterns,
            node_class=self.get_node_class_name(),
        )

    def _get_tool_category(self, tool_name: str) -> str:
        """
        Determine tool category from tool name for permission checking.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool category string: "db_tools", "mcp", "skills", or "tools"
        """
        # Check for skill-related tools
        if tool_name == "load_skill" or tool_name.startswith("skill_"):
            return "skills"

        # Check for database tools
        if tool_name.startswith("db_") or tool_name in [
            "list_tables",
            "describe_table",
            "execute_ddl",
            "execute_write",
            "transfer_query_result",
            "execute_sql",
            "get_sample_data",
        ]:
            return "db_tools"

        # Check for MCP tools (usually have mcp_ prefix or are in mcp_servers)
        mcp_tool_names = set()
        for server_name in self.mcp_servers.keys():
            mcp_tool_names.add(f"{server_name}_")
        for mcp_prefix in mcp_tool_names:
            if tool_name.startswith(mcp_prefix):
                return "mcp"

        # Default to generic tools category
        return "tools"

    def setup_input(self, workflow: "Workflow") -> Dict:
        """
        Setup input for agentic node from workflow context.

        Default implementation extracts common fields from workflow context
        and populates the input object. Subclasses can override for custom behavior.

        Args:
            workflow: Workflow instance containing context and task

        Returns:
            Dictionary with success status and message
        """
        if self.input is None:
            self.input = BaseInput()

        # Populate common fields from workflow context if input has these attributes
        if hasattr(self.input, "catalog"):
            self.input.catalog = workflow.task.catalog_name
        if hasattr(self.input, "database"):
            self.input.database = workflow.task.database_name
        if hasattr(self.input, "db_schema"):
            self.input.db_schema = workflow.task.schema_name
        if hasattr(self.input, "schemas"):
            self.input.schemas = workflow.context.table_schemas
        if hasattr(self.input, "metrics"):
            self.input.metrics = workflow.context.metrics

        return {"success": True, "message": f"Agentic node {self.type} input prepared"}

    def update_context(self, workflow: "Workflow") -> Dict:
        """
        Update workflow context with agentic node results.

        Default implementation stores SQL results if present.
        Subclasses can override for custom context updates.

        Args:
            workflow: Workflow instance to update

        Returns:
            Dictionary with success status and message
        """
        if not self.result:
            return {"success": False, "message": "No result to update context"}

        result = self.result

        # Store SQL generation results if present
        if hasattr(result, "sql") and result.sql:
            from datus.schemas.node_models import SQLContext

            new_record = SQLContext(
                sql_query=result.sql,
                explanation=getattr(result, "response", "") or getattr(result, "explanation", ""),
            )
            workflow.context.sql_contexts.append(new_record)

        return {"success": True, "message": "Agentic node context updated"}

    def execute(self) -> BaseResult:
        """
        Synchronous execution wrapper for agentic nodes.

        Agentic nodes are async by nature, so this wraps the async method
        to provide synchronous execution interface required by Node base class.

        Returns:
            BaseResult object with execution results
        """
        action_history_manager = ActionHistoryManager()

        async def _run_async():
            final_action = None
            async for action in self.execute_stream(action_history_manager):
                if action.status == ActionStatus.SUCCESS:
                    final_action = action
            return final_action

        try:
            # Get the final action from streaming execution
            final_action = asyncio.run(_run_async())

            # Extract result from final action output
            if final_action and final_action.output:
                output_data = final_action.output
                if isinstance(output_data, dict):
                    result_class = getattr(self, "result_class", None)
                    if result_class:
                        try:
                            self.result = result_class.model_validate(output_data)
                        except Exception as e:
                            logger.warning(f"Failed to validate result as {result_class.__name__}: {e}")
                            self.result = BaseResult(
                                success=output_data.get("success", True),
                                error=output_data.get("error"),
                            )
                    else:
                        self.result = BaseResult(
                            success=output_data.get("success", True),
                            error=output_data.get("error"),
                        )
                else:
                    self.result = output_data

            if not self.result:
                self.result = BaseResult(success=False, error="No result from execution")

            return self.result

        except Exception as e:
            logger.error(f"Agentic node execution error: {e}")
            self.result = BaseResult(success=False, error=str(e))
            return self.result

    # ── execute_stream template method ──────────────────────────────────
    #
    # The base class owns the streaming skeleton (input validation, session
    # setup, prompt assembly, retry loop, error handling). Subclasses customise
    # behaviour through the optional hooks declared below this method:
    #
    #   - ``_before_stream``               (async, side-effect init)
    #   - ``_build_template_context``      (extra render kwargs)
    #   - ``_compose_run_hooks``           (extra model hooks)
    #   - ``_maybe_rewrite_stream_action`` (live action rewrite, e.g. JSON→md)
    #   - ``_get_retry_policy``            (validate/retry strategy)
    #   - ``_build_success_result``        (REQUIRED — construct NodeResult)
    #
    # Subclasses also MUST declare ``result_class`` (a ``ClassVar[type[BaseResult]]``)
    # so the unified ``_build_error_result`` can construct the right NodeResult
    # subtype on the failure path.
    #
    # Subclasses MUST NOT override ``execute_stream`` itself — the template
    # contract is final. Add a new hook to ``AgenticNode`` if you encounter a
    # variation point that none of the existing hooks captures.

    # Declared as ``Optional`` so abstract intermediates (DeliverableAgenticNode,
    # BaseVisualArtifactAgenticNode) can leave it unset; concrete subclasses
    # must assign a real ``BaseResult`` subclass. Runtime check lives in
    # ``_build_error_result``.
    result_class: Any = None

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute the agentic node with streaming support — template method.

        Drives the full lifecycle: input validation → initial USER action →
        session setup → prompt assembly → optional retry loop around
        ``_stream_once`` → success/error result construction → closing
        ASSISTANT action. Subclasses customise via the hooks below, never by
        overriding this method.

        Args:
            action_history_manager: Optional action history manager. A fresh
                one is created when omitted.

        Yields:
            ActionHistory: Progress updates during execution.
        """
        from datus.agent.node.retry_policy import NoRetryPolicy
        from datus.agent.node.stream_run_context import StreamRunContext

        ahm = action_history_manager or ActionHistoryManager()
        if self.input is None:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message_args={"field_name": "input"},
            )

        ctx = StreamRunContext(
            user_input=self.input,
            action_history_manager=ahm,
            # Tolerate test doubles that bypass ``AgenticNode.__init__``
            # and therefore don't have ``pending_input_queue`` set.
            pending_input_queue=getattr(self, "pending_input_queue", None),
        )

        node_name = self.get_node_name()
        logger.info(
            "%s execute_stream start: session=%s msg=%r",
            node_name,
            getattr(self, "session_id", None),
            (getattr(self.input, "user_message", "") or "")[:120],
        )
        initial_action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=f"{node_name}_request",
            messages=f"User: {getattr(self.input, 'user_message', '')}",
            input_data=self.input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        ahm.add_action(initial_action)
        yield initial_action

        try:
            await self._before_stream(ctx)

            if self.execution_mode == "interactive":
                await self._auto_compact()
                ctx.session, ctx.conversation_summary = self._get_or_create_session()

            template_context = self._build_template_context(ctx)
            prompt_version = getattr(self.input, "prompt_version", None)
            if template_context:
                ctx.system_instruction = self._get_system_prompt(
                    conversation_summary=ctx.conversation_summary,
                    prompt_version=prompt_version,
                    template_context=template_context,
                )
            else:
                ctx.system_instruction = self._get_system_prompt(
                    conversation_summary=ctx.conversation_summary,
                    prompt_version=prompt_version,
                )

            # Compose the user prompt, optionally with a per-run override of
            # ``user_input.user_message`` set during ``_before_stream`` (used
            # by Compare and GenExtKnowledge to inject node-specific text
            # without mutating the caller's input object).
            if ctx.user_message_override is not None:
                original = self.input.user_message
                self.input.user_message = ctx.user_message_override
                try:
                    ctx.user_prompt = self._build_enhanced_message(self.input)
                finally:
                    self.input.user_message = original
            else:
                ctx.user_prompt = self._build_enhanced_message(self.input)

            policy = self._get_retry_policy() or NoRetryPolicy()
            max_attempts = max(1, getattr(policy, "max_attempts", 1))

            for attempt in range(1, max_attempts + 1):
                ctx.attempt = attempt
                policy.reset(ctx)

                async for stream_action in self._stream_once(ctx):
                    yield stream_action

                # Always probe ``should_retry`` so the policy can stash
                # per-attempt state (validation reports, verification flags)
                # for ``finalise`` to surface — even on the final attempt
                # when no further retry will actually fire.
                wants_retry = policy.should_retry(ctx)
                if not wants_retry or attempt >= max_attempts:
                    break
                for retry_action in policy.on_retry_actions(ctx):
                    ahm.add_action(retry_action)
                    yield retry_action
                next_prompt = policy.next_prompt(ctx)
                if next_prompt is not None:
                    ctx.user_prompt = next_prompt

            policy.finalise(ctx)

            result = self._build_success_result(ctx)
            self.result = result
            self.actions.extend(ahm.get_actions())

            # Optional post-build streaming hook. Visual-artifact subagents
            # override this to interleave finalize-progress messages around
            # their 10-15 s of post-validate LLM work; default is a no-op.
            # The hook may mutate ``result`` in place (e.g. populate
            # ``finalize_warnings``) and those mutations land in the
            # ``final_action`` dump below because ``model_dump()`` runs
            # after the hook is fully consumed.
            async for progress_action in self._stream_post_build(ctx, result):
                ahm.add_action(progress_action)
                yield progress_action

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=f"{node_name}_response",
                messages=(
                    f"{node_name} interaction completed successfully"
                    if getattr(result, "success", True)
                    else f"{node_name} interaction completed with failures"
                ),
                input_data=self.input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS if getattr(result, "success", True) else ActionStatus.FAILED,
            )
            ahm.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise
        except Exception as exc:
            error_msg = self._format_execution_error(exc)
            logger.error("%s execution error: %s", node_name, error_msg)

            error_result = self._build_error_result(exc, ctx)
            self.result = error_result
            ahm.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {error_msg}",
            )
            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="error",
                messages=f"{node_name} interaction failed: {error_msg}",
                input_data=self.input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            ahm.add_action(error_action)
            # Mirror the success path: persist this turn's actions onto the
            # node so cross-turn helpers (``_count_session_tokens``,
            # ``get_last_turn_usage``) don't keep reading stale state after
            # a failed attempt.
            self.actions.extend(ahm.get_actions())
            yield error_action

    async def _stream_once(self, ctx: "StreamRunContext") -> AsyncGenerator[ActionHistory, None]:
        """Run the model once and yield every action while collecting state.

        The template invokes this method ``max_attempts`` times (once for
        every retry-policy iteration). Each call:

        - Calls ``model.generate_with_tools_stream`` with the current
          ``ctx.user_prompt`` and per-node-configured tools / hooks.
        - Routes every emitted action through ``_maybe_rewrite_stream_action``
          so subclasses can re-shape items mid-flight (used by GenReport).
        - Updates ``ctx.response_content`` / ``ctx.last_successful_output``
          from assistant chunks (skipping ``is_thinking`` items so the
          model's internal monologue never lands in the final response) and
          ``ctx.last_tool_summary`` from successful tool actions.
        """
        async for stream_action in self.model.generate_with_tools_stream(
            prompt=ctx.user_prompt,
            tools=self.tools or [],
            mcp_servers=self.mcp_servers,
            instruction=ctx.system_instruction,
            max_turns=getattr(ctx.user_input, "max_turns", None) or self.max_turns,
            session=ctx.session,
            action_history_manager=ctx.action_history_manager,
            hooks=self._compose_run_hooks(ctx),
            agent_name=self.get_node_name(),
            interrupt_controller=self.interrupt_controller,
            pending_input_queue=ctx.pending_input_queue,
            # Defensive: test doubles that bypass ``AgenticNode.__init__``
            # may not have a broker; the model layer skips emit when None.
            interaction_broker=getattr(self, "interaction_broker", None),
        ):
            rewritten = self._maybe_rewrite_stream_action(stream_action, ctx)
            action_to_yield = rewritten or stream_action

            if (
                action_to_yield.role == ActionRole.ASSISTANT
                and action_to_yield.status == ActionStatus.SUCCESS
                and action_to_yield.output
            ):
                output = action_to_yield.output
                if isinstance(output, dict) and output.get("is_thinking") is not True:
                    ctx.last_successful_output = output
                    candidate = output.get("content", "") or output.get("response", "") or output.get("raw_output", "")
                    # Preserve dict candidates (used by Deliverable / ExtKnowledge
                    # for structured outputs); coerce only when the candidate is
                    # a non-empty non-string scalar.
                    if isinstance(candidate, str):
                        if candidate:
                            ctx.response_content = candidate
                    elif candidate:
                        ctx.response_content = candidate
            elif action_to_yield.role == ActionRole.TOOL and action_to_yield.status == ActionStatus.SUCCESS:
                tool_output = action_to_yield.output if isinstance(action_to_yield.output, dict) else {}
                summary = tool_output.get("summary") or tool_output.get("status_message") or ""
                if isinstance(summary, str) and summary.strip():
                    ctx.last_tool_summary = summary.strip()

            yield action_to_yield

    # ── optional hooks (subclasses override as needed) ──────────────────

    async def _before_stream(self, ctx: "StreamRunContext") -> None:
        """Hook: side-effect initialisation before the stream loop begins.

        Runs after input validation / initial action emission but BEFORE
        session setup and prompt assembly. Use this for async setup whose
        outcome affects tool selection, prompt building, or template context
        (e.g. parse ``user_input`` to derive ``ctx.user_message_override`` /
        ``ctx.extras``, enable/disable tools on ``self.tools``).
        """
        return None

    def _build_template_context(self, ctx: "StreamRunContext") -> Optional[Dict[str, Any]]:
        """Hook: extra keyword arguments forwarded to the system-prompt template.

        Return a dict to enable template-context rendering; return ``None``
        (default) when the node's template needs only the common variables
        injected by :meth:`_get_system_prompt`.

        Distinct from the per-subclass helper ``_prepare_template_context``
        (which various subclasses already define with ``(user_input, …)``
        signatures); subclasses that need template context override this hook
        and typically delegate: ``return self._prepare_template_context(ctx.user_input)``.
        """
        return None

    def _compose_run_hooks(self, ctx: "StreamRunContext") -> Any:
        """Hook: compose the final ``hooks`` argument passed to the model.

        Default: include ``self.hooks`` (typically a ``GenerationHooks``
        instance for todo/plan workflow nodes) only in interactive mode;
        otherwise return permission hooks alone. This covers SqlSummary,
        Feedback, GenSemanticModel, GenExtKnowledge, GenMetrics out of the
        box.

        Subclasses with non-``self.hooks`` extras (Deliverable's
        ``_validation_hook``) override to call ``self._compose_hooks(extra)``
        directly.
        """
        extra_hook = getattr(self, "hooks", None)
        if extra_hook is None or self.execution_mode != "interactive":
            return self._compose_hooks()
        return self._compose_hooks(extra_hook)

    def _maybe_rewrite_stream_action(self, action: ActionHistory, ctx: "StreamRunContext") -> Optional[ActionHistory]:
        """Hook: optionally replace a streamed action before it is yielded.

        Return a replacement :class:`ActionHistory` to substitute it (used by
        GenReport to swap JSON payloads for rendered markdown in real time)
        or ``None`` (default) to keep the original action unchanged.
        """
        return None

    def _get_retry_policy(self):
        """Hook: return a :class:`RetryPolicy` to drive validate/retry.

        Default returns :class:`NoRetryPolicy` (single execution). Override
        to return :class:`ValidationHookRetryPolicy` (deliverable_node.py) /
        :class:`VerifySqlRetryPolicy` (gen_ext_knowledge_agentic_node.py) when
        the node needs re-prompting on validation failure. Concrete policies
        live in their owning node's module — there is no shared ``policies/``
        package since each policy is bound to a specific node's internals.
        """
        from datus.agent.node.retry_policy import NoRetryPolicy

        return NoRetryPolicy()

    def _build_success_result(self, ctx: "StreamRunContext") -> BaseResult:
        """Hook (REQUIRED): construct the NodeResult on the success path.

        Subclasses parse ``ctx.response_content`` / ``ctx.last_successful_output``
        / ``ctx.last_tool_summary`` / ``ctx.extras`` and return an instance
        of ``self.result_class``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override _build_success_result(ctx)")

    async def _stream_post_build(
        self, ctx: "StreamRunContext", result: BaseResult
    ) -> AsyncGenerator[ActionHistory, None]:
        """Optional async-generator hook invoked after :meth:`_build_success_result`
        and before the final wrapper action is yielded.

        Visual-artifact subagents override this to interleave finalize-
        progress messages around their post-validate LLM work, so the
        chat panel doesn't sit silent through the 10-15 s finalize. The
        hook may mutate ``result`` in place (e.g. populate
        ``finalize_warnings`` / ``finalize_error``); those mutations
        flow into the final wrapper action's ``output_data``.

        Default: yield nothing.
        """
        # ``if False: yield`` keeps this an async-generator function (so
        # callers can ``async for``) while emitting zero items by default.
        if False:  # pragma: no cover - documented sentinel
            yield  # type: ignore[unreachable]

    def _build_error_result(self, exc: BaseException, ctx: "StreamRunContext") -> BaseResult:
        """Construct a uniform error NodeResult — base implementation final.

        Builds an instance of ``self.result_class`` populated with
        ``success=False``, ``error=<formatted>``, ``response=""``,
        ``tokens_used=0``. Automatically fills ``action_history`` for
        NodeResult subtypes that declare that field.

        Subclasses MUST declare ``result_class`` for this to work; the
        runtime guard below produces a clear error otherwise.
        """
        if self.result_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare a class-level "
                f"`result_class` attribute pointing to its BaseResult subtype"
            )

        kwargs: Dict[str, Any] = {
            "success": False,
            "error": self._format_execution_error(exc),
            "response": "",
            "tokens_used": 0,
        }
        if "action_history" in getattr(self.result_class, "model_fields", {}):
            kwargs["action_history"] = [a.model_dump() for a in ctx.action_history_manager.get_actions()]
        return self.result_class(**kwargs)

    def _get_or_create_broker(self) -> "InteractionBroker":
        """
        Get or create the interaction broker for this node.

        Resets the broker's asyncio.Queue so it binds to the current event loop.
        This is necessary because each asyncio.run() creates a new event loop,
        and asyncio.Queue is bound at creation time. Without this reset, reusing
        a node across multiple asyncio.run() calls would fail with
        'Queue is bound to a different event loop'.

        Returns:
            InteractionBroker instance for this node
        """
        self.interaction_broker.reset_queue()
        return self.interaction_broker

    async def execute_stream_with_interactions(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute with interaction support, merging execute_stream with broker
        and any tool action channels (e.g. sub-agent task actions).

        This is the method that UI components should call instead of execute_stream()
        when they want to handle interactions from hooks.

        Supports graceful interruption via self.interrupt_controller. When interrupted,
        yields an "interrupted" action and stops execution cleanly.

        Args:
            action_history_manager: Optional action history manager for tracking

        Yields:
            ActionHistory: Progress updates during execution, including
            INTERACTION actions and tool sub-actions.
        """
        self.interrupt_controller.reset()
        self.action_bus.reset()
        broker = self._get_or_create_broker()

        action_stream = self.execute_stream(action_history_manager)
        try:
            async for action in self.action_bus.merge(
                action_stream,
                broker.fetch(),
                on_primary_done=broker.close,
            ):
                self.interrupt_controller.check()
                yield action
        except ExecutionInterrupted:
            logger.info("Execution interrupted by user")
            yield ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="interrupted",
                messages="Execution interrupted. You can continue with additional information.",
                input_data={},
                status=ActionStatus.SUCCESS,
            )

    def clear_session(self) -> None:
        """Clear the current session."""
        if self.session_id:
            self.session_manager.clear_session(self.session_id)
            self._session = None
            logger.info(f"Cleared session: {self.session_id}")

    def delete_session(self) -> None:
        """Delete the current session completely.

        The node becomes unusable after this call — callers (REPL ``/delete``,
        API ``DELETE /sessions/{id}``) discard it and either build a fresh node
        or end the conversation. ``session_id`` stays set (it is immutable) so
        log lines and tracebacks can still reference which session was deleted.
        """
        if self.session_id:
            self.session_manager.delete_session(self.session_id)
            self._session = None
            logger.info("Deleted session: %s", self.session_id)

    async def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.

        Returns:
            Dictionary with session information
        """
        if not self.session_id:
            return {"session_id": None, "active": False}

        current_tokens = await self._count_session_tokens()

        return {
            "session_id": self.session_id,
            "active": self._session is not None,
            "token_count": current_tokens,
            "action_count": len(self.actions),
            "context_usage_ratio": current_tokens / self.context_length if self.context_length else 0,
            "context_remaining": self.context_length - current_tokens if self.context_length else 0,
            "context_length": self.context_length,
        }

    async def get_last_turn_usage(self) -> Optional[TokenUsage]:
        """Get token usage from the last assistant action that contains usage data."""
        from datus.schemas.token_usage import TokenUsage as _TokenUsage

        for action in reversed(self.actions):
            # Stop at the last root-level user message to scope to the current turn
            if action.role == ActionRole.USER and action.depth == 0:
                break
            if (
                action.role == ActionRole.ASSISTANT
                and action.depth == 0
                and isinstance(action.output, dict)
                and isinstance(action.output.get("usage"), dict)
            ):
                usage_dict = action.output["usage"]
                return _TokenUsage.from_usage_dict(
                    usage_dict,
                    session_total_tokens=usage_dict.get("last_call_input_tokens", 0)
                    or usage_dict.get("input_tokens", 0),
                    context_length=self.context_length or 0,
                )
        return None

    def _resolve_workspace_root(self) -> str:
        """
        Resolve workspace_root with priority: node-specific ``workspace_root`` >
        ``agent_config.project_root`` (which itself defaults to the launch CWD).

        Expands ``~`` to the user home directory if present.

        vscode short-circuit: the vscode front-end drives the daemon from a
        remote IDE that owns its own filesystem, so any concrete server-side
        path would just leak the daemon CWD. Return the literal ``"."``
        unexpanded for that source — callers that surface this value to the
        LLM (e.g. system prompt) stay neutral, and the proxied
        ``filesystem_tools.*`` route every real path through the client
        anyway. Web keeps its real root because web sessions still operate
        against a server-side workspace.
        """
        import os

        if getattr(self.agent_config, "_client_source", None) == "vscode":
            return "."

        node_workspace_root = self.node_config.get("workspace_root")
        if node_workspace_root:
            workspace_root = node_workspace_root
            logger.debug(f"Using node-specific workspace_root: {workspace_root}")
        elif self.agent_config and hasattr(self.agent_config, "project_root"):
            workspace_root = self.agent_config.project_root
            logger.debug(f"Using project_root as workspace_root: {workspace_root}")
        else:
            workspace_root = os.getcwd()
            logger.debug(f"Using current directory as workspace_root: {workspace_root}")

        expanded_path = os.path.expanduser(workspace_root)
        if expanded_path != workspace_root:
            logger.debug(f"Expanded workspace_root from '{workspace_root}' to '{expanded_path}'")
        return expanded_path

    def _resolve_filesystem_strict(self) -> bool:
        """Resolve the ``strict`` flag for this node's filesystem tool.

        Reads ``self.agent_config.filesystem_strict`` (process-wide default set
        by API / gateway bootstraps, or by ``agent.filesystem.strict`` / the
        ``--filesystem-strict`` CLI flag). CLI leaves it unset so EXTERNAL
        access falls back to broker-prompt behavior.
        """
        if self.agent_config is None:
            return False
        return bool(self.agent_config.filesystem_strict)

    def _make_filesystem_tool(self, **kwargs):
        """Construct a ``FilesystemFuncTool`` with this node's identity baked in.

        All production call sites go through this helper so ``root_path`` is
        uniformly ``_resolve_workspace_root()`` and ``current_node`` matches
        ``get_node_name()`` — the two inputs the path policy module expects to
        classify ``.datus/memory/{current_node}/**`` as a whitelist subtree
        for this node only. The ``strict`` flag is resolved from
        ``agent_config.filesystem_strict`` so API / gateway can opt out of
        interactive EXTERNAL prompts.
        """
        from datus.configuration.inherited_memory_overrides import get_inherited_memory
        from datus.tools.func_tool import FilesystemFuncTool

        root_path = kwargs.pop("root_path", None) or self._resolve_workspace_root()
        datus_home = kwargs.pop("datus_home", None)
        if datus_home is None and self.agent_config is not None:
            path_manager = getattr(self.agent_config, "path_manager", None)
            if path_manager is not None:
                try:
                    datus_home = str(path_manager.datus_home)
                except Exception:
                    datus_home = None
        strict = kwargs.pop("strict", None)
        if strict is None:
            strict = self._resolve_filesystem_strict()
        current_node = kwargs.pop("current_node", None) or self.get_node_name()
        inherited_memory_node = kwargs.pop("inherited_memory_node", None)
        if inherited_memory_node is None:
            inherited_memory_node = get_inherited_memory(current_node)
        return FilesystemFuncTool(
            root_path=root_path,
            current_node=current_node,
            datus_home=datus_home,
            strict=strict,
            inherited_memory_node=inherited_memory_node,
            **kwargs,
        )

    def _make_filesystem_policy(self):
        """Build a :class:`FilesystemPolicy` for ``PermissionHooks`` construction.

        Returns ``None`` when this node has no ``agent_config`` or the path
        manager cannot be resolved, so callers can treat the policy as opt-in
        and fall back to the pre-refactor category-level behavior.
        """
        if not self.agent_config:
            return None
        path_manager = getattr(self.agent_config, "path_manager", None)
        if path_manager is None:
            return None
        try:
            from pathlib import Path as _Path

            from datus.tools.permission.permission_hooks import FilesystemPolicy

            return FilesystemPolicy(
                root_path=_Path(self._resolve_workspace_root()).resolve(strict=False),
                current_node=self.get_node_name(),
                datus_home=_Path(path_manager.datus_home),
                strict=self._resolve_filesystem_strict(),
            )
        except Exception as e:
            logger.debug(f"Failed to build FilesystemPolicy: {e}")
            return None

    # ── Permission hook wiring ──────────────────────────────────────────
    # Subagent nodes historically passed ``hooks=None`` into
    # ``generate_with_tools_stream`` — meaning profile rules (DENY, ASK)
    # never fired for anything other than ``chat``. These helpers let
    # every subclass participate in the permission system with one call.

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Return ``{category: tools}`` for permission registration.

        Subclasses override this to declare which of ``self.tools`` belong
        to which permission category (``bi_tools``, ``scheduler_tools``,
        ``db_tools``, etc.). Categories not declared here fall back to the
        ``tools`` catch-all, which only matches explicit ``tools.*`` rules.

        The base implementation registers ``skill_func_tool`` under
        ``skills`` and ``bash_tool`` under ``bash_tools`` so the
        ``skills.*`` and ``bash_tools.*`` profile rules apply to every
        subagent that exposes them — overrides should ``super()`` +
        extend.
        """
        mapping: Dict[str, List[Any]] = {}
        if self.skill_func_tool:
            mapping["skills"] = list(self.skill_func_tool.available_tools())
        bash_tool = getattr(self, "bash_tool", None)
        if bash_tool:
            bash_tools = list(bash_tool.available_tools())
            if bash_tools:
                mapping["bash_tools"] = bash_tools
        return mapping

    def _populate_tool_registry(self) -> None:
        """Register every tool in :meth:`_tool_category_map` into
        :attr:`tool_registry`.

        Decoupled from :meth:`_ensure_permission_hooks` so callers that
        need the category map filled *before* the first LLM turn — most
        importantly :func:`apply_proxy_tools`, which inspects the
        registry to honour the ``_FS_DEPENDENT_NODES`` exclusion — can
        trigger it eagerly. Safe to call multiple times because
        :meth:`ToolRegistry.register_tools` is overwrite-write.

        Permission gating remains opt-in through
        :meth:`_ensure_permission_hooks`; this helper does **not** require
        a ``permission_manager`` and never builds ``PermissionHooks``.
        """
        try:
            for category, tools in self._tool_category_map().items():
                if tools:
                    self.tool_registry.register_tools(category, tools)
        except Exception:
            logger.debug(
                "Failed to populate tool_registry for %s; falling back to lazy registration.",
                self.get_node_name(),
                exc_info=True,
            )

    def _ensure_permission_hooks(self) -> None:
        """Build ``self.permission_hooks`` once tools are in place.

        Invoked lazily from :meth:`_compose_hooks` so subclasses don't have
        to remember the ordering (``setup_tools`` must happen first). Safe
        to call many times — short-circuits after the first successful
        build. Silently no-ops when no ``permission_manager`` exists
        (agent with permissions disabled).
        """
        if self.permission_hooks is not None:
            return
        if not self.permission_manager:
            return
        try:
            self._populate_tool_registry()
            from datus.tools.permission.permission_hooks import PermissionHooks

            # ``execution_mode="workflow"`` flows have no human in the loop, so
            # ASK / EXTERNAL fs hits short-circuit to ``PermissionDeniedException``
            # inside the hook rather than awaiting the broker indefinitely.
            non_interactive = getattr(self, "execution_mode", None) == "workflow"

            # Never call ``_get_or_create_broker`` here — it resets the queue
            # and orphans any parent CLI listener when running as a sub-agent.
            self.permission_hooks = PermissionHooks(
                broker=self.interaction_broker,
                permission_manager=self.permission_manager,
                node_name=self.get_node_name(),
                tool_registry=self.tool_registry,
                fs_policy=self._make_filesystem_policy(),
                non_interactive=non_interactive,
                proxied_tool_names=self.proxied_tool_names,
            )
            logger.debug(
                f"PermissionHooks attached to node '{self.get_node_name()}' "
                f"with {len(self.tool_registry)} tool mappings"
            )
        except Exception as e:
            # Fail closed: leaving ``permission_hooks=None`` with a
            # ``permission_manager`` present would silently bypass profile
            # DENY/ASK checks on every tool call. Raise so the node refuses
            # to run rather than executing with degraded enforcement.
            from datus.utils.exceptions import DatusException, ErrorCode

            logger.exception("Failed to build PermissionHooks for %s", self.get_node_name())
            self.permission_hooks = None
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Permission hook setup failed for {self.get_node_name()}: {e}"},
            ) from e

    @staticmethod
    def _extract_total_tokens(actions: List[ActionHistory]) -> int:
        """Walk the current root turn and return its assistant token count.

        Iterates in reverse from the most recent action, stopping at the
        last root-level user message so child/tool usage from sub-agents
        does not leak into the parent's per-turn total — same scoping as
        ``_count_session_tokens`` / ``get_last_turn_usage``. Tolerates
        ``total_tokens`` being a numeric string (some providers emit
        ``"1234"`` rather than ``1234``) — anything that fails an ``int``
        cast contributes ``0`` and the loop continues looking further back.
        Returns ``0`` when no assistant action carries a usable usage block.
        """
        for action in reversed(actions):
            if action.role == ActionRole.USER and action.depth == 0:
                break
            if action.role != ActionRole.ASSISTANT or action.depth != 0:
                continue
            output = action.output
            if not isinstance(output, dict):
                continue
            usage_info = output.get("usage")
            if not isinstance(usage_info, dict):
                continue
            total = usage_info.get("total_tokens")
            if not total:
                continue
            try:
                tokens = int(total)
            except (TypeError, ValueError):
                continue
            if tokens > 0:
                return tokens
        return 0

    @staticmethod
    def _format_execution_error(exc: BaseException) -> str:
        """Render an exception for error_result / error_action display.

        ``DatusException`` carries a structured error code that is normally
        lost when callers fall back to ``str(exc)``. Surface it as
        ``[CODE] <message>`` so logs and SSE error cards remain greppable.
        """
        from datus.utils.exceptions import DatusException

        if isinstance(exc, DatusException):
            return f"[{exc.code}] {exc}"
        return str(exc)

    def _compose_hooks(self, extra: Any = None) -> Any:
        """Combine permission hooks with an optional per-node hook.

        ``extra`` is typically ``self.hooks`` for workflow nodes
        (``feedback``, ``sql_summary``, …) that have their own todo/plan
        hooks. Returns a :class:`CompositeHooks` when both are present,
        a single hook when only one is present, or ``None`` when neither
        exists. Callers pass the result directly into
        ``generate_with_tools_stream(hooks=...)``.
        """
        self._ensure_permission_hooks()
        if self.permission_hooks and extra:
            from datus.tools.permission.permission_hooks import CompositeHooks

            return CompositeHooks([extra, self.permission_hooks])
        return self.permission_hooks or extra
