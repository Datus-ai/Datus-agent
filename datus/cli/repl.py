# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Datus-CLI REPL (Read-Eval-Print Loop) implementation.
This module provides the main interactive shell for the CLI.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import ConditionalCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import DynamicLexer, PygmentsLexer
from prompt_toolkit.styles import Style, merge_styles, style_from_pygments_cls
from pygments.lexers.shell import BashLexer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from datus.agent.workflow_runner import WorkflowRunner

from datus_db_core import BaseSqlConnector

from datus import __version__
from datus.cli._cli_utils import prompt_input
from datus.cli.agent_commands import AgentCommands
from datus.cli.autocomplete import (
    AtReferenceCompleter,
    BangCompleter,
    CustomPygmentsStyle,
    CustomSqlLexer,
    ServiceCommandCompleter,
    SlashCommandCompleter,
)
from datus.cli.bootstrap_bi_commands import BootstrapBiCommands
from datus.cli.build_kb_commands import BuildKbCommands
from datus.cli.chat_commands import ChatCommands
from datus.cli.cli_styles import (
    PASTE_COLLAPSE_THRESHOLD,
    STATUS_BAR_STYLE,
    print_error,
    print_info,
    print_success,
    print_warning,
    render_user_scrollback_text,
)
from datus.cli.context_commands import ContextCommands
from datus.cli.effort_commands import EffortCommands
from datus.cli.execution_state import ExecutionInterrupted, InterruptController
from datus.cli.init_commands import InitCommands
from datus.cli.input_modes import MODE_CHROME, InputMode, next_input_mode
from datus.cli.language_commands import LanguageCommands
from datus.cli.metadata_commands import MetadataCommands
from datus.cli.model_commands import ModelCommands
from datus.cli.sandbox_commands import SandboxCommands
from datus.cli.service_commands import ServiceCommands
from datus.cli.slash_registry import GROUP_ORDER, GROUP_TITLES, iter_visible, lookup
from datus.cli.status_bar import StatusBarProvider
from datus.cli.summarize_commands import MemoryOrganizeCommands, SessionSummarizeCommands
from datus.cli.todo_sidebar import TodoSidebarProvider
from datus.cli.tui import DatusApp, tui_enabled
from datus.cli.tui.app import EXIT_SENTINEL
from datus.cli.tui.live_display_state import LiveDisplayState
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.storage.embedding_diagnostics import format_context_degraded_warning
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS, DBType, SQLType
from datus.utils.exceptions import setup_exception_handler
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_sql_type

logger = get_logger(__name__)


DATUS_BANNER_TEXT = (
    "██████╗   █████╗  ████████╗ ██╗   ██╗ ███████╗\n"
    "██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║   ██║ ██╔════╝\n"
    "██║  ██║ ███████║    ██║    ██║   ██║ ███████╗\n"
    "██║  ██║ ██╔══██║    ██║    ██║   ██║ ╚════██║\n"
    "██████╔╝ ██║  ██║    ██║    ╚██████╔╝ ███████║\n"
    "╚═════╝  ╚═╝  ╚═╝    ╚═╝     ╚═════╝  ╚══════╝"
)
_BANNER_MIN_WIDTH = 60


class CommandType(Enum):
    """Type of command entered by the user."""

    SQL = "sql"  # Regular SQL statement (SQL mode)
    BASH = "bash"  # shell command (bash mode), run through the permission-gated bash tool
    SLASH = "slash"  # /command (session / metadata / context / agent / system)
    BANG = "bang"  # ``!<tool>`` / ``!<plugin>`` — run a tool or plugin CLI directly
    CHAT = "chat"  # bare text routed to the default agent
    EXIT = "exit"  # exit/quit command
    UNKNOWN = "unknown"  # unrecognized /command or renamed legacy prefix


_LEGACY_PREFIX_HINTS: dict[str, str] = {
    ".help": "/help",
    ".exit": "/exit",
    ".quit": "/quit",
    ".clear": "/clear",
    ".chat_info": "/chat_info",
    ".compact": "/compact",
    ".resume": "/resume",
    ".rewind": "/rewind",
    ".databases": "/databases",
    ".database": "/database",
    ".tables": "/tables",
    ".schemas": "/schemas",
    ".schema": "/schema",
    ".table_schema": "/table_schema",
    ".indexes": "/indexes",
    ".datasource": "/datasource",
    ".agent": "/agent",
    ".subagent": "/subagent",
    ".mcp": "/mcp",
    ".skill": "/skill",
    ".bootstrap-bi": "/bootstrap-bi",
    ".language": "/language",
    "@catalog": "/catalog",
    "@subject": "/subject",
}


class DatusCLI:
    """Main REPL for the Datus CLI application."""

    def __init__(self, args, interactive: bool = True):
        """Initialize the CLI with the given arguments."""
        self.args = args
        self.interactive = interactive
        self.console = Console(log_path=False)
        self.console_column_width = 16
        self.selected_catalog_path = ""
        self.selected_catalog_data = {}
        self.scope = getattr(args, "session_scope", None)

        setup_exception_handler(console_logger=self.console.print, prefix_wrap_func=lambda x: f"[red]{x}[/red]")
        self.db_connector: BaseSqlConnector | None = None

        self.agent = None
        self.agent_initializing = False
        self.agent_ready = False
        self._workflow_runner: WorkflowRunner | None = None
        self.startup_warnings: List[str] = []

        # Plan mode support
        self.plan_mode_active = False
        # Input mode: chat routes plain input to the model, SQL mode executes
        # it against the current datasource, bash mode runs it through the
        # permission-gated bash tool. Cycled with Tab on an empty TUI input
        # line (see the Tab key binding in ``_init_tui_app``); Esc / Ctrl+C
        # return to chat. Drives the mode-coloured prompt label, separators,
        # the mode hint line, and the contextual lexer/completer.
        self.input_mode: InputMode = InputMode.CHAT
        self._last_ctrl_c_time: float = 0.0
        # Default agent for /message routing ("" = chat node)
        self.default_agent = ""

        # Load agent config first so path-dependent helpers use the configured home.
        self.agent_config = load_agent_config(create_if_missing=True, **vars(self.args))
        # Stash the ``--report-dist`` CLI flag as a runtime override on
        # agent_config so ``gen_visual_report`` can pick it up at HTML compile
        # time without threading args through every layer (AgentConfig is a
        # plain class, dynamic attributes are fine — see datus/cli/main.py).
        report_dist_flag = getattr(self.args, "report_dist", None)
        if report_dist_flag:
            self.agent_config.report_dist_cli_override = report_dist_flag
        # REPL is interactive — auto-open the report HTML in a browser unless
        # the user opted out via ``--no-open-report``. Stash on agent_config
        # so the deeply-nested node reads it without arg threading.
        self.agent_config.report_auto_open = not bool(getattr(self.args, "no_open_report", False))
        self.configuration_manager = configuration_manager()

        # Active permission profile name. Initialized from agent_config;
        # mutated by /permission. StatusBarProvider reads this for display.
        self.active_profile: str = getattr(self.agent_config, "active_profile_name", "normal")

        # Bind the process-wide path-manager ContextVar once so implicit callers
        # (e.g. ``get_path_manager()`` inside storage init) resolve against the
        # loaded agent_config instead of an empty default.  Required before
        # background tasks are scheduled, since ContextVars are snapshotted at
        # task-creation / context-copy time.
        from datus.utils.path_manager import set_current_path_manager

        set_current_path_manager(agent_config=self.agent_config)

        # Background event loop for async init tasks.  A single daemon thread
        # hosts the loop; individual init work runs as coroutines that inherit
        # the current ContextVar snapshot (see ``_async_init_agent``).  Using
        # a managed loop instead of spawning ad-hoc ``threading.Thread`` means
        # we only pay the ContextVar-copy cost once per background task.
        self._bg_loop = asyncio.new_event_loop()
        self._bg_loop_thread = threading.Thread(
            target=self._bg_loop.run_forever,
            name="datus-cli-bg-loop",
            daemon=True,
        )
        self._bg_loop_thread.start()

        if args.history_file:
            history_file = Path(args.history_file).expanduser().resolve()
        else:
            history_file = self.agent_config.path_manager.history_file_path()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(history_file))
        self.session: PromptSession | None = None

        # Initialize available subagents early (needed by autocomplete)
        self.available_subagents = set(SYS_SUB_AGENTS)
        self.available_subagents.add("chat")
        if hasattr(self.agent_config, "agentic_nodes") and self.agent_config.agentic_nodes:
            self.available_subagents.update(name for name in self.agent_config.agentic_nodes.keys() if name != "chat")

        # TUI mode: use persistent prompt_toolkit Application with pinned
        # status bar + input. Requires a TTY on both stdin/stdout and can be
        # disabled via ``DATUS_TUI=0`` as an escape hatch.
        self._use_tui = self.interactive and tui_enabled()
        self.tui_app: Optional[DatusApp] = None
        self.live_state: Optional[LiveDisplayState] = None

        self.at_completer: AtReferenceCompleter
        if self.interactive:
            # Both paths build completers, lexers and styles via the same
            # helpers so feature parity is preserved.
            if self._use_tui:
                self._init_tui_app()
            else:
                self._init_prompt_session()
        else:
            self.at_completer = AtReferenceCompleter(
                self.agent_config,
                available_subagents=self.available_subagents,
                visibility_provider=self._visible_subagents_for_default,
            )

        # Last executed SQL and result
        self.last_sql = None
        self.last_result = None
        self._prefill_input = None  # For rewind: prefill input buffer with user message

        # Action history manager for tracking all CLI operations
        self.actions = ActionHistoryManager()

        # Initialize CLI context for state management
        from datus.cli.cli_context import CliContext

        self.cli_context = CliContext(
            current_db_name=getattr(args, "database", ""),
            current_catalog=getattr(args, "catalog", ""),
            current_schema=getattr(args, "schema", ""),
        )
        self.db_manager = db_manager_instance(self.agent_config.datasource_configs)

        # Initialize command handlers after cli_context is created
        self.agent_commands = AgentCommands(self, self.cli_context)
        self.chat_commands = ChatCommands(self)
        self.context_commands = ContextCommands(self)
        self.metadata_commands = MetadataCommands(self)
        self.bootstrap_bi_commands = BootstrapBiCommands(self)
        from datus.cli.bootstrap_commands import BootstrapCommands

        self.bootstrap_commands = BootstrapCommands(self)
        self.model_commands = ModelCommands(self)
        from datus.cli.plugin_commands import PluginCommands

        self.plugin_commands = PluginCommands(self)
        self.language_commands = LanguageCommands(self)
        self.effort_commands = EffortCommands(self)
        self.sandbox_commands = SandboxCommands(self)
        self.init_commands = InitCommands(self)
        self.build_kb_commands = BuildKbCommands(self)
        self.session_summarize_commands = SessionSummarizeCommands(self)
        self.memory_organize_commands = MemoryOrganizeCommands(self)
        self.service_commands = ServiceCommands(self)
        from datus.cli.bang_command import BangCommand
        from datus.cli.datasource_commands import DatasourceCommands

        self.bang_command = BangCommand(self)
        self.datasource_commands = DatasourceCommands(self)

        from datus.cli.background_sync import BackgroundSchemaSyncManager

        self.bg_sync = BackgroundSchemaSyncManager(self)
        self._status_bar_provider = StatusBarProvider(self)
        self._todo_sidebar_provider = TodoSidebarProvider(self)

        # Dictionary of available commands - created after handlers are initialized.
        # Only ``/`` slash commands populate this dict below; SQL/bash execution
        # is reached through the Tab-cycled input modes (see ``_parse_command``).
        self.commands: Dict[str, Any] = {}
        # Slash commands are driven by ``slash_registry.SLASH_COMMANDS`` so the
        # completer, help text, and dispatcher share one source of truth.
        for spec_name, handler in self._build_slash_handler_map().items():
            spec = lookup(spec_name)
            if spec is None:
                raise RuntimeError(f"Slash handler '{spec_name}' has no registry entry")
            self.commands[f"/{spec.name}"] = handler
            for alias in spec.aliases:
                self.commands[f"/{alias}"] = handler

        # Start agent initialization in background
        self._async_init_agent()
        self._init_connection()

    def _build_slash_handler_map(self) -> Dict[str, Any]:
        """Return the canonical-name -> handler map consumed by the commands dict.

        Kept alongside ``SLASH_COMMANDS`` ordering so the registry integrity
        test can assert every spec has a bound handler.
        """

        return {
            # session
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "clear": self.chat_commands.cmd_clear_chat,
            "chat_info": self.chat_commands.cmd_chat_info,
            "compact": self.chat_commands.cmd_compact,
            "resume": self.chat_commands.cmd_resume,
            "rewind": self.chat_commands.cmd_rewind,
            # metadata
            "databases": self.metadata_commands.cmd_list_databases,
            "database": self.metadata_commands.cmd_switch_database,
            "tables": self.metadata_commands.cmd_tables,
            "schemas": self.metadata_commands.cmd_schemas,
            "schema": self.metadata_commands.cmd_switch_schema,
            "table_schema": self.metadata_commands.cmd_table_schema,
            "indexes": self.metadata_commands.cmd_indexes,
            # context
            "catalog": self.context_commands.cmd_catalog,
            "subject": self.context_commands.cmd_subject,
            "save": self.agent_commands.cmd_save,
            # agent
            "agent": self._cmd_agent,
            "subagent": self._cmd_subagent,
            "datasource": self.datasource_commands.cmd,
            "language": self.language_commands.cmd_language,
            # system
            "mcp": self._cmd_mcp,
            "skill": self._cmd_skill,
            "bootstrap": self.bootstrap_commands.cmd,
            "bootstrap-bi": self.bootstrap_bi_commands.cmd,
            "model": self.model_commands.cmd_model,
            "plugins": self.plugin_commands.cmd_plugins,
            "effort": self.effort_commands.cmd_effort,
            "init": self.init_commands.cmd_init,
            "build-kb": self.build_kb_commands.cmd_build_kb,
            "session-summarize": self.session_summarize_commands.cmd_session_summarize,
            "memory-organize": self.memory_organize_commands.cmd_memory_organize,
            "services": self.service_commands.cmd_services,
            "permission": self._cmd_permission,
            "profile": self._cmd_profile,
            "sandbox": self.sandbox_commands.cmd_sandbox,
        }

    @property
    def workflow_runner(self) -> WorkflowRunner:
        if not self.check_agent_available():
            raise RuntimeError("Agent not initialized. Cannot create workflow runner.")
        if not self._workflow_runner:
            # use day as run_id in cli
            self._workflow_runner = self._create_workflow_runner()
        return self._workflow_runner

    def _create_custom_key_bindings(self):
        """Create custom key bindings for the REPL."""
        kb = KeyBindings()

        @kb.add("c-p")
        def _(event):
            """Ctrl+P: cycle the active permission profile."""
            self._cycle_permission_mode()
            event.app.invalidate()

        @kb.add("tab")
        def _(event):
            """Tab confirms the highlighted completion (arrow keys navigate)."""
            buffer = event.app.current_buffer

            if buffer.complete_state:
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    buffer.complete_next()
                    cs = buffer.complete_state
                    if cs and cs.current_completion is not None:
                        buffer.apply_completion(cs.current_completion)
            else:
                buffer.start_completion(select_first=True)

        @kb.add("s-tab")
        def _(event):
            """Shift+Tab: Toggle Plan Mode on/off"""
            self.plan_mode_active = not self.plan_mode_active

            # Clear current input buffer and force exit current prompt
            buffer = event.app.current_buffer
            buffer.reset()

            # Force the prompt to exit and restart with new prefix
            # This will cause the main loop to regenerate the prompt.
            # No scrollback announcement — the regenerated prompt reflects the
            # mode, matching the TUI path.
            buffer.validation_state = None
            event.app.exit()

        @kb.add("enter")
        def _(event):
            """
            Enter key:
                apply highlighted completion (if any), close the menu, then
                submit — in a single keystroke. The earlier two-step behaviour
                (first Enter only dismissed the auto-opened menu, second Enter
                submitted) was confusing for slash commands like ``/model``.
            """
            buffer = event.app.current_buffer

            if buffer.complete_state:
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    buffer.cancel_completion()

            buffer.validate_and_handle()

        @kb.add("c-o")
        def _(event):
            """Show details for display_actions"""
            event.app.exit(result="_open_chat_sql_details")

        return kb

    def _echo_user_input(self, prompt_text: str, user_input: str):
        """Re-echo user input in the scrollback with the unified user-row style."""
        self.console.print(render_user_scrollback_text(user_input, prompt_text))

    def _get_prompt_text(self):
        """Input-line prompt text for the active input mode.

        The Datus brand, plan mode, and current agent are rendered by the
        status bar on the line above, so the input line uses a minimal prompt:
        ``> `` in chat mode, ``sql> `` / ``bash> `` in the execution modes.
        The non-TUI PromptSession fallback has no mode cycling, so it always
        sees the chat prompt.
        """
        return MODE_CHROME[self.input_mode].prompt

    def _update_prompt(self):
        """Update the prompt display (called when mode changes)"""
        # The prompt will be updated on the next iteration of the main loop
        # This is a limitation of prompt_toolkit's PromptSession
        # For immediate feedback, we could force a redraw, but it's complex

    def _build_prompt_message(self, prompt_text: str):
        """Build multi-line prompt: status bar line + input prompt line."""
        try:
            state = self._status_bar_provider.current_state()
            tokens = state.to_formatted_tokens()
        except Exception as e:
            logger.debug(f"status bar render failed: {e}")
            tokens = []
        tokens.append(("", "\n"))
        tokens.append(("class:prompt", prompt_text))
        return tokens

    def _build_app_style(self) -> Style:
        """Return the prompt_toolkit Style used by both PromptSession and TUI.

        Declaring it once keeps status-bar/input coloring in sync between the
        two input paths and avoids drift when new status-bar segments are
        added.
        """
        return merge_styles(
            [
                style_from_pygments_cls(CustomPygmentsStyle),
                Style.from_dict(STATUS_BAR_STYLE),
            ]
        )

    def _status_tokens_for_tui(self) -> List[Tuple[str, str]]:
        """Build status-bar tokens for the persistent TUI layout.

        Shares :class:`StatusBarProvider` with the PromptSession path so both
        modes present the same brand/plan/agent/connector/model/tokens/ctx
        segments.
        """
        try:
            state = self._status_bar_provider.current_state()
            return state.to_formatted_tokens()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"status bar render failed: {e}")
            return []

    def _todo_tokens_for_tui(self) -> List[Tuple[str, str]]:
        """Build todo-sidebar tokens for the persistent TUI layout.

        Bound method wrapper so :meth:`_init_tui_app` can pass this as a
        callback *before* ``_todo_sidebar_provider`` is assigned in
        ``__init__`` — the actual provider lookup happens on each paint,
        by which point ``__init__`` has completed.
        """
        try:
            return self._todo_sidebar_provider.tokens()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"todo sidebar tokens failed: {e}")
            return []

    def _todo_has_items_for_tui(self) -> bool:
        """Filter callback for the todo-sidebar ``ConditionalContainer``.

        See :meth:`_todo_tokens_for_tui` for the deferred-binding
        rationale.
        """
        try:
            return self._todo_sidebar_provider.has_items()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"todo sidebar has_items failed: {e}")
            return False

    def _todo_line_count_for_tui(self) -> int:
        """Rendered row count for sizing the pinned todo sidebar."""
        try:
            return self._todo_sidebar_provider.line_count()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"todo sidebar line_count failed: {e}")
            return 0

    def _active_pending_input_queue(self):
        """Return the active chat node's :class:`PendingInputQueue` or None.

        Consulted by the TUI Enter handler and the pinned queue-preview
        ``ConditionalContainer`` while the agent is streaming. Defensive
        getattr chain because ``chat_commands`` is only created lazily.
        """
        chat_commands = getattr(self, "chat_commands", None)
        if chat_commands is None or getattr(chat_commands, "current_streaming_ctx", None) is None:
            return None
        current_node = getattr(chat_commands, "current_node", None)
        return getattr(current_node, "pending_input_queue", None) if current_node else None

    def _interrupt_agent(self, *, restore_unanswered_input: bool = False) -> None:
        chat_commands = getattr(self, "chat_commands", None)
        current_node = getattr(chat_commands, "current_node", None) if chat_commands else None
        controller = getattr(current_node, "interrupt_controller", None) if current_node else None
        editable_message: Optional[str] = None
        streaming_ctx = getattr(chat_commands, "current_streaming_ctx", None) if chat_commands else None
        has_model_response = bool(getattr(streaming_ctx, "has_model_response_started", False))
        if restore_unanswered_input and streaming_ctx is not None and not has_model_response:
            editable_message = getattr(streaming_ctx, "editable_user_message", None)
            streaming_ctx.request_unanswered_rollback()
        elif restore_unanswered_input and streaming_ctx is not None and has_model_response:
            streaming_ctx.request_interrupted_notice()
        if controller is not None:
            try:
                # Active Task.cancel reaches the node as CancelledError rather
                # than its cooperative ExecutionInterrupted branch. Preserve
                # the pre-existing Ctrl+C/partial-turn usage semantics; the
                # unanswered ESC rollback explicitly clears this snapshot.
                if current_node is not None:
                    current_node._drop_running_turn_usage_on_exit = False
                controller.interrupt()
            except Exception as exc:  # pragma: no cover - defensive
                if current_node is not None:
                    current_node._drop_running_turn_usage_on_exit = True
                logger.debug(f"interrupt_controller.interrupt failed: {exc}")
        if editable_message is not None and self.tui_app is not None:
            self.tui_app.restore_input_after_dispatch(editable_message)

    def _compute_pane_width(self, sidebar_visible: bool) -> int:
        """Width in cells available to the left output pane.

        Mirrors ``DatusApp._sidebar_target_width`` so Rich's wrap width
        matches the column prompt_toolkit will paint into. The scrollbar
        gutter (``_scrollbar_window`` in ``DatusApp``) is always present
        — its ``visible_filter`` is ``lambda: True`` — so 1 column is
        deducted regardless of sidebar visibility. Renderables that draw
        a right edge (e.g. ``Panel`` borders) would otherwise overlap or
        be clipped by the gutter.

        * Sidebar visible: ``cols - max(14, cols // 5) - 1`` (scrollbar)
        * Sidebar hidden: ``cols - 1`` (scrollbar)

        Floored at 20 so Rich can still format something on absurdly
        narrow terminals.
        """
        import shutil

        cols = shutil.get_terminal_size(fallback=(120, 30)).columns
        scrollbar_width = 1
        if sidebar_visible:
            sidebar_width = max(14, cols // 5)
            return max(20, cols - sidebar_width - scrollbar_width)
        return max(20, cols - scrollbar_width)

    def _reflow_for_sidebar(self, sidebar_visible: bool) -> None:
        """Reflow the output pane when the sidebar appears/disappears.

        Rich's ``Console.width`` is locked at construction, but the
        underlying ``_width`` attribute is what ``size``/``width`` read on
        every access. We **mutate** it in place rather than rebuild the
        Console so every consumer that captured the instance earlier
        (``chat_commands.console`` bound in ``ChatCommands.__init__``, any
        ``ActionHistoryDisplay`` constructed during a turn) immediately sees
        the new pane width on its next ``print``. Replacing the instance
        leaves those captures pointing at the old width and the freshly
        streamed tokens render past the new pane edge — exactly the
        "covered by the sidebar" symptom users see.

        Existing scrollback was wrapped at the old width. Clearing the
        buffer and replaying ``_full_screen_reprint`` lets banner + completed
        turns redraw cleanly at the new width; ``in_progress_actions``
        forwards the running turn's incremental actions so they survive
        the wipe.
        """
        new_width = self._compute_pane_width(sidebar_visible=sidebar_visible)
        current_console = getattr(self, "console", None)
        if current_console is None:
            return
        # Rich ignores an explicit width on dumb terminals unless height is
        # explicit too. The TUI output console is backed by an in-memory buffer,
        # so preserve the existing height to make width reflow deterministic.
        if (
            getattr(current_console, "_width", None) is not None
            and getattr(current_console, "_height", None) is None
            and getattr(current_console, "is_dumb_terminal", False)
        ):
            try:
                current_console._height = current_console.size.height
            except Exception:  # pragma: no cover - defensive
                current_console._height = 25
        if (
            getattr(current_console, "_width", None) == new_width
            or getattr(current_console, "width", None) == new_width
        ):
            return
        buffer = getattr(self, "_tui_output_buffer", None)
        if buffer is None:
            return
        # In-place width swap — see docstring for why we don't rebuild.
        current_console._width = new_width
        chat_commands = getattr(self, "chat_commands", None)
        if chat_commands is not None:
            try:
                verbose = bool(getattr(chat_commands, "_trace_verbose", False))
                in_progress = getattr(chat_commands, "_current_incremental_actions", None)
                chat_commands._full_screen_reprint(verbose=verbose, in_progress_actions=in_progress)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"sidebar reflow reprint failed: {exc}")
        else:
            # Early boot: no history yet, just drop whatever banner was
            # already written so the next paint starts clean at the new width.
            buffer.clear()
        tui_app = getattr(self, "tui_app", None)
        if tui_app is not None:
            tui_app.invalidate()

    def _init_tui_app(self) -> None:
        """Create the persistent ``DatusApp`` and register REPL bindings."""
        # Tab keeps the legacy completion behavior while a menu is open or
        # text is typed, and cycles the input mode (chat → sql → bash) on an
        # empty line. Additional bindings — Shift+Tab plan-mode toggle,
        # Ctrl+O trace details, ESC interrupt — are wired below.

        from datus.cli.tui.output_buffer import TUIOutputBuffer

        # The TUI path still relies on the same AtReferenceCompleter handle
        # that downstream code queries for subagent state, so attach it
        # before constructing the app.
        completer = self.create_combined_completer()

        # Build the shared live-render state first; DatusApp will wire its
        # own ``invalidate`` into it once constructed (see LiveDisplayState
        # docstring for the deferred-callback rationale).
        self.live_state = LiveDisplayState()

        # Build the in-memory output buffer and **swap self.console to write
        # into it** before any further banner / warning print. In
        # ``full_screen=True`` mode prompt_toolkit owns the entire terminal,
        # so anything printed via the original stdout-bound Console would be
        # erased the instant ``tui_app.run()`` starts. By rerouting now,
        # ``_print_welcome`` and friends land in the buffer and appear in the
        # scroll pane as the first paint renders. Rich's ``Console`` locks
        # ``file=`` and ``width=`` at construction so we replace the whole
        # instance.
        self._tui_output_buffer = TUIOutputBuffer(
            live_state_snapshot_fn=self.live_state.snapshot,
        )
        # Rich must be told the WIDTH OF THE OUTPUT PANE — *not* the full
        # terminal — otherwise Markdown borders, table grids, and Pygments
        # alignment break visibly the moment the sidebar takes its 20%
        # column. ``_compute_pane_width`` mirrors ``DatusApp._sidebar_target_width``.
        # Boot with ``sidebar_visible=False`` because no todo items exist at
        # startup; ``_reflow_for_sidebar`` will rebuild the Console at the
        # narrower width on the first transition.
        pane_width = self._compute_pane_width(sidebar_visible=False)
        self.console = Console(
            file=self._tui_output_buffer,
            force_terminal=True,
            color_system="256",
            width=pane_width,
            log_path=False,
        )
        # Propagate the new Console to subsystems that captured the old one
        # via bound method: setup_exception_handler kept a closure over
        # ``self.console.print``; reinstall it so global exceptions land in
        # the buffer too.
        setup_exception_handler(
            console_logger=lambda *a, **kw: self.console.print(*a, **kw),
            prefix_wrap_func=lambda x: f"[red]{x}[/red]",
        )

        # Syntax highlighting is contextual: the ``CustomSqlLexer`` applies
        # while SQL mode is active and ``BashLexer`` while bash mode is
        # active, so plain model-chat input shows no code colouring.
        # ``DynamicLexer`` re-evaluates the getter every paint.
        sql_lexer = PygmentsLexer(CustomSqlLexer)
        bash_lexer = PygmentsLexer(BashLexer)
        mode_lexers = {InputMode.SQL: sql_lexer, InputMode.BASH: bash_lexer}
        self.tui_app = DatusApp(
            status_tokens_fn=self._status_tokens_for_tui,
            dispatch_fn=self._dispatch_command_text,
            completer=completer,
            history=self.history,
            lexer=DynamicLexer(lambda: mode_lexers.get(self.input_mode)),
            input_mode_fn=lambda: self.input_mode,
            style=self._build_app_style(),
            input_prompt_fn=self._get_prompt_text,
            live_display_state=self.live_state,
            todo_tokens_fn=self._todo_tokens_for_tui,
            todo_has_items_fn=self._todo_has_items_for_tui,
            todo_line_count_fn=self._todo_line_count_for_tui,
            output_tokens_fn=self._tui_output_buffer.tokens,
            # Cursor positioning *must* read the count tied to the most
            # recent ``tokens()`` call — see ``render_line_count`` for the
            # IndexError race it avoids.
            output_line_count_fn=self._tui_output_buffer.render_line_count,
            # When passed, ``DatusApp`` swaps in :class:`BufferedOutputControl`
            # so prompt_toolkit's paint loop only materialises rows that fall
            # inside the viewport. This keeps type-latency flat even when the
            # verbose-mode scrollback grows into thousands of lines.
            output_buffer=self._tui_output_buffer,
            # Mid-run user-insert path. The TUI's Enter handler consults
            # this provider while the agent is streaming so typed messages
            # are queued for the next LLM turn (via the OpenAI Agents SDK
            # ``call_model_input_filter`` hook) instead of being dropped.
            pending_input_provider=self._active_pending_input_queue,
            # Dim ``<arg> [--opt]`` hint rendered after the input while typing a
            # ``!<tool>`` / ``!<plugin>`` command (see ``BangCommand.param_hint``).
            param_hint_fn=self._bang_param_hint,
        )

        # Now that the Application exists, wire the buffer's ``on_change``
        # callback to its ``invalidate`` so every Rich write triggers a
        # repaint (via ``loop.call_soon_threadsafe`` — thread-safe by
        # construction, see DatusApp.invalidate).
        self._tui_output_buffer.set_on_change(self.tui_app.invalidate)

        # Sidebar visibility transitions (Ctrl+T toggle, first task appearing,
        # all tasks cleared, terminal resize crossing the min-cols threshold)
        # require rebuilding the Console at the new pane width — see
        # ``_reflow_for_sidebar``.
        self.tui_app.set_sidebar_visibility_listener(self._reflow_for_sidebar)

        @self.tui_app.key_bindings.add("tab")
        def _tab(event):  # noqa: ANN001 - prompt_toolkit signature
            """Tab: cycle the input mode on an empty line, complete otherwise.

            With a completion menu open (or any text typed) Tab keeps the
            legacy completion behavior, so mode cycling never steals Tab from
            an in-flight completion. On the empty main composer while the
            agent is idle it advances chat → sql → bash → chat.
            """
            buffer = event.app.current_buffer
            if buffer.complete_state:
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    buffer.complete_next()
                    cs = buffer.complete_state
                    if cs and cs.current_completion is not None:
                        buffer.apply_completion(cs.current_completion)
                return
            if buffer is self.tui_app.input_buffer and not self.tui_app._agent_running.is_set() and not buffer.text:
                self._cycle_input_mode()
                event.app.invalidate()
                return
            buffer.start_completion(select_first=True)

        @self.tui_app.key_bindings.add("s-tab")
        def _s_tab(event):  # noqa: ANN001
            """Shift+Tab: Toggle Plan Mode on/off.

            Unlike the PromptSession handler, the TUI must not call
            ``event.app.exit()`` — that would tear down the persistent
            Application. Instead the REPL just flips the flag and asks the
            layout to repaint. No scrollback announcement: the status-bar's
            ``PLAN`` segment (driven by :meth:`StatusBarState.to_formatted_tokens`)
            already reflects the state, so a single ``invalidate`` suffices.
            """
            self.plan_mode_active = not self.plan_mode_active
            event.app.invalidate()

        @self.tui_app.key_bindings.add("c-p", eager=True)
        def _c_p(event):  # noqa: ANN001
            """Ctrl+P: cycle normal → auto → dangerous → normal.

            The persistent TUI continues receiving input while an agent turn
            is running, so the new profile takes effect for subsequent tool
            permission checks without interrupting the conversation.
            """
            self._cycle_permission_mode()
            event.app.invalidate()

        @self.tui_app.key_bindings.add("c-o")
        def _c_o(event):  # noqa: ANN001
            """Ctrl+O: toggle verbose during a live stream, or expand the
            last chat's inline trace details when idle."""
            from datus.cli.tui.console_bridge import run_in_terminal_sync

            chat_commands = getattr(self, "chat_commands", None)
            if chat_commands is None:
                return

            # Live stream active: toggle verbose on the streaming context
            # (mirrors the key_callbacks entry the termios listener used to
            # wire for Ctrl+O outside the TUI).
            streaming_ctx = getattr(chat_commands, "current_streaming_ctx", None)
            if streaming_ctx is not None:
                try:
                    streaming_ctx.toggle_verbose()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(f"toggle_verbose failed: {exc}")
                return

            last_actions = getattr(chat_commands, "last_actions", None)
            if not last_actions:
                return

            def _show() -> None:
                chat_commands.display_inline_trace_details(last_actions)

            run_in_terminal_sync(_show)

        @self.tui_app.key_bindings.add("c-t")
        def _c_t(event):  # noqa: ANN001
            """Ctrl+T: toggle the todo sidebar between visible and hidden.

            Flips ``DatusApp._sidebar_force_hidden``; the next render runs
            ``_sidebar_visible``, observes the transition against
            ``_last_sidebar_visible``, and schedules ``_reflow_for_sidebar``
            on the event loop. ``invalidate`` forces that render to happen
            on the very next tick instead of waiting for the next input.
            """
            self.tui_app.toggle_sidebar_hidden()
            event.app.invalidate()

        @self.tui_app.key_bindings.add("escape")
        def _esc(event):  # noqa: ANN001
            """Escape: interrupt the running agent loop.

            DatusApp keeps prompt_toolkit's escape-sequence debounce at 10 ms,
            which still distinguishes arrow keys (``\\x1b[A`` etc.) while
            giving standalone ESC the same immediate feel as Ctrl+C. While
            idle it returns to chat mode if a non-chat input mode is active,
            otherwise it is a no-op.
            """
            if not self.tui_app._agent_running.is_set():
                if self.input_mode is not InputMode.CHAT:
                    self._set_input_mode(InputMode.CHAT)
                    event.app.invalidate()
                return
            self._interrupt_agent(restore_unanswered_input=True)

        @self.tui_app.key_bindings.add("c-c")
        def _c_c(event):  # noqa: ANN001
            """Ctrl+C: double-press within 1s exits; single press interrupts
            a running agent or clears the buffer when idle.
            """
            now = time.monotonic()
            if now - self._last_ctrl_c_time < 1.0:
                self._last_ctrl_c_time = 0.0
                self._interrupt_agent()
                self.tui_app.exit(0)
                return
            self._last_ctrl_c_time = now

            if self.tui_app._agent_running.is_set():
                self._interrupt_agent()
                self.tui_app.show_ctrl_c_hint()
                return

            self.tui_app.clear_paste_state()
            event.app.current_buffer.reset()
            # A single Ctrl+C while idle also drops back to chat mode so the
            # user is never stuck in SQL/bash mode after clearing the line.
            if self.input_mode is not InputMode.CHAT:
                self._set_input_mode(InputMode.CHAT)
            self.tui_app.show_ctrl_c_hint()

    def _set_input_mode(self, mode: InputMode) -> None:
        """Switch the input mode.

        No scrollback announcement — the mode is already unmistakable from the
        coloured prompt (``>`` / ``sql>`` / ``bash>``), the bracketing
        separators, and the persistent hint line, all driven by the
        ``input_mode_fn`` callback the TUI reads every paint. Flipping the flag
        plus a single ``invalidate`` is enough to reflect the change.
        """
        mode = InputMode(mode)
        if self.input_mode is mode:
            return
        self.input_mode = mode

        tui_app = getattr(self, "tui_app", None)
        if tui_app is not None:
            tui_app.invalidate()

    def _cycle_input_mode(self) -> None:
        """Advance the input mode one step in the Tab cycle (chat → sql → bash)."""
        self._set_input_mode(next_input_mode(self.input_mode))

    def _init_prompt_session(self):
        # Setup prompt session with custom key bindings
        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            # No SQL highlighting in the non-TUI fallback — plain input is model
            # chat, ``!<tool>`` runs a tool/plugin, and ``/`` runs a command.
            lexer=None,
            completer=self.create_combined_completer(),
            multiline=True,
            key_bindings=self._create_custom_key_bindings(),
            enable_history_search=True,
            search_ignore_case=True,
            erase_when_done=True,
            style=self._build_app_style(),
            complete_while_typing=True,
        )

    # Create combined completer
    def create_combined_completer(self):
        """Build SlashCommandCompleter + AtReferenceCompleter + SqlCompleter."""
        from datus.cli.autocomplete import SQLCompleter

        # SQL keyword/function/table completion is contextual: it only fires
        # while SQL mode is active, so plain model-chat and bash input get no
        # SQL suggestion popups. Slash / @-reference / service completers stay
        # on in every mode.
        sql_completer = ConditionalCompleter(SQLCompleter(), Condition(lambda: self.input_mode is InputMode.SQL))
        self.service_completer = ServiceCommandCompleter(self)
        self.at_completer = AtReferenceCompleter(
            self.agent_config,
            available_subagents=self.available_subagents,
            visibility_provider=self._visible_subagents_for_default,
        )  # Router for @Table / @Metrics / @Sql / @Agent inline references
        self.slash_completer = SlashCommandCompleter()
        self.bang_completer = BangCompleter(self)

        # Use merge_completers to combine completers
        from prompt_toolkit.completion import merge_completers

        return merge_completers(
            [
                self.bang_completer,  # !<tool> / !<plugin> completer (bound to the ! prefix)
                self.service_completer,  # .<service>.<method> dispatcher completer (highest priority)
                self.slash_completer,  # Top-level slash commands
                self.at_completer,  # @Table / @Metrics / @Sql inline references
                sql_completer,  # SQL keyword completer (SQL mode only, lowest priority)
            ]
        )

    def _dispatch_command_text(self, user_input_raw: str) -> Optional[str]:
        """Parse and execute a single user command.

        Shared by both the PromptSession loop and the TUI worker thread. When
        invoked from the TUI, this function runs on a :class:`ThreadPoolExecutor`
        worker so ``asyncio.run(...)`` inside chat commands does not collide
        with the prompt_toolkit Application's event loop on the main thread.

        Returns :data:`EXIT_SENTINEL` when the user requested an exit so the
        caller can tear down the TUI; returns ``None`` otherwise.
        """
        if user_input_raw is None:
            return None
        user_input = user_input_raw.strip()
        if not user_input:
            return None

        try:
            cmd_type, cmd, args = self._parse_command(user_input)
            # CHAT commands render the user message via the action stream
            # (the node yields a depth-0 USER ActionHistory that the renderer
            # turns into the scrollback header). SQL / BASH modes also flow
            # through the chat stream (as an execution turn), so they render
            # their own styled block — no echo here. Only the remaining
            # non-CHAT commands (SLASH / UNKNOWN) echo the raw input.
            if cmd_type not in (CommandType.CHAT, CommandType.SQL, CommandType.BASH):
                prompt_text = self._get_prompt_text()
                try:
                    lines = user_input.split("\n")
                    if len(lines) > PASTE_COLLAPSE_THRESHOLD:
                        summary_line = f"[Pasted content: {len(lines)} lines]"
                        self._echo_user_input(prompt_text, summary_line)
                        preview = "\n".join(lines[:3]) + "\n..."
                        self.console.print(f"[dim]{preview}[/]")
                    else:
                        self._echo_user_input(prompt_text, user_input)
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(f"echo_user_input failed: {e}")
            if cmd_type == CommandType.EXIT:
                return EXIT_SENTINEL
            if cmd_type == CommandType.SQL:
                self._execute_sql_mode(args)
            elif cmd_type == CommandType.BASH:
                self._execute_bash_mode(args)
            elif cmd_type == CommandType.BANG:
                self._execute_bang_command(args)
            elif cmd_type == CommandType.SLASH:
                slash_result = self._execute_slash_command(cmd, args)
                # ``/rewind`` sets ``_prefill_input`` from inside the handler.
                # In TUI mode the buffer was already drained before dispatch,
                # so push the rewound message back into the live input area
                # here. ``set_input_text`` schedules the mutation onto the
                # prompt_toolkit loop, so it is safe from the worker.
                if self._use_tui and self.tui_app is not None and self._prefill_input:
                    self.tui_app.set_input_text(self._prefill_input)
                    self._prefill_input = None
                if slash_result == EXIT_SENTINEL:
                    return EXIT_SENTINEL
            elif cmd_type == CommandType.CHAT:
                self._execute_chat_command(args, subagent_name=cmd)
            elif cmd_type == CommandType.UNKNOWN:
                # ``cmd`` carries the full rejected token, ``args`` the hint
                # (renamed target or empty). Rendering lives here so parsing
                # stays side-effect free.
                self._render_unknown_command(cmd, args)
        except KeyboardInterrupt:
            # Interrupt during a single command dispatch is non-fatal: the
            # outer loop (or TUI event loop) stays alive.
            pass
        except Exception as e:
            if "exit" in str(e).lower() and "app" in str(e).lower():
                # Shift+Tab plan-mode toggle historically surfaced as an app
                # exit event; treat it as benign.
                pass
            else:
                logger.error(f"Error: {str(e)}")
                self.console.print(f"[red]Error:[/] {str(e)}")
        return None

    def run(self):
        """Run the REPL loop."""
        if self._use_tui and self.tui_app is not None:
            return self._run_tui()
        return self._run_prompt_session()

    def _run_prompt_session(self):
        """Classic ``PromptSession`` main loop (used for non-TTY fallback)."""
        self._print_welcome()
        self._check_for_upgrade()
        self._warn_no_model()
        self._warn_no_datasource()
        self._bootstrap_services()
        self._auto_resume_if_requested()

        while True:
            try:
                # Get dynamic prompt text
                prompt_text = self._get_prompt_text()

                # Get user input (with optional prefill from rewind)
                prefill = self._prefill_input or ""
                user_input_raw = self.session.prompt(
                    message=lambda pt=prompt_text: self._build_prompt_message(pt),
                    default=prefill,
                )
                if user_input_raw is None:
                    continue
                if user_input_raw == "_open_chat_sql_details":
                    if self.chat_commands and self.chat_commands.last_actions:
                        self.chat_commands.display_inline_trace_details(self.chat_commands.last_actions)
                    continue
                self._prefill_input = None

                result = self._dispatch_command_text(user_input_raw)
                if result == EXIT_SENTINEL:
                    return True

            except KeyboardInterrupt:
                now = time.monotonic()
                if now - self._last_ctrl_c_time < 1.0:
                    return 0
                self._last_ctrl_c_time = now
                self.console.print("[dim]Press Ctrl+C again to exit[/]")
                continue
            except EOFError:
                return 0
            except Exception as e:
                if "exit" in str(e).lower() and "app" in str(e).lower():
                    continue
                logger.error(f"Error: {str(e)}")
                self.console.print(f"[red]Error:[/] {str(e)}")

    def _pin_tui_to_bottom(self) -> None:
        """No-op kept for subclass compatibility."""

    def _run_tui(self):
        """Persistent TUI main loop.

        The prompt_toolkit Application owns the main thread; user input is
        dispatched to :meth:`_dispatch_command_text` on a worker thread so
        long-running agent loops do not block UI redraws, and so that
        ``asyncio.run(...)`` inside those handlers does not collide with the
        Application's event loop.
        """
        self._pin_tui_to_bottom()
        self._print_welcome()
        self._check_for_upgrade()
        self._warn_no_model()
        self._warn_no_datasource()
        self._bootstrap_services()
        self._auto_resume_if_requested()

        # Prefill support mirrors the PromptSession path: ``.rewind`` stores
        # the replayed user message in ``_prefill_input`` and expects the
        # next prompt to display it as pre-filled editable text.
        if self._prefill_input:
            self.tui_app.set_input_text(self._prefill_input)
            self._prefill_input = None

        try:
            self.tui_app.run()
        except KeyboardInterrupt:
            return 0
        return True

    def _async_init_agent(self):
        """Initialize the agent asynchronously as a background coroutine.

        The work itself is blocking (agent construction + storage pre-load),
        so it runs via ``loop.run_in_executor`` inside the coroutine.  Wrapping
        it in a coroutine lets us schedule it on our managed background loop
        and carry the caller's ContextVar snapshot across execution units,
        which the previous naked-``threading.Thread`` approach did not do.
        """
        if self.agent_initializing or self.agent_ready:
            return

        self.agent_initializing = True

        # Capture the current ContextVar state so the background task sees
        # ``set_current_path_manager`` bindings made in the main thread.
        ctx = contextvars.copy_context()

        async def _runner() -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, ctx.run, self._background_init_agent)

        # Schedule the coroutine on the managed background loop.  call_soon_threadsafe
        # is the standard way to bridge from a foreign thread into an asyncio loop.
        self._bg_loop.call_soon_threadsafe(lambda: self._bg_loop.create_task(_runner()))

    def run_on_bg_loop(self, coro, *, interrupt_controller: Optional[InterruptController] = None):
        """Run a coroutine on the persistent background event loop and block until done.

        ``asyncio.run(coro)`` creates and then tears down a fresh loop on every
        call. When a chat turn leaves asyncio Tasks owned by ``prompt_toolkit``
        (for example the ``wait_for_cpr_responses`` Task created while rendering
        an interactive ``ask_user`` prompt), the Future those Tasks await lives
        on the short-lived loop. Subsequent turns tear that loop down while the
        Task is still pending — Python's GC then raises
        ``got Future pending attached to a different loop`` when finalizing the
        orphaned Task. Routing chat stream coroutines through this single
        persistent background loop keeps every Future/Task on the same loop
        across turns, eliminating the cross-loop GC warning and the
        ``Press ENTER to continue`` terminal hang it triggers.

        Args:
            coro: Coroutine to execute on ``_bg_loop``.
            interrupt_controller: Optional controller whose interrupt signal
                actively cancels the background asyncio Task.  This wakes an
                API stream currently parked waiting for its next event instead
                of waiting for a later cooperative polling checkpoint.

        Returns:
            The coroutine's return value.
        """

        async def _run_cancellable():
            if interrupt_controller is None:
                return await coro

            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            if task is None:  # pragma: no cover - asyncio always supplies one
                return await coro
            cancel_requested = threading.Event()

            def _cancel_task() -> None:
                # ``interrupt()`` normally runs on the prompt_toolkit/input
                # thread.  Task.cancel is not cross-thread safe, so always
                # marshal it onto the background loop.
                cancel_requested.set()
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    # Loop already stopped during shutdown.
                    pass

            token = interrupt_controller.register_cancel_callback(_cancel_task)
            try:
                return await coro
            except asyncio.CancelledError:
                # Preserve ordinary cancellation (for example application
                # shutdown).  Only user-triggered cancellation becomes the
                # chat-layer's existing graceful interrupt exception.  Keep a
                # local latch because the node resets its reusable controller
                # at stream startup; a very early key press can otherwise win
                # registration but have its event flag cleared before
                # CancelledError is delivered.
                if cancel_requested.is_set() or interrupt_controller.is_interrupted:
                    raise ExecutionInterrupted("Execution interrupted by user") from None
                raise
            finally:
                interrupt_controller.unregister_cancel_callback(token)
                # The controller is reused by the next chat turn. Leaving the
                # event latched makes that turn's callback fire immediately
                # during registration, swallowing the first Enter after a
                # cancel. Reset only after this task has fully unwound, so
                # same-turn early interrupts remain reliable.
                if cancel_requested.is_set():
                    interrupt_controller.reset()

        future = asyncio.run_coroutine_threadsafe(_run_cancellable(), self._bg_loop)
        try:
            return future.result()
        except KeyboardInterrupt:
            # Main thread received Ctrl+C while the coroutine was running on
            # _bg_loop. Propagate cancellation to the bg loop so the running
            # Task stops promptly, then re-raise so callers still see the
            # KeyboardInterrupt they already handle.
            self._bg_loop.call_soon_threadsafe(future.cancel)
            raise

    def _background_init_agent(self):
        """Background function that initializes the agent (runs inside the
        background loop's executor)."""
        try:
            # Create a mock args object based on CLI args
            from argparse import Namespace

            agent_args = Namespace(
                temperature=0.7,
                top_p=0.9,
                max_tokens=8000,
                workflow="reflection",
                max_steps=20,
                debug=self.args.debug,
                load_cp=False,
                components=["metrics", "metadata", "table_lineage", "document"],
            )

            from datus.agent.agent import Agent

            self.agent = Agent(agent_args, self.agent_config)

            self.agent_ready = True
            self.agent_initializing = False

            self.agent_commands.update_agent_reference()
            self._pre_load_storage()
            self._workflow_runner = self._create_workflow_runner()
            self._maybe_schedule_startup_sync()
            self._warm_bang_node()
            # self.console.print("[dim]Agent initialized successfully in background[/]")
        except Exception as e:
            self.console.print(f"[red]Error:[/]Failed to initialize agent in background: {str(e)}")
            logger.error(f"Failed to initialize agent in background: {e}")
            self.agent_initializing = False
            self.agent = None

    def _pre_load_storage(self):
        """Preload rag to avoid unnecessary printing"""
        if self.at_completer:
            try:
                self.at_completer.reload_data()
                errors = [
                    error
                    for error in (
                        getattr(self.at_completer.table_completer, "last_error", None),
                        getattr(self.at_completer.metric_completer, "last_error", None),
                        getattr(self.at_completer.sql_completer, "last_error", None),
                    )
                    if error
                ]
                if errors:
                    warning = format_context_degraded_warning("; ".join(errors))
                    self.startup_warnings.append(warning)
                    logger.warning("REPL context autocomplete preload degraded: %s", warning)
                    print_warning(self.console, warning)
            except Exception as exc:
                warning = format_context_degraded_warning(exc)
                self.startup_warnings.append(warning)
                logger.warning("REPL context autocomplete preload failed: %s", warning)
                print_warning(self.console, warning)

    def _warm_bang_node(self) -> None:
        """Eagerly build the default chat node during background init so ``!``
        autocomplete can list the agent's tools before the first chat turn.

        ``current_node`` is otherwise created lazily on the first chat, leaving
        the ``!<tool>`` completer (which reads the live node's tools without
        forcing creation, to stay non-blocking) with nothing to show until then.
        Warming here mounts the full tool set on the same node the first chat
        reuses. Best-effort — a failure only degrades the ``!`` list, so it must
        never break agent init.
        """
        try:
            self.chat_commands.ensure_node_for_bang()
        except Exception as exc:  # noqa: BLE001 - warming is best-effort
            logger.debug("Bang-node warm skipped: %s", exc)

    def _maybe_schedule_startup_sync(self) -> None:
        """Kick off a one-shot background metadata sync for the default
        datasource so the first ``@Table`` completion after launch reflects
        any tables added since the last ``datus-agent bootstrap-kb``. Gated by
        ``agent.autocomplete.background_sync_on_startup``.
        """
        bg_sync = getattr(self, "bg_sync", None)
        if bg_sync is None:
            return
        ac = getattr(self.agent_config, "autocomplete", None)
        if ac is None or not getattr(ac, "background_sync_on_startup", False):
            return
        current = getattr(self.agent_config, "current_datasource", "")
        if not current:
            return
        bg_sync.schedule(datasource=current, reason="startup")

    # Historical ``_rebuild_llm_after_switch`` removed. ``/model`` now persists
    # the new target via :meth:`AgentConfig.set_active_*` and the Agent reads
    # :meth:`AgentConfig.active_model` lazily each time it needs an LLM, so
    # there is nothing to rebuild on a switch — see ``datus/agent/agent.py``
    # ``llm`` property for the dispatch point.

    def check_agent_available(self):
        """Check if agent is available, and inform the user if it's still initializing."""
        if self.agent_ready and self.agent:
            return True
        elif self.agent_initializing:
            self.console.print(
                "[yellow]AI features are still initializing in the background. Please try again shortly.[/]"
            )
            return False
        else:
            self.console.print("[red]Error:[/] AI features are not available. Agent initialization failed.")
            return False

    def _cmd_mcp(self, args):
        from datus.cli.mcp_commands import MCPCommands

        MCPCommands(self).cmd_mcp(args)

    def _cmd_skill(self, args):
        from datus.cli.skill_commands import SkillCommands

        SkillCommands(self).cmd_skill(args)

    def _smart_display_table(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Smart table display that handles wide tables by limiting columns and truncating content.

        Args:
            data: List of dictionaries representing table rows
            columns: The columns to display, if not provided, all columns will be displayed
        """
        if not data:
            self.console.print("[yellow]No data to display[/]")
            return

        if columns:
            all_columns_list = columns
        else:
            # Get all unique column names
            all_columns_list = []
            for row in data:
                all_columns_list.extend(list(row.keys()))
        # Calculate the maximum number of columns based on the terminal width.
        max_columns = max(4, self.console.width // self.console_column_width)

        # Smart column selection: show front + back + ellipsis based on terminal width
        if len(all_columns_list) > max_columns:
            show_back = max_columns // 2
            show_front = max_columns - show_back  # -1 for ellipsis

            # Select columns to display
            front_columns = all_columns_list[:show_front]
            back_columns = all_columns_list[-show_back:] if show_back > 0 else []
            display_columns = front_columns + ["..."] + back_columns
        else:
            display_columns = all_columns_list

        # Calculate dynamic column width based on number of columns
        # With folding enabled, we can use narrower columns and fit more on screen
        num_display_columns = len([col for col in display_columns if col != "..."])
        if num_display_columns <= 2:
            # For 1-2 columns, use moderate width (content will fold if needed)
            dynamic_column_width = max(25, self.console.width // max(2, num_display_columns) - 4)
        elif num_display_columns <= 4:
            # For 3-4 columns, use compact width
            dynamic_column_width = max(20, self.console.width // num_display_columns - 3)
        elif num_display_columns <= 8:
            # For 5-8 columns, use narrow width (content will fold if needed)
            dynamic_column_width = max(18, self.console.width // num_display_columns - 2)
        else:
            # For many columns, use the default compact width
            dynamic_column_width = self.console_column_width

        table = Table(show_header=True, header_style="bold green")

        # Add columns with width constraints and folding for overflow
        for col in display_columns:
            if col == "...":
                table.add_column(col, width=5, justify="center")
            else:
                # Use dynamic column width with folding enabled for long content
                table.add_column(col, width=dynamic_column_width, overflow="fold", no_wrap=False)

        # Add rows with truncated content
        for row in data:
            row_values: List[Any] = []
            for col in display_columns:
                if col == "...":
                    row_values.append("...")
                else:
                    row_value = row.get(col)
                    if isinstance(row_value, datetime):
                        row_value = row_value.strftime("%Y-%m-%d %H:%M:%S")
                    elif isinstance(row_value, date):
                        row_value = row_value.strftime("%Y-%m-%d")
                    else:
                        row_value = str(row_value)
                    row_values.append(row_value)
            table.add_row(*row_values)

        self.console.print(table)

    def reset_session(self):
        self.chat_commands.update_chat_node_tools()
        if self.at_completer:
            # Perhaps we should reload the data here.
            self.at_completer.reload_data()

    def _visible_subagents_for_default(self) -> set[str]:
        """Filter ``self.available_subagents`` to those eligible as default agent.

        Drops :data:`HIDDEN_SYS_SUB_AGENTS` (internal meta agents such as
        ``feedback``) and scoped agents whose datasource doesn't match the
        current one. Mirrors the previous ``SubagentCompleter._load_subagents``
        behaviour now that the completer no longer surfaces agents directly.
        """

        visible = {name for name in self.available_subagents if name not in HIDDEN_SYS_SUB_AGENTS}
        if hasattr(self.agent_config, "agentic_nodes") and self.agent_config.agentic_nodes:
            current_db = getattr(self.agent_config, "current_datasource", None)
            for name, sub_config in self.agent_config.agentic_nodes.items():
                sc = (sub_config or {}).get("scoped_context", {})
                scoped_ns = sc.get("datasource")
                if scoped_ns and scoped_ns != current_db:
                    visible.discard(name)
        return visible

    def _cmd_agent(self, args: str):
        """Open the unified agent management TUI (Custom tab seed).

        ``/agent`` with no args lands on the Custom tab — that's the
        actionable surface for switching the default agent. The Built-in
        tab is config-only in the TUI (``max_turns`` overrides), so
        seeding it would land users on a tab where ``Enter`` no longer
        sets a default. ``/agent <name>`` keeps the legacy direct-setter
        shortcut for scripting — no TUI is launched, and built-in names
        remain accepted for backward compatibility.
        """
        name = args.strip()
        if name:
            self._set_default_agent_by_name(name)
            return
        self._open_agent_app(seed_tab="custom")

    def _cmd_subagent(self, args: str):
        """Open the unified agent management TUI (Custom tab seed).

        Any arguments are ignored: the legacy ``add|list|remove|update``
        subcommands were removed and all operations now live inside the
        TUI (``a`` to add, ``d`` to delete, ``Enter`` to edit, ``s`` to
        set as default).
        """
        if args and args.strip():
            self.console.print("[dim]/subagent no longer accepts subcommands — opening the unified agent TUI.[/]")
        self._open_agent_app(seed_tab="custom")

    def _set_default_agent_by_name(self, name: str) -> None:
        visible = self._visible_subagents_for_default()
        if name not in visible:
            self.console.print(f"[red]Error:[/] Unknown agent '{name}'. Run '/agent' to see available agents.")
            return
        if name == "chat":
            self.default_agent = ""
            self.console.print("[green]Default agent reset to: chat[/]")
        else:
            self.default_agent = name
            self.console.print(f"[green]Default agent set to: {name}[/]")

    def _open_agent_app(self, *, seed_tab: str = "builtin") -> None:
        """Drive the unified :class:`AgentApp` with the follow-up handlers.

        The app itself only persists Built-in overrides internally. All
        other outcomes (wizard launch, deletion, default switch) are
        applied here so we never nest prompt_toolkit Applications.
        """
        from datus.cli.agent_app import AgentApp

        current_seed = seed_tab
        while True:
            visible_custom = self._visible_subagents_for_default() - SYS_SUB_AGENTS - {"chat"}
            app = AgentApp(
                agent_config=self.agent_config,
                console=self.console,
                default_agent=self.default_agent,
                visible_custom_agents=visible_custom,
                seed_tab=current_seed,
            )
            tui_app = getattr(self, "tui_app", None)
            if tui_app is not None and getattr(tui_app, "_loop", None) is not None:
                sel = tui_app.run_wizard(app.build_embedded_panel)
            else:
                sel = app.run()
            if sel is None:
                return
            if sel.kind == "set_default":
                self._set_default_agent_by_name(sel.name or "chat")
                return
            if sel.kind == "edit_custom" and sel.name:
                self._launch_sub_agent_wizard(existing=sel.name)
            elif sel.kind == "new_custom":
                self._launch_sub_agent_wizard(existing=None)
            elif sel.kind == "delete_custom" and sel.name:
                self._delete_custom_agent(sel.name)
            current_seed = sel.return_to_tab or "custom"

    def _launch_sub_agent_wizard(self, *, existing: Optional[str]) -> None:
        """Run :class:`SubAgentWizard` and persist the result.

        ``existing=None`` starts with an empty form; passing a name
        pre-fills the wizard from the current configuration. System
        subagents are never routed here — the unified TUI's Built-in tab
        handles them — so no :data:`SYS_SUB_AGENTS` guard is needed.
        """
        from datus.cli.cli_styles import print_error, print_success, print_warning
        from datus.cli.sub_agent_wizard import run_wizard
        from datus.schemas.agent_models import SubAgentConfig
        from datus.utils.sub_agent_manager import SubAgentManager

        sub_agent_manager = SubAgentManager(
            configuration_manager=self.configuration_manager,
            datasource=self.agent_config.current_datasource,
            agent_config=self.agent_config,
        )

        data: Optional[Dict[str, Any]] = None
        original_name: Optional[str] = None
        if existing:
            data = sub_agent_manager.get_agent(existing)
            if data is None:
                print_error(self.console, f"Agent '{existing}' not found.")
                return
            original_name = existing

        try:
            result = run_wizard(self, data)
        except Exception as exc:  # pragma: no cover - defensive
            print_error(self.console, f"An error occurred while running the wizard: {exc}", prefix=False)
            logger.error("Sub-agent wizard failed: %s", exc)
            return

        if result is None:
            print_warning(self.console, f"Agent cancelled {'creation' if not data else 'modification'}.")
            return

        agent_name = result.system_prompt
        if agent_name in SYS_SUB_AGENTS:
            print_error(
                self.console,
                f"'{agent_name}' is reserved for built-in sub-agents and cannot be used.",
            )
            return

        if original_name is None and isinstance(data, SubAgentConfig):
            original_name = data.system_prompt
        elif original_name is None and isinstance(data, dict):
            original_name = data.get("system_prompt")

        try:
            save_result = sub_agent_manager.save_agent(result, previous_name=original_name)
        except Exception as exc:
            print_error(self.console, f"Failed to persist sub-agent: {exc}", prefix=False)
            logger.error("Failed to persist sub-agent '%s': %s", agent_name, exc)
            return

        changed = save_result.get("changed", True)
        if not changed:
            print_warning(self.console, "No changes detected; skipping save.")
            return

        self._refresh_agent_config_after_subagent_change(sub_agent_manager)

        config_path = save_result.get("config_path")
        prompt_path = save_result.get("prompt_path")
        if config_path:
            self.console.print(f"- Updated configuration file: [cyan]{config_path}[/]")
        if prompt_path:
            self.console.print(f"- Created prompt template: [cyan]{prompt_path}[/]")
        if save_result.get("kb_action") == "cleared":
            print_warning(self.console, "- Cleared scoped knowledge base for previous configuration.")

        print_success(
            self.console,
            f"Sub-agent {agent_name} {'created' if not data else 'modified'} successfully.",
        )

    def _delete_custom_agent(self, name: str) -> None:
        from datus.cli.cli_styles import print_error, print_success
        from datus.utils.sub_agent_manager import SubAgentManager

        if name in SYS_SUB_AGENTS:
            print_error(self.console, f"System sub-agent '{name}' cannot be removed.")
            return
        sub_agent_manager = SubAgentManager(
            configuration_manager=self.configuration_manager,
            datasource=self.agent_config.current_datasource,
            agent_config=self.agent_config,
        )
        try:
            removed = sub_agent_manager.remove_agent(name)
        except Exception as exc:
            print_error(self.console, f"Error removing agent: {exc}", prefix=False)
            logger.error("Failed to remove agent '%s': %s", name, exc)
            return
        if not removed:
            print_error(self.console, f"Agent '{name}' not found.")
            return
        print_success(self.console, f"- Removed agent '{name}' from configuration.")
        self._refresh_agent_config_after_subagent_change(sub_agent_manager)

    def _refresh_agent_config_after_subagent_change(self, sub_agent_manager) -> None:
        """Mirror :meth:`SubAgentCommands._refresh_agent_config` from the
        retired text-based command so in-memory state stays consistent
        after wizard / delete flows."""
        try:
            if hasattr(self.agent_config, "agentic_nodes"):
                self.agent_config.agentic_nodes = sub_agent_manager.list_agents()
            if hasattr(self, "available_subagents"):
                self.available_subagents = set(SYS_SUB_AGENTS)
                self.available_subagents.add("chat")
                if self.agent_config.agentic_nodes:
                    self.available_subagents.update(
                        name for name in self.agent_config.agentic_nodes.keys() if name != "chat"
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to refresh in-memory agent config: %s", exc)

    def _run_profile_picker(self, current: str, notice: Optional[str] = None) -> Optional[str]:
        """Run ProfilePickerApp embedded in TUI when available."""
        from datus.cli.profile_picker_app import ProfilePickerApp

        app = ProfilePickerApp(console=self.console, current=current, notice=notice)
        tui_app = getattr(self, "tui_app", None)
        if tui_app is not None and getattr(tui_app, "_loop", None) is not None:
            return tui_app.run_wizard(app.build_embedded_panel)
        return app.run()

    def _run_dangerous_confirm(self) -> bool:
        """Run DangerousConfirmApp embedded in TUI when available.

        Returns True only if the user explicitly enabled Dangerous.
        """
        from datus.cli.profile_picker_app import DangerousConfirmApp

        app = DangerousConfirmApp(console=self.console)
        tui_app = getattr(self, "tui_app", None)
        if tui_app is not None and getattr(tui_app, "_loop", None) is not None:
            result = tui_app.run_wizard(app.build_embedded_panel)
            return result == "enable"
        return app.run()

    def _parse_command(self, text: str) -> Tuple[CommandType, str, str]:
        """Classify raw user input into a ``CommandType`` + canonical cmd + args.

        All side-effects (printing hints, running handlers) live in the
        dispatcher so this function stays deterministic and trivially
        unit-testable.

        Returns:
            Tuple ``(command_type, command, arguments)``:

            * ``SQL``    — ``command`` empty, ``arguments`` is the raw SQL
            * ``BANG``   — ``command`` empty, ``arguments`` is the text after ``!``
            * ``SLASH``  — ``command`` is the canonical ``"/name"`` (aliases resolved)
            * ``CHAT``   — ``command`` is the default agent, ``arguments`` is the message
            * ``EXIT``   — both empty
            * ``UNKNOWN`` — ``command`` is the rejected token, ``arguments`` is a hint
        """

        text = text.strip()
        bash_mode = self.input_mode is InputMode.BASH

        # Trailing ``;`` is only normalized on the SQL path (SQL mode), where it
        # is a redundant statement terminator. Chat keeps it (``Explain this;``
        # must reach the model verbatim) and bash keeps it (a trailing ``;`` is
        # legal shell syntax).

        # Exit: bare ``exit`` / ``quit`` still work; ``/exit`` and ``/quit`` flow
        # through the SLASH branch via the registry's alias map.
        if text.lower() in ("exit", "quit"):
            return CommandType.EXIT, "", ""

        # ``!`` prefix, chat mode only: run one of the agent's tools directly
        # (``!list_tables``) or an installed plugin's CLI (``!hello sync``),
        # dispatched by :class:`datus.cli.bang_command.BangCommand` (tools match
        # first). In SQL/bash mode a leading ``!`` is part of the statement
        # (e.g. bash history expansion), so the prefix is not interpreted there.
        if text.startswith("!") and self.input_mode is InputMode.CHAT:
            return CommandType.BANG, "", text[1:].strip()

        # Slash commands (/prefix). ``/<agent> <msg>`` was removed — agent
        # selection is now exclusively handled by ``/agent``. Unknown tokens
        # surface as ``UNKNOWN`` rather than silently flowing to chat so typos
        # fail loudly. In bash mode only *registered* slash commands are
        # intercepted, so ``/help`` stays reachable while absolute paths like
        # ``/usr/bin/ls`` fall through and run as bash.
        if text.startswith("/"):
            parts = text[1:].split(maxsplit=1)
            raw_token = parts[0] if parts and parts[0] else ""
            token = raw_token.lower()
            args = parts[1] if len(parts) > 1 else ""
            spec = lookup(token) if token else None
            if spec is not None:
                # ``/exit`` / ``/quit`` flow through SLASH dispatch so
                # ``_cmd_exit`` gets to close the DB connector before the
                # handler returns ``EXIT_SENTINEL`` to the outer loop.
                return CommandType.SLASH, f"/{spec.name}", args
            if bash_mode:
                return CommandType.BASH, "", text
            # Dynamic service routes (``/<service>`` for method listing or
            # ``/<service>.<method>`` for invocation) are resolved in
            # ``_execute_slash_command`` via ``ServiceCommands.dispatch``.
            # Preserve the raw token's casing because service / method
            # registry lookups respect the user's configured names.
            if raw_token:
                return CommandType.SLASH, f"/{raw_token}", args
            return CommandType.UNKNOWN, "/", ""

        # Legacy prefix hints: ``.xxx`` / ``@catalog`` / ``@subject`` used to
        # be live commands. Surface a rename hint instead of running them so
        # shell-history replay reports a clear error. Skipped in bash mode,
        # where a leading ``.`` is the POSIX source operator.
        if not bash_mode:
            first_token = text.split(maxsplit=1)[0].lower()
            legacy_target = _LEGACY_PREFIX_HINTS.get(first_token)
            if legacy_target is not None:
                return CommandType.UNKNOWN, first_token, legacy_target

        # Execution modes (cycled with Tab on an empty line in the TUI):
        # plain input runs as SQL against the current datasource or as a
        # shell command through the permission-gated bash tool. The slash
        # branch above runs first, so ``/help`` / ``/tables`` stay reachable
        # without leaving the mode.
        if self.input_mode is InputMode.SQL:
            sql = text[:-1].strip() if text.endswith(";") else text
            return CommandType.SQL, "", sql
        if bash_mode:
            return CommandType.BASH, "", text

        # Default: route free-form text to the model. SQL is never auto-detected
        # — the user opts into execution explicitly via SQL mode.
        return CommandType.CHAT, self.default_agent, text.strip()

    def _execute_sql_mode(self, sql: str) -> None:
        """Run an input-bar SQL statement, then send it as an execution turn.

        The statement first passes the same enforcement an LLM ``execute_sql``
        call gets — permission (``on_tool_start`` → ``_handle_sql_permission``:
        read auto-allow, write/DDL confirmation), plugin transformers, and
        ``_enforce_sql_policy`` — via :func:`datus.cli.bash_mode.run_sql_gate`.
        On approval the (possibly rewritten) statement executes and is packed
        into a marker-encoded chat message dispatched to the model
        (:meth:`_send_exec_turn`).
        """
        sql = sql.strip()
        if not sql:
            return
        from datus.cli.bash_mode import run_manual_sql_live

        # The live frame shows ``sql> <stmt> · running Ns`` while the gate +
        # query run, then the bordered result block. ``run_manual_sql_live``
        # returns ``(payload, dispatch)``; only a real execution feeds the model.
        payload, dispatch = run_manual_sql_live(self, sql, self._run_manual_sql)
        if dispatch and payload is not None:
            self._send_exec_turn(payload)

    def _run_manual_sql(self, sql: str):
        """Execute *sql* against the active connector and build an exec payload.

        Returns a :mod:`datus.cli.manual_exec` payload dict on success or SQL
        error (both worth showing the model), or ``None`` for infrastructure
        failures (no connector, empty/non-arrow result) after printing the
        error locally. Does not render the result itself — the exec block does.
        """
        import time

        from datus.cli.manual_exec import build_sql_error_payload, build_sql_message_payload, build_sql_payload

        logger.debug(f"Executing SQL query: '{sql}'")
        sql_action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="sql_execution",
            messages=f"Executing SQL: {sql[:100]}..." if len(sql) > 100 else f"Executing SQL: {sql}",
            input_data={"sql": sql},
            status=ActionStatus.PROCESSING,
        )
        self.actions.add_action(sql_action)

        try:
            if not self.db_connector:
                error_msg = "No database connection. Please initialize a connection first."
                self.console.print(f"[red]Error:[/] {error_msg}")
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return None

            start_time = time.time()
            result = self.db_connector.execute(input_params={"sql_query": sql}, result_format="arrow")
            exec_time = time.time() - start_time

            if not result:
                error_msg = "No result from the query."
                self.console.print(f"[red]Error:[/] {error_msg}")
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return None

            self.last_sql = sql
            self.last_result = result

            # For CONTENT_SET SQL (USE/SET), sync cli_context from connector state.
            if result.success:
                try:
                    sql_type = parse_sql_type(sql, getattr(self.db_connector, "dialect", ""))
                    if sql_type == SQLType.CONTENT_SET:
                        self.cli_context.current_catalog = getattr(self.db_connector, "catalog_name", "") or ""
                        self.cli_context.current_db_name = getattr(self.db_connector, "database_name", "") or ""
                        self.cli_context.current_schema = getattr(self.db_connector, "schema_name", "") or ""
                except Exception:
                    pass

            if result.success:
                # Non-arrow success paths. A successful statement without a
                # tabular payload is either DML (``row_count`` set — including a
                # zero-row UPDATE/DELETE) or DDL / a side-effecting statement
                # that returns no rows at all. Both are successes: ``result.success``
                # is already true here, so never fall through to a format error.
                if not hasattr(result.sql_return, "column_names"):
                    if result.row_count is not None:
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.SUCCESS,
                            output={"row_count": result.row_count, "execution_time": exec_time, "success": True},
                            messages=f"SQL executed successfully: {result.row_count} rows in {exec_time:.2f}s",
                        )
                        return build_sql_message_payload(
                            sql, True, f"{result.row_count} rows updated in {exec_time:.2f}s"
                        )
                    self.actions.update_action_by_id(
                        sql_action.action_id,
                        status=ActionStatus.SUCCESS,
                        output={"row_count": 0, "execution_time": exec_time, "success": True},
                        messages=f"SQL executed successfully in {exec_time:.2f}s",
                    )
                    return build_sql_message_payload(sql, True, f"success in {exec_time:.2f}s")

                rows = result.sql_return.to_pylist()
                columns = result.sql_return.column_names
                row_count = result.sql_return.num_rows
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.SUCCESS,
                    output={
                        "row_count": row_count,
                        "execution_time": exec_time,
                        "columns": columns,
                        "success": True,
                    },
                    messages=f"SQL executed successfully: {row_count} rows in {exec_time:.2f}s",
                )
                if self._workflow_runner and self._workflow_runner.workflow_ready:
                    self.workflow_runner.workflow.context.sql_contexts.append(
                        SQLContext(
                            sql_query=sql,
                            sql_return=str(result.sql_return),
                            row_count=row_count,
                            explanation=f"Manual sql: Returned {row_count} rows in {exec_time:.2f} seconds",
                        )
                    )
                return build_sql_payload(sql, columns, rows, row_count, exec_time)

            error_msg = result.error or "Unknown SQL error"
            self.actions.update_action_by_id(
                sql_action.action_id,
                status=ActionStatus.FAILED,
                output={"error": error_msg, "sql_error": True},
                messages=f"SQL error: {error_msg}",
            )
            if self._workflow_runner and self._workflow_runner.workflow_ready:
                self._workflow_runner.workflow.context.sql_contexts.append(
                    SQLContext(
                        sql_query=sql,
                        sql_return=str(result.error) if result.error else "Unknown error",
                        row_count=0,
                        explanation="Manual sql",
                    )
                )
            return build_sql_error_payload(sql, error_msg)
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            self.actions.update_action_by_id(
                sql_action.action_id,
                status=ActionStatus.FAILED,
                output={"error": str(e), "exception": True},
                messages=f"SQL execution exception: {str(e)}",
            )
            return build_sql_error_payload(sql, str(e))

    # ── bash mode ────────────────────────────────────────────────────────

    def _execute_bash_mode(self, command: str) -> None:
        """Run a bash-mode command with a live frame, then send it to the model.

        Permission gating + execution (BashTool + PermissionHooks, including
        the embedded permission wizard) live in :mod:`datus.cli.bash_mode`. The
        live frame shows ``bash> <cmd> · running Ns`` while it runs, then the
        bordered result block. Only a real execution is dispatched to the model
        (:meth:`_send_exec_turn`); a denied/cancelled command runs nothing.
        """
        command = command.strip()
        if not command:
            return
        from datus.cli.bash_mode import run_manual_bash_live

        payload, dispatch = run_manual_bash_live(self, command)
        if dispatch and payload is not None:
            self._send_exec_turn(payload)

    def _execute_bang_command(self, text: str) -> None:
        """Run a ``!<tool>`` / ``!<plugin>`` command via :class:`BangCommand`.

        Tools are gated + invoked directly and rendered locally; plugins run in a
        ``datus <plugin> ...`` subprocess. Neither is fed back to the model.
        """
        self.bang_command.dispatch(text)

    def _bang_param_hint(self, text: str) -> str:
        """Argument-name hint for the live input, consumed by the TUI's
        ``AfterInput`` processor. ``""`` while typing the tool/plugin name.

        Late-bound wrapper so ``self.bang_command`` need not exist when the app
        is constructed.
        """
        bang = getattr(self, "bang_command", None)
        if bang is None:
            return ""
        return bang.param_hint(text)

    # ── manual-execution chat turn ────────────────────────────────────────

    def _send_exec_turn(self, payload: dict) -> None:
        """Dispatch a manual-execution payload as a chat turn.

        The encoded message is fed through the normal chat path so the model
        sees the command + result and responds; its USER action renders as the
        styled SQL/bash block (never a plain user bubble). Plan mode never
        applies — a manual execution is not a planning task.
        """
        from datus.cli.manual_exec import encode_exec_message

        message = encode_exec_message(payload)
        try:
            self.chat_commands.execute_chat_command(message, plan_mode=False)
        except Exception as e:
            logger.error(f"Failed to dispatch execution turn: {e}")
            self.console.print(f"[red]Error:[/] {rich_escape(str(e))}")

    def _execute_chat_command(self, message: str, subagent_name: str = None):
        """Route free-form chat text to the configured default agent."""
        self.chat_commands.execute_chat_command(message, plan_mode=self.plan_mode_active, subagent_name=subagent_name)
        # Sync the REPL plan-mode toggle from node state — ``confirm_plan``
        # flips the node's ``plan_mode_active`` off when the user accepts
        # the plan, but the REPL switch otherwise stays on and would force
        # re-activation on the next prompt.
        node = getattr(self.chat_commands, "current_node", None)
        if node is not None and self.plan_mode_active and not getattr(node, "plan_mode_active", False):
            self.plan_mode_active = False
            logger.debug("REPL plan-mode toggle synced off after confirm_plan")

    def _execute_slash_command(self, cmd: str, args: str):
        """Execute a slash command resolved via ``SLASH_COMMANDS`` registry.

        Falls back to :meth:`ServiceCommands.dispatch` for dynamic
        ``/<service>.<method>`` routes (BI / scheduler / semantic methods
        enumerated at runtime, not statically registered in the registry).

        Returns ``EXIT_SENTINEL`` when the handler requested shutdown (``/exit``
        / ``/quit``) so the dispatcher can forward it to the outer loop.
        """
        logger.debug(f"Executing slash command: '{cmd}' with args: '{args}'")
        handler = self.commands.get(cmd)
        if handler is None:
            if self.service_commands.dispatch(cmd, args):
                return None
            self.console.print(f"[red]Unknown command:[/] {cmd}. Type /help.")
            return None
        result = handler(args)
        # ``/rewind`` returns a user message to prefill in the input buffer.
        if cmd == "/rewind" and result is not None:
            self._prefill_input = result
            return None
        if result == EXIT_SENTINEL:
            return EXIT_SENTINEL
        return None

    def _render_unknown_command(self, token: str, hint: str):
        """Report an unrecognised slash or renamed legacy prefix to the user."""
        if hint:
            self.console.print(f"[red]Unknown command:[/] '{token}' has been renamed to '{hint}'. Type /help.")
        else:
            self.console.print(f"[red]Unknown command:[/] {token}. Type /help.")

    def _wait_for_agent_available(self, max_attempts=5, delay=1):
        """Wait for the agent to become available, with timeout."""
        if self.check_agent_available():
            return True

        self.console.print("[yellow]Waiting for the agent to initialize...[/]")

        import time

        for _ in range(max_attempts):
            time.sleep(delay)
            if self.check_agent_available():
                return True

        self.console.print("[red]Agent initialization timed out. Try again later.[/]")
        return False

    def _cmd_help(self, args: str):
        """Display help for all CLI commands.

        Slash commands are rendered from :data:`SLASH_COMMANDS`. Tool commands
        and chat behaviour are described inline; use ``/<command>`` help output
        from the command itself for deeper usage (e.g. ``/mcp`` without args).
        """

        CMD_WIDTH = 30
        lines: list[str] = ["[green]Datus-CLI Help[/]\n"]
        lines.append("[bold]Chat:[/]")
        lines.append(f"    {'<message>':<{CMD_WIDTH}}Chat with the default agent (configure via /agent)")
        lines.append("")

        lines.append("[bold]Input modes:[/]")
        lines.append(f"    {'Tab':<{CMD_WIDTH}}On an empty line, cycle chat > / sql> / bash> modes")
        lines.append(f"    {'<sql>':<{CMD_WIDTH}}In SQL mode, Enter runs it; \\+Enter for a newline")
        lines.append(f"    {'<command>':<{CMD_WIDTH}}In bash mode, Enter runs it via the permission-gated bash tool")
        lines.append(f"    {'Esc / Ctrl+C':<{CMD_WIDTH}}Back to chat mode (plan mode applies to chat input only)")
        lines.append("")

        by_group: dict[str, list] = {group: [] for group in GROUP_ORDER}
        for spec in iter_visible():
            by_group.setdefault(spec.group, []).append(spec)
        for group in GROUP_ORDER:
            specs = by_group.get(group) or []
            if not specs:
                continue
            title = GROUP_TITLES.get(group, group.title())
            lines.append(f"[bold]{title} (/ prefix):[/]")
            for spec in specs:
                token = f"/{spec.name}"
                if spec.aliases:
                    token = token + ", " + ", ".join(f"/{alias}" for alias in spec.aliases)
                lines.append(f"    {token:<{CMD_WIDTH}}{spec.summary}")
            lines.append("")

        self.console.print("\n".join(lines).rstrip())

    def _cmd_exit(self, args: str) -> str:
        """Exit the CLI.

        Closes the DB connector and returns ``EXIT_SENTINEL`` so the dispatcher
        can signal both the PromptSession loop and the TUI application to shut
        down cleanly. Returning the sentinel (rather than calling
        ``sys.exit(0)``) matters in TUI mode where ``_cmd_exit`` runs on a
        worker thread — ``sys.exit`` would only kill the worker while the main
        prompt_toolkit Application kept running.
        """
        if self.db_connector:
            try:
                # Close the connection
                self.db_connector.close()
            except Exception as e:
                logger.warning(f"Database connection closed failed, reason:{e}")
        bg_sync = getattr(self, "bg_sync", None)
        if bg_sync is not None:
            bg_sync.shutdown()
        return EXIT_SENTINEL

    def _cmd_profile(self, args: str) -> None:
        """Deprecated alias for :meth:`_cmd_permission`."""

        return DatusCLI._cmd_permission(self, args, notice="/profile is deprecated; use /permission instead.")

    def _cmd_permission(self, args: str, notice: Optional[str] = None) -> None:
        """Open the permission profile picker and apply the choice.

        Delegates to ``_run_profile_picker`` / ``_run_dangerous_confirm``
        (inline prompt_toolkit pickers mirroring ``/agent``). Selecting
        ``dangerous`` triggers a second confirmation every session
        transition per spec decision #5.
        """
        current = getattr(self.agent_config, "active_profile_name", self.active_profile)
        choice = self._run_profile_picker(current, notice=notice)

        if choice is None:
            return

        DatusCLI._switch_permission_profile(self, choice, confirm_dangerous=True, announce=True)

    def _cycle_permission_mode(self) -> None:
        """Advance the permission profile without opening a modal picker.

        This is the Ctrl+P shortcut path.  Reaching ``dangerous`` requires a
        deliberate key press from ``auto``; keeping it modal-free is what
        allows the shortcut to work while a conversation is running.
        """
        from datus.tools.permission.profiles import PROFILE_NAMES

        current = getattr(self.agent_config, "active_profile_name", self.active_profile)
        try:
            next_index = (PROFILE_NAMES.index(current) + 1) % len(PROFILE_NAMES)
        except ValueError:
            next_index = 0
        DatusCLI._switch_permission_profile(
            self,
            PROFILE_NAMES[next_index],
            confirm_dangerous=False,
            announce=False,
        )

    def _switch_permission_profile(self, choice: str, *, confirm_dangerous: bool, announce: bool) -> None:
        """Apply *choice* to config and the live node's permission manager."""
        from datus.tools.permission.permission_config import PermissionConfig
        from datus.tools.permission.profiles import PROFILE_NAMES, build_effective_config, get_profile

        current = getattr(self.agent_config, "active_profile_name", self.active_profile)

        if choice not in PROFILE_NAMES:
            print_error(self.console, f"Unknown profile: {choice}")
            return

        if choice == current:
            print_info(self.console, f"Already on {choice}.")
            return

        # Dangerous second confirmation — every session transition re-confirms.
        if choice == "dangerous" and confirm_dangerous:
            confirmed = self._run_dangerous_confirm()
            if not confirmed:
                print_warning(self.console, "Dangerous mode cancelled.")
                return

        # Rebuild the user rules (exclude the profile key) and preserve the
        # new profile's default unless the user explicitly set one.
        # Mirrors build_effective_config in profiles.py. If you change one,
        # change both.
        raw_permissions = getattr(self.agent_config, "_raw_permissions", {}) or {}
        raw_user = {k: v for k, v in raw_permissions.items() if k != "profile"}
        plugin_rules_map = getattr(self.agent_config, "plugin_bash_rules", {}) or {}
        try:
            new_effective = build_effective_config(choice, raw_user, plugin_bash_rules=plugin_rules_map.get(choice))
        except Exception as e:
            # Mirror startup: if ``permissions.rules`` can't be parsed, refuse
            # to install a permissive profile base that would silently drop
            # restrictive overrides. Fail closed to ``normal`` with a clear
            # user-facing message instead of silently expanding privileges.
            logger.warning(f"Failed to rebuild effective permissions for {choice!r}: {e}. Falling back to 'normal'.")
            print_error(
                self.console,
                f"permissions.rules in agent.yml is malformed ({e}); refusing to switch to "
                f"{choice!r} and falling back to 'normal'.",
            )
            choice = "normal"
            new_effective = get_profile("normal")

        # Reconstruct the user_rules_cfg for switch_profile (it takes a
        # separable override, not a pre-merged config).
        user_rules_cfg: Optional[PermissionConfig] = None
        if raw_user:
            if "default" not in raw_user and "default_permission" not in raw_user:
                base = get_profile(choice)
                dp = base.default_permission
                raw_user = {
                    **raw_user,
                    "default_permission": dp.value if hasattr(dp, "value") else dp,
                }
            try:
                user_rules_cfg = PermissionConfig.from_dict(raw_user)
            except Exception as e:
                logger.warning(f"Malformed user rules for {choice!r}: {e}. Applying profile base only.")
                user_rules_cfg = None

        # Apply the runtime switch FIRST. If ``switch_profile`` raises
        # (unknown profile, malformed override, etc.) the config still
        # reports the *old* profile so the status bar and PermissionManager
        # stay consistent instead of publishing a half-applied state.
        prior_approvals = 0
        current_node = getattr(self.chat_commands, "current_node", None)
        if current_node is not None and hasattr(current_node, "permission_manager"):
            prior_approvals = len(getattr(current_node.permission_manager, "_session_approvals", {}))
            try:
                current_node.permission_manager.switch_profile(choice, user_overrides=user_rules_cfg)
            except Exception as e:
                print_error(self.console, f"Profile switch failed: {e}")
                return

        self.agent_config.permissions_config = new_effective
        self.agent_config.active_profile_name = choice
        self.active_profile = choice
        # Drop the CLI-owned PermissionManager built lazily for manual SQL/bash
        # before any chat node existed — it captured the *old* profile + session
        # approvals, so the next manual execution must rebuild it on the new
        # profile (otherwise a normal→dangerous or dangerous→normal switch would
        # not take effect for manual runs). Node-owned managers are switched
        # in place above.
        self._cli_bash_permission_manager = None
        if announce:
            print_success(self.console, f"Profile switched: {current} → {choice}")
            print_info(self.console, f"Session approvals cleared (was: {prior_approvals})")

    def catalogs_callback(self, selected_path: str = "", selected_data: Optional[Dict[str, Any]] = None):
        if not selected_path:
            return
        self.selected_catalog_path = selected_path
        self.selected_catalog_data = selected_data

    def _build_banner_panel(self) -> Panel:
        """Build the unified startup banner as a Rich Panel."""
        database = getattr(self.args, "datasource", "") or getattr(self.agent_config, "current_datasource", "")
        db_type = getattr(self.agent_config, "db_type", "") or ""

        if self.db_connector and database:
            db_line = f"[green]{database}[/]"
            if db_type:
                db_line += f"  [dim]({db_type})[/]"
            if self.cli_context.current_db_name and self.cli_context.current_db_name != database:
                db_line += f"  [dim]using {self.cli_context.current_db_name}[/]"
        elif database:
            db_line = f"[green]{database}[/]  [yellow]not connected[/]"
        else:
            db_line = "[yellow]not selected  (use /database to choose)[/]"

        context_summary = self.cli_context.get_context_summary() if self.db_connector else "No context available"
        show_context = context_summary and context_summary != "No context available"

        use_art = self.console.width >= _BANNER_MIN_WIDTH
        body = Table.grid(padding=(0, 0))
        body.add_column()

        if use_art:
            body.add_row(Text(DATUS_BANNER_TEXT, style="bold"))
        else:
            body.add_row(Text(f"DATUS v{__version__}", style="bold"))
        body.add_row(Text(""))
        body.add_row(Text("Data engineering agent builds evolvable context for your data system", style="bold"))
        body.add_row(Text(""))

        info = Table.grid(padding=(0, 2))
        info.add_column(style="dim", justify="left", no_wrap=True)
        info.add_column()
        info.add_row("Datasource", Text.from_markup(db_line))
        if show_context:
            info.add_row("Context", Text.from_markup(f"[dim]{context_summary}[/]"))
        body.add_row(info)
        body.add_row(Text(""))
        body.add_row(Text.from_markup("[dim]Type / for commands, /help for the full list, /exit to quit[/]"))

        return Panel(
            body,
            title=f"v{__version__}",
            title_align="left",
            padding=(1, 2),
        )

    def _print_welcome(self):
        """Print the unified startup banner.

        Also used as the Ctrl+O clear-screen header callback so the banner
        reappears at the top after verbose-mode toggles redraw the terminal.
        """
        self.console.print(self._build_banner_panel())

    def _warn_no_model(self):
        """Print a one-time hint when no active model is configured."""
        try:
            self.agent_config.active_model()
        except Exception:
            self.console.print("[yellow]No model configured. Use /model to set up a model.[/]")

    def _warn_no_datasource(self):
        """Print a one-time hint when no datasource is configured."""
        if not self.agent_config.services.datasources:
            self.console.print("[yellow]No datasources configured. Use /datasource to add one.[/]")

    def _check_for_upgrade(self) -> None:
        """Hint that a newer ``datus-agent`` is available, then refresh the cache.

        Two-step, deliberately non-blocking and non-racy:

        * **Synchronous fast path** — if a *fresh* on-disk cache already
          knows of a newer release, print a one-line yellow hint right
          after the banner. A cold cache shows nothing (no network call on
          the main thread).
        * **Background refresh** — a daemon thread re-queries PyPI and
          updates the cache only. It never prints (which would interleave
          with the prompt); the hint surfaces from the next launch onward.

        Skipped entirely for non-interactive sessions or when
        ``DATUS_DISABLE_VERSION_CHECK`` is set. Any failure is swallowed —
        a version check must never abort startup.
        """
        import os

        if os.environ.get("DATUS_DISABLE_VERSION_CHECK"):
            return
        try:
            from datus.cli.service_bootstrap import _is_interactive

            if not _is_interactive():
                return
        except Exception:  # pragma: no cover - defensive
            return

        from datus import __version__
        from datus.cli import upgrade_service as upgrade_svc

        try:
            newer = upgrade_svc.newer_version_available(__version__, agent_config=self.agent_config, cached_only=True)
            if newer:
                self.console.print(
                    f"[yellow]A new datus-agent is available: {__version__} -> {newer}. "
                    f"Run [bold]datus upgrade[/] to update.[/]"
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cached version check failed: %s", exc)

        def _worker() -> None:
            try:
                upgrade_svc.get_latest_version(agent_config=self.agent_config)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("background version fetch failed: %s", exc)

        try:
            threading.Thread(target=_worker, daemon=True).start()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not start version-check thread: %s", exc)

    def _bootstrap_services(self) -> None:
        """Pin project defaults and kick off background adapter installs.

        Skipped silently in non-interactive sessions (CI, ``echo |
        datus-cli``, MCP / API hosts) — see ``service_bootstrap._is_interactive``
        for the exact guard. Failures inside the bootstrap never abort
        startup.
        """
        try:
            from datus.cli import service_bootstrap

            service_bootstrap.run(self)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"service bootstrap failed: {exc}")

    def _auto_resume_if_requested(self) -> None:
        """Auto-trigger ``/resume`` when ``--resume <session_id>`` is set.

        Same effect as entering the REPL and typing ``/resume <session_id>``:
        rehydrates the node and replays history before the first prompt.
        """
        session_id = getattr(self.args, "resume", None)
        if not session_id:
            return
        self.chat_commands.cmd_resume(session_id)

    def prompt_input(self, message: str, default: str = "", choices: list = None, multiline: bool = False):
        """
        Unified input method using prompt_toolkit to avoid conflicts with rich.Prompt.ask().

        Args:
            message: The prompt message to display
            default: Default value if user presses Enter without input
            choices: List of valid choices (validates input)
            multiline: Whether to allow multiline input

        Returns:
            User input string or default value
        """
        session_style = self.session.style if self.session is not None else Style.from_dict({})
        return prompt_input(
            self.console, message, default=default, choices=choices, multiline=multiline, style=session_style
        )

    def _init_connection(self, timeout_seconds: int = 30):
        """Initialize database connection with timeout control.

        Args:
            timeout_seconds: Maximum time to wait for connection (default: 30 seconds)
        """
        current_datasource = self.agent_config.current_datasource
        if not current_datasource:
            self.db_connector = None
            return

        def _do_init_connection():
            """Inner function to perform connection initialization."""
            if not self.cli_context.current_db_name:
                db_name, connector = self.db_manager.first_conn_with_name(current_datasource)
                return db_name, connector
            else:
                connector = self.db_manager.get_conn(current_datasource, self.cli_context.current_db_name)
                return self.cli_context.current_db_name, connector

        try:
            # Use ThreadPoolExecutor with timeout for connection initialization
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_init_connection)
                try:
                    db_name, self.db_connector = future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    self.console.print(
                        f"[red]Error:[/] Database connection timed out after {timeout_seconds} seconds. "
                        f"Please check if the database server for datasource '{current_datasource}' is running "
                        "and accessible."
                    )
                    logger.error(f"Database connection timeout for datasource: {current_datasource}")
                    self.db_connector = None
                    return

            if not self.db_connector:
                self.console.print("[red]Error:[/] No database connection.")
                return

            # Update context based on dialect
            if self.db_connector.dialect in (DBType.SQLITE, DBType.DUCKDB):
                self.cli_context.update_database_context(db_name=self.db_connector.database_name)
            else:
                self.cli_context.update_database_context(
                    catalog=self.db_connector.catalog_name,
                    db_name=self.db_connector.database_name,
                )

            # Test the connection with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.db_connector.test_connection)
                try:
                    connection_result = future.result(timeout=timeout_seconds)
                    logger.debug(f"Connection test result: {connection_result}")
                except FuturesTimeoutError:
                    self.console.print(
                        f"[red]Error:[/] Connection test timed out after {timeout_seconds} seconds. "
                        f"The database server for datasource '{current_datasource}' may be unresponsive."
                    )
                    logger.error(f"Connection test timeout for datasource: {current_datasource}")
                    self.db_connector = None

        except Exception as e:
            self.console.print(f"[red]Error:[/] Failed to connect to database: {str(e)}")
            logger.error(f"Database connection failed for datasource {current_datasource}: {e}")
            self.db_connector = None

    def _create_workflow_runner(self) -> WorkflowRunner:
        return self.agent.create_workflow_runner(run_id=datetime.now().strftime("%Y%m%d"))
