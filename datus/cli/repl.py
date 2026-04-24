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
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style, merge_styles, style_from_pygments_cls
from rich.console import Console
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
    CustomPygmentsStyle,
    CustomSqlLexer,
    ServiceCommandCompleter,
    SlashCommandCompleter,
)
from datus.cli.bi_dashboard import BiDashboardCommands
from datus.cli.chat_commands import ChatCommands
from datus.cli.cli_styles import PASTE_COLLAPSE_THRESHOLD, print_error, print_info, print_success, print_warning
from datus.cli.context_commands import ContextCommands
from datus.cli.effort_commands import EffortCommands
from datus.cli.language_commands import LanguageCommands
from datus.cli.metadata_commands import MetadataCommands
from datus.cli.model_commands import ModelCommands
from datus.cli.service_commands import ServiceCommands
from datus.cli.slash_registry import GROUP_ORDER, GROUP_TITLES, iter_visible, lookup
from datus.cli.status_bar import StatusBarProvider
from datus.cli.tui import DatusApp, tui_enabled
from datus.cli.tui.app import EXIT_SENTINEL
from datus.cli.tui.live_display_state import LiveDisplayState
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
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

    SQL = "sql"  # Regular SQL statement
    TOOL = "tool"  # !command (tool/workflow)
    SLASH = "slash"  # /command (session / metadata / context / agent / system)
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

        # Plan mode support
        self.plan_mode_active = False
        self._last_ctrl_c_time: float = 0.0
        # Default agent for /message routing ("" = chat node)
        self.default_agent = ""

        # Load agent config first so path-dependent helpers use the configured home.
        self.agent_config = load_agent_config(create_if_missing=True, **vars(self.args))
        self.configuration_manager = configuration_manager()

        # Active permission profile name. Initialized from agent_config;
        # mutated by /profile. StatusBarProvider reads this for display.
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
            self.at_completer = AtReferenceCompleter(self.agent_config, available_subagents=self.available_subagents)

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
        self.bi_dashboard_commands = BiDashboardCommands(self)
        self.model_commands = ModelCommands(self)
        self.language_commands = LanguageCommands(self)
        self.effort_commands = EffortCommands(self)
        self.service_commands = ServiceCommands(self)
        from datus.cli.datasource_commands import DatasourceCommands

        self.datasource_commands = DatasourceCommands(self)
        self._status_bar_provider = StatusBarProvider(self)

        # Dictionary of available commands - created after handlers are initialized
        self.commands: Dict[str, Any] = {
            # "!run": self.agent_commands.cmd_darun_screen,
            "!sl": self.agent_commands.cmd_schema_linking,
            "!schema_linking": self.agent_commands.cmd_schema_linking,
            "!sm": self.agent_commands.cmd_search_metrics,
            "!search_metrics": self.agent_commands.cmd_search_metrics,
            "!sq": self.agent_commands.cmd_search_reference_sql,
            "!search_sql": self.agent_commands.cmd_search_reference_sql,
            "!sd": self.agent_commands.cmd_doc_search,
            "!search_document": self.agent_commands.cmd_doc_search,
            # "!gen": self.agent_commands.cmd_gen,
            # "!fix": self.agent_commands.cmd_fix,
            "!save": self.agent_commands.cmd_save,
            "!bash": self._cmd_bash,
            # to be deprecated when sub agent is read
            # "!reason": self.agent_commands.cmd_reason_stream,
            # "!compare": self.agent_commands.cmd_compare_stream,
        }
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
            # agent
            "agent": self._cmd_agent,
            "subagent": self._cmd_subagent,
            "datasource": self.datasource_commands.cmd,
            "language": self.language_commands.cmd_language,
            # system
            "mcp": self._cmd_mcp,
            "skill": self._cmd_skill,
            "bootstrap-bi": self.bi_dashboard_commands.cmd,
            "model": self.model_commands.cmd_model,
            "effort": self.effort_commands.cmd_effort,
            "services": self.service_commands.cmd_services,
            "profile": self._cmd_profile,
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

        @kb.add("tab")
        def _(event):
            """The Tab key triggers completion only, not navigation."""
            buffer = event.app.current_buffer

            if buffer.complete_state:
                # If the menu is already open, close it.
                buffer.complete_next()
            else:
                # If the menu is incomplete, trigger completion.
                buffer.start_completion(select_first=False)

        @kb.add("s-tab")
        def _(event):
            """Shift+Tab: Toggle Plan Mode on/off"""
            self.plan_mode_active = not self.plan_mode_active

            # Clear current input buffer and force exit current prompt
            buffer = event.app.current_buffer
            buffer.reset()

            # Force the prompt to exit and restart with new prefix
            # This will cause the main loop to regenerate the prompt
            buffer.validation_state = None
            event.app.exit()

            # Show mode change message
            if self.plan_mode_active:
                self.console.print("[green]Plan Mode Activated![/]")
                self.console.print("[dim]Enter your planning task and press Enter to generate plan[/]")
            else:
                self.console.print("[yellow]Plan Mode Deactivated[/]")

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
        """Re-echo user input with Pygments syntax highlighting matching prompt_toolkit style."""
        from pygments import highlight
        from pygments.formatters import TerminalTrueColorFormatter
        from rich.text import Text

        highlighted = highlight(user_input, CustomSqlLexer(), TerminalTrueColorFormatter(style=CustomPygmentsStyle))
        echoed = Text(prompt_text, style="green bold")
        echoed.append_text(Text.from_ansi(highlighted.rstrip("\n")))
        self.console.print(echoed)

    def _get_prompt_text(self):
        """Input-line prompt text.

        The Datus brand, plan mode, and current agent are now rendered by the
        status bar on the line above, so the input line keeps a single minimal
        indicator. Legacy call sites that expect a textual prompt still receive
        a non-empty string here.
        """
        return "> "

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
                Style.from_dict(
                    {
                        "prompt": "ansigreen bold",
                        "input-prompt": "ansigreen bold",
                        "input-prompt.busy": "ansibrightblack",
                        "input-prompt.hint": "italic #9a9aaa",
                        "input-area": "",
                        "status-bar": "#9a9aaa",
                        "status-bar.brand": "#ffd866 bold",
                        "status-bar.plan": "#9a9aaa",
                        "status-bar.profile": "#9a9aaa",
                        "status-bar.profile.auto": "ansicyan",
                        "status-bar.profile.dangerous": "ansired",
                        "status-bar.sep": "#9a9aaa",
                        "status-bar.agent": "#9a9aaa",
                        "status-bar.connector": "#9a9aaa",
                        "status-bar.model": "#9a9aaa",
                        "status-bar.tokens": "#9a9aaa",
                        "status-bar.ctx": "#9a9aaa",
                        "status-bar.running": "#ffb86c bold",
                        "status-bar.dot": "#ffb86c bold",
                        "separator": "#444444",
                        # Slash-command autocomplete popup. Every row pins
                        # ``bg:default`` so the menu blends into the terminal
                        # palette instead of prompt_toolkit's stock teal-on-
                        # white block. prompt_toolkit's default style for
                        # ``.current`` ships with ``reverse`` (swaps fg/bg,
                        # producing a highlighted bar); ``noreverse`` strips
                        # that so the selection is conveyed by text color
                        # alone — bold bright cyan — with no colored band.
                        "completion-menu": "bg:default",
                        "completion-menu.completion": "bg:default fg:default",
                        "completion-menu.completion.current": "noreverse bg:default fg:ansibrightcyan bold",
                        "completion-menu.meta.completion": "bg:default fg:ansibrightblack",
                        "completion-menu.meta.completion.current": "noreverse bg:default fg:ansibrightcyan bold",
                        "hint": "#9a9aaa italic",
                        # Pinned subagent/tool rolling-window lines. Dim grey
                        # to match the scrollback ``[dim]`` styling used by the
                        # Rich renderers so the visual weight stays consistent
                        # between the pinned region and the append area.
                        "subagent-live": "#6e6e6e",
                        "processing-live": "#6e6e6e",
                        # Pinned subagent header: plain cyan on the
                        # "⏺ subagent_type" prefix, default colour on the
                        # parenthesised goal so the prompt text does not
                        # compete visually with the subagent name.
                        "subagent-header-live": "ansicyan",
                        "subagent-header-goal-live": "",
                    }
                ),
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

    def _interrupt_agent(self) -> None:
        chat_commands = getattr(self, "chat_commands", None)
        current_node = getattr(chat_commands, "current_node", None) if chat_commands else None
        controller = getattr(current_node, "interrupt_controller", None) if current_node else None
        if controller is not None:
            try:
                controller.interrupt()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"interrupt_controller.interrupt failed: {exc}")

    def _init_tui_app(self) -> None:
        """Create the persistent ``DatusApp`` and register REPL bindings."""
        # The Tab handler matches the legacy PromptSession behavior
        # (trigger completion only, no navigation). Additional bindings —
        # Shift+Tab plan-mode toggle, Ctrl+O trace details, ESC interrupt —
        # are wired in later phases.
        from prompt_toolkit.lexers import PygmentsLexer

        # The TUI path still relies on the same AtReferenceCompleter handle
        # that downstream code queries for subagent state, so attach it
        # before constructing the app.
        completer = self.create_combined_completer()

        # Build the shared live-render state first; DatusApp will wire its
        # own ``invalidate`` into it once constructed (see LiveDisplayState
        # docstring for the deferred-callback rationale).
        self.live_state = LiveDisplayState()

        self.tui_app = DatusApp(
            status_tokens_fn=self._status_tokens_for_tui,
            dispatch_fn=self._dispatch_command_text,
            completer=completer,
            history=self.history,
            lexer=PygmentsLexer(CustomSqlLexer),
            style=self._build_app_style(),
            input_prompt_fn=self._get_prompt_text,
            live_display_state=self.live_state,
        )

        @self.tui_app.key_bindings.add("tab")
        def _tab(event):  # noqa: ANN001 - prompt_toolkit signature
            buffer = event.app.current_buffer
            if buffer.complete_state:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=False)

        @self.tui_app.key_bindings.add("s-tab")
        def _s_tab(event):  # noqa: ANN001
            """Shift+Tab: Toggle Plan Mode on/off.

            Unlike the PromptSession handler, the TUI must not call
            ``event.app.exit()`` — that would tear down the persistent
            Application. Instead the REPL just flips the flag and asks the
            layout to repaint; the status-bar's ``PLAN`` segment is driven
            by :meth:`StatusBarState.to_formatted_tokens` so a single
            ``invalidate`` is enough to reflect the change.
            """
            from datus.cli.tui.console_bridge import run_in_terminal_sync

            self.plan_mode_active = not self.plan_mode_active
            active = self.plan_mode_active

            def _announce() -> None:
                if active:
                    self.console.print("[green]Plan Mode Activated![/]")
                    self.console.print("[dim]Enter your planning task and press Enter to generate plan[/]")
                else:
                    self.console.print("[yellow]Plan Mode Deactivated[/]")

            # Printing via ``run_in_terminal`` keeps the pinned status-bar +
            # input intact: prompt_toolkit temporarily moves them out of the
            # way, emits the message, then restores them at the bottom.
            run_in_terminal_sync(_announce)
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

        @self.tui_app.key_bindings.add("escape")
        def _esc(event):  # noqa: ANN001
            """Escape: interrupt the running agent loop.

            prompt_toolkit debounces ESC so this handler only fires for a
            standalone key press, not for the leading byte of arrow-key
            escape sequences (``\\x1b[A`` etc.). While idle the binding is
            a no-op so default Buffer behavior (no-op for ESC in insert
            mode) is preserved.
            """
            if not self.tui_app._agent_running.is_set():
                return
            self._interrupt_agent()

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
            self.tui_app.show_ctrl_c_hint()

    def _init_prompt_session(self):
        # Setup prompt session with custom key bindings
        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            lexer=PygmentsLexer(CustomSqlLexer),
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

        sql_completer = SQLCompleter()
        self.service_completer = ServiceCommandCompleter(self)
        self.at_completer = AtReferenceCompleter(
            self.agent_config, available_subagents=self.available_subagents
        )  # Router for @Table / @Metrics / @Sql inline references
        self.slash_completer = SlashCommandCompleter()

        # Use merge_completers to combine completers
        from prompt_toolkit.completion import merge_completers

        return merge_completers(
            [
                self.service_completer,  # .<service>.<method> dispatcher completer (highest priority)
                self.slash_completer,  # Top-level slash commands
                self.at_completer,  # @Table / @Metrics / @Sql inline references
                sql_completer,  # SQL keyword completer (lowest priority)
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

        # Re-echo user input with syntax highlighting. In TUI mode the input
        # TextArea clears on Enter, so echoing via the patched stdout keeps a
        # transcript of what was submitted. In PromptSession mode
        # ``erase_when_done=True`` removes the prompt line, so the echo is
        # still useful.
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

        try:
            cmd_type, cmd, args = self._parse_command(user_input)
            if cmd_type == CommandType.EXIT:
                return EXIT_SENTINEL
            if cmd_type == CommandType.SQL:
                self._execute_sql(user_input)
            elif cmd_type == CommandType.TOOL:
                self._execute_tool_command(cmd, args)
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
        self._warn_no_model()
        self._warn_no_datasource()

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
        self._warn_no_model()
        self._warn_no_datasource()

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

    def run_on_bg_loop(self, coro):
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

        Returns:
            The coroutine's return value.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
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
            # self.console.print("[dim]Agent initialized successfully in background[/]")
        except Exception as e:
            self.console.print(f"[red]Error:[/]Failed to initialize agent in background: {str(e)}")
            logger.error(f"Failed to initialize agent in background: {e}")
            self.agent_initializing = False
            self.agent = None

    def _pre_load_storage(self):
        """Preload rag to avoid unnecessary printing"""
        if self.at_completer:
            self.at_completer.reload_data()

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
        """Open the unified agent management TUI (Built-in tab seed).

        ``/agent`` with no args lands on the Built-in tab so users can
        tweak ``model`` / ``max_turns`` overrides for system subagents.
        ``/agent <name>`` keeps the legacy direct-setter shortcut for
        scripting — no TUI is launched.
        """
        name = args.strip()
        if name:
            self._set_default_agent_by_name(name)
            return
        self._open_agent_app(seed_tab="builtin")

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
            if tui_app is not None:
                with tui_app.suspend_input():
                    sel = app.run()
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

    def _run_profile_picker(self, current: str) -> Optional[str]:
        """Run the standalone ProfilePickerApp; return selection or None."""
        from datus.cli.profile_picker_app import ProfilePickerApp

        app = ProfilePickerApp(console=self.console, current=current)
        tui_app = getattr(self, "tui_app", None)
        if tui_app is not None:
            with tui_app.suspend_input():
                return app.run()
        return app.run()

    def _run_dangerous_confirm(self) -> bool:
        """Run the standalone DangerousConfirmApp; return True only if
        the user explicitly enabled Dangerous."""
        from datus.cli.profile_picker_app import DangerousConfirmApp

        app = DangerousConfirmApp(console=self.console)
        tui_app = getattr(self, "tui_app", None)
        if tui_app is not None:
            with tui_app.suspend_input():
                return app.run()
        return app.run()

    def _parse_command(self, text: str) -> Tuple[CommandType, str, str]:
        """Classify raw user input into a ``CommandType`` + canonical cmd + args.

        All side-effects (printing hints, running handlers) live in the
        dispatcher so this function stays deterministic and trivially
        unit-testable.

        Returns:
            Tuple ``(command_type, command, arguments)``:

            * ``SQL``    — ``command`` empty, ``arguments`` is the raw SQL
            * ``TOOL``   — ``command`` is ``"!name"`` (lowercased)
            * ``SLASH``  — ``command`` is the canonical ``"/name"`` (aliases resolved)
            * ``CHAT``   — ``command`` is the default agent, ``arguments`` is the message
            * ``EXIT``   — both empty
            * ``UNKNOWN`` — ``command`` is the rejected token, ``arguments`` is a hint
        """

        text = text.strip()

        # Remove trailing semicolons (common in SQL)
        if text.endswith(";"):
            text = text[:-1].strip()

        # Exit: bare ``exit`` / ``quit`` still work; ``/exit`` and ``/quit`` flow
        # through the SLASH branch via the registry's alias map.
        if text.lower() in ("exit", "quit"):
            return CommandType.EXIT, "", ""

        # Tool commands (!prefix). Unchanged by this refactor.
        if text.startswith("!"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.TOOL, cmd, args

        # Slash commands (/prefix). ``/<agent> <msg>`` was removed — agent
        # selection is now exclusively handled by ``/agent``. Unknown tokens
        # surface as ``UNKNOWN`` rather than silently flowing to chat so typos
        # fail loudly.
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
        # shell-history replay reports a clear error.
        first_token = text.split(maxsplit=1)[0].lower()
        legacy_target = _LEGACY_PREFIX_HINTS.get(first_token)
        if legacy_target is not None:
            return CommandType.UNKNOWN, first_token, legacy_target

        # Determine if text is SQL or chat using parse_sql_type
        try:
            # Get current database dialect from agent_config.db_type (set from current datasource)
            dialect = self.agent_config.db_type if self.agent_config.db_type else "snowflake"
            sql_type = parse_sql_type(text, dialect)

            # If parse_sql_type returns a valid SQL type (not UNKNOWN), treat as SQL
            if sql_type != SQLType.UNKNOWN:
                return CommandType.SQL, "", text
            return CommandType.CHAT, self.default_agent, text.strip()
        except Exception:
            # If any exception occurs, treat as chat
            return CommandType.CHAT, self.default_agent, text.strip()

    def _execute_sql(self, sql: str, system: bool = False):
        """Execute a SQL query and display results."""
        logger.debug(f"Executing SQL query: '{sql}'")

        # Create action for SQL execution
        sql_action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="sql_execution",
            messages=f"Executing SQL: {sql[:100]}..." if len(sql) > 100 else f"Executing SQL: {sql}",
            input_data={"sql": sql, "system": system},
            status=ActionStatus.PROCESSING,
        )
        self.actions.add_action(sql_action)

        try:
            if not self.db_connector:
                error_msg = "No database connection. Please initialize a connection first."
                self.console.print(f"[red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return

            # Execute the query
            import time

            start_time = time.time()
            result = self.db_connector.execute(input_params={"sql_query": sql}, result_format="arrow")
            end_time = time.time()
            exec_time = end_time - start_time

            if not result:
                error_msg = "No result from the query."
                self.console.print(f"[red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return

            # Save for later reference
            self.last_sql = sql
            self.last_result = result

            # For CONTENT_SET SQL (USE/SET statements), update cli_context in-place from connector state
            if result.success:
                try:
                    sql_type = parse_sql_type(sql, getattr(self.db_connector, "dialect", ""))
                    if sql_type == SQLType.CONTENT_SET:
                        self.cli_context.current_catalog = getattr(self.db_connector, "catalog_name", "") or ""
                        self.cli_context.current_db_name = getattr(self.db_connector, "database_name", "") or ""
                        self.cli_context.current_schema = getattr(self.db_connector, "schema_name", "") or ""
                except Exception:
                    pass

            # Display results and update action
            if result.success:
                if not hasattr(result.sql_return, "column_names"):
                    if result.row_count is not None and result.row_count > 0:
                        # Update action with success
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.SUCCESS,
                            output={
                                "row_count": result.row_count,
                                "execution_time": exec_time,
                                "success": True,
                            },
                            messages=f"SQL executed successfully: {result.row_count} rows in {exec_time:.2f}s",
                        )
                        self.console.print(f"[dim]Update {result.sql_return} rows in {exec_time:.2f} seconds[/]")
                    elif result.sql_return:
                        self.console.print(f"[dim]SQL execution successful in {exec_time:.2f} seconds[/]")
                        # Update action with success
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.SUCCESS,
                            output={
                                "row_count": 0,
                                "execution_time": exec_time,
                                "success": True,
                            },
                            messages=f"SQL executed successfully in {exec_time:.2f}s",
                        )
                    else:
                        error_msg = (
                            f"Query execution failed - received string instead of Arrow data:"
                            f" {result.error or 'Unknown error'}"
                        )
                        self.console.print(f"[red]Error:[/] {error_msg}")

                        # Update action with error
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.FAILED,
                            output={"error": error_msg, "result_type_error": True},
                            messages=f"Result format error: {error_msg}",
                        )
                    return
                # Convert Arrow data to list of dictionaries for smart display
                rows = result.sql_return.to_pylist()
                self._smart_display_table(data=rows, columns=result.sql_return.column_names)

                row_count = result.sql_return.num_rows
                self.console.print(f"[dim]Returned {row_count} rows in {exec_time:.2f} seconds[/]")

                # Update action with success
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.SUCCESS,
                    output={
                        "row_count": row_count,
                        "execution_time": exec_time,
                        "columns": result.sql_return.column_names,
                        "success": True,
                    },
                    messages=f"SQL executed successfully: {row_count} rows in {exec_time:.2f}s",
                )
                workflow_ready = self._workflow_runner and self._workflow_runner.workflow_ready
                if not system and workflow_ready:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.sql_return),
                        row_count=row_count,
                        explanation=f"Manual sql: Returned {row_count} rows in {exec_time:.2f} seconds",
                    )
                    self.workflow_runner.workflow.context.sql_contexts.append(new_record)

            else:
                error_msg = result.error or "Unknown SQL error"
                self.console.print(f"[red]SQL Error:[/] {error_msg}")

                # Update action with SQL error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "sql_error": True},
                    messages=f"SQL error: {error_msg}",
                )
                workflow_ready = self._workflow_runner and self._workflow_runner.workflow_ready
                if not system and workflow_ready:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.error) if result.error else "Unknown error",
                        row_count=0,
                        explanation="Manual sql",
                    )
                    self._workflow_runner.workflow.context.sql_contexts.append(new_record)
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            self.console.print(f"[red]Error:[/] {str(e)}")

            # Update action with exception
            self.actions.update_action_by_id(
                sql_action.action_id,
                status=ActionStatus.FAILED,
                output={"error": str(e), "exception": True},
                messages=f"SQL execution exception: {str(e)}",
            )

    def _execute_tool_command(self, cmd: str, args: str):
        """Execute a tool command (! prefix)."""
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            self.console.print(f"[red]Unknown command:[/] {cmd}")

    def _execute_chat_command(self, message: str, subagent_name: str = None):
        """Route free-form chat text to the configured default agent."""
        self.chat_commands.execute_chat_command(message, plan_mode=self.plan_mode_active, subagent_name=subagent_name)

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

    def _cmd_bash(self, args: str):
        """Execute a bash command."""
        # Define a whitelist of allowed commands
        whitelist = ["pwd", "ls", "cat", "head", "tail", "echo"]

        if not args.strip():
            self.console.print("[yellow]Please provide a bash command.[/]")
            return

        # Parse the command to check against whitelist
        cmd_parts = args.split()
        base_cmd = cmd_parts[0]

        if base_cmd not in whitelist:
            self.console.print(
                f"[red]Security:[/] Command '{base_cmd}' not in whitelist. Allowed: {', '.join(whitelist)}"
            )
            return

        try:
            # Execute the command
            import subprocess

            result = subprocess.run(args, shell=True, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                if result.stdout:
                    self.console.print(result.stdout)
            else:
                self.console.print(f"[red]Command failed with code {result.returncode}:[/]\n{result.stderr}")

        except subprocess.TimeoutExpired:
            self.console.print("[red]Error:[/] Command timed out after 10 seconds.")
        except Exception as e:
            self.console.print(f"[red]Error:[/] {str(e)}")

    def _cmd_help(self, args: str):
        """Display help for all CLI commands.

        Slash commands are rendered from :data:`SLASH_COMMANDS`. Tool commands
        and chat behaviour are described inline; use ``/<command>`` help output
        from the command itself for deeper usage (e.g. ``/mcp`` without args).
        """

        CMD_WIDTH = 30
        lines: list[str] = ["[green]Datus-CLI Help[/]\n"]
        lines.append("[bold]SQL:[/]")
        lines.append(f"    {'<sql>':<{CMD_WIDTH}}Execute SQL query directly")
        lines.append("")

        lines.append("[bold]Chat:[/]")
        lines.append(f"    {'<message>':<{CMD_WIDTH}}Chat with the default agent (configure via /agent)")
        lines.append("")

        lines.append("[bold]Tool Commands (! prefix):[/]")
        tool_cmds = [
            ("!sl, !schema_linking", "Schema linking: recommended tables and values"),
            ("!sm, !search_metrics", "Search metrics by natural language"),
            ("!sq, !search_sql", "Search reference SQL by natural language"),
            ("!sd, !search_document", "Search platform documentation by keywords"),
            ("!save", "Save the last result to a file"),
            ("!bash <command>", "Execute a bash command (limited to safe commands)"),
        ]
        for cmd, desc in tool_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
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
        return EXIT_SENTINEL

    def _cmd_profile(self, args: str) -> None:
        """Open the profile selection picker and apply the choice.

        Delegates to ``_run_profile_picker`` / ``_run_dangerous_confirm``
        (inline prompt_toolkit pickers mirroring ``/agent``). Selecting
        ``dangerous`` triggers a second confirmation every session
        transition per spec decision #5.
        """
        from datus.tools.permission.permission_config import PermissionConfig
        from datus.tools.permission.profiles import PROFILE_NAMES, build_effective_config, get_profile

        current = getattr(self.agent_config, "active_profile_name", self.active_profile)
        choice = self._run_profile_picker(current)

        if choice is None:
            return

        if choice not in PROFILE_NAMES:
            print_error(self.console, f"Unknown profile: {choice}")
            return

        if choice == current:
            print_info(self.console, f"Already on {choice}.")
            return

        # Dangerous second confirmation — every session transition re-confirms.
        if choice == "dangerous":
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
        try:
            new_effective = build_effective_config(choice, raw_user)
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
                self.cli_context.update_database_context(db_name=self.db_connector.database_name, db_logic_name=db_name)
            else:
                self.cli_context.update_database_context(
                    catalog=self.db_connector.catalog_name,
                    db_name=self.db_connector.database_name,
                    db_logic_name=db_name or self.db_connector.database_name or current_datasource,
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
