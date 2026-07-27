# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Persistent prompt_toolkit Application that pins the status bar + input.

The Datus REPL historically used ``PromptSession.prompt()`` and re-rendered
the status bar as a prefix each round. During agent loops, the session was
not active and the status bar + input disappeared. :class:`DatusApp` replaces
that pull model with a long-lived ``Application(full_screen=False)`` whose
layout keeps the status bar (1 row) and input :class:`TextArea` at the bottom
of the terminal for the entire REPL lifetime. Agent work runs on a dedicated
worker thread; its output is captured by ``patch_stdout(raw=True)`` and
scrolls in the area above the pinned bottom.

Concurrency contract:

* The prompt_toolkit Application owns the main thread and its asyncio loop.
* User input is accepted via an ``enter`` key binding; when the agent is idle,
  the input text is passed to ``dispatch_fn`` on a ``ThreadPoolExecutor``
  (``max_workers=1``). While that future is pending, ``agent_running`` is set,
  Enter is swallowed, and the status bar/input reflect the busy state.
* ``dispatch_fn`` runs in the worker thread and may use ``asyncio.run(...)``
  internally without clashing with the main loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done, to_filter
from prompt_toolkit.formatted_text import FormattedText, to_formatted_text
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    AnyContainer,
    ConditionalContainer,
    DynamicContainer,
    HSplit,
    ScrollOffsets,
    VSplit,
    Window,
    to_container,
)
from prompt_toolkit.layout.controls import BufferControl, DummyControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

from datus.cli.cli_styles import PASTE_COLLAPSE_THRESHOLD
from datus.cli.input_modes import MODE_CHROME, InputMode
from datus.cli.tui.clipboard import copy_to_clipboard
from datus.cli.tui.live_display_state import LiveDisplayState, compute_pinned_max_rows
from datus.cli.tui.output_buffer import BufferedOutputControl, TUIOutputBuffer, extract_selection_text
from datus.cli.tui.region_selection import SelectableRegion
from datus.cli.tui.scrollbar import ScrollbarController, build_scrollbar_window
from datus.cli.tui.search import SearchState, find_matches
from datus.cli.tui.selection import (
    MultiClickTracker,
    SelectionAutoscroll,
    SelectionPoint,
    TranscriptSelection,
    fragment_plain_text,
    word_bounds_at,
)
from datus.cli.tui.wizard_host import EmbeddedWizard
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Sentinel returned by ``dispatch_fn`` to request clean shutdown of the TUI.
EXIT_SENTINEL = "__datus_tui_exit__"


def tui_enabled() -> bool:
    """Decide whether the TUI path should be used for this invocation.

    Returns ``True`` only when both stdin and stdout are TTYs and the
    ``DATUS_TUI`` environment variable is not set to a falsy value
    (``0``/``false``/``no``/``off``, case-insensitive).
    """
    env = os.environ.get("DATUS_TUI", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _is_jediterm(env: Optional[dict] = None) -> bool:
    """Detect JetBrains' JediTerm host terminal.

    JediTerm advertises SGR mouse support but echoes the escape sequences
    back as raw input while still driving its native scrollback in
    parallel, producing the symptoms reported by the user: scroll wheel
    moves both the app viewport AND the terminal scrollback, the
    in-app scrollbar pixel-snaps to the wrong cell, and overshoot
    rubber-bands at the bottom. The cleanest fix is to keep mouse
    capture off there and let JediTerm's DECSET 1007 behaviour
    translate the wheel into ↑/↓ keys, which the app handles directly.
    """
    src = env if env is not None else os.environ
    return src.get("TERMINAL_EMULATOR", "").strip().lower() == "jetbrains-jediterm"


def _resolve_mouse_support(env: Optional[dict] = None) -> bool:
    """Resolve whether prompt_toolkit's ``mouse_support`` should be on.

    Order of precedence:

    1. ``DATUS_FORCE_MOUSE_CAPTURE`` truthy → force on (escape hatch for
       JediTerm users on a build that handles the SGR echo themselves).
    2. ``DATUS_FORCE_MOUSE_CAPTURE`` falsy (``0``/``false``/``no``/``off``)
       → force off.
    3. Otherwise: off under JediTerm, on everywhere else.
    """
    src = env if env is not None else os.environ
    override = src.get("DATUS_FORCE_MOUSE_CAPTURE", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return not _is_jediterm(src)


def _select_buffer_line(buffer: Buffer) -> None:
    """Select the whole line under the cursor via the buffer's native selection.

    Triple-click path for buffer-backed inputs: prompt_toolkit's
    ``BufferControl`` only implements double-click word selection natively,
    so the third click widens it to the current line here. Uses the same
    ``selection_state`` machinery, so the highlight paints identically.
    """
    document = buffer.document
    line_start = document.cursor_position - document.cursor_position_col
    line_length = len(document.current_line)
    buffer.cursor_position = line_start
    buffer.start_selection()
    buffer.cursor_position = line_start + line_length


def _buffer_selection_text(buffer: Buffer) -> str:
    """Plain text covered by a buffer's native selection.

    Mirrors :meth:`prompt_toolkit.document.Document.cut_selection`'s
    slicing (per-range ``text[from:to]`` joined with ``\\n``) without
    mutating the buffer or clearing ``selection_state`` — the highlight
    should stay visible after the copy, like the output pane's.
    """
    if buffer.selection_state is None:
        return ""
    document = buffer.document
    return "\n".join(document.text[start:end] for start, end in document.selection_ranges())


class DatusApp:
    """Wrapper around a persistent prompt_toolkit Application.

    The Application is built eagerly in ``__init__`` so callers can attach
    additional key bindings (via :meth:`key_bindings`) or introspect the
    input buffer before :meth:`run` is invoked.
    """

    def __init__(
        self,
        *,
        status_tokens_fn: Callable[[], List[Tuple[str, str]]],
        dispatch_fn: Callable[[str], Optional[str]],
        completer: Optional[Completer] = None,
        history: Optional[History] = None,
        lexer: Optional[Lexer] = None,
        style: Optional[Style] = None,
        placeholder_fn: Optional[Callable[[], str]] = None,
        param_hint_fn: Optional[Callable[[str], str]] = None,
        input_prompt_fn: Optional[Callable[[], str]] = None,
        input_mode_fn: Optional[Callable[[], str]] = None,
        live_display_state: Optional[LiveDisplayState] = None,
        todo_tokens_fn: Optional[Callable[[], List[Tuple[str, str]]]] = None,
        todo_has_items_fn: Optional[Callable[[], bool]] = None,
        todo_line_count_fn: Optional[Callable[[], int]] = None,
        output_tokens_fn: Optional[Callable[[], List[Tuple[str, str]]]] = None,
        output_line_count_fn: Optional[Callable[[], int]] = None,
        output_buffer: Optional["TUIOutputBuffer"] = None,
        pending_input_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._status_tokens_fn = status_tokens_fn
        self._dispatch_fn = dispatch_fn
        # Returns the active :class:`PendingInputQueue` (or ``None``) for
        # the running chat session. Consulted by the Enter handler when
        # ``_agent_running`` is set so the user can stage messages
        # mid-stream, and by the queue-preview ``ConditionalContainer``
        # pinned above the status bar.
        self._pending_input_provider = pending_input_provider
        self._placeholder_fn = placeholder_fn or (lambda: "")
        # Given the live input text, returns a dim ``<arg> [--opt]`` hint rendered
        # after the input while typing a ``!<tool>`` / ``!<plugin>`` command.
        self._param_hint_fn = param_hint_fn or (lambda text: "")
        self._input_prompt_fn = input_prompt_fn or (lambda: "> ")
        # Returns the REPL's active input mode ("chat" / "sql" / "bash").
        # Drives the mode-coloured prompt label, the separators bracketing
        # the input, and the mode hint line — all re-evaluated on every paint.
        self._input_mode_fn = input_mode_fn or (lambda: InputMode.CHAT.value)
        self._live_state = live_display_state
        self._todo_tokens_fn = todo_tokens_fn
        self._todo_has_items_fn = todo_has_items_fn
        self._todo_line_count_fn = todo_line_count_fn
        # Sidebar visibility model. ``_sidebar_force_hidden`` is the user-
        # toggled override (Ctrl+T). ``_last_sidebar_visible`` caches the
        # last value returned by :meth:`_sidebar_visible` so the filter can
        # detect transitions and notify ``_on_sidebar_visibility_change``
        # via the event loop (filter runs on the render path, so reflow
        # work must be deferred). The listener is wired by ``DatusCLI``
        # to rebuild the Rich Console at the new pane width and re-render
        # the scrollback so existing rows wrap to the new width.
        self._sidebar_force_hidden: bool = False
        self._last_sidebar_visible: Optional[bool] = None
        self._on_sidebar_visibility_change: Optional[Callable[[bool], None]] = None
        # Scroll-pane output (full_screen=True). Replaces patch_stdout —
        # all console.print output is captured into an in-memory buffer
        # which feeds ``output_tokens_fn``. ``output_line_count_fn`` is
        # consulted by the sticky-bottom auto-scroll logic.
        self._output_tokens_fn = output_tokens_fn or (lambda: [])
        self._output_line_count_fn = output_line_count_fn or (lambda: 0)
        self._output_buffer = output_buffer
        # Sticky-bottom scroll model: ``_output_at_bottom=True`` (the
        # default) means ``_get_output_scroll`` returns the max possible
        # offset every frame, so new output is always in view. Wheel-up
        # / PgUp snapshot the current top-of-viewport into
        # ``_output_scroll_offset`` and disengage. Wheel-down / PgDn
        # past the last row re-engages. Wheel events come through the
        # output Window's ``mouse_handler`` (FormattedTextControl has
        # no built-in scroll behaviour).
        self._output_at_bottom: bool = True
        self._output_scroll_offset: int = 0

        # Software-painted text selection inside the output pane. Mouse
        # capture (``mouse_support=True``) preempts terminal-native
        # shift+drag selection, so :class:`TranscriptSelection` tracks
        # an anchor + head pair updated from MOUSE_DOWN / MOUSE_MOVE /
        # MOUSE_UP and :class:`BufferedOutputControl` consults it to
        # paint a reverse-video highlight on the selected rows.
        self._selection = TranscriptSelection()
        # Direction-state for "drag past the viewport edge → keep
        # extending the selection while auto-scrolling". The actual
        # ticking loop lives in :meth:`_selection_autoscroll_loop`.
        self._selection_autoscroll = SelectionAutoscroll()
        # Every other text pane gets its own independent software
        # selection (status bar, todo sidebar, queue preview) so any
        # visible text can be drag-copied, not just the scrollback.
        # A drag is confined to the region it started in; beginning a
        # selection anywhere clears the others (see
        # :meth:`_clear_other_selections`). Buffer-backed inputs (the
        # composer, the search field) use prompt_toolkit's native
        # selection instead and only need copy-on-release, tracked in
        # ``_selection_buffers``.
        self._selection_regions: dict[str, SelectableRegion] = {}
        self._selection_buffers: dict[str, Buffer] = {}
        # Double-click → word, triple-click → line in the scrollback pane.
        # Regions carry their own tracker; buffer-backed inputs get one in
        # :meth:`_install_buffer_copy_handler`.
        self._output_click_tracker = MultiClickTracker()
        # Registry keys created by :meth:`_wire_wizard_selection` for the
        # currently mounted embedded wizard; popped on unmount.
        self._wizard_selection_names: List[str] = []

        # Ctrl+F find-in-scrollback. ``_search_state`` is the shared
        # data structure the renderer (via ``search_provider``) and the
        # search bar handlers both read/write; ``_search_active``
        # controls the ``ConditionalContainer`` that materialises the
        # bottom search row. The state machine lives in :meth:`_open_search`
        # / :meth:`_close_search` / :meth:`_on_search_text_changed` /
        # :meth:`_jump_to_match`.
        self._search_state = SearchState()
        self._search_active: bool = False
        self._search_buffer = Buffer(multiline=False, on_text_changed=self._on_search_text_changed)

        self._agent_running = threading.Event()
        self._dispatch_state_lock = threading.Lock()
        self._input_restore_after_dispatch: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="datus-tui-worker")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._exit_code: int = 0
        self._last_ctrl_c_time: float = 0.0
        # Single-line transient hint above the input area (Ctrl+C confirm,
        # "Copied to clipboard", etc.). Each show_hint call replaces the
        # previous text and resets the auto-clear timer.
        self._hint_text: str = ""
        self._hint_clear_handle: Optional[asyncio.TimerHandle] = None

        self._stored_paste: Optional[str] = None
        self._paste_collapsed: bool = False

        self._input_area = TextArea(
            height=self._get_input_height,
            multiline=True,
            wrap_lines=True,
            completer=completer,
            history=history,
            lexer=lexer,
            focus_on_click=True,
            complete_while_typing=True,
            auto_suggest=None,
            style="class:input-area",
            prompt=self._get_input_prompt,
            # Dim ``<arg> [--opt]`` hint rendered after a ``!<tool>`` / ``!<plugin>``
            # command. ``AfterInput`` only paints on the last line, so it never
            # collides with multi-line chat input; it collapses to nothing when
            # the hint is empty. Gated so the transformation is skipped entirely
            # outside ``!`` input.
            input_processors=[
                ConditionalProcessor(
                    AfterInput(self._bang_hint_text, style="class:bang-arg-hint"),
                    filter=Condition(lambda: self._input_area.text.startswith("!")),
                )
            ],
        )
        self._input_area.window.dont_extend_height = to_filter(True)
        self._input_area.buffer.on_text_changed += self._on_buffer_text_changed
        # The composer already gets native drag-selection from prompt_toolkit's
        # BufferControl; this wrapper adds copy-on-release + cross-region rules.
        self._install_buffer_copy_handler("input", self._input_area.control, self._input_area.buffer)

        self._status_region = self._make_selection_region("status", self._safe_status_tokens)
        self._status_window = Window(
            content=FormattedTextControl(
                text=self._status_region.render_tokens,
                focusable=False,
                show_cursor=False,
            ),
            height=1,
            style="class:status-bar",
            wrap_lines=False,
        )
        # While a selection drag is active, the status bar becomes the
        # bottom-edge autoscroll trigger. See :meth:`_status_mouse_handler`.
        self._status_window.content.mouse_handler = self._status_mouse_handler

        # In ``full_screen=False`` mode the Application renders only the rows
        # it needs (status bar + input) at the bottom of the terminal. All
        # program output above that region is emitted via ``patch_stdout``,
        # which inserts new lines above the Application's rendered area, so
        # no explicit output window is required here.
        #
        # Inlines the layout of prompt_toolkit's ``CompletionsMenu`` but drops
        # the ``ScrollbarMargin`` so the slash-command popup never renders a
        # right-hand scrollbar column. Styling is controlled via
        # ``completion-menu.*`` keys in ``DatusCLI._build_app_style``.
        self._completions_menu = ConditionalContainer(
            content=Window(
                content=CompletionsMenuControl(),
                width=Dimension(min=8),
                height=Dimension(min=1, max=10),
                scroll_offsets=ScrollOffsets(top=1, bottom=1),
                right_margins=[],
                dont_extend_width=True,
                style="class:completion-menu",
                z_index=10**8,
            ),
            filter=has_completions & ~is_done,
        )

        hint_body = Window(
            content=FormattedTextControl(lambda: [("class:hint", self._hint_text)]),
            height=1,
            wrap_lines=False,
        )
        # Bottom-most row of the layout: off-window releases clamp here.
        hint_body.content.mouse_handler = self._decoration_mouse_handler
        self._hint_window = ConditionalContainer(
            content=hint_body,
            filter=Condition(lambda: bool(self._hint_text)),
        )

        # Persistent indicator pinned directly under the input while a
        # non-chat mode is active. A single window whose text/style follow
        # the active mode; collapses to zero rows in chat mode.
        def _mode_hint_tokens() -> List[Tuple[str, str]]:
            chrome = MODE_CHROME[self._current_input_mode()]
            if not chrome.hint:
                return []
            return [(chrome.hint_style, chrome.hint)]

        self._mode_hint_window = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(_mode_hint_tokens),
                height=1,
                wrap_lines=False,
            ),
            filter=Condition(lambda: self._current_input_mode() is not InputMode.CHAT),
        )

        self._search_bar = self._build_search_bar()

        # Scrollable output pane (left, weight=4). Replaces the old
        # patch_stdout-driven scrollback. Every byte Rich emits flows into
        # an in-memory ``TUIOutputBuffer``; ``output_tokens_fn`` (wired by
        # ``DatusCLI``) returns the buffer's token stream on every paint.
        # The streaming markdown tail and subagent rolling window — still
        # written to ``LiveDisplayState`` — are concatenated by the buffer
        # so they appear at the bottom of the same pane, exactly where the
        # cursor would have rendered them in ``full_screen=False`` mode.
        #
        # Scrolling model: explicit ``get_vertical_scroll`` callback +
        # ``wrap_lines=False``. Rich is configured to wrap output at the
        # pane width (see ``DatusCLI._init_tui_app`` setting
        # ``Console(width=...)``), so each logical line in the buffer is
        # already one visual row — ``vertical_scroll`` (counted in
        # visual rows) maps 1:1 to our buffer's line count. This avoids
        # the cursor-driven scroll's "wait for cursor to leave the
        # viewport edge before shifting" hesitation that made trackpad
        # scrolling feel like it stalled before suddenly jumping.
        # Prefer the lazy-row :class:`BufferedOutputControl` when a buffer is
        # wired. It feeds prompt_toolkit a ``UIContent.get_line`` callable so
        # only the rows that intersect the viewport are materialised — the
        # ``FormattedTextControl`` fallback hashes the entire fragment stream
        # on every paint, which dominates type-latency once the scrollback
        # contains thousands of lines (the verbose-mode case). The fallback
        # path is preserved so tests / non-TUI callers that only pass a
        # ``tokens_fn`` still work.
        if self._output_buffer is not None:
            output_control = BufferedOutputControl(
                self._output_buffer,
                focusable=True,
                show_cursor=False,
                get_cursor_position=self._output_cursor_position,
                selection_provider=lambda: self._selection,
                # Surface the ``SearchState`` only while the search bar is
                # open *or* still has matches in flight, so the overlay
                # disappears cleanly when ``_close_search`` runs.
                search_provider=lambda: (
                    self._search_state if (self._search_active or self._search_state.matches) else None
                ),
            )
        else:
            output_control = FormattedTextControl(
                text=self._output_tokens_fn,
                focusable=True,
                show_cursor=False,
                get_cursor_position=self._output_cursor_position,
            )
        self._output_window = Window(
            content=output_control,
            width=Dimension(weight=4),
            wrap_lines=False,
            scroll_offsets=ScrollOffsets(),
            get_vertical_scroll=self._get_output_scroll,
            always_hide_cursor=to_filter(True),
            style="class:output-pane",
        )
        # Neither control type accepts ``mouse_handler`` via constructor — it
        # is an instance method that the Window consults. Override it in place
        # so scroll-wheel events drive our sticky-bottom-aware scroll motion
        # instead of the default Window scroll (which would be undone by
        # ``get_cursor_position`` re-anchoring to the last line on the next
        # paint).
        self._output_window.content.mouse_handler = self._output_mouse_handler

        # Custom scrollbar — built as its own Window because prompt_toolkit's
        # built-in :class:`ScrollbarMargin` never receives mouse events
        # (Window's mouse-handler range excludes the margin columns).
        # The widget converts click / drag rows into scroll offsets via
        # :class:`ScrollbarController` and shares the sticky-bottom model
        # used by the wheel handler.
        self._scrollbar_controller = ScrollbarController(
            # Pass attribute-dispatched lambdas (rather than direct bound
            # methods) so tests that monkeypatch ``_output_viewport_rows``
            # are observed by the controller too; otherwise the bound
            # method captured at construction would freeze the test view
            # of the layout at fixture-setup time.
            viewport_rows_fn=lambda: self._output_viewport_rows(),
            total_rows_fn=lambda: int(self._output_line_count_fn()),
            get_scroll_fn=lambda: self._get_output_scroll(self._output_window),
            set_scroll_fn=lambda offset: self._set_output_scroll_offset(offset),
            invalidate_fn=lambda: self.invalidate(),
        )
        self._scrollbar_window = build_scrollbar_window(
            self._scrollbar_controller,
            # Always render the gutter — even when content fits the
            # viewport — so the layout doesn't jitter as scrollback
            # grows past the first screenful. The track + thumb logic in
            # ``ScrollbarController.thumb_geometry`` handles the
            # no-overflow case by filling the column.
            visible_filter=lambda: True,
        )
        # The gutter's per-fragment handlers only know the scrollbar. Wrap
        # the control so an in-flight text-selection drag that ends over
        # the gutter (terminals clamp off-window releases onto edge
        # columns like this one) still finalises + copies. The scrollbar's
        # own drag keeps absolute priority.
        scrollbar_control = self._scrollbar_window.content.content
        original_scrollbar_handler = scrollbar_control.mouse_handler

        def _scrollbar_mouse_handler(event: MouseEvent):  # noqa: ANN202
            if not self._scrollbar_controller.dragging:
                intercepted = self._inflight_drag_intercept(event)
                if intercepted is not NotImplemented:
                    return intercepted
            return original_scrollbar_handler(event)

        scrollbar_control.mouse_handler = _scrollbar_mouse_handler

        # Right-side todo-list sidebar for the pinned output row. Wired via
        # callbacks so non-TUI callers (and tests that pass no todo hooks)
        # keep an unchanged single-column layout. Hidden when items are empty
        # or the terminal is too narrow to fit a useful 20% column.
        self._todo_sidebar = self._build_todo_sidebar()

        # Top output row: scrollable output (weight=4) + scrollbar gutter
        # (1 col) + todo sidebar (weight=1). The scrollbar is wedged
        # between content and sidebar so it remains visible regardless of
        # whether the sidebar is filtered out.
        top_row = VSplit([self._output_window, self._scrollbar_window, self._todo_sidebar])

        # Bottom section is dynamic so sub-wizards can replace it. In
        # normal operation we render status bar + input + hint; while
        # an embedded wizard is active ``_active_wizard.container``
        # takes the slot instead, hiding status + input and pushing the
        # output row upward as much as the wizard needs. See
        # ``datus/cli/tui/wizard_host.py`` for the embedding contract.
        self._queue_preview = self._build_queue_preview()
        self._normal_bottom_section = HSplit(
            [
                self._queue_preview,
                self._make_separator(),
                self._status_window,
                # The two rules bracketing the input take the active mode's
                # colour (red in SQL mode, yellow in bash mode).
                self._make_separator(mode_aware=True),
                self._input_area,
                self._completions_menu,
                self._search_bar,
                self._make_separator(mode_aware=True),
                self._mode_hint_window,
                self._hint_window,
            ]
        )
        self._active_wizard: Optional[EmbeddedWizard] = None
        self._stashed_focus: Optional[Window] = None

        root = HSplit(
            [
                top_row,
                DynamicContainer(self._bottom_section),
            ]
        )

        self._kb = self._build_default_key_bindings()
        # Mutable wizard-kb layer. Wizards push their bindings into this
        # KeyBindings instance when mounted via :meth:`mount_wizard` and
        # they are removed in :meth:`unmount_wizard`. We merge this into
        # the app's key_bindings at construction time so it always wins
        # over the focused widget's local bindings (e.g. a TextArea's
        # built-in typing bindings don't shadow the wizard's Tab /
        # Enter / Esc navigation).
        self._wizard_kb_layer = KeyBindings()

        # ``full_screen=True`` so the output pane and sidebar share the
        # full terminal vertical real estate. ``mouse_support=True`` lights
        # up the scroll wheel for the output pane (and click-to-focus); per
        # plan the user accepted losing terminal-native Shift+drag select
        # inside the rendered area in exchange.
        #
        # JetBrains' JediTerm advertises SGR mouse support but leaks the
        # escape sequences back as raw input AND keeps driving its native
        # scrollback in parallel — the user sees ghost scrolling, jumpy
        # scrollbars, and rubber-band bounce at the bottom. We disable
        # mouse capture there and rely on JediTerm's DECSET 1007 behaviour
        # (wheel-in-alt-screen → ↑/↓ keys), which the Up/Down handlers
        # below translate into in-app scroll without any terminal-side
        # scrollback being touched. Set ``DATUS_FORCE_MOUSE_CAPTURE=1`` to
        # opt back in.
        self._mouse_support_enabled = _resolve_mouse_support()
        self._app: Application = Application(
            layout=Layout(root, focused_element=self._input_area),
            key_bindings=merge_key_bindings([self._kb, self._wizard_kb_layer]),
            style=style or Style([]),
            full_screen=True,
            mouse_support=self._mouse_support_enabled,
            erase_when_done=False,
        )
        # prompt_toolkit defaults to waiting 500 ms before deciding that a
        # lone ESC byte is not the prefix of an arrow/function-key sequence.
        # Keep a tiny sequencing window, but make standalone ESC cancellation
        # perceptually as immediate as Ctrl+C.
        self._app.ttimeoutlen = 0.01

        # Live state now drives the streaming tail inside the scrollable
        # output pane (via ``TUIOutputBuffer.tokens()``) — its invalidate
        # callback still wakes the main loop the same way.
        if self._live_state is not None:
            self._live_state.set_invalidate(self.invalidate)
            self._live_state.set_max_rows_provider(self._pinned_max_rows)

    def _current_input_mode(self) -> InputMode:
        """The REPL's active input mode, defaulting to chat on any error."""
        try:
            return InputMode(self._input_mode_fn())
        except Exception:  # pragma: no cover - defensive
            return InputMode.CHAT

    def _make_separator(self, mode_aware: bool = False) -> Window:
        """Full-width horizontal rule rendered with a box-drawing character.

        When ``mode_aware`` is set the rule takes the active input mode's
        colour (red in SQL mode, yellow in bash mode) so the input area is
        visibly bracketed; otherwise it uses the default separator colour.

        Wired to :meth:`_decoration_mouse_handler` because off-window releases
        get clamped onto edge rows like these \u2014 the handler closes out any
        in-flight selection drag so it still copies.
        """
        if mode_aware:

            def _sep_style() -> str:
                return MODE_CHROME[self._current_input_mode()].separator_style

            window = Window(height=1, char="\u2500", style=_sep_style)
        else:
            window = Window(height=1, char="\u2500", style="class:separator")
        window.content.mouse_handler = self._decoration_mouse_handler
        return window

    # Minimum terminal width before the sidebar becomes too narrow to
    # be readable (20% of 60 columns ≈ 12, with a 1-col rule on its
    # left). Below this threshold the entire right column is hidden.
    _SIDEBAR_MIN_TERMINAL_COLS = 60

    # Max number of pending lines rendered in the queue preview before
    # the trailing items are collapsed into an "(+N more)" line. The
    # preview's job is just to confirm "your text is queued" — it is
    # not a transcript and should never push the layout around.
    _QUEUE_PREVIEW_MAX_LINES = 5

    def _build_queue_preview(self) -> ConditionalContainer:
        """Build the pinned queue-preview box rendered above the status bar.

        Visibility is driven by :class:`Condition` so the entire container
        collapses to zero rows when the queue is empty — no layout churn,
        no manual show/hide. When :attr:`_pending_input_provider` is not
        wired (non-chat dispatch), the filter never reports visible.
        """

        queue_region = self._make_selection_region("queue", self._queue_preview_tokens)
        window = Window(
            content=FormattedTextControl(queue_region.render_tokens),
            height=Dimension(min=1, max=self._QUEUE_PREVIEW_MAX_LINES + 2),
            style="class:queue-preview.box",
            always_hide_cursor=True,
        )
        window.content.mouse_handler = lambda event: self._region_mouse_dispatch(queue_region, event)
        return ConditionalContainer(content=window, filter=Condition(self._queue_preview_visible))

    def _enqueue_pending_input(self, buffer, app) -> bool:
        """Push the input-buffer text into the running agent's pending queue.

        Returns ``True`` when the text was successfully queued, ``False``
        otherwise (no queue wired, blank text). The OpenAI Agents SDK
        caches ``_model_input_items`` for the lifetime of a streamed run,
        so dropping text into the session mid-stream is invisible to the
        model. The queue is drained by the model layer's
        ``call_model_input_filter`` before the next LLM turn.
        """
        queue = self._pending_input_provider() if self._pending_input_provider else None
        if queue is None:
            return False
        text = buffer.text
        if self._stored_paste:
            placeholder = self._paste_placeholder(self._stored_paste.count("\n") + 1)
            if placeholder in text:
                text = text.replace(placeholder, self._stored_paste, 1)
            self._stored_paste = None
            self._paste_collapsed = False
        stripped = text.strip()
        if not stripped:
            return False
        queue.push(stripped)
        buffer.reset()
        if app is not None:
            app.invalidate()
        return True

    def _queue_preview_visible(self) -> bool:
        """Visibility predicate for the pinned queue-preview container."""
        if self._pending_input_provider is None:
            return False
        queue = self._pending_input_provider()
        if queue is None:
            return False
        try:
            return len(queue) > 0
        except TypeError:
            return False

    def _queue_preview_tokens(self) -> List[Tuple[str, str]]:
        """Render the formatted-text rows for the queue-preview container."""
        if self._pending_input_provider is None:
            return []
        queue = self._pending_input_provider()
        if queue is None:
            return []
        try:
            items = queue.snapshot()
        except AttributeError:
            return []
        if not items:
            return []
        lines: List[Tuple[str, str]] = [
            ("class:queue-preview.header", f"⏎ queued for agent ({len(items)}):\n"),
        ]
        visible_items = items[: self._QUEUE_PREVIEW_MAX_LINES]
        for idx, text in enumerate(visible_items, 1):
            preview = text if len(text) <= 80 else text[:77] + "..."
            lines.append(("class:queue-preview.item", f"  {idx}. {preview}\n"))
        overflow = len(items) - len(visible_items)
        if overflow > 0:
            lines.append(("class:queue-preview.item", f"  (+{overflow} more)\n"))
        return lines

    def _build_todo_sidebar(self) -> ConditionalContainer:
        """Construct the TodoList sidebar column pinned above the status bar.

        When ``todo_tokens_fn`` is None (non-TUI callers, tests) the
        sidebar's visibility filter returns False permanently, which
        makes the VSplit collapse to a single column at render time —
        no behavioural change for callers that don't opt in.

        The outer width is **hard-pinned** to ``terminal_cols // 5``
        (min 14) via a callable ``Dimension``. Without this, in idle
        mode (live region collapsed to 0 rows / 0 preferred width)
        prompt_toolkit's VSplit splitter ignores the weights and lets
        the sidebar's content preferred-width drive the column size —
        long CJK tasks push the sidebar far past 20%. Pinning min=max
        keeps the split deterministic. Height is also capped to the pinned
        output row budget so a long todo list cannot push the status bar/input
        stack away from the bottom of the terminal.
        """
        tokens_fn = self._todo_tokens_fn or (lambda: [])
        has_items_fn = self._todo_has_items_fn or (lambda: False)

        def _sidebar_visible() -> bool:
            if self._sidebar_force_hidden:
                visible = False
            elif self._terminal_columns() < self._SIDEBAR_MIN_TERMINAL_COLS:
                visible = False
            else:
                try:
                    visible = bool(has_items_fn())
                except Exception:  # pragma: no cover - defensive
                    visible = False
            self._note_sidebar_visibility(visible)
            return visible

        def _sidebar_width() -> Dimension:
            target = self._sidebar_target_width()
            return Dimension(min=target, max=target, preferred=target)

        # In full_screen mode the top output row absorbs all remaining
        # vertical space, so the sidebar Window must NOT cap its height —
        # otherwise a 4-task sidebar would shrink the row to 4 rows and
        # let the output pane below it collapse. ``dont_extend_height``
        # is also dropped: the sidebar can fill the column with empty
        # rows beneath the last task.
        #
        # ``wrap_lines=True`` lets a single long task wrap across as many
        # visual rows as the column needs — the provider intentionally
        # does NOT truncate content so wrapping is the only way the user
        # sees the full task text on a narrow sidebar.
        todo_region = self._make_selection_region("todo", tokens_fn)
        sidebar_body = Window(
            content=FormattedTextControl(
                text=todo_region.render_tokens,
                focusable=False,
                show_cursor=False,
            ),
            wrap_lines=True,
            dont_extend_width=False,
            style="class:todo-sidebar",
        )
        sidebar_body.content.mouse_handler = lambda event: self._region_mouse_dispatch(todo_region, event)
        sidebar_rule = Window(width=Dimension.exact(1), char="│", style="class:separator")
        sidebar_rule.content.mouse_handler = self._decoration_mouse_handler
        return ConditionalContainer(
            content=VSplit(
                [
                    sidebar_rule,
                    sidebar_body,
                ],
                width=_sidebar_width,
            ),
            filter=Condition(_sidebar_visible),
        )

    def set_sidebar_visibility_listener(self, callback: Optional[Callable[[bool], None]]) -> None:
        """Register a callback invoked whenever the sidebar's visibility flips.

        The listener is scheduled on the Application's event loop via
        ``call_soon_threadsafe`` because the filter that detects the
        transition runs on prompt_toolkit's render path — replacing the
        Rich Console / clearing the buffer there would either re-enter
        rendering or race with paint. The current visibility value is
        passed to the callback.
        """
        self._on_sidebar_visibility_change = callback

    def toggle_sidebar_hidden(self) -> bool:
        """Flip the manual hide override. Returns the new ``force_hidden`` value.

        The actual reflow is driven by the visibility listener: the next
        ``_sidebar_visible`` evaluation will observe the flipped flag, see
        a transition against ``_last_sidebar_visible``, and notify the
        listener. Callers should invalidate the app after toggling so the
        filter re-runs on the next render tick.
        """
        self._sidebar_force_hidden = not self._sidebar_force_hidden
        return self._sidebar_force_hidden

    def _note_sidebar_visibility(self, visible: bool) -> None:
        """Detect transitions and schedule the listener on the event loop.

        First call after construction records the value without firing —
        this avoids a spurious reflow at startup before any history has
        accumulated. Subsequent transitions schedule the listener via
        ``call_soon_threadsafe`` when available; if no loop is attached
        yet (e.g. unit tests inspecting the filter directly) the listener
        is skipped — the next real transition while running will fire it.
        """
        if self._last_sidebar_visible is None:
            self._last_sidebar_visible = visible
            return
        if visible == self._last_sidebar_visible:
            return
        self._last_sidebar_visible = visible
        callback = self._on_sidebar_visibility_change
        if callback is None:
            return
        loop = self._loop
        if loop is None:
            app = getattr(self, "_app", None)
            loop = getattr(app, "loop", None) if app is not None else None
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(callback, visible)
        except RuntimeError:  # pragma: no cover - defensive (loop closed)
            pass

    def _sidebar_target_width(self) -> int:
        """Total cells reserved for the right-side todo column.

        Returns ``max(14, terminal_cols // 5)`` so a narrow terminal
        still gets a 14-cell minimum, and wider terminals get exactly
        20% of the column count. The corresponding line-content cell
        budget (subtracting the 1-col ``│`` separator and the
        ``" ✓ "`` prefix) is computed in :meth:`DatusCLI._sidebar_content_width`
        and passed to ``TodoSidebarProvider`` so truncation tracks the
        actual rendered width.
        """
        return max(14, self._terminal_columns() // 5)

    def _live_state_is_active(self) -> bool:
        return self._live_state is not None and self._live_state.is_active()

    def _terminal_rows(self) -> int:
        """Best-effort current terminal row count.

        Prefers the live ``Application.output`` (accurate and resize-aware),
        falls back to a sensible default when the app hasn't attached to
        a terminal yet (e.g. early construction, unit tests).
        """
        app = getattr(self, "_app", None)
        if app is not None:
            try:
                size = app.output.get_size()
                if size and size.rows > 0:
                    return int(size.rows)
            except Exception:
                pass
        try:
            import shutil

            size = shutil.get_terminal_size(fallback=(80, 24))
            return int(size.lines)
        except Exception:
            return 24

    def _pinned_max_rows(self) -> int:
        """Current row ceiling for the pinned region (terminal-aware)."""
        return compute_pinned_max_rows(self._terminal_rows())

    def _output_viewport_rows(self) -> int:
        """Visible row count of the output pane (live render_info if available).

        Used by PgUp / PgDn / wheel handlers to size each scroll step
        against the actual rendered viewport. Falls back to a coarse
        estimate before the first paint so the very first key-press
        still produces a reasonable jump.
        """
        try:
            render_info = self._output_window.render_info
        except AttributeError:
            render_info = None
        if render_info is not None and getattr(render_info, "window_height", 0) > 0:
            return int(render_info.window_height)
        return max(1, self._terminal_rows() - 4)  # status + 2 separators + input

    def _output_page_size(self) -> int:
        """Rows to scroll on a single PageUp / PageDown keystroke."""
        return max(1, self._output_viewport_rows() - 1)

    def _output_max_scroll(self) -> int:
        """Max scroll offset given current content + viewport.

        Used by both ``_get_output_scroll`` (sticky-bottom anchor) and
        the wheel/PgDn handlers (clamp). When content fits entirely in
        the viewport this is ``0``; otherwise it's the offset that
        puts the last line at the bottom row.
        """
        total = int(self._output_line_count_fn())
        viewport = self._output_viewport_rows()
        return max(0, total - viewport)

    def _get_output_scroll(self, window) -> int:  # noqa: ANN001
        """Window scroll callback — returns the top-row offset.

        Sticky-bottom: always pin to ``_output_max_scroll``. User
        scrolled mode: return the snapshotted ``_output_scroll_offset``
        clamped to current max so terminal resize / output growth
        can't push the viewport into a negative gap.
        """
        if self._output_at_bottom:
            return self._output_max_scroll()
        return max(0, min(self._output_scroll_offset, self._output_max_scroll()))

    def _output_cursor_position(self):
        """Match the cursor to our chosen scroll offset.

        After ``get_vertical_scroll`` sets the viewport, prompt_toolkit's
        ``do_scroll`` (in ``Window._scroll_without_linewrapping``) ALSO
        pulls the scroll back to keep ``cursor_position`` visible. If
        we leave the cursor at ``Point(0, 0)`` it ends up clamping the
        viewport to row 0 every frame — sticky-bottom snaps to the top
        the instant it's evaluated.
        Returning a cursor synced to ``vertical_scroll`` (top of
        viewport when scrolled, bottom of content when sticky) means
        ``do_scroll`` sees the cursor as already-visible and leaves
        the offset alone.
        """
        from prompt_toolkit.data_structures import Point

        total = int(self._output_line_count_fn())
        if total <= 0:
            return None
        scroll = self._get_output_scroll(self._output_window)
        if self._output_at_bottom:
            return Point(x=0, y=total - 1)  # last row → bottom of viewport
        # Top of viewport — keeps do_scroll from pulling the offset down.
        return Point(x=0, y=max(0, min(total - 1, scroll)))

    def _scroll_output_up(self, rows: int) -> None:
        """Shift viewport up by ``rows``; disengage sticky-bottom."""
        rows = max(1, rows)
        if self._output_at_bottom:
            # Snapshot the current bottom-anchored position so the
            # very first wheel-up tick moves visible content
            # immediately (otherwise we'd start from a stale 0 offset
            # and the user would feel a pause before the jump).
            self._output_scroll_offset = self._output_max_scroll()
            self._output_at_bottom = False
        self._output_scroll_offset = max(0, self._output_scroll_offset - rows)

    def _scroll_output_down(self, rows: int) -> None:
        """Shift viewport down by ``rows``; re-engage sticky-bottom on overshoot."""
        rows = max(1, rows)
        if self._output_at_bottom:
            return  # already at bottom; nothing to do
        max_off = self._output_max_scroll()
        new_off = self._output_scroll_offset + rows
        if new_off >= max_off:
            self._output_at_bottom = True
            self._output_scroll_offset = max_off
        else:
            self._output_scroll_offset = new_off

    def _set_output_scroll_offset(self, offset: int) -> None:
        """Absolute-position setter used by the scrollbar drag handler.

        Mirrors the sticky-bottom rules of :meth:`_scroll_output_up` /
        :meth:`_scroll_output_down`: clamping to ``[0, max_scroll]`` and
        re-engaging sticky-bottom only when the new offset is at the
        very bottom — a single mid-track click should not snap back to
        following live output.
        """
        max_off = self._output_max_scroll()
        offset = max(0, min(int(offset), max_off))
        self._output_scroll_offset = offset
        self._output_at_bottom = offset >= max_off and max_off > 0
        # Special case: when content fits the viewport (max_off == 0) we
        # stay in sticky-bottom mode by default so new output continues
        # to land at the bottom row.
        if max_off == 0:
            self._output_at_bottom = True

    # Fixed step: every wheel event scrolls exactly one row. macOS
    # trackpads emit dense streams of small events so they accumulate
    # into smooth motion; a discrete mouse wheel click moves one row,
    # which matches prompt_toolkit's built-in Window scroll cadence.
    _OUTPUT_WHEEL_STEP = 1

    def _output_mouse_handler(self, event: MouseEvent):  # noqa: ANN001
        """Mouse dispatcher for the scrollback pane.

        Handles four concerns in one entry point so we can share
        precedence + state-flag housekeeping across them:

        * **Scroll wheel** — sticky-bottom aware up/down ticks.
        * **MOUSE_DOWN + LEFT** — begin a fresh selection. Anchor lives
          at the *snapshot row* (``vertical_scroll + position.y``) so a
          subsequent scroll does not warp the selection.
        * **MOUSE_MOVE + LEFT** — extend the head while dragging. When
          the pointer is at or past the viewport edge, arm
          :class:`SelectionAutoscroll` and let the background task
          continue advancing the offset on its own.
        * **MOUSE_UP** — finalise the selection. If non-empty, plain
          text is copied to the system clipboard (pyperclip → OSC 52).
        """
        et = event.event_type
        if et == MouseEventType.SCROLL_UP:
            self._scroll_output_up(self._OUTPUT_WHEEL_STEP)
            self._app.invalidate()
            return None
        if et == MouseEventType.SCROLL_DOWN:
            self._scroll_output_down(self._OUTPUT_WHEEL_STEP)
            self._app.invalidate()
            return None

        # Forward mouse events to the scrollbar while the user is mid-
        # drag, even when the pointer wanders off the 1-col gutter. The
        # scrollbar window only registers handlers within its own 1-col
        # screen range, so without this forwarding a horizontal pixel of
        # jitter into the output pane breaks the drag — the scroll
        # offset freezes and the thumb only catches up when the cursor
        # snaps back onto the gutter. Translating ``event.position.y``
        # (a UIContent line index) into a scrollbar-relative row uses
        # the fact that the scrollbar and the output Window share the
        # same parent VSplit row, so their viewport heights are equal.
        if self._scrollbar_controller.dragging:
            if et in (MouseEventType.MOUSE_MOVE, MouseEventType.MOUSE_UP):
                self._forward_to_scrollbar(event)
                return None
            if et == MouseEventType.MOUSE_DOWN:
                # A separate MOUSE_DOWN landed on the output pane while
                # we thought scrollbar was still being dragged — most
                # likely the OS dropped a release event somewhere. Cancel
                # the implicit drag so the new click is treated normally.
                self._scrollbar_controller._dragging = False  # noqa: SLF001
            return NotImplemented

        # In-flight drag bookkeeping — both the pane's own drag and drags
        # owned by other regions. Any event proving the button was
        # released (including releases that happened outside the terminal
        # window) finalises + copies; foreign drag-moves are confined.
        intercepted = self._inflight_drag_intercept(event, own_surface="output")
        if intercepted is not NotImplemented:
            return intercepted

        if et == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
            point = self._selection_point_from_event(event)
            if point is not None:
                clicks = self._output_click_tracker.register(point.line, point.column)
                self._clear_other_selections(keep="output")
                self._selection_autoscroll.disarm()
                # Disengage sticky-bottom so output growth doesn't yank
                # the rows out from under the dragging pointer.
                if self._output_at_bottom:
                    self._output_scroll_offset = self._output_max_scroll()
                    self._output_at_bottom = False
                if clicks >= 2:
                    self._select_output_span(point, whole_line=clicks >= 3)
                else:
                    self._selection.begin(point)
                self._app.invalidate()
            return None

        if et == MouseEventType.MOUSE_MOVE and event.button == MouseButton.LEFT:
            if not self._selection.dragging:
                return NotImplemented
            point = self._selection_point_from_event(event)
            if point is not None:
                self._selection.update_head(point)
            # Edge-driven autoscroll: only the **top** edge is detectable
            # via ``position.y`` because the row equals ``vertical_scroll``
            # only when the mouse is on the topmost visible line. Bottom
            # edge cannot be inferred from y alone (prompt_toolkit clamps
            # past-bottom events to the last rendered row, which is
            # indistinguishable from a legitimate click on that row), so
            # downward autoscroll is delegated to the status bar's mouse
            # handler — see :meth:`_status_mouse_handler`.
            self._maybe_arm_top_edge_autoscroll(event)
            self._app.invalidate()
            return None

        if et == MouseEventType.MOUSE_UP:
            # An in-flight drag never reaches here — the intercept above
            # finalises it. A bare release with nothing in flight is not
            # ours to handle.
            return NotImplemented

        return NotImplemented

    def _select_output_span(self, point: SelectionPoint, *, whole_line: bool) -> None:
        """Select (and copy) the word or whole line at ``point`` in the scrollback.

        Multi-click path: the selection is created already-finished so a
        MOUSE_MOVE with the button still held does not warp it;
        :meth:`_finalize_output_selection` handles copy + repaint exactly
        like a drag release would.
        """
        if self._output_buffer is None:
            return
        snap = self._output_buffer.snapshot()
        fragments = snap.get_line(point.line) if 0 <= point.line < snap.total else []
        text = fragment_plain_text(fragments)
        if whole_line:
            bounds = (0, len(text)) if text else None
        else:
            bounds = word_bounds_at(text, point.column)
        if bounds is None or bounds[0] >= bounds[1]:
            self._selection.clear()
            return
        self._selection.begin(SelectionPoint(line=point.line, column=bounds[0]))
        self._selection.update_head(SelectionPoint(line=point.line, column=bounds[1]))
        self._finalize_output_selection()

    def _selection_point_from_event(self, event: MouseEvent) -> Optional[SelectionPoint]:
        """Translate a control-relative MouseEvent into a buffer coordinate.

        prompt_toolkit hands the control a ``position`` already rebased
        into UIContent line/column space, so ``position.y`` *is* the
        snapshot row index — no scroll-offset addition needed.

        Returns ``None`` for events we cannot trust:

        * ``Point(0, 0)`` when the viewport's top row is *not* line 0
          (i.e. the user has scrolled). prompt_toolkit's ``Window._mouse_handler``
          emits exactly this sentinel when its ``rowcol_to_yx`` lookup
          fails — most commonly for a click on a blank row that paints
          no characters. Accepting it would snap the selection head all
          the way to the top of the buffer.
        * Negative coordinates (defensive).
        * Empty buffer.

        For a valid event past the last content row, anchor at the last
        line so a downward drag still highlights something visible.
        """
        line = int(event.position.y)
        column = int(event.position.x)
        if line < 0:
            return None
        total = int(self._output_line_count_fn())
        if total <= 0:
            return None
        # Reject the prompt_toolkit fallback sentinel when the viewport
        # is scrolled off the top of the buffer. (0, 0) is only a legit
        # mouse position when line 0 is currently visible.
        if line == 0 and column == 0:
            vertical_scroll = self._get_output_scroll(self._output_window)
            if vertical_scroll > 0:
                return None
        if line >= total:
            line = max(0, total - 1)
        return SelectionPoint(line=line, column=column)

    def _maybe_arm_top_edge_autoscroll(self, event: MouseEvent) -> None:
        """Arm scroll-up when a drag pulls the cursor onto the top visible row.

        Only the top edge is detected here — the bottom edge is handled
        by :meth:`_status_mouse_handler` because prompt_toolkit clamps
        past-bottom y coordinates to the last rendered row, making
        equality-based detection ambiguous on that side.

        Any motion landing inside the body also clears a previously
        armed autoscroll in *either* direction: returning from the
        status bar (which arms +1) needs an explicit disarm here
        because that handler never sees the pointer leaving.
        """
        top = self._get_output_scroll(self._output_window)
        row = int(event.position.y)
        if row <= top and top > 0:
            self._selection_autoscroll.arm(-1)
        elif self._selection_autoscroll.is_active():
            self._selection_autoscroll.disarm()

    def _status_mouse_handler(self, event: MouseEvent):  # noqa: ANN201
        """Mouse handler attached to the status bar window.

        Three responsibilities, checked in precedence order:

        * **Scrollbar drag forwarding** — when the user is dragging the
          scrollbar and their pointer crosses below the output pane onto
          the status bar, forward the event so scroll keeps following
          the cursor. Releasing on the status bar must also clear the
          scrollbar drag flag.
        * **Output-selection autoscroll** — during a scrollback selection
          drag, motion/press on the status bar arms scroll-down. Release
          here finalises the selection (and copies to clipboard).
        * **Status-bar-local selection** — with no drag in flight, the
          status bar is a selectable region of its own: drag across it to
          copy the status text.
        """
        et = event.event_type
        if self._scrollbar_controller.dragging:
            if et in (MouseEventType.MOUSE_MOVE, MouseEventType.MOUSE_UP):
                # The status bar is 1 row tall and sits immediately
                # below the output / scrollbar row. Forward as the
                # last scrollbar row so a drag past the bottom snaps
                # the scroll to its max.
                self._forward_to_scrollbar(event, row_override="bottom")
                return None
            return NotImplemented
        if self._selection.dragging:
            if et == MouseEventType.MOUSE_MOVE and event.button == MouseButton.LEFT:
                self._selection_autoscroll.arm(+1)
                self._app.invalidate()
                return None
            if self._drag_release_event(event):
                # MOUSE_UP here, or proof the release happened elsewhere
                # (fresh press / buttonless move) — close out the drag.
                self._finalize_output_selection()
                if et != MouseEventType.MOUSE_DOWN:
                    return None
                # A fresh press falls through so it can start a
                # status-bar selection of its own.
            else:
                return NotImplemented
        return self._region_mouse_dispatch(self._status_region, event)

    # ── Per-region selection plumbing ──────────────────────────────

    def _make_selection_region(self, name: str, tokens_fn: Callable[[], List[Tuple[str, str]]]) -> SelectableRegion:
        """Create + register an independent selection region for one pane."""
        region = SelectableRegion(
            name,
            tokens_fn,
            on_begin=lambda: self._clear_other_selections(keep=name),
            on_copy=self._copy_selection_text,
            invalidate=lambda: self._app.invalidate(),
        )
        self._selection_regions[name] = region
        return region

    def _copy_selection_text(self, text: str) -> None:
        """Clipboard + hint plumbing shared by every selection surface."""
        if text and copy_to_clipboard(text):
            self.show_hint("✓ Copied to clipboard")

    def _dragging_selection_region(self) -> Optional[SelectableRegion]:
        """The auxiliary region currently mid-drag, or ``None``."""
        for region in self._selection_regions.values():
            if region.selection.dragging:
                return region
        return None

    def _clear_other_selections(self, keep: Optional[str] = None) -> None:
        """Drop every selection except ``keep``'s.

        Selections are independent per region, and exactly one highlight
        should be visible at a time — beginning a drag anywhere clears
        the rest. ``keep`` is ``"output"`` for the scrollback pane, a
        region name for the auxiliary panes, a buffer name for the
        native-selection inputs, or ``None`` to clear everything
        (Escape).
        """
        if keep != "output":
            self._selection.clear()
            self._selection_autoscroll.disarm()
        for name, region in self._selection_regions.items():
            if name != keep:
                region.selection.clear()
        for name, buffer in self._selection_buffers.items():
            if name != keep and buffer.selection_state is not None:
                buffer.exit_selection()

    def _finalize_output_selection(self) -> None:
        """Close out the scrollback drag; copy the covered text if any."""
        self._selection.finish()
        self._selection_autoscroll.disarm()
        if not self._selection.is_empty() and self._output_buffer is not None:
            self._copy_selection_text(extract_selection_text(self._output_buffer, self._selection))
        self._app.invalidate()

    def _region_mouse_dispatch(self, region: SelectableRegion, event: MouseEvent):  # noqa: ANN201
        """Route a pane's mouse event through the cross-region drag rules.

        The intercept below closes out any in-flight drag (this pane's or
        a foreign one) and confines foreign drag-moves; everything else
        falls through to the pane's own selection state machine.
        """
        intercepted = self._inflight_drag_intercept(event, own_surface=region)
        if intercepted is not NotImplemented:
            return intercepted
        return region.handle_mouse(event)

    def _inflight_drag_intercept(self, event: MouseEvent, *, own_surface: object = None):  # noqa: ANN201
        """Shared pre-dispatch for every mouse surface while a drag is in flight.

        prompt_toolkit has no mouse capture: each event goes to whatever
        window the pointer is over, and a release outside the terminal is
        either clamped onto an edge cell by the emulator or never
        delivered at all. Every wired surface runs this first so an
        in-flight drag can always be closed out (and its text copied):

        * ``MOUSE_UP`` — normal release, wherever it lands. Consumed.
        * ``MOUSE_MOVE`` without LEFT held — hover after an off-terminal
          release (terminals in any-event tracking mode report these as
          soon as the pointer re-enters). Finalise; consumed.
        * ``MOUSE_DOWN`` — a fresh press proves the previous release was
          never delivered. Finalise, then return ``NotImplemented`` so
          the caller can start the interaction the press represents.
        * ``MOUSE_MOVE`` with LEFT over a **foreign** surface — swallowed:
          selections stay confined to the surface they started on. The
          owner (``own_surface`` is ``"output"`` for the scrollback pane,
          the :class:`SelectableRegion` for region dispatch) gets
          ``NotImplemented`` so it keeps extending its own selection.

        Returns ``None`` when the event was consumed, ``NotImplemented``
        when the caller should continue with its own handling.
        """
        output_dragging = self._selection.dragging
        region = self._dragging_selection_region()
        if not output_dragging and region is None:
            return NotImplemented
        if self._drag_release_event(event):
            if output_dragging:
                self._finalize_output_selection()
            elif region is not None:
                region.finish_drag()
            return NotImplemented if event.event_type == MouseEventType.MOUSE_DOWN else None
        owns_drag = (output_dragging and own_surface == "output") or (region is not None and region is own_surface)
        if owns_drag:
            return NotImplemented
        if event.event_type == MouseEventType.MOUSE_MOVE:
            return None
        return NotImplemented

    @staticmethod
    def _drag_release_event(event: MouseEvent) -> bool:
        """Whether ``event`` proves the drag button is no longer held.

        A ``MOUSE_UP`` is the obvious case. A ``MOUSE_DOWN`` means a fresh
        press, so the previous release must have been missed. A
        ``MOUSE_MOVE`` whose button is not LEFT is a hover — the button
        was released somewhere we couldn't see (off-terminal, or over a
        cell without a handler).
        """
        et = event.event_type
        if et in (MouseEventType.MOUSE_UP, MouseEventType.MOUSE_DOWN):
            return True
        return et == MouseEventType.MOUSE_MOVE and event.button != MouseButton.LEFT

    def _decoration_mouse_handler(self, event: MouseEvent):  # noqa: ANN201
        """Mouse handler for decorative surfaces (separators, hint row).

        Terminal emulators clamp off-window mouse coordinates onto the
        nearest edge cell, which is exactly where these rows sit — a drag
        released below the app lands its ``MOUSE_UP`` here. Their only
        mouse duty is to keep such a drag from dangling uncopied.
        """
        return self._inflight_drag_intercept(event)

    def _install_buffer_copy_handler(self, name: str, control, buffer: Buffer) -> None:  # noqa: ANN001
        """Add copy-on-release + cross-region rules to a ``BufferControl``.

        prompt_toolkit's ``BufferControl`` already implements native
        drag-selection (MOUSE_DOWN moves the cursor, MOUSE_MOVE extends
        ``buffer.selection_state``, double-click selects a word) and the
        highlight paints via ``class:selected``. This wrapper adds the
        write to the system clipboard on release and a triple-click →
        select-line gesture, without disturbing the native behaviour.
        """
        self._selection_buffers[name] = buffer
        original_handler = control.mouse_handler
        # prompt_toolkit counts double-clicks on MOUSE_UP, so the tracker
        # does too — its third rapid UP widens the native word selection
        # to the whole line.
        click_tracker = MultiClickTracker()

        def handler(event: MouseEvent):  # noqa: ANN202
            et = event.event_type
            intercepted = self._inflight_drag_intercept(event)
            if intercepted is not NotImplemented:
                return intercepted
            if et == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
                self._clear_other_selections(keep=name)
            result = original_handler(event)
            if et == MouseEventType.MOUSE_UP:
                clicks = click_tracker.register(int(event.position.y), int(event.position.x))
                if clicks >= 3:
                    _select_buffer_line(buffer)
                text = _buffer_selection_text(buffer)
                if text:
                    self._copy_selection_text(text)
                    return None
            return result

        control.mouse_handler = handler

    def _any_selection_active(self) -> bool:
        """Whether any pane (scrollback, region, or input buffer) has a selection."""
        if not self._selection.is_empty():
            return True
        if any(not region.selection.is_empty() for region in self._selection_regions.values()):
            return True
        return any(buffer.selection_state is not None for buffer in self._selection_buffers.values())

    def _forward_to_scrollbar(self, event: MouseEvent, *, row_override: Optional[str] = None) -> None:
        """Send ``event`` to :class:`ScrollbarController` from a non-gutter window.

        ``event.position.y`` arriving via the output pane is the UIContent
        line index, so subtracting ``vertical_scroll`` recovers the row
        within the shared viewport — which equals the row index in the
        scrollbar's own 1-col content. For events from the status bar
        (always 1 row tall, sitting *below* the gutter) pass
        ``row_override="bottom"`` so the scrollbar clamps to its last
        track row. ``row_override="top"`` is also accepted for symmetry
        when adding more forwarders.
        """
        viewport = max(1, self._output_viewport_rows())
        if row_override == "bottom":
            row = viewport - 1
        elif row_override == "top":
            row = 0
        else:
            vertical_scroll = self._get_output_scroll(self._output_window)
            row = max(0, min(viewport - 1, int(event.position.y) - vertical_scroll))
        forwarded = MouseEvent(
            position=Point(x=0, y=row),
            event_type=event.event_type,
            button=event.button,
            modifiers=event.modifiers,
        )
        self._scrollbar_controller.handle_event(forwarded)

    # ── Find-in-scrollback (Ctrl+F) ────────────────────────────────

    def _build_search_bar(self) -> ConditionalContainer:
        """1-row search bar shown beneath the input when ``_search_active``.

        Three columns laid out left → right:

        * ``" Find: "`` prompt label (fixed width, ``class:search-prompt``).
        * The :class:`Buffer`-backed input field (``BufferControl``) where
          the user types the query.
        * Status block on the right edge — ``"1/12"`` / ``"No matches"`` /
          ``"type to search…"`` so the user always sees where they are.

        The container is wrapped in a :class:`ConditionalContainer` keyed
        on ``_search_active``, so it occupies zero vertical space outside
        of an active search session and the existing bottom layout stays
        unchanged for everyone else.
        """
        prompt_window = Window(
            content=FormattedTextControl(text=lambda: [("class:search-prompt", " Find: ")]),
            height=1,
            dont_extend_width=True,
            wrap_lines=False,
        )
        search_control = BufferControl(
            buffer=self._search_buffer,
            focusable=True,
            key_bindings=self._build_search_kb(),
        )
        self._install_buffer_copy_handler("search", search_control, self._search_buffer)
        input_window = Window(
            content=search_control,
            height=1,
            wrap_lines=False,
            style="class:search-input",
        )
        status_window = Window(
            content=FormattedTextControl(text=self._search_status_tokens),
            height=1,
            dont_extend_width=True,
            wrap_lines=False,
        )
        return ConditionalContainer(
            content=VSplit([prompt_window, input_window, status_window]),
            filter=Condition(lambda: self._search_active),
        )

    def _build_search_kb(self) -> KeyBindings:
        """Search-buffer-local key bindings.

        * ``Enter`` / ``Down`` → next match
        * ``Shift+Tab`` / ``Up`` → previous match
        * ``Escape``      → close search, drop highlights
        * ``Ctrl+C``      → same as Escape
        * ``Ctrl+G``      → same as Escape (Readline-style cancel)
        * ``Ctrl+F``      → idempotent reset — clear the query so the
                            user can retype without first closing.

        Note: ``Shift+Enter`` was the obvious choice for "previous", but
        most terminals (including the default macOS Terminal) send the
        same byte sequence for Enter and Shift+Enter, so prompt_toolkit
        does not expose an ``s-enter`` key. ``Shift+Tab`` and ``Up`` are
        both well-supported and unambiguous.
        """
        kb = KeyBindings()

        @kb.add("enter")
        def _next(event) -> None:  # noqa: ANN001
            self._jump_to_match(+1)

        @kb.add("down")
        def _next_down(event) -> None:  # noqa: ANN001
            self._jump_to_match(+1)

        @kb.add("s-tab")
        def _prev(event) -> None:  # noqa: ANN001
            self._jump_to_match(-1)

        @kb.add("up")
        def _prev_up(event) -> None:  # noqa: ANN001
            self._jump_to_match(-1)

        @kb.add("escape", eager=True)
        def _close_esc(event) -> None:  # noqa: ANN001
            self._close_search()

        @kb.add("c-c")
        def _close_c(event) -> None:  # noqa: ANN001
            self._close_search()

        @kb.add("c-g")
        def _close_g(event) -> None:  # noqa: ANN001
            self._close_search()

        @kb.add("c-f")
        def _retype(event) -> None:  # noqa: ANN001
            # Already open; reset the query so a second Ctrl+F is a "clear
            # and retype" gesture rather than dispatching to the global
            # binding (which would just re-open us).
            self._search_buffer.text = ""

        return kb

    def _open_search(self) -> None:
        """Show the search bar, focus its input, and clear any prior state."""
        # Drop any leftover search results from a previous session so a
        # fresh Ctrl+F always starts from "type to search…".
        self._search_state.clear()
        # Reset the buffer *before* flipping ``_search_active`` so the
        # ``on_text_changed`` callback (which fires synchronously on
        # ``text =``) doesn't see a stale active flag.
        self._search_buffer.text = ""
        self._search_active = True
        try:
            self._app.layout.focus(self._search_buffer)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("_open_search: focus(search buffer) failed: %s", exc)
        self._app.invalidate()

    def _close_search(self) -> None:
        """Hide the search bar, clear the overlay, return focus to input."""
        self._search_active = False
        self._search_state.clear()
        self._search_buffer.text = ""
        try:
            self._app.layout.focus(self._input_area)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("_close_search: focus(input) failed: %s", exc)
        self._app.invalidate()

    def _on_search_text_changed(self, buffer: Buffer) -> None:
        """Re-scan the scrollback on every keystroke and jump to first match."""
        query = buffer.text
        if not query:
            # Empty query: clear results but keep the bar open so the
            # user can keep typing.
            self._search_state.update(query="", matches=[], current_idx=-1)
            self._app.invalidate()
            return
        if self._output_buffer is None:
            self._search_state.update(query=query, matches=[], current_idx=-1)
            self._app.invalidate()
            return
        snap = self._output_buffer.snapshot()
        matches = find_matches(snap.get_line, snap.total, query)
        self._search_state.update(query=query, matches=matches, current_idx=0 if matches else -1)
        if matches:
            self._center_on_line(matches[0].line)
        self._app.invalidate()

    def _jump_to_match(self, direction: int) -> None:
        """Move ``current_idx`` by ``direction`` (wraps) and re-centre."""
        if not self._search_state.matches:
            return
        new_idx = (self._search_state.current_idx + direction) % len(self._search_state.matches)
        self._search_state.set_current(new_idx)
        match = self._search_state.current()
        if match is not None:
            self._center_on_line(match.line)
        self._app.invalidate()

    def _center_on_line(self, line_idx: int) -> None:
        """Scroll the viewport so ``line_idx`` sits roughly in the middle.

        Re-uses :meth:`_set_output_scroll_offset` which already handles
        sticky-bottom + bounds clamping. ``viewport // 2`` is a coarse
        centring heuristic — exact centring would need wrapping-aware
        accounting that the rest of the codebase doesn't bother with.
        """
        viewport = max(1, self._output_viewport_rows())
        target = max(0, line_idx - viewport // 2)
        self._set_output_scroll_offset(target)

    def _search_status_tokens(self) -> List[Tuple[str, str]]:
        """Right-aligned status label inside the search bar."""
        state = self._search_state
        if not state.query:
            return [("class:search-meta", " type to search… ")]
        if not state.matches:
            return [("class:search-meta.no-match", " No matches ")]
        return [("class:search-meta", f" {state.current_idx + 1}/{len(state.matches)} ")]

    def _terminal_columns(self) -> int:
        """Best-effort current terminal column count.

        Mirrors :meth:`_terminal_rows`: prefer the live ``Application.output``
        (resize-aware), fall back to ``shutil.get_terminal_size`` and finally
        a sane default. Used by :meth:`_input_visual_line_count` so the input
        bar grows when long text wraps onto extra visual rows.
        """
        app = getattr(self, "_app", None)
        if app is not None:
            try:
                size = app.output.get_size()
                if size and size.columns > 0:
                    return int(size.columns)
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            import shutil

            size = shutil.get_terminal_size(fallback=(80, 24))
            return int(size.columns)
        except Exception:  # pragma: no cover - defensive
            return 80

    def _input_prompt_display_width(self) -> int:
        """Display width (in cells) of the rendered input prompt.

        Walks the FormattedText fragments returned by :meth:`_get_input_prompt`
        and sums :func:`get_cwidth` per character so wide / CJK glyphs are
        accounted for. Used to subtract the prompt cells from the first visual
        line's available width when computing wrap counts.
        """
        try:
            fragments = self._get_input_prompt()
        except Exception:  # pragma: no cover - defensive
            return 0
        width = 0
        for fragment in fragments:
            try:
                _, text = fragment[0], fragment[1]
            except (IndexError, TypeError):  # pragma: no cover - defensive
                continue
            for ch in text:
                width += get_cwidth(ch)
        return width

    def _input_visual_line_count(self) -> int:
        """Rows the input would occupy after wrapping at the current width.

        Splits the buffer text on hard newlines and, for each segment,
        ceil-divides its display width by the available column count
        (full terminal width minus the prompt width on the first segment).
        Empty hard lines still count as one row, matching prompt_toolkit's
        own renderer.
        """
        try:
            text = self._input_area.buffer.text
        except AttributeError:  # pragma: no cover - defensive
            return 1
        columns = max(self._terminal_columns(), 1)
        prompt_width = self._input_prompt_display_width()
        total = 0
        for idx, line in enumerate(text.split("\n")):
            usable = columns - (prompt_width if idx == 0 else 0)
            usable = max(usable, 1)
            line_width = sum(get_cwidth(ch) for ch in line)
            total += max(1, -(-line_width // usable))
        return max(total, 1)

    def show_hint(self, text: str, duration: float = 1.0) -> None:
        """Show a transient one-line hint above the input area.

        A second call replaces the previous text and restarts the timer,
        so back-to-back events (e.g. Ctrl+C then a mouse copy) don't get
        their hints cut short by an earlier countdown.
        """
        if self._hint_clear_handle is not None:
            self._hint_clear_handle.cancel()
            self._hint_clear_handle = None
        self._hint_text = text
        self._app.invalidate()
        if self._loop is not None and duration > 0:
            self._hint_clear_handle = self._loop.call_later(duration, self._clear_hint)

    def show_ctrl_c_hint(self) -> None:
        self.show_hint("Press Ctrl+C again to exit", 1.0)

    def _clear_hint(self) -> None:
        self._hint_text = ""
        self._hint_clear_handle = None
        self._app.invalidate()

    # -- public API --------------------------------------------------------

    def _get_input_height(self) -> Dimension:
        visual_lines = self._input_visual_line_count()
        preferred = min(visual_lines, 15)
        return Dimension(min=1, preferred=max(preferred, 1), max=15)

    @staticmethod
    def _paste_placeholder(line_count: int) -> str:
        return f"[Pasted content: {line_count} lines]"

    def _on_buffer_text_changed(self, buffer: Buffer) -> None:
        if self._stored_paste:
            placeholder = self._paste_placeholder(self._stored_paste.count("\n") + 1)
            if placeholder not in buffer.text:
                self._stored_paste = None
                self._paste_collapsed = False

    @property
    def application(self) -> Application:
        return self._app

    @property
    def input_buffer(self) -> Buffer:
        return self._input_area.buffer

    @property
    def key_bindings(self) -> KeyBindings:
        """Shared KeyBindings. Callers may attach additional handlers."""
        return self._kb

    @property
    def agent_running(self) -> threading.Event:
        return self._agent_running

    @property
    def paste_collapsed(self) -> bool:
        return self._paste_collapsed

    def clear_paste_state(self) -> None:
        self._stored_paste = None
        self._paste_collapsed = False

    def set_input_text(self, text: str) -> None:
        """Prefill the input buffer (e.g. for ``.rewind``). Thread-safe."""

        def _apply() -> None:
            self._set_input_text_now(text)

        if self._loop is None:
            # Application has not started yet — direct mutation is safe because
            # no event loop owns the buffer.
            _apply()
            return
        try:
            self._loop.call_soon_threadsafe(_apply)
        except RuntimeError:
            # Loop already closed; the buffer won't be observed anyway.
            pass

    def restore_input_after_dispatch(self, text: str) -> None:
        """Restore cancelled input only after the worker releases run state.

        Showing the text while ``_agent_running`` is still set makes an
        immediate Enter route it into the pending-input queue. Deferring the
        mutation until ``_on_dispatch_done`` makes restoration and the idle
        transition atomic from the prompt loop's point of view.
        """
        with self._dispatch_state_lock:
            if self._agent_running.is_set():
                self._input_restore_after_dispatch = text
                return
        self.set_input_text(text)

    def _set_input_text_now(self, text: str) -> None:
        buffer = self._input_area.buffer
        document_cls = buffer.document.__class__
        buffer.document = document_cls(text)
        self._app.invalidate()

    def invalidate(self) -> None:
        """Trigger a redraw from any thread."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._app.invalidate)
        except RuntimeError:
            # Loop already closed; redraw is not meaningful anymore.
            pass

    # ── Embedded sub-wizard host ─────────────────────────────────

    def _bottom_section(self) -> AnyContainer:
        """DynamicContainer callback for the root layout's bottom half.

        Returns the active wizard's container when one is mounted,
        otherwise the standard status+input+hint stack. Called on
        every render so swapping the slot is a single field write
        plus an ``invalidate``.
        """
        if self._active_wizard is not None:
            return self._active_wizard.container
        return self._normal_bottom_section

    def mount_wizard(self, panel: EmbeddedWizard) -> None:
        """Mount ``panel`` in the bottom slot. **Main-loop only**.

        Worker threads must use :meth:`run_wizard` instead — it
        marshals the call onto the loop and blocks on ``done_future``.

        The wizard's key bindings are pushed into ``_wizard_kb_layer``
        so they fire even when focus is on a TextArea (whose own
        ``BufferControl`` has built-in typing bindings that would
        otherwise shadow the wizard's Tab / Enter / Esc navigation).
        ``unmount_wizard`` empties the layer.
        """
        self._active_wizard = panel
        try:
            self._stashed_focus = self._app.layout.current_window
        except Exception:  # pragma: no cover - defensive
            self._stashed_focus = None
        # Inject the wizard's bindings into the global layer. We mutate
        # ``bindings`` in place (the ``KeyBindings`` API doesn't expose
        # a public ``extend``; the attribute is a public list).
        if panel.key_bindings is not None:
            self._wizard_kb_layer.bindings.extend(panel.key_bindings.bindings)
            self._wizard_kb_layer._clear_cache()
        # Give the wizard's text panes the same drag/multi-click copy the
        # rest of the TUI has — the main Application's mouse capture stays
        # on while a wizard is mounted, so this is the only copy path.
        self._wire_wizard_selection(panel.container)
        if panel.first_focus is not None:
            try:
                self._app.layout.focus(panel.first_focus)
            except Exception as exc:
                logger.debug("mount_wizard: focus(panel.first_focus) failed: %s", exc)
        self._app.invalidate()

    def unmount_wizard(self) -> None:
        """Restore the normal bottom section and input focus. Main-loop only."""
        self._active_wizard = None
        # Clear all wizard bindings from the layer.
        self._wizard_kb_layer.bindings.clear()
        self._wizard_kb_layer._clear_cache()
        self._unwire_wizard_selection()
        target: Optional[Any] = self._stashed_focus
        if target is None:
            target = self._input_area
        try:
            self._app.layout.focus(target)
        except Exception:
            try:
                self._app.layout.focus(self._input_area)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("unmount_wizard: focus restore failed: %s", exc)
        self._stashed_focus = None
        self._app.invalidate()

    def _wire_wizard_selection(self, container: AnyContainer) -> None:
        """Give every text control inside an embedded wizard drag-copy support.

        Embedded wizards (ask_user / permission prompts, slash-command
        pickers) render inside the main Application, so terminal-native
        selection is unavailable there exactly like everywhere else under
        mouse capture. Walk the wizard's container tree and wire each
        display control to its own :class:`SelectableRegion` (drag /
        double-click word / triple-click line, copy on release), each
        input control to the copy-on-release wrapper, and each decorative
        window to the dangling-drag finaliser. Undone by
        :meth:`_unwire_wizard_selection` when the wizard unmounts.
        """
        for window in self._walk_windows(container):
            control = window.content
            # An instance-level handler means the wizard wired its own
            # mouse behaviour — don't fight it.
            if "mouse_handler" in control.__dict__:
                continue
            if isinstance(control, FormattedTextControl):
                source_text = control.text
                if self._fragments_carry_mouse_handlers(source_text):
                    # Click-driven widgets (CheckboxList / RadioList /
                    # Button rows) embed per-fragment handlers; wiring
                    # selection over them would steal their clicks.
                    continue
                name = f"wizard:{len(self._wizard_selection_names)}"
                region = self._make_selection_region(name, lambda text=source_text: list(to_formatted_text(text)))
                original_handler = control.mouse_handler

                def handler(event: MouseEvent, *, _region=region, _orig=original_handler):  # noqa: ANN202
                    result = self._region_mouse_dispatch(_region, event)
                    if result is NotImplemented:
                        # Preserve fragment-embedded handlers / Window
                        # fallback for events selection doesn't consume.
                        return _orig(event)
                    return result

                control.text = region.render_tokens
                control.mouse_handler = handler
                self._wizard_selection_names.append(name)
            elif isinstance(control, BufferControl):
                name = f"wizard:{len(self._wizard_selection_names)}"
                self._install_buffer_copy_handler(name, control, control.buffer)
                self._wizard_selection_names.append(name)
            elif isinstance(control, DummyControl):
                # Separator rules etc.: close out in-flight drags whose
                # release lands on them (same duty as the main layout's
                # decorative rows).
                control.mouse_handler = self._decoration_mouse_handler

    @staticmethod
    def _fragments_carry_mouse_handlers(text: Any) -> bool:
        """Whether a control's current fragments embed per-fragment handlers.

        Evaluated once at wire time. Returns ``True`` (→ leave the
        control alone) when inspection fails, so an exotic text callable
        can never be broken by selection wiring.
        """
        try:
            fragments = to_formatted_text(text)
        except Exception:
            return True
        return any(len(fragment) >= 3 for fragment in fragments)

    @staticmethod
    def _walk_windows(container: AnyContainer) -> Iterator[Window]:
        """Yield every :class:`Window` in ``container``'s tree."""
        stack = [to_container(container)]
        while stack:
            node = stack.pop()
            if isinstance(node, Window):
                yield node
                continue
            try:
                stack.extend(node.get_children())
            except Exception:  # pragma: no cover - defensive
                continue

    def _unwire_wizard_selection(self) -> None:
        """Drop the selection registrations of the unmounted wizard."""
        for name in self._wizard_selection_names:
            self._selection_regions.pop(name, None)
            self._selection_buffers.pop(name, None)
        self._wizard_selection_names.clear()

    def run_wizard(
        self,
        panel_factory: Callable[[asyncio.Future], EmbeddedWizard],
    ) -> Any:
        """Mount a wizard, block the calling thread until it resolves.

        Worker-thread entry point for embedded sub-wizards. The
        factory receives an ``asyncio.Future`` bound to the main loop
        and returns the :class:`EmbeddedWizard` instance (its key
        bindings should resolve the future via
        :func:`datus.cli.tui.wizard_host.resolve_with` /
        :func:`resolve_cancel`). The wizard sees the same event loop
        as the parent app, so no nested ``asyncio.run`` is needed.

        Returns the value passed to ``set_result``; ``None`` means
        cancel.
        """
        loop = self._loop
        if loop is None:
            raise RuntimeError("DatusApp.run_wizard requires an active event loop")

        done_future = loop.create_future()
        panel = panel_factory(done_future)

        loop.call_soon_threadsafe(self.mount_wizard, panel)
        try:
            cf = asyncio.run_coroutine_threadsafe(self._await_wizard_done(done_future), loop)
            return cf.result()
        finally:
            loop.call_soon_threadsafe(self.unmount_wizard)

    @staticmethod
    async def _await_wizard_done(done_future: asyncio.Future) -> Any:
        try:
            return await done_future
        except asyncio.CancelledError:
            return None

    @contextmanager
    def suspend_input(self, ready_timeout: float = 2.0) -> Iterator[None]:
        """Release stdin so a nested Application run on the worker can own it.

        Bridges the worker thread into the main loop's ``in_terminal()``
        context: the main :class:`Application` erases its UI, detaches its
        input reader, and switches the tty to cooked mode. The worker runs
        its own interactive sub-Application inside the ``with`` block with
        exclusive access to stdin; on exit the main app redraws itself.

        The handshake uses two :class:`threading.Event` objects so either
        side can observe failure without leaking the paused coroutine:
        ``released`` is set by the coroutine once ``in_terminal()`` is
        active; ``resume`` is set by the worker when it's done.

        No-op outside TUI mode (``self._loop`` is ``None``), so callers
        can wrap unconditionally.
        """
        if self._loop is None or self._app is None:
            yield
            return

        released = threading.Event()
        resume = threading.Event()

        async def _paused() -> None:
            async with in_terminal():
                released.set()
                # Off-loop wait so the asyncio loop stays responsive while
                # the worker owns stdin.
                await asyncio.get_running_loop().run_in_executor(None, resume.wait)

        try:
            fut = asyncio.run_coroutine_threadsafe(_paused(), self._loop)
        except RuntimeError as exc:
            # Loop already closed; nothing to suspend.
            logger.debug("suspend_input: loop unavailable (%s)", exc)
            yield
            return

        if not released.wait(timeout=ready_timeout):
            resume.set()
            try:
                fut.result(timeout=ready_timeout)
            except Exception:  # pragma: no cover - cleanup path
                pass
            raise RuntimeError("DatusApp failed to release stdin within timeout")

        try:
            yield
        finally:
            resume.set()
            try:
                fut.result(timeout=ready_timeout)
            except Exception as exc:  # pragma: no cover - cleanup path
                logger.debug("suspend_input cleanup raised: %s", exc)

    def submit_user_input(self, text: str) -> Optional[Future]:
        """Dispatch user input to the worker thread.

        Returns the :class:`Future` tracking the worker task (or ``None`` if
        the input was rejected because the agent is already running or the
        text is blank).
        """
        if not text.strip():
            return None
        with self._dispatch_state_lock:
            if self._agent_running.is_set():
                return None
            self._agent_running.set()
        if self._loop is None:
            # Application has not started yet — run synchronously.
            try:
                self._safe_dispatch(text)
            finally:
                with self._dispatch_state_lock:
                    self._agent_running.clear()
            return None

        self._app.invalidate()
        # Snapshot ContextVars on the prompt_toolkit loop thread so the worker
        # sees bindings (e.g. ``set_current_path_manager``) set during startup.
        # ``run_in_executor`` does not propagate ContextVars on its own.
        ctx = contextvars.copy_context()
        future = self._loop.run_in_executor(self._executor, ctx.run, self._safe_dispatch, text)
        future.add_done_callback(self._on_dispatch_done)
        return future

    def exit(self, code: int = 0) -> None:
        """Request Application exit (thread-safe)."""
        self._exit_code = code
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._app.exit)
        except RuntimeError:
            pass

    def run(self) -> int:
        """Run the Application in full_screen mode. Blocks until exit.

        ``patch_stdout`` is gone: all Rich console output now flows into
        :class:`TUIOutputBuffer` (set up by ``DatusCLI``), which feeds the
        scrollable output pane on the left of the layout. The full-screen
        Application owns the entire terminal until ``exit()`` is called.
        """

        async def _main() -> None:
            self._loop = asyncio.get_running_loop()
            blink_task = asyncio.create_task(self._blink_invalidate_loop())
            autoscroll_task = asyncio.create_task(self._selection_autoscroll_loop())
            try:
                await self._app.run_async()
            finally:
                for task in (blink_task, autoscroll_task):
                    task.cancel()
                for task in (blink_task, autoscroll_task):
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("background task cleanup raised: %s", exc)
                self._loop = None

        try:
            asyncio.run(_main())
        finally:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except KeyboardInterrupt:
                self._executor.shutdown(wait=False, cancel_futures=True)
        return self._exit_code

    # -- internals ---------------------------------------------------------

    # Cadence of the periodic invalidate that drives the status-bar running
    # indicator blink. Matched to ``status_bar._RUNNING_BLINK_HALF_PERIOD`` so
    # one full glyph cycle takes ~1s. Only fires while the agent is running,
    # so idle REPLs do not redraw the layout.
    _BLINK_INTERVAL_SECONDS = 0.5

    # Polling cadence for the drag-past-the-edge auto-scroll. Matches
    # the ``SelectionAutoscroll.interval_seconds`` default so a fast
    # cursor near the viewport edge feels continuous without saturating
    # the renderer.
    _AUTOSCROLL_POLL_SECONDS = 0.015

    async def _selection_autoscroll_loop(self) -> None:
        """Tick :class:`SelectionAutoscroll` while the user is dragging past an edge.

        Only runs when :class:`SelectionAutoscroll.direction` is set and
        the user is still dragging a selection; otherwise the coroutine
        just sleeps. Every fired tick advances the viewport by one row
        and updates the selection head to follow the new bottom/top
        edge, so the highlight extends as if the cursor stayed pinned
        to the off-screen mouse position. Auto-disarms once the viewport
        hits the top (offset=0) or bottom (sticky-bottom re-engaged).
        """
        try:
            while True:
                await asyncio.sleep(self._AUTOSCROLL_POLL_SECONDS)
                if not self._selection_autoscroll.is_active():
                    continue
                if not self._selection.dragging:
                    self._selection_autoscroll.disarm()
                    continue
                if not self._selection_autoscroll.due():
                    continue
                direction = self._selection_autoscroll.direction
                if direction < 0:
                    if self._get_output_scroll(self._output_window) <= 0:
                        # Already at the top — nothing to scroll into view.
                        self._selection_autoscroll.disarm()
                        continue
                    self._scroll_output_up(1)
                    new_row = self._get_output_scroll(self._output_window)
                else:
                    if self._output_at_bottom:
                        # Sticky-bottom: no further content below.
                        self._selection_autoscroll.disarm()
                        continue
                    self._scroll_output_down(1)
                    viewport = max(1, self._output_viewport_rows())
                    new_row = self._get_output_scroll(self._output_window) + viewport - 1
                total = int(self._output_line_count_fn())
                if total > 0:
                    new_row = max(0, min(new_row, total - 1))
                # Extend the head to the new edge column-anchored on the
                # head's last x so the highlight grows in a straight
                # vertical band rather than zig-zagging.
                head_col = self._selection.head.column if self._selection.head is not None else 0
                self._selection.update_head(SelectionPoint(line=new_row, column=head_col))
                try:
                    self._app.invalidate()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("autoscroll invalidate raised: %s", exc)
        except asyncio.CancelledError:
            raise

    async def _blink_invalidate_loop(self) -> None:
        """Keep the status-bar ``running`` dot pulsing by periodic re-renders.

        prompt_toolkit only re-renders on invalidate; the running-indicator
        glyph is derived from ``time.monotonic()`` so it only animates when
        the layout is refreshed. This task provides that cadence, but stays
        idle when no agent is running so no unnecessary layout work happens
        on the REPL prompt path.
        """
        try:
            while True:
                await asyncio.sleep(self._BLINK_INTERVAL_SECONDS)
                if self._agent_running.is_set():
                    try:
                        self._app.invalidate()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("blink invalidate raised: %s", exc)
        except asyncio.CancelledError:
            raise

    def _safe_status_tokens(self) -> FormattedText:
        try:
            tokens = self._status_tokens_fn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("status_tokens_fn raised: %s", exc)
            tokens = []
        return FormattedText(tokens)

    def _get_input_prompt(self) -> FormattedText:
        # Non-chat modes override the prompt with a distinct coloured marker
        # (red ``sql> `` / yellow ``bash> ``) so it is unmistakable that
        # Enter will execute the input, not chat. This holds even when pasted
        # content is collapsed — Enter still executes in SQL/bash mode, so the
        # prompt must not misleadingly show the green chat ``> ``.
        mode = self._current_input_mode()
        if self._paste_collapsed:
            prompt_tokens: List[Tuple[str, str]] = (
                [(MODE_CHROME[mode].prompt_style, MODE_CHROME[mode].prompt)]
                if mode is not InputMode.CHAT
                else [("class:input-prompt", "> ")]
            )
            prompt_tokens.append(("class:input-prompt.hint", "(Ctrl+E to expand) "))
            return FormattedText(prompt_tokens)
        if mode is not InputMode.CHAT:
            chrome = MODE_CHROME[mode]
            return FormattedText([(chrome.prompt_style, chrome.prompt)])
        try:
            text = self._input_prompt_fn() or "> "
        except Exception:  # pragma: no cover - defensive
            text = "> "
        style_class = "class:input-prompt.busy" if self._agent_running.is_set() else "class:input-prompt"
        return FormattedText([(style_class, text)])

    def _bang_hint_text(self) -> str:
        """Dim argument hint appended after a ``!<tool>`` / ``!<plugin>`` command.

        Delegates to the REPL's ``param_hint_fn`` (``BangCommand.param_hint``),
        which returns ``""`` while the tool/plugin name is still being typed.
        Runs on every paint, so it never raises.
        """
        try:
            hint = self._param_hint_fn(self._input_area.text)
        except Exception:  # pragma: no cover - a hint must never break rendering
            return ""
        return f"  {hint}" if hint else ""

    def _safe_dispatch(self, text: str) -> Optional[str]:
        try:
            return self._dispatch_fn(text)
        except SystemExit:
            raise
        except BaseException:  # pragma: no cover - defensive
            logger.exception("dispatch_fn raised for input: %r", text)
            return None

    def _on_dispatch_done(self, future: Future) -> None:
        with self._dispatch_state_lock:
            self._agent_running.clear()
            restored_input = self._input_restore_after_dispatch
            self._input_restore_after_dispatch = None
        # This callback runs on the prompt_toolkit event-loop thread. Apply
        # the document before returning to input processing, so the first
        # Enter after cancellation submits normally instead of seeing a
        # transient busy state or an empty buffer.
        if restored_input is not None:
            self._set_input_text_now(restored_input)
        else:
            self.invalidate()
        try:
            result = future.result()
        except BaseException:  # pragma: no cover - already logged
            return
        if result == EXIT_SENTINEL:
            self.exit(0)

    def _build_default_key_bindings(self) -> KeyBindings:
        """Default bindings: Enter submit/swallow, Ctrl+D exit, Ctrl+C cancel.

        REPL-specific bindings (Tab completion, Shift+Tab Plan Mode, Ctrl+O
        trace details, ESC interrupt) are attached by the caller via
        :attr:`key_bindings` so DatusCLI can keep its existing handlers close
        to the state they mutate.
        """
        kb = KeyBindings()

        @kb.add(Keys.BracketedPaste)
        def _bracketed_paste(event) -> None:  # noqa: ANN001
            data = event.data.replace("\r\n", "\n").replace("\r", "\n")
            line_count = data.count("\n") + 1
            buffer = event.app.current_buffer

            if line_count > PASTE_COLLAPSE_THRESHOLD:
                cur_text = buffer.text
                cur_pos = buffer.cursor_position
                if self._stored_paste:
                    old_ph = self._paste_placeholder(self._stored_paste.count("\n") + 1)
                    if old_ph in cur_text:
                        expanded = cur_text.replace(old_ph, self._stored_paste, 1)
                        cur_pos += len(self._stored_paste) - len(old_ph)
                        cur_text = expanded
                self._stored_paste = data
                self._paste_collapsed = True
                placeholder = self._paste_placeholder(line_count)
                new_text = cur_text[:cur_pos] + placeholder + cur_text[cur_pos:]
                buffer.document = Document(new_text, cur_pos + len(placeholder))
            else:
                buffer.insert_text(data)

        has_stored_paste = Condition(lambda: self._stored_paste is not None)

        @kb.add("c-e", filter=has_stored_paste)
        def _ctrl_e_toggle(event) -> None:  # noqa: ANN001
            buffer = event.app.current_buffer
            placeholder = self._paste_placeholder(self._stored_paste.count("\n") + 1)
            if placeholder in buffer.text:
                expanded = buffer.text.replace(placeholder, self._stored_paste, 1)
                buffer.document = Document(expanded, len(expanded))
            self._stored_paste = None
            self._paste_collapsed = False
            event.app.invalidate()

        @kb.add("enter")
        def _enter(event) -> None:  # noqa: ANN001
            buffer = event.app.current_buffer

            # ``\`` immediately before the cursor + Enter inserts a newline
            # instead of submitting — the only way to compose multi-line input
            # (most terminals do not deliver a distinct Shift+Enter). Skipped
            # while a completion menu is open so Enter still accepts a pick.
            if not buffer.complete_state and buffer.document.char_before_cursor == "\\":
                buffer.delete_before_cursor(1)
                buffer.insert_text("\n")
                return

            if buffer.complete_state:
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    buffer.cancel_completion()

            if self._agent_running.is_set():
                self._enqueue_pending_input(buffer, event.app)
                return

            text = buffer.text
            if self._stored_paste:
                placeholder = self._paste_placeholder(self._stored_paste.count("\n") + 1)
                if placeholder in text:
                    text = text.replace(placeholder, self._stored_paste, 1)
                self._stored_paste = None
                self._paste_collapsed = False

            if text.strip():
                history = buffer.history
                strings = history.get_strings()
                if not strings or strings[-1] != text:
                    history.append_string(text)
            buffer.reset()
            self.submit_user_input(text)

        @kb.add("c-d")
        def _ctrl_d(event) -> None:  # noqa: ANN001
            if self._agent_running.is_set():
                # A worker task still owns the executor; tearing down the
                # Application here would drop the pinned TUI while the agent
                # keeps running. Ignore Ctrl+D until the worker finishes.
                return
            if event.app.current_buffer.text:
                # Standard readline semantics: Ctrl+D with content does nothing.
                return
            self.exit(0)

        @kb.add("c-c")
        def _ctrl_c(event) -> None:  # noqa: ANN001
            now = time.monotonic()
            if now - self._last_ctrl_c_time < 1.0:
                self._last_ctrl_c_time = 0.0
                self.exit(0)
                return
            self._last_ctrl_c_time = now

            if not self._agent_running.is_set():
                self._stored_paste = None
                self._paste_collapsed = False
                event.app.current_buffer.reset()
                self.show_ctrl_c_hint()

        @kb.add("pageup")
        def _page_up(event) -> None:  # noqa: ANN001
            self._scroll_output_up(self._output_page_size())
            event.app.invalidate()

        @kb.add("pagedown")
        def _page_down(event) -> None:  # noqa: ANN001
            self._scroll_output_down(self._output_page_size())
            event.app.invalidate()

        # When mouse capture is disabled (e.g. JediTerm — see
        # ``_resolve_mouse_support``), the host terminal's DECSET 1007
        # "alternate scroll mode" translates wheel events into ↑/↓ key
        # sequences. Route those into the output viewport so the wheel
        # *feels* like in-app scroll. Gated on:
        #  * mouse capture being off (terminals with real mouse capture
        #    keep TextArea's default Up=history behaviour);
        #  * composer being empty (multi-line cursor navigation wins);
        #  * no autocomplete popup open (Up/Down must keep selecting).
        wheel_via_keys = Condition(lambda: not self._mouse_support_enabled)
        composer_empty = Condition(lambda: not self._input_area.buffer.text)
        no_completion = Condition(lambda: not self._input_area.buffer.complete_state)
        scroll_keys_active = wheel_via_keys & composer_empty & no_completion

        @kb.add("up", filter=scroll_keys_active)
        def _scroll_up_arrow(event) -> None:  # noqa: ANN001
            self._scroll_output_up(self._OUTPUT_WHEEL_STEP)
            event.app.invalidate()

        @kb.add("down", filter=scroll_keys_active)
        def _scroll_down_arrow(event) -> None:  # noqa: ANN001
            self._scroll_output_down(self._OUTPUT_WHEEL_STEP)
            event.app.invalidate()

        # Escape clears an active selection (in whichever pane holds
        # one). The filter gates this so we never compete with
        # ``DatusCLI``'s Esc → agent-interrupt binding (registered later
        # on the same KeyBindings instance); ``eager=True`` lets
        # prompt_toolkit pick this handler over the interrupt one when
        # both filters are true, which is the intuitive precedence —
        # clearing the visible highlight is a local UI concern that the
        # user explicitly asked for.
        has_selection = Condition(self._any_selection_active)

        @kb.add("escape", filter=has_selection, eager=True)
        def _clear_selection(event) -> None:  # noqa: ANN001
            self._clear_other_selections(keep=None)
            event.app.invalidate()

        # Ctrl+F opens the find-in-scrollback bar. ``eager=True`` jumps
        # ahead of prompt_toolkit's built-in BufferControl bindings
        # (the default ``c-f`` is "cursor forward" inside a TextArea —
        # which would otherwise swallow the keystroke without our
        # ever seeing it).
        @kb.add("c-f", eager=True)
        def _open_find(event) -> None:  # noqa: ANN001
            self._open_search()

        return kb
