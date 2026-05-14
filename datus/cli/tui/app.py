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
from typing import Callable, Iterator, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done, to_filter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, ScrollOffsets, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

from datus.cli.cli_styles import PASTE_COLLAPSE_THRESHOLD
from datus.cli.tui.live_display_state import LiveDisplayState, compute_pinned_max_rows
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
        input_prompt_fn: Optional[Callable[[], str]] = None,
        live_display_state: Optional[LiveDisplayState] = None,
        todo_tokens_fn: Optional[Callable[[], List[Tuple[str, str]]]] = None,
        todo_has_items_fn: Optional[Callable[[], bool]] = None,
        todo_line_count_fn: Optional[Callable[[], int]] = None,
        output_tokens_fn: Optional[Callable[[], List[Tuple[str, str]]]] = None,
        output_line_count_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self._status_tokens_fn = status_tokens_fn
        self._dispatch_fn = dispatch_fn
        self._placeholder_fn = placeholder_fn or (lambda: "")
        self._input_prompt_fn = input_prompt_fn or (lambda: "> ")
        self._live_state = live_display_state
        self._todo_tokens_fn = todo_tokens_fn
        self._todo_has_items_fn = todo_has_items_fn
        self._todo_line_count_fn = todo_line_count_fn
        # Scroll-pane output (full_screen=True). Replaces patch_stdout —
        # all console.print output is captured into an in-memory buffer
        # which feeds ``output_tokens_fn``. ``output_line_count_fn`` is
        # consulted by the sticky-bottom auto-scroll logic.
        self._output_tokens_fn = output_tokens_fn or (lambda: [])
        self._output_line_count_fn = output_line_count_fn or (lambda: 0)
        # Sticky-bottom is driven by the FormattedTextControl cursor
        # position. ``_output_at_bottom=True`` (the default) means the
        # cursor tracks the very last logical line of content — every
        # invalidate re-anchors the scroll viewport so freshly committed
        # output stays in view. PgUp / wheel-up snapshot the logical
        # cursor into ``_output_cursor_y`` and disengage; PgDn / wheel
        # back to the last line re-engages. Wheel events come through the
        # output Window's ``mouse_handler`` because FormattedTextControl
        # has no built-in scroll behaviour.
        self._output_at_bottom: bool = True
        self._output_cursor_y: int = 0

        self._agent_running = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="datus-tui-worker")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._exit_code: int = 0
        self._last_ctrl_c_time: float = 0.0
        self._ctrl_c_hint: str = ""

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
        )
        self._input_area.window.dont_extend_height = to_filter(True)
        self._input_area.buffer.on_text_changed += self._on_buffer_text_changed

        self._status_window = Window(
            content=FormattedTextControl(
                text=self._safe_status_tokens,
                focusable=False,
                show_cursor=False,
            ),
            height=1,
            style="class:status-bar",
            wrap_lines=False,
        )

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

        self._hint_window = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(lambda: [("class:hint", self._ctrl_c_hint)]),
                height=1,
                wrap_lines=False,
            ),
            filter=Condition(lambda: bool(self._ctrl_c_hint)),
        )

        # Scrollable output pane (left, weight=4). Replaces the old
        # patch_stdout-driven scrollback. Every byte Rich emits flows into
        # an in-memory ``TUIOutputBuffer``; ``output_tokens_fn`` (wired by
        # ``DatusCLI``) returns the buffer's token stream on every paint.
        # The streaming markdown tail and subagent rolling window — still
        # written to ``LiveDisplayState`` — are concatenated by the buffer
        # so they appear at the bottom of the same pane, exactly where the
        # cursor would have rendered them in ``full_screen=False`` mode.
        #
        # Sticky-bottom auto-scroll is achieved by pointing
        # ``get_cursor_position`` at the last logical line of content and
        # letting prompt_toolkit's built-in "scroll to keep cursor visible"
        # do the work. PgUp / mouse-wheel-up disengage by moving the
        # logical cursor up; PgDn / wheel-down back to the end re-engages.
        # Using ``get_cursor_position`` instead of ``get_vertical_scroll``
        # is required because ``wrap_lines=True`` makes logical-row counts
        # mismatch visible-row counts, so a hand-rolled offset would
        # systematically under-scroll on wrapped output.
        self._output_window = Window(
            content=FormattedTextControl(
                text=self._output_tokens_fn,
                focusable=True,
                show_cursor=False,
                get_cursor_position=self._output_cursor_position,
            ),
            width=Dimension(weight=4),
            wrap_lines=True,
            scroll_offsets=ScrollOffsets(top=1, bottom=1),
            always_hide_cursor=to_filter(True),
            style="class:output-pane",
        )
        # FormattedTextControl has no ``mouse_handler`` constructor kwarg —
        # it's an instance method consulted by the Window. Override it
        # in place so scroll-wheel events trigger our sticky-bottom-aware
        # cursor movement instead of the default Window scroll (which
        # would be undone by ``get_cursor_position`` re-anchoring to the
        # last line on the next paint).
        self._output_window.content.mouse_handler = self._output_mouse_handler

        # Right-side todo-list sidebar for the pinned output row. Wired via
        # callbacks so non-TUI callers (and tests that pass no todo hooks)
        # keep an unchanged single-column layout. Hidden when items are empty
        # or the terminal is too narrow to fit a useful 20% column.
        self._todo_sidebar = self._build_todo_sidebar()

        # Top output row: scrollable output (weight=4) + todo sidebar
        # (weight=1). No fixed height — the row absorbs all remaining
        # terminal rows above the status bar / input / hint stack.
        top_row = VSplit([self._output_window, self._todo_sidebar])

        root = HSplit(
            [
                top_row,
                self._make_separator(),
                self._status_window,
                self._make_separator(),
                self._input_area,
                self._completions_menu,
                self._make_separator(),
                self._hint_window,
            ]
        )

        self._kb = self._build_default_key_bindings()

        # ``full_screen=True`` so the output pane and sidebar share the
        # full terminal vertical real estate. ``mouse_support=True`` lights
        # up the scroll wheel for the output pane (and click-to-focus); per
        # plan the user accepted losing terminal-native Shift+drag select
        # inside the rendered area in exchange.
        self._app: Application = Application(
            layout=Layout(root, focused_element=self._input_area),
            key_bindings=self._kb,
            style=style or Style([]),
            full_screen=True,
            mouse_support=True,
            erase_when_done=False,
        )

        # Live state now drives the streaming tail inside the scrollable
        # output pane (via ``TUIOutputBuffer.tokens()``) — its invalidate
        # callback still wakes the main loop the same way.
        if self._live_state is not None:
            self._live_state.set_invalidate(self.invalidate)
            self._live_state.set_max_rows_provider(self._pinned_max_rows)

    @staticmethod
    def _make_separator() -> Window:
        """Full-width horizontal rule rendered with box-drawing character."""
        return Window(height=1, char="\u2500", style="class:separator")

    # Minimum terminal width before the sidebar becomes too narrow to
    # be readable (20% of 60 columns ≈ 12, with a 1-col rule on its
    # left). Below this threshold the entire right column is hidden.
    _SIDEBAR_MIN_TERMINAL_COLS = 60

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
            if self._terminal_columns() < self._SIDEBAR_MIN_TERMINAL_COLS:
                return False
            try:
                return bool(has_items_fn())
            except Exception:  # pragma: no cover - defensive
                return False

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
        sidebar_body = Window(
            content=FormattedTextControl(
                text=tokens_fn,
                focusable=False,
                show_cursor=False,
            ),
            wrap_lines=True,
            dont_extend_width=False,
            style="class:todo-sidebar",
        )
        return ConditionalContainer(
            content=VSplit(
                [
                    Window(width=Dimension.exact(1), char="\u2502", style="class:separator"),
                    sidebar_body,
                ],
                width=_sidebar_width,
            ),
            filter=Condition(_sidebar_visible),
        )

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

    def _output_last_line_index(self) -> int:
        """Index of the last logical line in the output token stream.

        Returns 0 when there is no content so the cursor stays at the
        origin instead of going negative (which would crash
        prompt_toolkit's coordinate math).
        """
        return max(0, int(self._output_line_count_fn()) - 1)

    def _output_cursor_position(self) -> Optional[Point]:
        """Cursor position fed to ``FormattedTextControl``.

        When sticky-bottom is engaged, the cursor sits on the last
        logical line and prompt_toolkit's default behaviour (keep the
        cursor visible within ``scroll_offsets``) auto-scrolls the pane
        to show new output every paint. When the user has paged up the
        cursor is clamped to ``_output_cursor_y`` so the viewport stays
        where they left it. The cursor column is irrelevant; the
        control is read-only and the cursor is hidden via
        ``always_hide_cursor``.

        Returns ``None`` when the buffer is empty — otherwise
        prompt_toolkit's render loop dereferences ``fragment_lines[0]``
        on an empty list (``controls.py:413``) and crashes the TUI.
        """
        line_count = int(self._output_line_count_fn())
        if line_count <= 0:
            return None
        last = line_count - 1
        if self._output_at_bottom:
            return Point(x=0, y=last)
        return Point(x=0, y=max(0, min(last, self._output_cursor_y)))

    def _scroll_output_up(self, rows: int) -> None:
        """Move the logical cursor up by ``rows``; disengage sticky-bottom."""
        last = self._output_last_line_index()
        # Snapshot the current scroll position into the explicit cursor
        # state on the first PgUp so subsequent jumps are relative.
        current = last if self._output_at_bottom else min(last, self._output_cursor_y)
        self._output_cursor_y = max(0, current - max(1, rows))
        self._output_at_bottom = False

    def _scroll_output_down(self, rows: int) -> None:
        """Move the logical cursor down by ``rows``; re-engage sticky-bottom on overshoot."""
        last = self._output_last_line_index()
        current = last if self._output_at_bottom else min(last, self._output_cursor_y)
        new_y = current + max(1, rows)
        if new_y >= last:
            self._output_at_bottom = True
            self._output_cursor_y = last
        else:
            self._output_cursor_y = new_y

    # Wheel events scroll a small fixed step — matches what feels native
    # in most terminal emulators.
    _OUTPUT_WHEEL_STEP = 3

    def _output_mouse_handler(self, event: MouseEvent):  # noqa: ANN001
        """Translate scroll-wheel events on the output pane into scroll motion.

        FormattedTextControl has no built-in mouse handling, so we wire
        the wheel directly. Click / drag fall through (return
        ``NotImplemented``) so prompt_toolkit's default focus-on-click
        and selection still apply once we add a selectable buffer.
        """
        if event.event_type == MouseEventType.SCROLL_UP:
            self._scroll_output_up(self._OUTPUT_WHEEL_STEP)
            self._app.invalidate()
            return None
        if event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll_output_down(self._OUTPUT_WHEEL_STEP)
            self._app.invalidate()
            return None
        return NotImplemented

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

    def show_ctrl_c_hint(self) -> None:
        self._ctrl_c_hint = "Press Ctrl+C again to exit"
        self._app.invalidate()
        if self._loop is not None:
            self._loop.call_later(1.0, self._clear_ctrl_c_hint)

    def _clear_ctrl_c_hint(self) -> None:
        self._ctrl_c_hint = ""
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
        buffer = self._input_area.buffer
        document_cls = buffer.document.__class__

        def _apply() -> None:
            buffer.document = document_cls(text)
            self._app.invalidate()

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

    def invalidate(self) -> None:
        """Trigger a redraw from any thread."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._app.invalidate)
        except RuntimeError:
            # Loop already closed; redraw is not meaningful anymore.
            pass

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
        if self._agent_running.is_set():
            return None
        if not text.strip():
            return None
        if self._loop is None:
            # Application has not started yet — run synchronously.
            self._safe_dispatch(text)
            return None

        self._agent_running.set()
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
            try:
                await self._app.run_async()
            finally:
                blink_task.cancel()
                try:
                    await blink_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("blink_invalidate_loop cleanup raised: %s", exc)
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
        if self._paste_collapsed:
            return FormattedText(
                [
                    ("class:input-prompt", "> "),
                    ("class:input-prompt.hint", "(Ctrl+E to expand) "),
                ]
            )
        try:
            text = self._input_prompt_fn() or "> "
        except Exception:  # pragma: no cover - defensive
            text = "> "
        style_class = "class:input-prompt.busy" if self._agent_running.is_set() else "class:input-prompt"
        return FormattedText([(style_class, text)])

    def _safe_dispatch(self, text: str) -> Optional[str]:
        try:
            return self._dispatch_fn(text)
        except SystemExit:
            raise
        except BaseException:  # pragma: no cover - defensive
            logger.exception("dispatch_fn raised for input: %r", text)
            return None

    def _on_dispatch_done(self, future: Future) -> None:
        self._agent_running.clear()
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

            if buffer.complete_state:
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    buffer.cancel_completion()

            if self._agent_running.is_set():
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

        return kb
