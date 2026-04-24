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
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done, to_filter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, ScrollOffsets, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
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
    ) -> None:
        self._status_tokens_fn = status_tokens_fn
        self._dispatch_fn = dispatch_fn
        self._placeholder_fn = placeholder_fn or (lambda: "")
        self._input_prompt_fn = input_prompt_fn or (lambda: "> ")
        self._live_state = live_display_state

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

        # Pinned live-render region for subagent rolling window, processing-
        # tool blink, and streaming-markdown tail. Sits between the
        # patch_stdout scroll area and the status bar.
        #
        # Height is derived from the *current* terminal size on every render
        # (see :meth:`_pinned_max_rows`) so a larger terminal gets a larger
        # pinned area — enough room for a full markdown table to stay
        # box-rendered — while a narrow terminal is protected from having
        # the input row squeezed off screen. ``dont_extend_height=True``
        # plus ``wrap_lines=False`` means we never steal more than our
        # declared budget, and prompt_toolkit shrinks us first when the
        # input area needs to grow (multi-line paste, completions, etc.).
        live_state_active = Condition(self._live_state_is_active)
        self._live_region = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(self._render_live_region),
                height=self._pinned_height_dimension,
                dont_extend_height=True,
                wrap_lines=False,
            ),
            filter=live_state_active,
        )

        root = HSplit(
            [
                self._live_region,
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

        self._app: Application = Application(
            layout=Layout(root, focused_element=self._input_area),
            key_bindings=self._kb,
            style=style or Style([]),
            full_screen=False,
            mouse_support=False,
            erase_when_done=False,
        )

        # Wire the live-region repaint callback now that ``self.invalidate`` is
        # bound; :class:`LiveDisplayState` defaults to a no-op callback so
        # constructing it before the app is still safe. The row-budget
        # provider delegates to :meth:`_pinned_max_rows` so writers (the
        # streaming refresh daemon) shape their line lists against the
        # same ceiling prompt_toolkit applies to the Window.
        if self._live_state is not None:
            self._live_state.set_invalidate(self.invalidate)
            self._live_state.set_max_rows_provider(self._pinned_max_rows)

    @staticmethod
    def _make_separator() -> Window:
        """Full-width horizontal rule rendered with box-drawing character."""
        return Window(height=1, char="\u2500", style="class:separator")

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

    def _pinned_height_dimension(self) -> Dimension:
        """Height dimension callback for the pinned Window.

        ``Dimension.preferred`` is anchored to the current content size so a
        short markdown tail doesn't grab extra rows it doesn't need, while
        ``max`` rises with the terminal so a full table can live-render in
        box form once the body has enough visible rows.
        """
        cap = self._pinned_max_rows()
        preferred = 1
        if self._live_state is not None:
            preferred = max(1, min(cap, self._live_state.line_count()))
        return Dimension(min=1, preferred=preferred, max=cap)

    def _render_live_region(self) -> FormattedText:
        """Flatten every pinned line into a single multiline FormattedText.

        Lines are joined with ``\\n`` so the hosting Window sizes naturally
        to the pinned-line count (each line contributes one terminal row).
        """
        if self._live_state is None:
            return FormattedText([])
        snap = self._live_state.snapshot()
        if not snap:
            return FormattedText([])
        fragments: List[Tuple[str, str]] = []
        for idx, line in enumerate(snap):
            if idx > 0:
                fragments.append(("", "\n"))
            fragments.extend(line.segments)
        return FormattedText(fragments)

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
        try:
            line_count = self._input_area.buffer.document.line_count
        except AttributeError:
            line_count = 1
        preferred = min(line_count, 15)
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
        """Run the Application under ``patch_stdout``. Blocks until exit.

        Pinning the layout to the bottom of the terminal is done by the
        caller *before* printing the banner (see ``DatusCLI._pin_to_bottom``),
        so the banner still lands in the visible area. Anchoring inside this
        method would push the already-printed banner off screen.
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
            with patch_stdout(raw=True):
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

        return kb
