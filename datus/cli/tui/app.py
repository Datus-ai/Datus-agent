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
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import has_completions, is_done, to_filter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, ScrollOffsets, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

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
    ) -> None:
        self._status_tokens_fn = status_tokens_fn
        self._dispatch_fn = dispatch_fn
        self._placeholder_fn = placeholder_fn or (lambda: "")
        self._input_prompt_fn = input_prompt_fn or (lambda: "> ")

        self._agent_running = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="datus-tui-worker")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._exit_code: int = 0

        # Input: multi-line TextArea. We intentionally do not set an
        # ``accept_handler`` — Enter is handled by our own key binding so we
        # can swallow it while the agent is running.
        # ``preferred=1`` keeps the input collapsed to a single row by default
        # so HSplit doesn't allocate the full remaining terminal height to
        # the TextArea. ``max=6`` lets multi-line pastes expand up to a
        # reasonable cap before the content itself needs to scroll inside the
        # buffer. The inner Window created by TextArea defaults to
        # ``dont_extend_height=not multiline`` (i.e. False when multiline is
        # on), which makes the Window ignore ``preferred`` in favour of
        # ``max`` when space is plentiful — override it after construction so
        # the collapsed-by-default behaviour is actually honoured.
        self._input_area = TextArea(
            height=Dimension(min=1, preferred=1, max=6),
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
        # ``Window.dont_extend_height`` is stored as a Filter instance
        # (``to_filter`` runs in ``Window.__init__``), so assigning a plain
        # ``True`` would break the callable the renderer later expects. Wrap
        # the boolean so the override is honoured.
        self._input_area.window.dont_extend_height = to_filter(True)

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

        root = HSplit(
            [
                self._make_separator(),
                self._status_window,
                self._make_separator(),
                self._input_area,
                self._completions_menu,
                self._make_separator(),
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

    @staticmethod
    def _make_separator() -> Window:
        """Full-width horizontal rule rendered with box-drawing character."""
        return Window(height=1, char="\u2500", style="class:separator")

    # -- public API --------------------------------------------------------

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
            try:
                await self._app.run_async()
            finally:
                self._loop = None

        try:
            with patch_stdout(raw=True):
                asyncio.run(_main())
        finally:
            self._executor.shutdown(wait=False)
        return self._exit_code

    # -- internals ---------------------------------------------------------

    def _safe_status_tokens(self) -> FormattedText:
        try:
            tokens = self._status_tokens_fn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("status_tokens_fn raised: %s", exc)
            tokens = []
        return FormattedText(tokens)

    def _get_input_prompt(self) -> FormattedText:
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

        @kb.add("enter")
        def _enter(event) -> None:  # noqa: ANN001 - prompt_toolkit signature
            buffer = event.app.current_buffer

            # Completion menu interaction mirrors the legacy PromptSession.
            if buffer.complete_state:
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    buffer.cancel_completion()
                return

            if self._agent_running.is_set():
                # Editable but not submittable. Silently ignore Enter.
                return

            text = buffer.text
            if text.strip():
                buffer.append_to_history()
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
            # When the agent is idle, clear any typed text (like bash).
            # Agent-running handling is injected by the REPL (it has the
            # interrupt_controller handle) via the shared KeyBindings.
            if not self._agent_running.is_set():
                event.app.current_buffer.reset()

        return kb
