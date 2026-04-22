# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/agent`` picker rendered as a single prompt_toolkit
:class:`Application`.

The legacy implementation called :func:`select_choice` while the outer
REPL's ``PromptSession`` state was still alive, so the nested
:class:`Application` fought with the previous session's cursor-position
request (CPR) Future. That produced the noisy ``got Future pending
attached to a different loop`` trace and the terminal-blocking
``Press ENTER to continue`` prompt whenever a user typed ``/agent``.

Here the whole picker runs inside **one** fully-owned Application,
mirroring :mod:`datus.cli.language_app` / :mod:`datus.cli.model_app` so
``erase_when_done=True`` tears down cleanly and the caller can wrap
``app.run()`` with :meth:`DatusApp.suspend_input` when the outer TUI is
active.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class AgentPickerApp:
    """Single-phase picker for the default agent.

    Returns the selected agent name on Enter, or ``None`` if the user
    cancels (Escape / Ctrl-C).
    """

    def __init__(
        self,
        console: Console,
        agents: List[str],
        current: str = "",
    ):
        if not agents:
            raise ValueError("AgentPickerApp requires at least one agent")

        self._console = console
        self._agents = list(agents)
        self._current = current or ""
        self._idx = self._agents.index(current) if current in self._agents else 0
        self._app = self._build_app()

    def run(self) -> Optional[str]:
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("AgentPickerApp crashed: %s", exc)
            self._console.print(f"[bold red]/agent error:[/] {exc}")
            return None

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            self._idx = max(0, self._idx - 1)

        @kb.add("down")
        def _down(event):
            self._idx = min(len(self._agents) - 1, self._idx + 1)

        @kb.add("pageup")
        def _page_up(event):
            self._idx = max(0, self._idx - 10)

        @kb.add("pagedown")
        def _page_down(event):
            self._idx = min(len(self._agents) - 1, self._idx + 10)

        @kb.add("enter")
        def _enter(event):
            event.app.exit(self._agents[self._idx])

        @kb.add("escape")
        def _escape(event):
            event.app.exit(None)

        @kb.add("c-c")
        def _ctrl_c(event):
            event.app.exit(None)

        header_window = Window(
            content=FormattedTextControl(self._render_header, focusable=False),
            height=Dimension(min=1, max=2),
        )

        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )

        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        root = HSplit(
            [
                header_window,
                Window(height=1, char="─"),
                list_window,
                Window(height=1, char="─"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root, focused_element=list_window),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    def _render_header(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = [("bold", "  Select default agent")]
        if self._current:
            lines.append(("", f"  [current: {self._current}]"))
        else:
            lines.append(("", "  [current: chat]"))
        return lines

    def _render_list(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        for i, name in enumerate(self._agents):
            label = name
            if name == self._current:
                label = f"{name}  ← current"
            if i == self._idx:
                lines.append(("reverse", f"  → {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  ↑↓ navigate   Enter select   Esc cancel")]
