# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/effort`` picker rendered as a single prompt_toolkit
:class:`Application`.

Two-step flow:
1. Pick a reasoning effort level (``off|minimal|low|medium|high``).
2. Pick a persistence scope (*project* or *global*).

Runs inside one Application so the outer TUI only needs to release
``stdin`` once via :meth:`DatusApp.suspend_input`.  Visual style mirrors
:class:`datus.cli.language_app.LanguageApp`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, print_error, render_tui_title_bar
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


EFFORT_CHOICES: Dict[str, str] = {
    "off": "Disable reasoning (no thinking)",
    "minimal": "Minimal effort (fast, gpt-5 family)",
    "low": "Low effort",
    "medium": "Medium effort (balanced)",
    "high": "High effort (deep reasoning)",
}

SCOPE_CHOICES: Dict[str, str] = {
    "project": ".datus/config.yml (this project only)",
    "global": "agent.yml (all projects)",
}


class _Phase(Enum):
    EFFORT = "effort"
    SCOPE = "scope"


@dataclass
class EffortSelection:
    """Outcome of an :class:`EffortApp` run.

    ``code`` is the selected effort level (``off|minimal|low|medium|high``).
    ``scope`` is ``"project"`` or ``"global"``.
    """

    code: str
    scope: str = "project"


class EffortApp:
    """Two-step effort picker: level -> persistence scope.

    The caller wraps ``app.run()`` in ``tui_app.suspend_input()`` when the
    REPL is in TUI mode. Returns ``None`` on cancel (Escape / Ctrl-C).
    """

    def __init__(
        self,
        console: Console,
        current_effort: str = "",
        current_source: str = "not set",
        scope_only: Optional[str] = None,
    ):
        self._console = console
        self._current = current_effort
        self._current_source = current_source

        self._effort_keys: List[str] = list(EFFORT_CHOICES.keys())
        self._scope_keys: List[str] = list(SCOPE_CHOICES.keys())
        self._effort_idx: int = self._default_effort_index()
        self._scope_idx: int = 0

        if scope_only is not None:
            self._phase = _Phase.SCOPE
            self._selected_code = scope_only
        else:
            self._phase = _Phase.EFFORT
            self._selected_code = ""

        self._app = self._build_app()

    def run(self) -> Optional[EffortSelection]:
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("EffortApp crashed: %s", exc)
            print_error(self._console, f"/effort error: {exc}")
            return None

    def _default_effort_index(self) -> int:
        if self._current in self._effort_keys:
            return self._effort_keys.index(self._current)
        return self._effort_keys.index("medium")

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            if self._phase == _Phase.EFFORT:
                total = len(self._effort_keys)
                self._effort_idx = (self._effort_idx - 1) % total
            else:
                self._scope_idx = max(0, self._scope_idx - 1)

        @kb.add("down")
        def _down(event):
            if self._phase == _Phase.EFFORT:
                total = len(self._effort_keys)
                self._effort_idx = (self._effort_idx + 1) % total
            else:
                self._scope_idx = min(len(self._scope_keys) - 1, self._scope_idx + 1)

        @kb.add("enter")
        def _enter(event):
            if self._phase == _Phase.EFFORT:
                self._selected_code = self._effort_keys[self._effort_idx]
                self._phase = _Phase.SCOPE
                self._scope_idx = 0
            else:
                scope = self._scope_keys[self._scope_idx]
                event.app.exit(EffortSelection(code=self._selected_code, scope=scope))

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

        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Reasoning Effort")),
            height=1,
        )

        root = HSplit(
            [
                title_bar,
                header_window,
                Window(height=1, char="\u2500"),
                list_window,
                Window(height=1, char="\u2500"),
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
        lines: List[Tuple[str, str]] = []
        if self._phase == _Phase.EFFORT:
            lines.append(("bold", "  Select reasoning effort"))
            if self._current:
                lines.append(("", f"  [current: {self._current}, source: {self._current_source}]"))
            else:
                lines.append(("", "  [current: not set (model defaults apply)]"))
        else:
            lines.append(("bold", f"  Save '{self._selected_code}' to"))
        return lines

    def _render_list(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        if self._phase == _Phase.EFFORT:
            for i, key in enumerate(self._effort_keys):
                label = f"{key:<8} {EFFORT_CHOICES[key]}"
                is_current = key == self._current
                if is_current:
                    label += "  \u2190 current"
                if i == self._effort_idx:
                    lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
                elif is_current:
                    lines.append((CLR_CURRENT, f"    {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
        else:
            for i, key in enumerate(self._scope_keys):
                label = f"{key:<10} {SCOPE_CHOICES[key]}"
                if i == self._scope_idx:
                    lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  \u2191\u2193 navigate   Enter select   Esc cancel")]
