# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/language`` picker rendered as a single prompt_toolkit
:class:`Application`.

Two-step flow:
1. Pick a language (or *auto* to clear the override).
2. Pick a persistence scope (*project* or *global*).

Runs inside one Application so the outer TUI only needs to release
``stdin`` once via :meth:`DatusApp.suspend_input`.  Visual style mirrors
:class:`datus.cli.model_app.ModelApp`: ``reverse`` highlight, ``→`` cursor,
separator lines, and a footer hint row.
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

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


LANGUAGE_CHOICES: Dict[str, str] = {
    "auto": "Model decides (clear override)",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}

SCOPE_CHOICES: Dict[str, str] = {
    "project": ".datus/config.yml (this project only)",
    "global": "agent.yml (all projects)",
}


class _Phase(Enum):
    LANGUAGE = "language"
    SCOPE = "scope"


@dataclass
class LanguageSelection:
    """Outcome of a :class:`LanguageApp` run.

    ``code`` is the selected language code (e.g. ``"zh"``).
    ``scope`` is ``"project"`` or ``"global"``.
    If ``code`` is ``"auto"``, the caller should clear the override.
    """

    code: str
    scope: str = "project"


class LanguageApp:
    """Two-step language picker: language code -> persistence scope.

    The caller wraps ``app.run()`` in ``tui_app.suspend_input()`` when the
    REPL is in TUI mode. Returns ``None`` on cancel (Escape / Ctrl-C).
    """

    def __init__(
        self,
        console: Console,
        current_language: str = "",
        current_source: str = "not set",
        scope_only: Optional[str] = None,
    ):
        self._console = console
        self._current = current_language
        self._current_source = current_source

        self._lang_keys: List[str] = list(LANGUAGE_CHOICES.keys())
        self._scope_keys: List[str] = list(SCOPE_CHOICES.keys())
        self._lang_idx: int = self._default_lang_index()
        self._scope_idx: int = 0

        if scope_only is not None:
            self._phase = _Phase.SCOPE
            self._selected_code = scope_only
        else:
            self._phase = _Phase.LANGUAGE
            self._selected_code = ""

        self._app = self._build_app()

    def run(self) -> Optional[LanguageSelection]:
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("LanguageApp crashed: %s", exc)
            self._console.print(f"[bold red]/language error:[/] {exc}")
            return None

    def _default_lang_index(self) -> int:
        if self._current in self._lang_keys:
            return self._lang_keys.index(self._current)
        return 0

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            if self._phase == _Phase.LANGUAGE:
                self._lang_idx = max(0, self._lang_idx - 1)
            else:
                self._scope_idx = max(0, self._scope_idx - 1)

        @kb.add("down")
        def _down(event):
            if self._phase == _Phase.LANGUAGE:
                self._lang_idx = min(len(self._lang_keys) - 1, self._lang_idx + 1)
            else:
                self._scope_idx = min(len(self._scope_keys) - 1, self._scope_idx + 1)

        @kb.add("enter")
        def _enter(event):
            if self._phase == _Phase.LANGUAGE:
                self._selected_code = self._lang_keys[self._lang_idx]
                self._phase = _Phase.SCOPE
            else:
                scope = self._scope_keys[self._scope_idx]
                event.app.exit(LanguageSelection(code=self._selected_code, scope=scope))

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
        if self._phase == _Phase.LANGUAGE:
            lines.append(("bold", "  Select response language"))
            if self._current:
                display = LANGUAGE_CHOICES.get(self._current, self._current)
                lines.append(("", f"  [current: {self._current} ({display}), source: {self._current_source}]"))
            else:
                lines.append(("", "  [current: not set (model decides)]"))
        else:
            lines.append(("bold", f"  Save '{self._selected_code}' to"))
        return lines

    def _render_list(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        if self._phase == _Phase.LANGUAGE:
            for i, key in enumerate(self._lang_keys):
                label = f"{key:<6} {LANGUAGE_CHOICES[key]}"
                if key == self._current:
                    label += "  \u2190 current"
                if i == self._lang_idx:
                    lines.append(("reverse", f"  \u2192 {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
        else:
            for i, key in enumerate(self._scope_keys):
                label = f"{key:<10} {SCOPE_CHOICES[key]}"
                if i == self._scope_idx:
                    lines.append(("reverse", f"  \u2192 {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  \u2191\u2193 navigate   Enter select   Esc cancel")]
