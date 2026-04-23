# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Reusable single-select list picker rendered as one prompt_toolkit
:class:`Application`.

Used by ``/agent``, ``/resume``, ``/rewind``, and any future command that
needs "pick one item from a list" with consistent UI/UX.  Visual style
mirrors :class:`LanguageApp` and :class:`ModelApp`: ``CLR_CURSOR``
highlight, ``→`` cursor, separator lines, footer hint row, and
CJK-aware text clipping.

Callers wrap ``app.run()`` in ``tui_app.suspend_input()`` when the REPL
is in TUI mode, exactly like ``/model`` and ``/language``.
"""

from __future__ import annotations

import shutil
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, render_tui_title_bar
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MAX_VISIBLE = 15


@dataclass
class ListItem:
    """A single item in the list selector.

    ``key`` is the opaque identifier returned on selection.
    ``primary`` is the main display text (line 1).
    ``secondary`` is optional metadata text (line 2, shown dim).
    ``is_current`` marks the currently active item with green style.
    """

    key: str
    primary: str
    secondary: str = ""
    is_current: bool = False


@dataclass
class ListSelection:
    """Outcome of a :class:`ListSelectorApp` run."""

    key: str


def _display_width(text: str) -> int:
    w = 0
    for ch in text:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _clip(text: str, width: int) -> str:
    w = 0
    for i, ch in enumerate(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width:
            return text[:i]
        w += cw
    return text


class ListSelectorApp:
    """Generic single-select list picker.

    Returns :class:`ListSelection` on confirm, ``None`` on cancel.
    """

    def __init__(self, title: str, items: List[ListItem]) -> None:
        self._title = title
        self._items = items
        self._total = len(items)

        term = shutil.get_terminal_size((120, 40))
        self._content_width = term.columns - 8
        self._max_visible = min(MAX_VISIBLE, max(1, (term.lines - 3) // (3 if self._has_secondary() else 2)))

        self._cursor = 0
        self._offset = 0

        current_idx = next((i for i, item in enumerate(items) if item.is_current), None)
        if current_idx is not None:
            self._cursor = current_idx
            if self._cursor >= self._max_visible:
                self._offset = min(
                    max(0, self._cursor - self._max_visible // 2), max(0, self._total - self._max_visible)
                )

        self._app = self._build_app()

    def run(self) -> Optional[ListSelection]:
        if not self._items:
            return None
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("ListSelectorApp crashed: %s", exc)
            return None

    def _has_secondary(self) -> bool:
        return any(item.secondary for item in self._items)

    def _ensure_visible(self) -> None:
        if self._cursor < self._offset:
            self._offset = self._cursor
        elif self._cursor >= self._offset + self._max_visible:
            self._offset = self._cursor - self._max_visible + 1

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            self._cursor = (self._cursor - 1) % self._total if self._total else 0
            if self._cursor == self._total - 1:
                self._offset = max(0, self._total - self._max_visible)
            else:
                self._ensure_visible()

        @kb.add("down")
        def _down(event):
            self._cursor = (self._cursor + 1) % self._total if self._total else 0
            if self._cursor == 0:
                self._offset = 0
            else:
                self._ensure_visible()

        @kb.add("pageup")
        def _page_up(event):
            self._cursor = max(0, self._cursor - self._max_visible)
            self._ensure_visible()

        @kb.add("pagedown")
        def _page_down(event):
            self._cursor = min(self._total - 1, self._cursor + self._max_visible)
            self._ensure_visible()

        @kb.add("enter")
        def _enter(event):
            if 0 <= self._cursor < self._total:
                event.app.exit(ListSelection(key=self._items[self._cursor].key))

        @kb.add("escape")
        def _escape(event):
            event.app.exit(None)

        @kb.add("c-c")
        def _ctrl_c(event):
            event.app.exit(None)

        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar(self._title)),
            height=1,
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
                title_bar,
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

    def _render_list(self) -> List[Tuple[str, str]]:
        if not self._items:
            return [("ansibrightblack", "  (nothing to show)\n")]

        lines: List[Tuple[str, str]] = []
        visible_end = min(self._offset + self._max_visible, self._total)

        if self._total > self._max_visible:
            lines.append(("ansiyellow", f"  ({self._offset + 1}-{visible_end} of {self._total})\n"))

        show_secondary = self._has_secondary()
        for i in range(self._offset, visible_end):
            item = self._items[i]
            primary = _clip(item.primary, self._content_width)
            is_sel = i == self._cursor

            if is_sel:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {primary}\n"))
                if show_secondary and item.secondary:
                    secondary = _clip(item.secondary, self._content_width)
                    lines.append((CLR_CURSOR, f"      {secondary}\n"))
            elif item.is_current:
                lines.append((CLR_CURRENT, f"    {primary}  \u2190 current\n"))
                if show_secondary and item.secondary:
                    secondary = _clip(item.secondary, self._content_width)
                    lines.append((CLR_CURRENT, f"      {secondary}\n"))
            else:
                lines.append(("", f"    {primary}\n"))
                if show_secondary and item.secondary:
                    secondary = _clip(item.secondary, self._content_width)
                    lines.append(("ansibrightblack", f"      {secondary}\n"))

            if show_secondary:
                lines.append(("", "\n"))

        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  \u2191\u2193 navigate   Enter select   Esc cancel")]
