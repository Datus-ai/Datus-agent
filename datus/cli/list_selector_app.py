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

import asyncio
import shutil
import unicodedata
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, render_tui_title_bar
from datus.cli.tui.wizard_host import EmbeddedWizard, resolve_cancel, resolve_with
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

        self._cursor = 0
        self._offset = 0
        # Active list Window (built by _build_root_container); used by
        # ``_max_visible`` to read the actual rendered row count instead
        # of a static terminal-height cap — important when embedded,
        # where the wizard only owns the bottom slice of the screen.
        self._list_window: Optional[Window] = None

        current_idx = next((i for i, item in enumerate(items) if item.is_current), None)
        if current_idx is not None:
            self._cursor = current_idx
            mv = self._max_visible
            if self._cursor >= mv:
                self._offset = min(max(0, self._cursor - mv // 2), max(0, self._total - mv))

        # ``_on_done`` swaps between ``Application.exit`` (standalone)
        # and ``done_future.set_result`` (embedded). See EffortApp for
        # the dual-mode pattern.
        self._on_done: Optional[Callable[[Optional[ListSelection]], None]] = None

    # ── Standalone entry point ────────────────────────────────────

    def run(self) -> Optional[ListSelection]:
        if not self._items:
            return None
        kb = self._build_key_bindings()
        root, list_window = self._build_root_container(kb)
        app = Application(
            layout=Layout(root, focused_element=list_window),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )
        self._on_done = lambda result: app.exit(result=result)
        try:
            return app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("ListSelectorApp crashed: %s", exc)
            return None
        finally:
            self._on_done = None

    # ── Embedded entry point ──────────────────────────────────────

    def build_embedded_panel(self, done_future: "asyncio.Future") -> EmbeddedWizard:
        if not self._items:
            # Empty list: resolve immediately and skip the selection callbacks
            # so the host unmounts without dispatching keypresses through us.
            resolve_cancel(done_future)
            kb = self._build_key_bindings()
            root, list_window = self._build_root_container(kb)
            return EmbeddedWizard(
                container=root,
                key_bindings=kb,
                first_focus=list_window,
                done_future=done_future,
            )
        self._on_done = lambda result: (
            resolve_cancel(done_future) if result is None else resolve_with(done_future, result)
        )
        kb = self._build_key_bindings()
        root, list_window = self._build_root_container(kb)
        return EmbeddedWizard(
            container=root,
            key_bindings=kb,
            first_focus=list_window,
            done_future=done_future,
        )

    # ── Internals ────────────────────────────────────────────────

    def _has_secondary(self) -> bool:
        return any(item.secondary for item in self._items)

    @property
    def _max_visible(self) -> int:
        """Items visible at once — adapts to actual rendered window height.

        Each item costs ``per_item`` rows (3 with secondary metadata,
        2 otherwise). Tries the live Window ``render_info`` first so
        the cap shrinks in embedded mode where the wizard only owns
        the bottom slice of the screen. Falls back to a conservative
        fraction of the terminal height when ``render_info`` is None
        (very first paint, or no Window built yet).
        """
        per_item = 3 if self._has_secondary() else 2
        # Live render_info → use actual Window height.
        win = self._list_window
        if win is not None:
            info = getattr(win, "render_info", None)
            if info is not None and getattr(info, "window_height", 0) > 0:
                # Reserve 2 rows for title bar + footer hint.
                avail = max(1, int(info.window_height) - 2)
                return min(MAX_VISIBLE, max(1, avail // per_item))
        # Fallback: assume embedded mode takes roughly half the screen.
        term_rows = shutil.get_terminal_size((120, 40)).lines
        return min(MAX_VISIBLE, max(1, (term_rows // 2 - 3) // per_item))

    def _ensure_visible(self) -> None:
        if self._cursor < self._offset:
            self._offset = self._cursor
        elif self._cursor >= self._offset + self._max_visible:
            self._offset = self._cursor - self._max_visible + 1

    def _finish(self, result: Optional[ListSelection]) -> None:
        if self._on_done is None:
            return
        self._on_done(result)

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):  # noqa: ANN001
            self._cursor = (self._cursor - 1) % self._total if self._total else 0
            if self._cursor == self._total - 1:
                self._offset = max(0, self._total - self._max_visible)
            else:
                self._ensure_visible()

        @kb.add("down")
        def _down(event):  # noqa: ANN001
            self._cursor = (self._cursor + 1) % self._total if self._total else 0
            if self._cursor == 0:
                self._offset = 0
            else:
                self._ensure_visible()

        @kb.add("pageup")
        def _page_up(event):  # noqa: ANN001
            self._cursor = max(0, self._cursor - self._max_visible)
            self._ensure_visible()

        @kb.add("pagedown")
        def _page_down(event):  # noqa: ANN001
            self._cursor = min(self._total - 1, self._cursor + self._max_visible)
            self._ensure_visible()

        @kb.add("enter")
        def _enter(event):  # noqa: ANN001
            if 0 <= self._cursor < self._total:
                self._finish(ListSelection(key=self._items[self._cursor].key))

        @kb.add("escape")
        def _escape(event):  # noqa: ANN001
            self._finish(None)

        @kb.add("c-c")
        def _ctrl_c(event):  # noqa: ANN001
            self._finish(None)

        return kb

    def _build_root_container(self, kb: KeyBindings) -> Tuple[HSplit, Window]:
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar(self._title)),
            height=1,
        )
        self._list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True, key_bindings=kb),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )
        list_window = self._list_window
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
        return root, list_window

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
