# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unified TUI for user interactions during agent execution.

Renders one or more :class:`InteractionEvent` as a tab-based prompt_toolkit
Application following the same visual conventions as :class:`ModelApp`.
"""

from __future__ import annotations

import io
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from prompt_toolkit.application import Application
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from rich.console import Console as RichConsole
from rich.markdown import Markdown as RichMarkdown

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, CODE_THEME, SYM_CHECK
from datus.schemas.interaction_event import InteractionEvent
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_FREE_TEXT_KEY = "__free_text__"


@dataclass
class InteractionResult:
    """Outcome of an :class:`InteractionApp` run."""

    answers: List[List[str]] = field(default_factory=list)


class InteractionApp:
    """Tab-based TUI for collecting user answers to InteractionEvent questions.

    Follows :class:`ModelApp` visual conventions: ``reverse bold`` active tab,
    ``CLR_CURSOR`` for highlighted row, footer hints, ``─`` separators.
    """

    def __init__(self, events: List[InteractionEvent]) -> None:
        self._events = events
        self._idx = 0

        n = len(events)
        self._answers: List[Optional[List[str]]] = [None] * n
        self._cursors: List[int] = [0] * n
        self._offsets: List[int] = [0] * n
        self._checked: List[Set[str]] = [set() for _ in range(n)]
        self._text_values: List[str] = [""] * n
        self._on_text_row: bool = False
        self._content_offsets: List[int] = [0] * n
        self._content_lines_cache: List[Optional[List[str]]] = [None] * n

        self._error_message: Optional[str] = None

        term_size = shutil.get_terminal_size((120, 40))
        self._max_vis: int = max(3, min(15, term_size.lines - 7))
        self._content_render_width: int = max(40, min(term_size.columns - 6, 120))
        self._max_content_vis: int = max(6, min(40, term_size.lines - 11))
        self._content_file_paths: List[Optional[Path]] = [None] * n

        self._text_buf = Buffer(name="interaction_text", on_text_changed=self._on_text_changed)

        self._choices_win: Optional[Window] = None
        self._text_win: Optional[Window] = None
        self._app = self._build()

    # ── public ──────────────────────────────────────────────────

    def run(self) -> InteractionResult:
        """Run the app. Always returns a result (ESC fills defaults)."""
        try:
            from datus.cli._cli_utils import _run_sub_application

            result = _run_sub_application(self._app)
            if isinstance(result, InteractionResult):
                return result
        except (KeyboardInterrupt, EOFError):
            pass
        except Exception as exc:
            logger.error("InteractionApp crashed: %s", exc)
        finally:
            self._cleanup_content_files()
        return self._esc_result()

    def _cleanup_content_files(self) -> None:
        for path in self._content_file_paths:
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Failed to remove interaction temp file %s: %s", path, exc)

    # ── properties ──────────────────────────────────────────────

    @property
    def _ev(self) -> InteractionEvent:
        return self._events[self._idx]

    @property
    def _is_batch(self) -> bool:
        return len(self._events) > 1

    def _keys(self) -> List[str]:
        keys = list(self._ev.choices.keys())
        if self._ev.allow_free_text:
            keys.append(_FREE_TEXT_KEY)
        return keys

    # ── result helpers ──────────────────────────────────────────

    def _esc_result(self) -> InteractionResult:
        out: List[List[str]] = []
        for i, ev in enumerate(self._events):
            if self._answers[i] is not None:
                out.append(self._answers[i])
            elif ev.default_choice:
                out.append([ev.default_choice])
            else:
                out.append([""])
        return InteractionResult(answers=out)

    def _all_done(self) -> bool:
        return all(a is not None for a in self._answers)

    def _next_unanswered(self) -> Optional[int]:
        for i in range(len(self._events)):
            if self._answers[i] is None:
                return i
        return None

    # ── state transitions ───────────────────────────────────────

    def _switch_tab(self, idx: int) -> None:
        if self._on_text_row:
            self._text_values[self._idx] = self._text_buf.text
        self._idx = idx
        self._error_message = None
        ev = self._events[idx]
        if not ev.choices and ev.allow_free_text:
            self._focus_text()
        else:
            self._text_buf.text = self._text_values[idx]
            self._focus_choices()

    def _focus_choices(self) -> None:
        self._on_text_row = False
        if self._choices_win:
            self._app.layout.focus(self._choices_win)

    def _focus_text(self) -> None:
        self._on_text_row = True
        self._text_buf.text = self._text_values[self._idx]
        if self._text_win:
            self._app.layout.focus(self._text_win)

    def _on_text_changed(self, buf: Buffer) -> None:
        self._text_values[self._idx] = buf.text

    # ── content rendering ──────────────────────────────────────

    def _render_to_ansi(self, content: str, content_type: str) -> str:
        buf = io.StringIO()
        console = RichConsole(file=buf, force_terminal=True, width=self._content_render_width, color_system="256")
        if content_type == "markdown":
            console.print(RichMarkdown(content))
        elif content_type in ("sql", "yaml", "python", "json"):
            from rich.syntax import Syntax

            console.print(Syntax(content, content_type, theme=CODE_THEME, word_wrap=True))
        else:
            console.print(content)
        return buf.getvalue()

    def _content_lines(self) -> List[str]:
        idx = self._idx
        if self._content_lines_cache[idx] is not None:
            return self._content_lines_cache[idx]
        ev = self._events[idx]
        if not ev.content:
            self._content_lines_cache[idx] = []
        elif ev.content_type == "text":
            self._content_lines_cache[idx] = [f"  {line}" for line in ev.content.split("\n")]
        else:
            ansi_str = self._render_to_ansi(ev.content, ev.content_type)
            lines = ansi_str.split("\n")
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            self._content_lines_cache[idx] = lines
        return self._content_lines_cache[idx]

    def _scroll_content(self, delta: int) -> None:
        total = len(self._content_lines())
        max_off = max(0, total - self._max_content_vis)
        off = self._content_offsets[self._idx]
        self._content_offsets[self._idx] = max(0, min(off + delta, max_off))

    def _content_file_path(self) -> Path:
        idx = self._idx
        existing = self._content_file_paths[idx]
        if existing and existing.exists():
            return existing

        ev = self._events[idx]
        suffix = {
            "sql": ".sql",
            "yaml": ".yaml",
            "python": ".py",
            "json": ".json",
            "markdown": ".md",
        }.get(ev.content_type, ".txt")
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            prefix="datus-interaction-",
            suffix=suffix,
        ) as f:
            f.write(ev.content or "")
            f.write("\n")
            path = Path(f.name)
        self._content_file_paths[idx] = path
        return path

    def _open_full_content(self) -> None:
        path = self._content_file_path()

        def _view() -> None:
            pager = os.environ.get("PAGER", "less -R")
            argv = shlex.split(pager) or ["less", "-R"]
            executable = shutil.which(argv[0])
            if executable:
                try:
                    subprocess.run([executable, *argv[1:], str(path)], check=False)
                    return
                except Exception as exc:
                    logger.warning("Failed to open pager for interaction content: %s", exc)
            print(path.read_text(encoding="utf-8", errors="replace"))

        run_in_terminal(_view, render_cli_done=True)

    def _confirm(self) -> None:
        i = self._idx
        ev = self._ev
        keys = self._keys()

        if self._on_text_row:
            self._text_values[i] = self._text_buf.text

        if ev.multi_select:
            result = sorted(self._checked[i])
            text_val = self._text_values[i].strip()
            if text_val:
                result.append(text_val)
            self._answers[i] = result
        elif self._on_text_row:
            self._answers[i] = [self._text_values[i].strip()]
        elif keys:
            cur = self._cursors[i]
            if cur < len(keys):
                k = keys[cur]
                if k != _FREE_TEXT_KEY:
                    self._answers[i] = [k]
        else:
            self._answers[i] = [""]

        if self._all_done():
            self._app.exit(result=InteractionResult(answers=[a for a in self._answers]))
            return

        nxt = self._next_unanswered()
        if nxt is not None:
            self._switch_tab(nxt)

    # ── layout ──────────────────────────────────────────────────

    def _build(self) -> Application:
        tab_win = Window(
            FormattedTextControl(self._render_tabs, focusable=False),
            height=1,
        )

        content_win = Window(
            FormattedTextControl(self._render_content, focusable=False),
            wrap_lines=True,
            dont_extend_height=True,
        )

        self._choices_win = Window(
            FormattedTextControl(self._render_choices, focusable=True),
            always_hide_cursor=True,
            dont_extend_height=True,
        )

        def _text_prefix_label():
            if self._ev.choices:
                keys = self._keys()
                idx = keys.index(_FREE_TEXT_KEY) + 1 if _FREE_TEXT_KEY in keys else len(keys) + 1
                return [(CLR_CURSOR, f"  {idx}. ")]
            return [(CLR_CURSOR, "  Answer: ")]

        self._text_win = Window(
            BufferControl(buffer=self._text_buf),
            height=1,
            left_margins=[],
            style=CLR_CURSOR,
        )
        text_prefix_win = Window(
            FormattedTextControl(_text_prefix_label),
            dont_extend_width=True,
            height=1,
            style=CLR_CURSOR,
        )
        from prompt_toolkit.layout.containers import VSplit

        text_row = VSplit([text_prefix_win, self._text_win])
        text_container = ConditionalContainer(
            content=text_row,
            filter=Condition(lambda: self._on_text_row),
        )

        error_win = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("ansired", f"  {self._error_message or ''}")]),
                height=1,
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )

        footer_win = Window(
            FormattedTextControl(self._render_footer, focusable=False),
            height=1,
        )

        def _sep():
            return Window(height=1, char="\u2500")

        title_bar = Window(height=1, char="\u2500")

        root = HSplit(
            [
                title_bar,
                tab_win,
                _sep(),
                content_win,
                _sep(),
                self._choices_win,
                text_container,
                error_win,
                _sep(),
                footer_win,
            ]
        )

        initial_focus = self._choices_win
        if not self._ev.choices and self._ev.allow_free_text:
            self._on_text_row = True
            initial_focus = self._text_win

        return Application(
            layout=Layout(root, focused_element=initial_focus),
            key_bindings=self._kb(),
            full_screen=False,
            erase_when_done=True,
        )

    # ── rendering ───────────────────────────────────────────────

    def _render_tabs(self) -> List:
        parts = []
        for i, ev in enumerate(self._events):
            label = ev.title or f"Q{i + 1}"
            if i == self._idx:
                parts.append(("reverse bold", f" {label} "))
            elif self._answers[i] is not None:
                parts.append((CLR_CURRENT, f" {SYM_CHECK} {label} "))
            else:
                parts.append(("", f"   {label} "))
            parts.append(("", " "))
        if self._is_batch:
            parts.append(("ansibrightblack", "  (Tab or \u2190 \u2192 to switch)"))
        return parts

    def _render_content(self):
        lines = self._content_lines()
        if not lines:
            return []
        total = len(lines)
        mv = self._max_content_vis
        if total <= mv:
            return ANSI("\n".join(lines))
        off = self._content_offsets[self._idx]
        off = max(0, min(off, max(0, total - mv)))
        self._content_offsets[self._idx] = off
        end = min(off + mv, total)
        visible = "\n".join(lines[off:end])
        indicator = f"\x1b[90m  ({off + 1}\u2013{end} of {total} lines, Shift+\u2191\u2193 scroll, v view full)\x1b[0m"
        return ANSI(indicator + "\n" + visible)

    def _render_choices(self) -> List:
        ev = self._ev
        keys = self._keys()
        total = len(keys)

        if total == 0:
            return []

        cursor = self._cursors[self._idx]
        offset = self._offsets[self._idx]
        mv = self._max_vis

        if cursor < offset:
            offset = cursor
        elif cursor >= offset + mv:
            offset = cursor - mv + 1
        offset = max(0, min(offset, max(0, total - mv)))
        self._offsets[self._idx] = offset

        lines = []
        end = min(offset + mv, total)

        if total > mv:
            lines.append(("ansibrightblack", f"  ({offset + 1}-{end} of {total})\n"))

        for i in range(offset, end):
            k = keys[i]
            is_cur = i == cursor and not self._on_text_row

            seq = i + 1

            if k == _FREE_TEXT_KEY:
                if not self._on_text_row:
                    tv = self._text_values[self._idx]
                    if tv:
                        display_text = tv if len(tv) <= 40 else tv[:37] + "..."
                        label = f"{seq}. {display_text}"
                    else:
                        label = f"{seq}. Custom input..."
                    if is_cur:
                        lines.append((CLR_CURSOR, f"  {label}\n"))
                    else:
                        lines.append(("ansibrightblack", f"  {label}\n"))
                continue

            display = ev.choices.get(k, k)

            if ev.multi_select:
                is_checked = k in self._checked[self._idx]
                check = "[x]" if is_checked else "[ ]"
                label = f"{seq}. {check} {display}"
                if is_cur:
                    lines.append((CLR_CURSOR, f"  {label}\n"))
                elif is_checked:
                    lines.append((CLR_CURRENT, f"  {label}\n"))
                else:
                    lines.append(("", f"  {label}\n"))
            else:
                label = f"{seq}. {display}"
                if is_cur:
                    lines.append((CLR_CURSOR, f"  {label}\n"))
                elif self._answers[self._idx] is not None and k in self._answers[self._idx]:
                    lines.append((CLR_CURRENT, f"  {label}\n"))
                else:
                    lines.append(("", f"  {label}\n"))

        if lines and lines[-1][1].endswith("\n"):
            lines[-1] = (lines[-1][0], lines[-1][1][:-1])
        return lines

    def _render_footer(self) -> List:
        if self._on_text_row:
            parts = ["  Enter submit"]
            if self._ev.choices:
                parts.append("   \u2191 back to choices")
            if self._is_batch:
                parts.append("   \u2190\u2192 switch")
            parts.append("   Esc cancel")
            return [("ansibrightblack", "".join(parts))]
        parts = ["  \u2191\u2193 navigate   Enter confirm"]
        if self._ev.multi_select:
            parts.append("   Space toggle")
        if len(self._content_lines()) > self._max_content_vis:
            parts.append("   Shift+\u2191\u2193 scroll   v view full")
        if self._is_batch:
            parts.append("   \u2190\u2192 switch")
        parts.append("   Esc cancel")
        return [("ansibrightblack", "".join(parts))]

    # ── key bindings ────────────────────────────────────────────

    def _kb(self) -> KeyBindings:
        kb = KeyBindings()
        on_choices = Condition(lambda: not self._on_text_row)
        on_text = Condition(lambda: self._on_text_row)
        is_multi = Condition(lambda: self._ev.multi_select and not self._on_text_row)
        is_batch = Condition(lambda: self._is_batch)
        is_single_q = Condition(lambda: not self._is_batch and not self._ev.multi_select)

        def _move_cursor(delta: int) -> None:
            total = len(self._keys())
            if total == 0:
                return
            self._cursors[self._idx] = (self._cursors[self._idx] + delta) % total
            keys = self._keys()
            if keys[self._cursors[self._idx]] == _FREE_TEXT_KEY:
                self._focus_text()

        @kb.add("up", filter=on_choices)
        def _up(event):
            _move_cursor(-1)

        @kb.add("down", filter=on_choices)
        def _down(event):
            _move_cursor(1)

        @kb.add("pageup", filter=on_choices)
        def _pgup(event):
            self._cursors[self._idx] = max(0, self._cursors[self._idx] - self._max_vis)

        @kb.add("pagedown", filter=on_choices)
        def _pgdn(event):
            total = len(self._keys())
            self._cursors[self._idx] = min(total - 1, self._cursors[self._idx] + self._max_vis)

        has_long_content = Condition(lambda: len(self._content_lines()) > self._max_content_vis)

        @kb.add("s-down", filter=on_choices & has_long_content)
        def _content_pgdn(event):
            self._scroll_content(self._max_content_vis)

        @kb.add("s-up", filter=on_choices & has_long_content)
        def _content_pgup(event):
            self._scroll_content(-self._max_content_vis)

        @kb.add("v", filter=on_choices & has_long_content)
        def _view_full(event):
            self._open_full_content()

        @kb.add("enter", filter=on_choices)
        def _enter(event):
            self._confirm()

        @kb.add("enter", filter=on_text)
        def _submit_text(event):
            self._confirm()

        @kb.add("up", filter=on_text)
        def _text_up(event):
            keys = self._keys()
            ft_idx = keys.index(_FREE_TEXT_KEY) if _FREE_TEXT_KEY in keys else len(keys) - 1
            self._cursors[self._idx] = (ft_idx - 1) % len(keys) if len(keys) > 1 else 0
            self._focus_choices()

        @kb.add("space", filter=is_multi)
        def _toggle(event):
            keys = self._keys()
            cur = self._cursors[self._idx]
            if cur < len(keys) and keys[cur] != _FREE_TEXT_KEY:
                ch = self._checked[self._idx]
                k = keys[cur]
                ch.symmetric_difference_update({k})

        @kb.add("a", filter=is_multi)
        def _toggle_all(event):
            ch = self._checked[self._idx]
            all_k = set(self._ev.choices.keys())
            if ch == all_k:
                ch.clear()
            else:
                ch.update(all_k)

        @kb.add("left", filter=is_batch & on_choices)
        def _prev(event):
            self._switch_tab((self._idx - 1) % len(self._events))

        @kb.add("right", filter=is_batch & on_choices)
        def _next(event):
            self._switch_tab((self._idx + 1) % len(self._events))

        @kb.add("s-tab", filter=is_batch & on_choices)
        def _stab(event):
            self._switch_tab((self._idx - 1) % len(self._events))

        @kb.add("tab", filter=is_batch & on_choices)
        def _tab(event):
            self._switch_tab((self._idx + 1) % len(self._events))

        # Single-char shortcuts (single question, single-select, on choices)
        for _c in "0123456789abcdefghijklmnopqrstuvwxyz":
            if _c in {"a", "v"}:
                continue

            @kb.add(_c, filter=is_single_q & on_choices)
            def _shortcut(event, ch=_c):
                if ch in self._ev.choices:
                    self._answers[self._idx] = [ch]
                    event.app.exit(result=InteractionResult(answers=[a for a in self._answers]))

        @kb.add("escape", filter=on_text)
        def _esc_text(event):
            self._focus_choices()

        @kb.add("escape", filter=on_choices)
        def _esc(event):
            event.app.exit(result=self._esc_result())

        @kb.add("c-c")
        def _cc(event):
            event.app.exit(result=self._esc_result())

        return kb
