# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Right-side pinned TUI sidebar that renders the current session's TodoList.

Reads the JSON file persisted by
:class:`datus.tools.func_tool.plan_tools.SessionTodoStorage` (at
``~/.datus/data/{project}/todos/{session_id}.json``) and converts the
items into prompt_toolkit token tuples. The TUI mounts those tokens as
the right column of the pinned output row immediately above the status
bar. The column is hidden via a ``ConditionalContainer`` when the file
is missing or empty so the output row reclaims full width.

The provider is deliberately decoupled from any prompt_toolkit
container construction — :class:`datus.cli.tui.app.DatusApp` builds the
Window and wires the callbacks. That keeps imports cheap for the
non-TUI code paths and avoids a circular import between this module
and ``datus.cli.tui.app``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from pydantic import ValidationError

from datus.tools.func_tool.plan_tools import TodoItem, TodoList, TodoStatus
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class TodoSidebarProvider:
    """Renders the persisted TodoList of the current session as tokens.

    Designed to be called once per TUI paint. An ``os.stat`` mtime
    cache short-circuits the JSON parse when the file has not changed
    since the previous frame, so the typical idle-paint cost is a
    single ``stat`` syscall.

    Cache is also invalidated when the session_id changes (via
    ``cmd_resume`` / ``rewind`` / ``switch``) — without that, the
    sidebar would keep rendering the previous session's todos after a
    resume.
    """

    def __init__(self, cli: "DatusCLI") -> None:
        self._cli = cli
        self._cached_mtime: Optional[float] = None
        self._cached_items: List[TodoItem] = []
        self._cached_session_id: Optional[str] = None

    # ── public API used by the prompt_toolkit Window ──────────────

    def has_items(self) -> bool:
        """Filter for ``ConditionalContainer`` — True shows the sidebar."""
        return bool(self._refresh_items())

    def tokens(self) -> List[Tuple[str, str]]:
        """Build the FormattedText token list for the sidebar Window."""
        return self._format_tokens(self._refresh_items())

    def line_count(self) -> int:
        """Logical row count of the rendered sidebar.

        Layout: hint row + title row + one row per task. The Window itself
        wraps long content with ``wrap_lines=True``, so the actual on-screen
        row count may be larger — line_count tracks logical rows only.
        """
        items = self._refresh_items()
        if not items:
            return 0
        return len(items) + 2

    # ── internal ──────────────────────────────────────────────────

    def _current_session_id(self) -> Optional[str]:
        chat_commands = getattr(self._cli, "chat_commands", None)
        node = getattr(chat_commands, "current_node", None) if chat_commands else None
        return getattr(node, "session_id", None) if node else None

    def _todo_path(self) -> Optional[Path]:
        sid = self._current_session_id()
        if not sid:
            return None
        try:
            return get_path_manager().todo_list_path(sid)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("TodoSidebar: cannot resolve todo_list_path: %s", exc)
            return None

    def _refresh_items(self) -> List[TodoItem]:
        sid = self._current_session_id()
        if sid != self._cached_session_id:
            self._cached_session_id = sid
            self._cached_mtime = None
            self._cached_items = []
        path = self._todo_path()
        if path is None or not path.exists():
            self._cached_items = []
            self._cached_mtime = None
            return self._cached_items
        try:
            mtime = path.stat().st_mtime_ns
        except OSError as exc:  # pragma: no cover - defensive
            logger.debug("TodoSidebar: stat failed on %s: %s", path, exc)
            return self._cached_items
        if mtime == self._cached_mtime:
            return self._cached_items
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            todo_list = TodoList(**data)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            logger.debug("TodoSidebar: failed to load %s: %s", path, exc)
            return self._cached_items
        self._cached_items = list(todo_list.items)
        self._cached_mtime = mtime
        return self._cached_items

    def _format_tokens(self, items: List[TodoItem]) -> List[Tuple[str, str]]:
        """Emit one styled fragment per task, tightly stacked under the title.

        Tasks are **not** truncated — the hosting Window has
        ``wrap_lines=True`` and a hard-pinned column width, so prompt_toolkit
        wraps the content across as many visual rows as it needs.

        Items are reordered so **incomplete tasks (pending + failed)
        render first**, then completed tasks. The sidebar Window has a
        bounded height (the top output row's slice of the terminal), so
        content past the bottom of the visible column is clipped by
        prompt_toolkit. Putting completed tasks last means a narrow
        terminal naturally hides done items first, keeping the most
        actionable work in view. ``sorted`` is stable so within each
        group the original ``TodoList`` order is preserved — adding a
        new pending task still puts it after the older pending tasks.
        """
        if not items:
            return []
        total = len(items)
        done = sum(1 for it in items if it.status == TodoStatus.COMPLETED)
        ordered = sorted(items, key=lambda it: it.status == TodoStatus.COMPLETED)
        # Hint row sits above the title so the user always knows the
        # Ctrl+T toggle exists. The hosting Window has ``wrap_lines=True``,
        # so on a 14-cell sidebar (the narrowest the layout allows before
        # the column is force-hidden) the hint wraps to 2 visual rows
        # rather than being clipped.
        out: List[Tuple[str, str]] = [
            ("class:todo-sidebar.hint", " Ctrl+T toggle\n"),
            ("class:todo-sidebar.title", f" Tasks ({done}/{total})\n"),
        ]
        last_idx = len(ordered) - 1
        for idx, it in enumerate(ordered):
            symbol, style = self._status_glyph(it.status)
            out.append((style, f" {symbol} {it.title}"))
            if idx != last_idx:
                out.append(("", "\n"))
        return out

    @staticmethod
    def _status_glyph(status: TodoStatus) -> Tuple[str, str]:
        if status == TodoStatus.COMPLETED:
            return "\u2713", "class:todo-sidebar.completed"  # ✓
        if status == TodoStatus.FAILED:
            return "\u2717", "class:todo-sidebar.failed"  # ✗
        if status == TodoStatus.IN_PROGRESS:
            return "\u25d0", "class:todo-sidebar.in_progress"  # ◐
        return "\u25cb", "class:todo-sidebar.pending"  # ○
