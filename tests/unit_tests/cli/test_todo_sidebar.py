# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.todo_sidebar.TodoSidebarProvider`.

Covers:
- Reads the persisted ``TodoList`` and emits a title token plus one
  glyph-prefixed line per item.
- ``has_items()`` flips True/False with the underlying file's presence.
- mtime cache prevents repeated JSON parses when the file is unchanged.
- Session-id change invalidates the cache so resume / switch / rewind
  scenarios pick up the new session's todos.
- CJK-heavy content gets cell-aware truncation (no half-charged glyphs).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def path_manager(tmp_path):
    from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager

    reset_path_manager()
    pm = DatusPathManager(
        datus_home=str(tmp_path / "datus"),
        project_name="proj",
        project_root=str(tmp_path / "project"),
    )
    set_current_path_manager(pm)
    yield pm
    reset_path_manager()


def _make_cli(session_id):
    """Build a minimal CLI stub with ``chat_commands.current_node.session_id``."""
    node = SimpleNamespace(session_id=session_id)
    chat_commands = SimpleNamespace(current_node=node)
    return SimpleNamespace(chat_commands=chat_commands), node


def _write_todo_file(path, items):
    payload = {"items": [{"id": f"id-{i}", "content": c, "status": s} for i, (c, s) in enumerate(items)]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_has_items_false_when_file_missing(path_manager):
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli("session_missing")
    provider = TodoSidebarProvider(cli)

    assert provider.has_items() is False
    assert provider.tokens() == []


def test_has_items_false_without_session_id(path_manager):
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli(None)
    provider = TodoSidebarProvider(cli)

    assert provider.has_items() is False
    assert provider.tokens() == []


def test_tokens_render_title_and_glyphs(path_manager):
    """Incomplete (pending + failed) tasks render before completed,
    in stable order within each group."""
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli("session_render")
    _write_todo_file(
        path_manager.todo_list_path("session_render"),
        [("write SQL", "completed"), ("review PR", "pending"), ("ship", "failed")],
    )
    provider = TodoSidebarProvider(cli)

    assert provider.has_items() is True
    tokens = provider.tokens()

    # Title + 3 task fragments + 2 newline separators = 6 fragments.
    assert len(tokens) == 6
    title_style, title_text = tokens[0]
    assert title_style == "class:todo-sidebar.title"
    assert "Tasks (1/3)" in title_text

    # Tasks at indices 1, 3, 5. Ordering: pending, failed, completed
    # (incomplete first, stable within group → review PR before ship).
    task_styles = [tokens[1][0], tokens[3][0], tokens[5][0]]
    assert task_styles == [
        "class:todo-sidebar.pending",
        "class:todo-sidebar.failed",
        "class:todo-sidebar.completed",
    ]
    task_texts = [tokens[1][1], tokens[3][1], tokens[5][1]]
    assert "\u25cb" in task_texts[0] and "review PR" in task_texts[0]  # ○
    assert "\u2717" in task_texts[1] and "ship" in task_texts[1]  # ✗
    assert "\u2713" in task_texts[2] and "write SQL" in task_texts[2]  # ✓


def test_incomplete_tasks_render_before_completed_stable_order(path_manager):
    """Regression: stable sort preserves source order within each group.

    The user wants completed tasks pushed to the bottom of the sidebar
    so a tall todo list clips already-done work first when the column
    overflows. Pending and failed are both treated as "incomplete" and
    share the top of the list in their original ordering."""
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli("session_priority")
    _write_todo_file(
        path_manager.todo_list_path("session_priority"),
        [
            ("done-old", "completed"),
            ("pending-1", "pending"),
            ("done-new", "completed"),
            ("failed-1", "failed"),
            ("pending-2", "pending"),
        ],
    )
    provider = TodoSidebarProvider(cli)

    rendered = "".join(text for _, text in provider.tokens())
    task_lines = [line.strip() for line in rendered.splitlines() if line.strip() and not line.startswith(" Tasks")]
    # Incomplete first (stable: pending-1, done-new is completed so skipped,
    # failed-1, pending-2), then completed (done-old, done-new).
    assert task_lines == [
        "\u25cb pending-1",
        "\u2717 failed-1",
        "\u25cb pending-2",
        "\u2713 done-old",
        "\u2713 done-new",
    ]


def test_tasks_are_packed_without_blank_separators(path_manager):
    """Tasks are stacked tightly under the title row — the user
    explicitly asked for a compact layout, so the inter-task blank
    line was removed."""
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli("session_blank")
    _write_todo_file(
        path_manager.todo_list_path("session_blank"),
        [("one", "pending"), ("two", "pending"), ("three", "pending")],
    )
    provider = TodoSidebarProvider(cli)

    rendered = "".join(text for _, text in provider.tokens())
    assert rendered.splitlines() == [
        " Tasks (0/3)",
        " \u25cb one",
        " \u25cb two",
        " \u25cb three",
    ]
    # 1 title + 3 tasks = 4 logical rows.
    assert provider.line_count() == 4


def test_long_content_is_not_truncated(path_manager):
    """The Window wraps via ``wrap_lines=True`` — the provider must
    emit the full content with no ellipsis truncation."""
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli("session_long")
    long_text = "x" * 50
    _write_todo_file(
        path_manager.todo_list_path("session_long"),
        [(long_text, "pending")],
    )
    provider = TodoSidebarProvider(cli)

    tokens = provider.tokens()
    rendered = "".join(text for _, text in tokens)
    assert long_text in rendered
    assert "\u2026" not in rendered  # no ellipsis


def test_mtime_cache_skips_repeated_parse(path_manager):
    import datus.cli.todo_sidebar as sidebar_mod
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli("session_cache")
    _write_todo_file(
        path_manager.todo_list_path("session_cache"),
        [("a", "pending"), ("b", "pending")],
    )
    provider = TodoSidebarProvider(cli)

    # Prime the cache.
    provider.tokens()

    call_count = {"n": 0}
    real_loads = json.loads

    def counting_loads(s, *args, **kwargs):
        call_count["n"] += 1
        return real_loads(s, *args, **kwargs)

    with patch.object(sidebar_mod.json, "loads", side_effect=counting_loads):
        provider.tokens()
        provider.tokens()
        provider.tokens()

    assert call_count["n"] == 0, "mtime cache should suppress repeated JSON parses"


def test_session_id_change_invalidates_cache(path_manager):
    """resume/switch/rewind: a new session_id must dump the previous cache."""
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, node = _make_cli("session_a")
    _write_todo_file(
        path_manager.todo_list_path("session_a"),
        [("task A", "pending")],
    )
    _write_todo_file(
        path_manager.todo_list_path("session_b"),
        [("task B1", "completed"), ("task B2", "pending")],
    )
    provider = TodoSidebarProvider(cli)

    tokens_a = provider.tokens()
    rendered_a = "".join(text for _, text in tokens_a)
    assert "task A" in rendered_a
    assert "Tasks (0/1)" in rendered_a

    # Simulate cmd_resume swapping in a different session.
    node.session_id = "session_b"
    tokens_b = provider.tokens()
    rendered_b = "".join(text for _, text in tokens_b)
    assert "task B1" in rendered_b
    assert "task B2" in rendered_b
    assert "Tasks (1/2)" in rendered_b


def test_corrupted_json_keeps_previous_state(path_manager):
    """Bad JSON shouldn't crash the paint loop — provider returns last known good list."""
    from datus.cli.todo_sidebar import TodoSidebarProvider

    cli, _ = _make_cli("session_corrupt")
    todo_path = path_manager.todo_list_path("session_corrupt")
    _write_todo_file(todo_path, [("good", "pending")])
    provider = TodoSidebarProvider(cli)
    primed = provider.tokens()
    assert "good" in "".join(text for _, text in primed)

    # Overwrite with garbage and bump mtime so the cache reloads.
    todo_path.write_text("{ not valid", encoding="utf-8")
    import os

    stat_result = todo_path.stat()
    os.utime(todo_path, (stat_result.st_atime + 1, stat_result.st_mtime + 1))

    tokens = provider.tokens()
    # Last-known-good cached items survive the parse failure.
    assert "good" in "".join(text for _, text in tokens)
