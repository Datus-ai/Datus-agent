# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""End-to-end tests for the user-turn-bounded minor compact pass on AgenticNode.

Exercises ``_minor_compact`` against a real in-memory session-like mock and a
hermetic ``ToolArchive`` so the on-disk side effects are verifiable.

The pass keeps the original tool I/O of the most recent
``keep_recent_user_turns`` user-message turns intact and offloads everything
older to disk via a single-line ``[DATUS_ARCHIVED]`` marker.
"""

from pathlib import Path
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.compact_archive import ARCHIVED_MARKER, ToolArchive
from datus.configuration.agent_config import CompactConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager


class _Node(AgenticNode):
    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        yield  # pragma: no cover

    def get_node_name(self) -> str:
        return "test_chat"


def _build_node(tmp_path, *, keep_recent_user_turns=2, archive_threshold=100):
    """Construct a node with ``__init__`` bypassed, then wire just the compact
    state — bypasses the full Node base class to keep the test hermetic.
    """
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _Node.__new__(_Node)
    node.agent_config = None
    node.session_id = "sid_test"
    node.actions = []
    node._compact_cfg = CompactConfig()
    node._compact_cfg.minor.keep_recent_user_turns = keep_recent_user_turns
    node._compact_cfg.minor.archive_threshold = archive_threshold
    node._compact_cfg.minor.archive_preview_chars = 50
    node._compacted_until = 0
    node._archive = ToolArchive(
        project_name="proj",
        session_id="sid_test",
        base_dir=tmp_path / "data",
        preview_chars=50,
    )
    node._compact_lock = None
    return node


def _mock_session(items):
    """Return an async-session-like mock whose ``get_items`` yields ``items``
    and whose ``clear_session``/``add_items`` are AsyncMocks for assertion.
    """
    session = MagicMock()
    session.get_items = AsyncMock(return_value=items)
    session.clear_session = AsyncMock()
    session.add_items = AsyncMock()
    return session


def _user_message(text: str = "hi"):
    return {"type": "message", "role": "user", "content": text}


def _tool_pair(idx, args_text, output_text):
    """Build one (function_call, function_call_output) pair the SDK would emit."""
    return [
        {
            "type": "function_call",
            "name": "read_query",
            "call_id": f"c{idx}",
            "arguments": args_text,
        },
        {
            "type": "function_call_output",
            "call_id": f"c{idx}",
            "output": output_text,
        },
    ]


def _build_turns(num_turns, *, args_long=True, output_long=False):
    """Build ``num_turns`` user turns, each followed by one tool pair.

    Layout per turn (3 items): [user_message, function_call, function_call_output].

    ``args_long=True`` (the typical case) makes args exceed threshold; arguments
    are never archived, so this only exercises the "args preserved verbatim"
    assertions. ``output_long`` defaults to False so existing tests that assert
    "kept window output stays as ``{"success": 1}``" continue to hold; set it
    True when the test wants the output archived.
    """
    items = []
    for i in range(num_turns):
        items.append(_user_message(f"q{i}"))
        args_text = "x" * 500 if args_long else "short"
        out_text = ("y" * 500) if output_long else '{"success": 1}'
        items.extend(_tool_pair(i, args_text, out_text))
    return items


@pytest.mark.asyncio
async def test_minor_compact_archives_tool_io_of_older_user_turns(tmp_path):
    """Older user turns' output gets archived; arguments and the kept window
    stay verbatim. ``function_call.arguments`` is never archived.
    """
    node = _build_node(tmp_path, keep_recent_user_turns=2, archive_threshold=100)
    # 4 user turns; latest 2 must be preserved. Outputs of the archived turns
    # are long; arguments are long too but must NOT be archived.
    items = _build_turns(4, output_long=True)
    node._session = _mock_session(items)

    result = await node._minor_compact(reason="t")

    assert result["success"]
    # First 2 user turns × 1 tool pair × output only = 2 archives (args never archived).
    assert result["archived_count"] == 2
    rewritten = node._session.add_items.await_args.args[0]
    # Turn 0 (items 1-2) and turn 1 (items 4-5): output archived, args preserved.
    assert rewritten[1]["arguments"] == "x" * 500
    assert rewritten[2]["output"].startswith(ARCHIVED_MARKER)
    assert rewritten[4]["arguments"] == "x" * 500
    assert rewritten[5]["output"].startswith(ARCHIVED_MARKER)
    # Latest 2 user turns (items 6+) untouched — args still the original.
    assert rewritten[7]["arguments"] == "x" * 500
    assert rewritten[10]["arguments"] == "x" * 500


@pytest.mark.asyncio
async def test_minor_compact_preserves_recent_user_turns(tmp_path):
    """The most-recent ``keep_recent_user_turns`` user turns are never
    compacted, even if their output crosses ``archive_threshold``. (Older
    turns' long output is archived, which is what triggers the rewrite.)
    """
    node = _build_node(tmp_path, keep_recent_user_turns=2, archive_threshold=100)
    items = _build_turns(4, output_long=True)
    node._session = _mock_session(items)

    await node._minor_compact(reason="t")
    rewritten = node._session.add_items.await_args.args[0]
    # User-turn 2 (items 6-8) and user-turn 3 (items 9-11) must remain raw —
    # args were never archive-eligible, and recent output stays verbatim.
    assert rewritten[7]["arguments"] == "x" * 500
    assert rewritten[8]["output"] == "y" * 500
    assert rewritten[10]["arguments"] == "x" * 500


@pytest.mark.asyncio
async def test_minor_compact_idempotent_second_pass(tmp_path):
    """Running minor twice on the same session must not produce duplicate
    archive files or rewrap markers — the marker-prefix detection in
    ``maybe_truncate_item`` is the canonical idempotency guarantee.
    """
    node = _build_node(tmp_path, keep_recent_user_turns=1, archive_threshold=100)
    items = _build_turns(3, output_long=True)
    node._session = _mock_session(items)

    first = await node._minor_compact(reason="t1")
    assert first["archived_count"] > 0
    files_after_first = sorted(p.name for p in node._archive.dir.iterdir())

    # Second pass — feed the rewritten items back as if loaded from session.
    rewritten = node._session.add_items.await_args.args[0]
    node._session = _mock_session(rewritten)
    # Reset high-water mark so the scan would naively revisit the prefix.
    node._compacted_until = 0
    second = await node._minor_compact(reason="t2")

    files_after_second = sorted(p.name for p in node._archive.dir.iterdir())
    # No new files; archive_count is 0 because every candidate was a marker.
    assert second["archived_count"] == 0
    assert files_after_first == files_after_second


@pytest.mark.asyncio
async def test_minor_compact_correct_when_state_lost(tmp_path):
    """If state.json disappears between runs (``_compacted_until=0``), the
    next compact still yields the right output because every previously-
    archived item is detected via its in-message ``[DATUS_ARCHIVED]`` marker.
    """
    node = _build_node(tmp_path, keep_recent_user_turns=1, archive_threshold=100)
    items = _build_turns(3, output_long=True)
    node._session = _mock_session(items)
    await node._minor_compact(reason="first")
    rewritten = node._session.add_items.await_args.args[0]

    # Simulate node rebuild on resume with no state file.
    fresh = _build_node(tmp_path, keep_recent_user_turns=1, archive_threshold=100)
    fresh._session = _mock_session(rewritten)
    result = await fresh._minor_compact(reason="resume")
    # Eligible region is fully archived already → archived_count == 0 even
    # though we re-scanned from index 0.
    assert result["archived_count"] == 0


@pytest.mark.asyncio
async def test_minor_compact_noop_when_too_few_user_turns(tmp_path):
    """A session with ≤ keep_recent_user_turns user messages has nothing
    older than the kept window — compact returns success with no writes.
    """
    node = _build_node(tmp_path, keep_recent_user_turns=3, archive_threshold=10)
    items = _build_turns(2)  # only 2 user turns
    node._session = _mock_session(items)

    result = await node._minor_compact(reason="t")
    assert result["archived_count"] == 0
    node._session.add_items.assert_not_called()


@pytest.mark.asyncio
async def test_minor_compact_disabled_returns_noop(tmp_path):
    node = _build_node(tmp_path)
    node._compact_cfg.minor.enabled = False
    node._session = _mock_session([])
    result = await node._minor_compact(reason="t")
    assert result["success"]
    assert result["archived_count"] == 0
    assert result["window"] is None


@pytest.mark.asyncio
async def test_minor_compact_archives_to_disk_byte_equal(tmp_path):
    """Archived content must be byte-equal to the original — this is the
    "zero information loss" invariant that lets the LLM recover the original
    via ``read_file(path)``.
    """
    node = _build_node(tmp_path, keep_recent_user_turns=1, archive_threshold=100)
    original = "abcdef" * 200  # 1200 chars
    items = (
        [_user_message("first ask")]
        + _tool_pair(0, "short", original)
        + [_user_message("second ask")]
        + _tool_pair(1, "short", "short")
    )
    node._session = _mock_session(items)

    await node._minor_compact(reason="t")
    rewritten = node._session.add_items.await_args.args[0]
    output_marker = rewritten[2]["output"]
    archived_path = output_marker.split("path=", 1)[1].split(" preview=", 1)[0]
    assert Path(archived_path).read_bytes() == original.encode("utf-8")


@pytest.mark.asyncio
async def test_minor_compact_advances_high_water_mark(tmp_path):
    """After a successful pass ``_compacted_until`` advances to the cutoff
    so the next pass doesn't re-scan the same prefix.
    """
    node = _build_node(tmp_path, keep_recent_user_turns=1, archive_threshold=10)
    items = _build_turns(3)
    node._session = _mock_session(items)

    assert node._compacted_until == 0
    result = await node._minor_compact(reason="t")
    assert result["window"][0] == 0
    assert node._compacted_until == result["window"][1]
    assert node._compacted_until > 0
