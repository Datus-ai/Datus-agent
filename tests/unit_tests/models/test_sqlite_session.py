# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Atomic replacement contracts using real SQLite messages and metadata."""

import sqlite3
from types import SimpleNamespace

import pytest
from agents.usage import Usage

from datus.models.session_manager import SessionManager
from datus.storage.session_state import ContextState


@pytest.mark.asyncio
async def test_failed_metadata_insert_keeps_history_and_measurement(tmp_path):
    """A database error after deleting old rows rolls back the entire rewrite."""
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("atomic")
    original = [{"role": "user", "content": "keep"}, {"role": "assistant", "content": "history"}]
    await session.add_items(original)
    manager.save_context_state("atomic", ContextState(700, 1000, True))
    with sqlite3.connect(tmp_path / "atomic.db") as conn:
        conn.execute(
            "CREATE TRIGGER reject_rewrite BEFORE INSERT ON message_structure "
            "BEGIN SELECT RAISE(ABORT, 'metadata rejected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="metadata rejected"):
        await session.replace_items([{"role": "assistant", "content": "replacement"}])
    assert await session.get_items() == original
    assert manager.load_context_state("atomic") == ContextState(700, 1000, True)
    session.close()


@pytest.mark.asyncio
async def test_rewrite_preserves_billing_and_monotonic_turns(tmp_path):
    """Compaction neither refunds usage nor overwrites it on the next turn."""
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("billing")
    await session.add_items([{"role": "user", "content": "first"}])
    await session.store_run_usage(
        SimpleNamespace(
            context_wrapper=SimpleNamespace(
                usage=Usage(requests=1, input_tokens=100, output_tokens=20, total_tokens=120)
            )
        )
    )
    manager.save_context_state("billing", ContextState(100, 1000, True))
    await session.replace_items([{"role": "assistant", "content": "recap"}])
    assert manager.load_context_state("billing") == ContextState(0, 1000, False)
    await session.add_items([{"role": "user", "content": "second"}])
    await session.store_run_usage(
        SimpleNamespace(
            context_wrapper=SimpleNamespace(usage=Usage(requests=1, input_tokens=50, output_tokens=10, total_tokens=60))
        )
    )
    usage = await session.get_turn_usage()
    assert [row["user_turn_number"] for row in usage] == [1, 2]
    assert manager.get_detailed_usage("billing")["total"]["total_tokens"] == 180
    session.close()


@pytest.mark.asyncio
async def test_rewrite_invalidates_live_measurement_without_refunding_spend(tmp_path):
    """Resume must not recover pre-rewrite occupancy from the live usage table."""
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("running")
    manager.upsert_running_turn_usage(
        "running",
        1,
        {
            "requests": 2,
            "input_tokens": 1200,
            "output_tokens": 100,
            "total_tokens": 1300,
            "last_call_input_tokens": 900,
            "context_usage_ratio": 0.9,
        },
        1000,
    )
    await session.replace_items([{"role": "assistant", "content": "recap"}])
    cumulative = manager.get_running_turn_usage("running")["cumulative"]
    assert cumulative["last_call_input_tokens"] == 0
    assert cumulative["context_usage_ratio"] == 0
    assert cumulative["total_tokens"] == 1300
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [None, "{not json", '["a", "list"]', '"a string"', "17"])
async def test_an_unreadable_usage_snapshot_never_blocks_the_rewrite(tmp_path, snapshot):
    """No shape of ``cumulative_json`` may abort the compaction transaction.

    The rewrite rolls back on any exception, so a snapshot the reader cannot
    interpret would keep the oversized history that compaction exists to
    remove. An uninterpretable snapshot is treated as absent instead.
    """
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("unreadable")
    await session.add_items([{"role": "user", "content": "oversized"}])
    manager.upsert_running_turn_usage("unreadable", 1, {"total_tokens": 1300}, 1000)
    with sqlite3.connect(tmp_path / "unreadable.db") as conn:
        conn.execute("UPDATE running_turn_usage SET cumulative_json = ?", (snapshot,))

    await session.replace_items([{"role": "assistant", "content": "recap"}])

    assert await session.get_items() == [{"role": "assistant", "content": "recap"}]
    cumulative = manager.get_running_turn_usage("unreadable")["cumulative"]
    assert cumulative["last_call_input_tokens"] == 0
    assert cumulative["context_usage_valid"] is False
    session.close()


# ===========================================================================
# Session title
#
# A session is listed under its first user message. That message used to be
# found by scanning the history, which a major compact deletes: the scan then
# reports nothing, or — once the conversation continues — a mid-conversation
# message that silently retitles the chat. The title is now recorded as it
# arrives and read back from ``session_meta``, which no rewrite touches.
# ===========================================================================


@pytest.mark.asyncio
async def test_the_first_user_message_names_the_session(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("named")

    await session.add_items([{"role": "user", "content": "How many buses ran today?"}])

    assert manager.get_session_info("named")["first_user_message"] == "How many buses ran today?"
    session.close()


@pytest.mark.asyncio
async def test_a_compact_cannot_erase_the_session_title(tmp_path):
    """The post-compact shape: an assistant recap and no user rows at all."""
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("compacted")
    await session.add_items(
        [
            {"role": "user", "content": "How many buses ran today?"},
            {"role": "assistant", "content": "412."},
        ]
    )

    await session.replace_items([{"role": "assistant", "content": "Recap of the conversation."}])

    assert await session.get_items() == [{"role": "assistant", "content": "Recap of the conversation."}]
    assert manager.get_session_info("compacted")["first_user_message"] == "How many buses ran today?"
    session.close()


@pytest.mark.asyncio
async def test_a_later_message_cannot_retitle_a_compacted_session(tmp_path):
    """After a compact the earliest surviving user row is mid-conversation.

    The session is reopened first, so the write-once guard is enforced by the
    stored row rather than by an in-memory flag that a resume would reset.
    """
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("continued")
    await session.add_items([{"role": "user", "content": "How many buses ran today?"}])
    await session.replace_items([{"role": "assistant", "content": "Recap of the conversation."}])
    manager.close_all_sessions()

    reopened = manager.get_session("continued")
    await reopened.add_items([{"role": "user", "content": "And by route?"}])

    assert manager.get_session_info("continued")["first_user_message"] == "How many buses ran today?"
    reopened.close()


@pytest.mark.asyncio
async def test_a_session_recorded_before_the_title_existed_still_names_itself(tmp_path):
    """Sessions older than ``session_meta`` keep the scan as their only source."""
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("legacy")
    await session.add_items([{"role": "user", "content": "What is SQL?"}])
    with sqlite3.connect(tmp_path / "legacy.db") as conn:
        conn.execute("DROP TABLE session_meta")

    assert manager.get_session_info("legacy")["first_user_message"] == "What is SQL?"
    session.close()


@pytest.mark.asyncio
async def test_a_first_request_compact_still_names_the_session(tmp_path):
    """Compaction on the very first request persists the input itself.

    The SDK re-appends that same input afterwards and ``add_items`` drops it as
    a duplicate, so the opening message only ever appears in the dropped part.
    A later compact then removes the stored copy too, leaving nothing to scan.
    """
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("firstreq")
    opening = [{"role": "user", "content": "How many buses ran today?"}]
    await session.replace_items(opening)
    session.skip_persisted_input_once(opening)
    await session.add_items(opening)

    await session.replace_items([{"role": "assistant", "content": "Recap of the conversation."}])

    assert manager.get_session_info("firstreq")["first_user_message"] == "How many buses ran today?"
    session.close()


@pytest.mark.asyncio
async def test_a_tool_result_never_names_the_session(tmp_path):
    """Claude-native tool results arrive as user messages but are not turns.

    ``_is_user_message`` already rejects them for turn numbering; the title has
    to honour the same definition or a resumed session would be listed under a
    blob of tool output.
    """
    manager = SessionManager(session_dir=str(tmp_path))
    session = manager.get_session("tooled")

    await session.add_items(
        [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "412"}]},
            {"role": "user", "content": "How many buses ran today?"},
        ]
    )

    assert manager.get_session_info("tooled")["first_user_message"] == "How many buses ran today?"
    session.close()
