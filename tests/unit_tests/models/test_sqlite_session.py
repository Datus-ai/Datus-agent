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
