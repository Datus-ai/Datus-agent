# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``SessionManager``'s running-turn usage side table.

The side table holds an in-flight turn's cumulative usage so the CLI status
bar and resume can show partial progress before ``store_run_usage`` commits
the final ``turn_usage`` row. These tests run against a real SQLite file in
``tmp_path`` to lock the persistence contract end-to-end.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest
from agents.extensions.memory import AdvancedSQLiteSession

from datus.models.session_manager import SessionManager


@pytest.fixture
def sm(tmp_path: Path) -> Iterator[SessionManager]:
    manager = SessionManager(session_dir=str(tmp_path / "sessions"))
    yield manager
    manager.close_all_sessions()


def _bootstrap_session(sm: SessionManager, session_id: str) -> str:
    """Materialise an empty session DB so the side-table methods have a file
    to write to. Mirrors what the agent runtime does on first SDK call."""
    db_path = os.path.join(sm.session_dir, f"{session_id}.db")
    AdvancedSQLiteSession(session_id=session_id, db_path=db_path, create_tables=True)
    return db_path


class TestUpsertAndGet:
    def test_upsert_then_get_round_trip(self, sm: SessionManager) -> None:
        session_id = "chat_session_aaa"
        _bootstrap_session(sm, session_id)
        sm.upsert_running_turn_usage(
            session_id,
            user_turn_number=2,
            cumulative={
                "requests": 3,
                "input_tokens": 800,
                "output_tokens": 200,
                "total_tokens": 1000,
                "cached_tokens": 100,
            },
            context_length=200_000,
        )
        running = sm.get_running_turn_usage(session_id)
        assert isinstance(running, dict)
        assert running["user_turn_number"] == 2
        assert running["context_length"] == 200_000
        assert running["cumulative"]["total_tokens"] == 1000
        assert running["cumulative"]["cached_tokens"] == 100
        # ISO-8601 UTC stamp produced by ``to_utc_iso`` — pins the date /
        # time separator so a regression that strips the time component
        # (and would still pass a bare truthiness check) is caught.
        assert isinstance(running["updated_at"], str)
        assert "T" in running["updated_at"]

    def test_upsert_overwrites_previous_snapshot(self, sm: SessionManager) -> None:
        """Each LLM call replaces the prior snapshot — we never want stale
        partial counts polluting ``get_detailed_usage``'s ``running`` field."""
        session_id = "chat_session_bbb"
        _bootstrap_session(sm, session_id)
        sm.upsert_running_turn_usage(session_id, 1, {"total_tokens": 100}, 200_000)
        sm.upsert_running_turn_usage(session_id, 1, {"total_tokens": 250}, 200_000)
        running = sm.get_running_turn_usage(session_id)
        assert running["cumulative"]["total_tokens"] == 250

    def test_upsert_silently_skips_when_db_missing(self, sm: SessionManager) -> None:
        """Early ``on_llm_end`` can fire before the SDK has materialised the
        session DB. The method must no-op so the hook never crashes the
        run loop."""
        # No bootstrap → no .db file
        sm.upsert_running_turn_usage("chat_session_ccc", 1, {"total_tokens": 50}, 0)
        assert sm.get_running_turn_usage("chat_session_ccc") is None

    def test_get_returns_none_when_not_persisted(self, sm: SessionManager) -> None:
        session_id = "chat_session_ddd"
        _bootstrap_session(sm, session_id)
        # No upsert call → row absent → ``None``
        assert sm.get_running_turn_usage(session_id) is None


class TestClear:
    def test_clear_removes_row_so_future_status_bar_reads_zero(self, sm: SessionManager) -> None:
        session_id = "chat_session_eee"
        _bootstrap_session(sm, session_id)
        sm.upsert_running_turn_usage(session_id, 1, {"total_tokens": 75}, 100_000)
        sm.clear_running_turn_usage(session_id)
        assert sm.get_running_turn_usage(session_id) is None

    def test_clear_is_idempotent(self, sm: SessionManager) -> None:
        session_id = "chat_session_fff"
        _bootstrap_session(sm, session_id)
        # First call on an empty side-table must not raise and must leave
        # the row absent; second call on the same empty state must keep it
        # absent and equally not raise. Both observations matter: a
        # surprise CREATE TABLE side-effect from a stray DDL would still
        # leave the row "absent" but would silently bloat the schema.
        sm.clear_running_turn_usage(session_id)
        sm.clear_running_turn_usage(session_id)
        assert sm.get_running_turn_usage(session_id) is None

    def test_clear_silently_skips_when_db_missing(self, sm: SessionManager) -> None:
        """When the SDK has not yet materialised the session DB the clear
        method must no-op AND a subsequent ``get_*`` must still return
        ``None`` — i.e. the method does not lazily create the file."""
        session_id = "chat_session_ggg"
        sm.clear_running_turn_usage(session_id)
        # No bootstrap happened, so the DB must not have been created as a
        # side-effect; ``get_running_turn_usage`` re-confirms by returning
        # ``None`` instead of an empty dict.
        db_path = os.path.join(sm.session_dir, f"{session_id}.db")
        assert not os.path.exists(db_path)
        assert sm.get_running_turn_usage(session_id) is None


class TestGetDetailedUsageMerges:
    def test_running_snapshot_is_folded_into_total(self, sm: SessionManager) -> None:
        """``get_detailed_usage`` must add the running row's cumulative
        counters to ``total`` so a status bar that reads ``total.total_tokens``
        sees the live progress, not just historical turns."""
        session_id = "chat_session_hhh"
        db_path = _bootstrap_session(sm, session_id)
        # Seed one persisted turn so we can verify the merge adds on top.
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO turn_usage "
                "(session_id, branch_id, user_turn_number, requests, input_tokens, "
                "output_tokens, total_tokens, input_tokens_details, output_tokens_details, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, "main", 1, 1, 100, 50, 150, "{}", "{}", "2025-01-01T00:00:00"),
            )
            conn.commit()
        sm.upsert_running_turn_usage(
            session_id,
            user_turn_number=2,
            cumulative={
                "requests": 2,
                "input_tokens": 400,
                "output_tokens": 100,
                "total_tokens": 500,
                "cached_tokens": 20,
            },
            context_length=200_000,
        )

        detailed = sm.get_detailed_usage(session_id)
        assert detailed["turn_count"] == 1  # historical turns only — running is not a row
        # ``running`` must be a fully-shaped snapshot, not just truthy —
        # downstream consumers iterate its ``cumulative`` field directly so
        # a bare ``is not None`` would not catch a regression that drops
        # the wrapping ``{"cumulative": ...}`` envelope.
        running = detailed["running"]
        assert running["cumulative"]["total_tokens"] == 500
        assert running["user_turn_number"] == 2
        # Persisted + running merged into ``total``.
        assert detailed["total"]["total_tokens"] == 150 + 500
        assert detailed["total"]["requests"] == 1 + 2
        assert detailed["total"]["cached_tokens"] == 0 + 20

    def test_running_absent_returns_none_running_field(self, sm: SessionManager) -> None:
        session_id = "chat_session_iii"
        _bootstrap_session(sm, session_id)
        detailed = sm.get_detailed_usage(session_id)
        assert detailed["running"] is None
        assert detailed["total"]["total_tokens"] == 0
