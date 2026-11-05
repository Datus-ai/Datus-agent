# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Atomic host acceptance against the real SDK history schema."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from datus.models.accepted_turn import AcceptedOpeningSession, AcceptedTurnStore
from datus.models.session_manager import SessionManager
from datus.models.sqlite_session import DatusSQLiteSession
from datus.utils.exceptions import DatusException


def accept(store, request_id="request", message="Analyze", evidence="frozen"):
    """Accept a stable opening with immutable evidence validation."""

    def validate(existing):
        """Reject retries that replace the original evidence."""
        if existing["evidence"] != evidence:
            raise ValueError("Evidence changed")

    return store.accept(request_id, message, metadata={"evidence": evidence}, validate_existing=validate)


@pytest.mark.asyncio
async def test_concurrent_acceptance_recovers_same_receipt_and_visible_opening(tmp_path):
    """Concurrent acceptance and reopening yield one receipt and one visible user turn."""
    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda _: accept(store), range(8)))
    assert all(receipt == receipts[0] for receipt in receipts)
    reopened = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    assert accept(reopened) == receipts[0]
    history = DatusSQLiteSession(session_id="stable", db_path=str(store.path))
    assert await history.get_items() == [{"role": "user", "content": "Analyze"}]
    assert SessionManager(session_dir=str(tmp_path)).get_max_user_turn_number("stable") == 1
    history.close()


@pytest.mark.parametrize("change", ["request", "message", "evidence"])
def test_retry_cannot_rebind_request_message_or_evidence(tmp_path, change):
    """Retries cannot change request identity, input or evidence."""
    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    original = accept(store)
    with pytest.raises((DatusException, ValueError)):
        accept(
            store,
            **{
                "request": {"request_id": "another"},
                "message": {"message": "another"},
                "evidence": {"evidence": "another"},
            }[change],
        )
    assert store.read() == original


@pytest.mark.asyncio
async def test_failed_receipt_insert_rolls_back_message_and_turn(tmp_path):
    """A receipt failure leaves no partial history and permits a clean retry."""
    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    session = DatusSQLiteSession(session_id="stable", db_path=str(store.path), create_tables=True)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TABLE session_meta(session_id TEXT, key TEXT, value TEXT, PRIMARY KEY(session_id,key))"
        )
        connection.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON session_meta BEGIN SELECT RAISE(ABORT, 'receipt rejected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="receipt rejected"):
        accept(store)
    assert await session.get_items() == []
    assert store.read() is None
    assert SessionManager(session_dir=str(tmp_path)).get_max_user_turn_number("stable") == 0
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER reject_receipt")
    assert accept(store)["status"] == "pending"
    session.close()


@pytest.mark.asyncio
async def test_existing_chat_is_not_adopted_as_host_request(tmp_path):
    """An unrelated chat remains unchanged when host acceptance is refused."""
    store = AcceptedTurnStore(str(tmp_path), "ordinary", namespace="host")
    session = DatusSQLiteSession(session_id="ordinary", db_path=str(store.path), create_tables=True)
    await session.add_items([{"role": "user", "content": "Original"}])
    with pytest.raises(DatusException, match="conflict"):
        accept(store)
    assert await session.get_items() == [{"role": "user", "content": "Original"}]
    assert store.read() is None
    session.close()


def test_fence_does_not_block_history_or_metadata_and_releases_after_failure(tmp_path):
    """Execution fencing excludes duplicate dispatch without blocking receipt reads."""
    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    accept(store)
    other = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    with pytest.raises(RuntimeError):
        with store.execution_lock() as acquired:
            assert acquired
            with other.execution_lock() as duplicate:
                assert not duplicate
            store.update("running")
            assert other.read()["status"] == "running"
            raise RuntimeError("dispatcher stopped")
    with other.execution_lock() as acquired:
        assert acquired
        other.update("failed", "INTERRUPTED")
    assert store.read()["safeErrorCode"] == "INTERRUPTED"


@pytest.mark.parametrize("identity", ["../escape", "/absolute", "", "session.db", "a/b", "a\\b"])
def test_reject_path_identities(tmp_path, identity):
    """Unsafe session identifiers are rejected before filesystem writes."""
    with pytest.raises(DatusException):
        AcceptedTurnStore(str(tmp_path), identity, namespace="host")
    assert list(tmp_path.iterdir()) == []


def test_reject_symlink_and_reserved_metadata(tmp_path):
    """Acceptance cannot follow a symlink or replace reserved receipt fields."""
    target = tmp_path / "other.db"
    target.touch()
    (tmp_path / "link.db").symlink_to(target)
    with pytest.raises(DatusException):
        AcceptedTurnStore(str(tmp_path), "link", namespace="host")
    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    with pytest.raises(DatusException):
        store.accept("request", "Analyze", metadata={"status": "completed"}, validate_existing=lambda _: None)
    assert not store.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Analyze", [{"type": "input_text", "text": "Analyze"}]])
async def test_preaccepted_opening_enters_model_and_history_once_after_sdk_reset(tmp_path, content):
    """SDK input persistence and history rewrites retain one durable opening."""
    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    accept(store)
    session = AcceptedOpeningSession(session_id="stable", db_path=str(store.path), opening="Analyze")
    assert await session.get_items() == []
    session.skip_persisted_input_once([])
    await session.add_items([{"role": "user", "content": content}, {"role": "assistant", "content": "Answer"}])
    history = DatusSQLiteSession(session_id="stable", db_path=str(store.path))
    assert await history.get_items() == [
        {"role": "user", "content": "Analyze"},
        {"role": "assistant", "content": "Answer"},
    ]
    await session.replace_items([{"role": "user", "content": content}, {"role": "assistant", "content": "Summary"}])
    assert await history.get_items() == [
        {"role": "user", "content": "Analyze"},
        {"role": "assistant", "content": "Summary"},
    ]
    session.close()
    history.close()


@pytest.mark.parametrize("directory", ["question?mark", "hash#mark", "percent%20mark", "relative ?#%目录"])
def test_read_receipt_from_uri_sensitive_directory(tmp_path, monkeypatch, directory):
    """Read-only receipt lookup treats URI delimiters as literal path characters."""
    monkeypatch.chdir(tmp_path)
    store = AcceptedTurnStore(directory, "stable", namespace="host")
    receipt = accept(store)
    assert store.read() == receipt
    assert store.update("completed")["status"] == "completed"
    assert store.read()["status"] == "completed"


@pytest.mark.parametrize("operation", ["read", "update", "read_error", "update_error"])
def test_receipt_connections_close_on_success_and_failure(tmp_path, monkeypatch, operation):
    """Receipt access closes real connections on returns and invalid stored JSON."""
    from contextlib import closing

    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    accept(store)
    if operation.endswith("_error"):
        with closing(sqlite3.connect(store.path)) as conn, conn:
            conn.execute("UPDATE session_meta SET value='invalid' WHERE key='host'")
    connections = []
    connect = sqlite3.connect

    def capture(*args, **kwargs):
        """Retain connections so garbage collection cannot hide missing closes."""
        conn = connect(*args, **kwargs)
        connections.append(conn)
        if operation.startswith("read"):
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("CREATE TABLE forbidden_write(value)")
        return conn

    monkeypatch.setattr(sqlite3, "connect", capture)
    try:
        if operation.endswith("_error"):
            import json

            with pytest.raises(json.JSONDecodeError):
                store.read() if operation.startswith("read") else store.update("completed")
        else:
            result = store.read() if operation == "read" else store.update("completed")
            assert result["status"] == ("pending" if operation == "read" else "completed")
        assert len(connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
    finally:
        for conn in connections:
            conn.close()


def test_update_missing_receipt_does_not_create_database(tmp_path):
    """Updating an expired or absent session reports unavailability without recreating it."""
    store = AcceptedTurnStore(str(tmp_path), "missing", namespace="host")
    with pytest.raises(DatusException, match="Accepted turn is unavailable"):
        store.update("running")
    assert not store.path.exists()


def test_begin_failure_releases_session_connection_and_lock_reference(tmp_path, monkeypatch):
    """A competing SQLite writer cannot leak the SDK session when acceptance fails."""
    from contextlib import closing

    store = AcceptedTurnStore(str(tmp_path), "stable", namespace="host")
    seed = DatusSQLiteSession(session_id="stable", db_path=str(store.path), create_tables=True)
    seed.close()
    sessions = []
    lock_path = store.path.resolve()
    initial_count = DatusSQLiteSession._file_lock_counts.get(lock_path, 0)
    with closing(sqlite3.connect(store.path)) as blocker:

        def session_factory(**kwargs):
            """Create a real session, then hold a competing write transaction."""
            session = DatusSQLiteSession(**kwargs)
            session._get_connection().execute("PRAGMA busy_timeout=0")
            sessions.append(session)
            blocker.execute("BEGIN IMMEDIATE")
            return session

        monkeypatch.setattr("datus.models.accepted_turn.DatusSQLiteSession", session_factory)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                accept(store)
            assert DatusSQLiteSession._file_lock_counts.get(lock_path, 0) == initial_count
            with pytest.raises(RuntimeError, match="closed"):
                sessions[0]._get_connection()
        finally:
            blocker.rollback()
            for session in sessions:
                session.close()
