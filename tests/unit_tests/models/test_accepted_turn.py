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
