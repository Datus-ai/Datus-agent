"""Behavior of durable chat receipts across independent service instances."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from datus.models.chat_request_store import (
    CHAT_REQUEST_WINDOW_MS,
    ChatRequestExpired,
    ChatRequestReceipt,
    ChatRequestStore,
)


def test_only_one_concurrent_dispatch_wins_and_expired_requests_stay_expired(tmp_path):
    now = 1_795_000_000.0
    request_id = f"{int(now * 1000)}-{'a' * 32}"
    barrier = Barrier(2, timeout=5)

    def make_store(key=request_id, project="project-1"):
        return ChatRequestStore(str(tmp_path), key, project_id=project, user_id="owner", clock=lambda: now)

    def dispatch(session):
        store = make_store()
        receipt = store.reserve(session, f"{session}-run")
        barrier.wait()
        return receipt, store.begin(receipt.run_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(dispatch, "first")
        second = pool.submit(dispatch, "second")
        (first_receipt, first_started), (second_receipt, second_started) = first.result(), second.result()

    assert first_receipt == second_receipt
    assert first_started != second_started
    assert make_store().find() == ChatRequestReceipt(first_receipt.session_id, first_receipt.run_id, "running")
    assert make_store(project="project-2").reserve("third", "third-run") == ChatRequestReceipt(
        "third", "third-run", "pending"
    )

    now += CHAT_REQUEST_WINDOW_MS / 1000 + 1
    next_store = make_store(f"{int(now * 1000)}-{'b' * 32}")
    assert next_store.reserve("fresh", "fresh-run").state == "pending"
    with pytest.raises(ChatRequestExpired):
        make_store().reserve("duplicate", "duplicate-run")
