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


def test_only_one_concurrent_claim_wins_and_expired_requests_stay_expired(tmp_path):
    now = 1_795_000_000.0
    request_id = f"{int(now * 1000)}-{'a' * 32}"
    barrier = Barrier(2, timeout=5)

    def make_store(key=request_id, project="project-1"):
        return ChatRequestStore(str(tmp_path), key, project_id=project, user_id="owner", clock=lambda: now)

    def claim(session):
        store = make_store()
        barrier.wait()
        return store.claim(session, f"{session}-run")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(claim, "first")
        second = pool.submit(claim, "second")
        losing_receipts = [receipt for receipt in (first.result(), second.result()) if receipt is not None]

    assert len(losing_receipts) == 1
    assert make_store().find() == losing_receipts[0]
    assert make_store(project="project-2").claim("third", "third-run") is None
    assert make_store(project="project-2").find() == ChatRequestReceipt("third", "third-run")

    now += CHAT_REQUEST_WINDOW_MS / 1000 + 1
    next_store = make_store(f"{int(now * 1000)}-{'b' * 32}")
    assert next_store.claim("fresh", "fresh-run") is None
    with pytest.raises(ChatRequestExpired):
        make_store().claim("duplicate", "duplicate-run")
