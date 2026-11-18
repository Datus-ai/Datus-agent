# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import threading

import pytest

from datus.storage.semantic_model.artifact_file import _dosi_file_lock, semantic_artifact_lock

posix_only = pytest.mark.skipif(os.name != "posix", reason="Windows keeps the lock file")


@posix_only
def test_semantic_artifact_lock_leaves_nothing_behind(tmp_path):
    target = tmp_path / "semantic_models" / "orders.yml"

    with semantic_artifact_lock(target):
        assert not target.exists()
        assert [p.name for p in target.parent.iterdir()] == [".orders.yml.dosi.lock"]

    assert not target.exists()
    assert list(target.parent.iterdir()) == []


@posix_only
def test_a_writer_that_wins_a_removed_lock_takes_the_current_one(tmp_path):
    """Another process removed its lock file before unlocking, and a third one locked a fresh file.

    Winning the removed file would exclude no one, so the writer must wait on the current one.
    """
    import fcntl

    lock_path = tmp_path / ".orders.yml.dosi.lock"
    first = lock_path.open("a+b")
    fcntl.flock(first.fileno(), fcntl.LOCK_EX)
    entered = threading.Event()
    release = threading.Event()

    def writer():
        with _dosi_file_lock(lock_path):
            entered.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=writer)
    thread.start()
    # Let the writer open `first` and wait on it.
    assert not entered.wait(timeout=0.2)
    lock_path.unlink()
    second = lock_path.open("a+b")
    fcntl.flock(second.fileno(), fcntl.LOCK_EX)
    first.close()

    assert not entered.wait(timeout=0.3), "the writer entered while another one held the lock"

    lock_path.unlink()
    second.close()
    assert entered.wait(timeout=10)
    assert lock_path.exists()
    release.set()
    thread.join(timeout=10)
    assert not lock_path.exists()
