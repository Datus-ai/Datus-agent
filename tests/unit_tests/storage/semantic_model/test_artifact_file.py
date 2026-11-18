# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import errno
import os
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

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
def test_a_writer_that_wins_a_removed_lock_takes_the_current_one(tmp_path, monkeypatch):
    """Another process removed its lock file before unlocking, and a third one locked a fresh file.

    Winning the removed file would exclude no one, so the writer must lock the current one. The
    writer reports each file it asks to lock, so the test acts only once the writer is really
    waiting; timeouts only bound a failure.
    """
    import fcntl

    lock_path = tmp_path / ".orders.yml.dosi.lock"
    first = lock_path.open("a+b")
    fcntl.flock(first.fileno(), fcntl.LOCK_EX)

    real_flock = fcntl.flock
    requested = queue.Queue()  # inodes the writer asks to lock, in order
    writer = {}

    def reporting_flock(fd, operation):
        if threading.get_ident() == writer.get("ident") and operation == fcntl.LOCK_EX:
            requested.put(os.fstat(fd).st_ino)
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", reporting_flock)
    entered = threading.Event()
    release = threading.Event()

    def write():
        writer["ident"] = threading.get_ident()
        with _dosi_file_lock(lock_path):
            entered.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=write, daemon=True)
    thread.start()
    assert requested.get(timeout=10) == os.fstat(first.fileno()).st_ino

    lock_path.unlink()
    second = lock_path.open("a+b")
    real_flock(second.fileno(), fcntl.LOCK_EX)
    first.close()

    assert requested.get(timeout=10) == os.fstat(second.fileno()).st_ino, "the writer kept the removed file"
    assert not entered.is_set()

    lock_path.unlink()
    second.close()
    assert entered.wait(timeout=10)
    assert requested.get(timeout=10) == lock_path.stat().st_ino
    release.set()
    thread.join(timeout=10)
    assert not lock_path.exists()


@posix_only
def test_a_failed_lock_attempt_closes_its_file(tmp_path, monkeypatch):
    import fcntl

    opened = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    def no_locks(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(Path, "open", tracking_open)
    monkeypatch.setattr(fcntl, "flock", no_locks)

    with pytest.raises(OSError, match="No locks available"):
        with _dosi_file_lock(tmp_path / ".orders.yml.dosi.lock"):
            pytest.fail("entered without the lock")

    assert len(opened) == 1 and opened[0].closed


@posix_only
def test_a_lock_file_that_cannot_be_removed_does_not_fail_the_write(tmp_path, monkeypatch):
    target = tmp_path / "semantic_models" / "orders.yml"

    def read_only_directory(self, missing_ok=False):
        raise PermissionError(errno.EACCES, "Permission denied", str(self))

    with semantic_artifact_lock(target):
        target.write_text("written")
        monkeypatch.setattr(Path, "unlink", read_only_directory)

    assert target.read_text() == "written"
    assert sorted(p.name for p in target.parent.iterdir()) == [".orders.yml.dosi.lock", "orders.yml"]


def test_windows_locks_the_first_byte_and_keeps_the_file(tmp_path, monkeypatch):
    lock_path = tmp_path / ".orders.yml.dosi.lock"
    calls = []
    msvcrt = SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=lambda fd, mode, nbytes: calls.append((mode, nbytes)))

    # Only the lock runs as Windows: a Path built while os.name is "nt" would be a WindowsPath.
    with monkeypatch.context() as windows:
        windows.setattr(os, "name", "nt")
        windows.setitem(sys.modules, "msvcrt", msvcrt)
        with _dosi_file_lock(lock_path):
            assert calls == [(msvcrt.LK_LOCK, 1)]

    assert calls == [(msvcrt.LK_LOCK, 1), (msvcrt.LK_UNLCK, 1)]
    # msvcrt locks a byte range, so an empty file gets one byte to lock; the file stays.
    assert lock_path.read_bytes() == b"\0"
