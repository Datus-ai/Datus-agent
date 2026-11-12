# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Concurrency-safe file primitives for semantic artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

#: Weak values so the registry does not grow for the life of the process. A sandbox pod that
#: edits thousands of files would otherwise keep a lock object per path forever. This is safe
#: because every holder keeps a strong local reference for the duration of the ``with``: an entry
#: can only be collected once nobody is inside it, and a thread arriving after that creates a
#: fresh lock with no one to exclude.
_ARTIFACT_LOCKS: "weakref.WeakValueDictionary[str, threading.RLock]" = weakref.WeakValueDictionary()
_ARTIFACT_LOCKS_GUARD = threading.Lock()


def artifact_revision(content: bytes) -> str:
    """Return the stable revision token exposed by semantic-model APIs."""

    return f"sha256:{hashlib.sha256(content).hexdigest()}"


@contextmanager
def _thread_mutation_lock(target_path: Path) -> Iterator[None]:
    """Serialize mutations of one file across threads in this process.

    One registry for every writer, deliberately. The agent framework dispatches tool calls with
    ``asyncio.gather``, so two edits of the same file can run at once, and
    semantic-model tools and generic ``FilesystemFuncTool`` can target the same file.
    A second registry would leave those writer paths free to interleave.
    """

    target_path = target_path.resolve(strict=False)
    key = str(target_path)
    with _ARTIFACT_LOCKS_GUARD:
        thread_lock = _ARTIFACT_LOCKS.get(key)
        if thread_lock is None:
            thread_lock = threading.RLock()
            _ARTIFACT_LOCKS[key] = thread_lock

    with thread_lock:
        yield


@contextmanager
def path_mutation_lock(target_path: Path) -> Iterator[None]:
    """Serialize managed model edits with native loading and other file writers."""
    path = Path(target_path)
    if "semantic_models" in path.parts and path.suffix.lower() in {".yaml", ".yml", ".json"}:
        with semantic_artifact_lock(path):
            yield
    else:
        with _thread_mutation_lock(path):
            yield


_MODEL_LOCK_DEPTH = threading.local()


def model_lock_held(path: str | Path) -> bool:
    return str(Path(path).resolve()) in getattr(_MODEL_LOCK_DEPTH, "paths", set())


@contextmanager
def semantic_artifact_lock(target_path: Path) -> Iterator[None]:
    """Load before editing, then coordinate with Dosi's atomic file replacement."""
    target_path = Path(target_path).resolve(strict=False)
    if model_lock_held(target_path):
        with _thread_mutation_lock(target_path):
            yield
        return
    if target_path.is_file():
        from datus.tools.semantic_tools.dosi.files import file_engine

        file_engine(target_path, allow_invalid=True)
    with _thread_mutation_lock(target_path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with (target_path.parent / f".{target_path.name}.dosi.lock").open("a+b") as lock:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            else:
                import msvcrt

                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            paths = getattr(_MODEL_LOCK_DEPTH, "paths", set())
            _MODEL_LOCK_DEPTH.paths = paths
            paths.add(str(target_path))
            try:
                yield
            finally:
                paths.remove(str(target_path))
                if os.name == "posix":
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                else:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def atomic_write_bytes(target_path: Path, content: bytes, *, mode: Optional[int] = None) -> None:
    """Create or replace ``target_path`` atomically while preserving its mode."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent)
    temp_path = Path(temp_name)
    try:
        target_mode = mode
        if target_mode is None:
            target_mode = stat.S_IMODE(target_path.stat().st_mode) if target_path.exists() else 0o644
        os.chmod(temp_path, target_mode)
        stream = os.fdopen(fd, "wb")
        fd = -1
        with stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target_path)
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


def atomic_write_text(target_path: Path, content: str, *, mode: Optional[int] = None) -> None:
    """Atomically write UTF-8 semantic artifact text."""

    atomic_write_bytes(target_path, content.encode("utf-8"), mode=mode)
