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
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_ARTIFACT_LOCKS: dict[str, threading.RLock] = {}
_ARTIFACT_LOCKS_GUARD = threading.Lock()


def artifact_revision(content: bytes) -> str:
    """Return the stable revision token exposed by semantic-model APIs."""

    return f"sha256:{hashlib.sha256(content).hexdigest()}"


@contextmanager
def semantic_artifact_lock(target_path: Path) -> Iterator[None]:
    """Serialize one semantic artifact across threads in this process."""

    target_path = target_path.resolve(strict=False)
    key = str(target_path)
    with _ARTIFACT_LOCKS_GUARD:
        thread_lock = _ARTIFACT_LOCKS.setdefault(key, threading.RLock())

    with thread_lock:
        yield


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
