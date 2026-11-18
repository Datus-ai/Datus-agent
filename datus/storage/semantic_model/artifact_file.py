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
from contextlib import ExitStack, contextmanager, suppress
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
    elif "ontology" in path.parts and path.suffix.lower() in {".yaml", ".yml", ".json"}:
        ontology_dir = next(
            (parent for parent in path.parents if parent.name == "ontology" and parent.parent.name == "subject"),
            path.parent,
        )
        with ontology_bundle_lock(ontology_dir, create=True), _thread_mutation_lock(path):
            yield
    else:
        with _thread_mutation_lock(path):
            yield


_MODEL_LOCK_DEPTH = threading.local()


def model_lock_held(path: str | Path) -> bool:
    return str(Path(path).resolve()) in getattr(_MODEL_LOCK_DEPTH, "paths", set())


def ontology_dir_for_model(path: str | Path | None) -> Path | None:
    """Find the project lock for a model under ``subject/semantic_models``."""
    if not path:
        return None
    target = Path(path).resolve(strict=False)
    for parent in (target, *target.parents):
        if parent.name == "semantic_models" and parent.parent.name == "subject":
            return parent.parent / "ontology"
    return None


@contextmanager
def semantic_bundle_read_lock(model_path: str | Path | None, ontology_dir: str | Path | None = None) -> Iterator[None]:
    """Coordinate discovery with both the model project and explicit ontology inputs."""
    directories = {
        Path(directory).resolve() for directory in (ontology_dir_for_model(model_path), ontology_dir) if directory
    }
    with ExitStack() as stack:
        for directory in sorted(directories):
            stack.enter_context(ontology_bundle_lock(directory))
        yield


@contextmanager
def semantic_model_load_lock(target_path: str | Path, *, ontology_dir: str | Path | None = None) -> Iterator[None]:
    """Take the project lock before the model's thread lock, without loading it.

    Native loading may migrate a historical model while holding the GIL. Wait
    in Python before entering the native loader so a publisher can release its
    file locks. The same order applies to model editors and ontology publication.
    """
    target_path = Path(target_path).resolve(strict=False)
    with semantic_bundle_read_lock(target_path, ontology_dir), _thread_mutation_lock(target_path):
        yield


@contextmanager
def semantic_artifact_lock(target_path: Path, *, load_before_lock: bool = True) -> Iterator[None]:
    """Coordinate with Dosi's atomic file replacement.

    Editors load before taking the native file lock to allow model migration.
    Read-only dependencies must set ``load_before_lock=False`` to avoid migration.
    """
    target_path = Path(target_path).resolve(strict=False)
    if model_lock_held(target_path):
        with _thread_mutation_lock(target_path):
            yield
        return
    with semantic_model_load_lock(target_path):
        # Loading may migrate a model, so it must also wait for read-only holders.
        if load_before_lock and target_path.is_file():
            from datus.tools.semantic_tools.dosi.files import file_engine

            file_engine(target_path, allow_invalid=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with _dosi_file_lock(target_path.parent / f".{target_path.name}.dosi.lock"):
            paths = getattr(_MODEL_LOCK_DEPTH, "paths", set())
            _MODEL_LOCK_DEPTH.paths = paths
            paths.add(str(target_path))
            try:
                yield
            finally:
                paths.remove(str(target_path))


@contextmanager
def _dosi_file_lock(lock_path: Path) -> Iterator[None]:
    """Hold the lock file Dosi shares with every writer of a model, then remove it.

    The file exists only while held, the protocol Dosi follows too. A holder removes it before
    unlocking, so whoever then wins the removed file finds the path no longer names it and locks
    the current one instead. Every writer has to follow both halves: one that keeps a lock on a
    removed file excludes no one. Windows keeps the file, as these handles are opened without
    delete sharing.
    """
    if os.name != "posix":
        import msvcrt

        with lock_path.open("a+b") as lock:
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    while True:
        lock = lock_path.open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            held = os.fstat(lock.fileno())
            try:
                current = os.stat(lock_path)
            except FileNotFoundError:
                current = None
        except BaseException:
            lock.close()
            raise
        if current is not None and (current.st_dev, current.st_ino) == (held.st_dev, held.st_ino):
            break
        lock.close()
    try:
        yield
    finally:
        # A directory the writer cannot change (read-only, say) keeps the file; the write stands.
        with suppress(OSError):
            lock_path.unlink(missing_ok=True)
        lock.close()


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
    except BaseException:
        if fd >= 0:
            os.close(fd)
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(target_path: Path, content: str, *, mode: Optional[int] = None) -> None:
    """Atomically write UTF-8 semantic artifact text."""

    atomic_write_bytes(target_path, content.encode("utf-8"), mode=mode)


_ONTOLOGY_LOCK_DEPTH = threading.local()


@contextmanager
def ontology_bundle_lock(ontology_dir: str | Path | None, *, create: bool = False) -> Iterator[None]:
    """Coordinate ontology bundle publication with Agent discovery and engine reloads.

    Reentrant in one thread and coordinated between processes. POSIX readers
    and ordinary per-model editors share the lock; ``create=True`` marks a
    bundle writer and takes it exclusively. Native loaders follow the same order.
    """
    if not ontology_dir:
        yield
        return
    directory = Path(ontology_dir).resolve()
    lock_root = directory.parent.parent if directory.parent.name == "subject" else directory.parent
    key = str(lock_root if os.name == "posix" else directory)
    held = getattr(_ONTOLOGY_LOCK_DEPTH, "paths", set())
    if key in held:
        yield
        return
    # POSIX can lock the existing project directory itself. Readers therefore
    # coordinate with first publication without creating files in the project.
    # Windows needs a sidecar because its byte-range locks cannot lock a directory.
    lock_path = lock_root / f".{directory.name}.bundle.lock"
    if create:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not lock_path.parent.is_dir():
        yield
        return
    if os.name == "posix":
        import fcntl

        # Ordinary readers/editors share the project lock and keep independent
        # per-model locks. Only coordinated bundle publication is exclusive.
        # Separate open descriptions also coordinate threads in one process.
        descriptor = os.open(lock_root, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if create else fcntl.LOCK_SH)
            _ONTOLOGY_LOCK_DEPTH.paths = held
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        return
    import msvcrt

    with _thread_mutation_lock(lock_path):
        try:
            stream = lock_path.open("rb")
        except FileNotFoundError:
            stream = lock_path.open("a+b")
        # Byte-range locks accept a read-only descriptor and can extend beyond
        # EOF. Readers can coordinate with publishers without reading a locked byte.
        with stream:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            _ONTOLOGY_LOCK_DEPTH.paths = held
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
