"""Bounded, disk-backed staging for complete file exports."""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary, ref

import pyarrow as pa

from datus.utils.project_files import ProjectDirectory, _same_file

ACTIVE_BULK_CANCEL: ContextVar[threading.Event | None] = ContextVar("bulk_query_cancel", default=None)
_slots_lock = threading.Lock()
_read_slots = WeakKeyDictionary()
_async_slots = WeakKeyDictionary()


def check_cancelled() -> None:
    event = ACTIVE_BULK_CANCEL.get()
    if event is not None and event.is_set():
        raise InterruptedError("File export cancelled; no complete file was published")


@asynccontextmanager
async def data_tool_slot():
    """Wait before to_thread, so queued bulk calls cannot exhaust its pool."""
    loop = asyncio.get_running_loop()
    with _slots_lock:
        slot = _async_slots.get(loop)
        semaphore = slot() if slot is not None else None
        if semaphore is None:
            semaphore = asyncio.Semaphore(2)
            _async_slots[loop] = ref(semaphore)
    async with semaphore:
        yield


@contextmanager
def bulk_slot(connector):
    """Bound concurrent file exports per connector."""
    with _slots_lock:
        registry = _read_slots
        if connector not in registry:
            registry[connector] = threading.BoundedSemaphore(2)
        semaphore = registry[connector]
    while not semaphore.acquire(timeout=0.25):
        check_cancelled()
    try:
        check_cancelled()
        yield
    finally:
        semaphore.release()


@dataclass(frozen=True)
class QueryLimits:
    batch_rows: int = 5000
    max_rows: int = 1_000_000
    max_bytes: int = 1024 * 1024 * 1024
    max_batch_bytes: int = 32 * 1024 * 1024

    @classmethod
    def from_dict(cls, raw: Any) -> QueryLimits:
        if raw is None:
            return cls()
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("bulk_query must contain only batch_rows, max_rows, max_bytes, max_batch_bytes")
        for value in raw.values():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("bulk_query limits must be positive integers")
        return cls(**raw)


class QuerySpool:
    """Stage independent Arrow batches, reconciling null-only batch types.

    Publish only after the query finishes and all result policies pass.
    """

    def __init__(self, directory: ProjectDirectory, limits: QueryLimits):
        directory.verify_current()
        self.directory = directory
        self._temporary_name = ".query-" + uuid4().hex
        self._stage = directory.child(self._temporary_name)
        try:
            self._stage_identity = os.fstat(self._stage.fd)
        except BaseException:
            try:
                self._stage.close()
            finally:
                with suppress(OSError):
                    directory.remove_child(self._temporary_name)
            raise
        self._owned_files = {}
        self.limits = limits
        self.schema: pa.Schema | None = None
        self.rows = 0
        self.bytes = 0
        self.paths: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            for name, identity in self._owned_files.items():
                with suppress(OSError):
                    self._stage.unlink(name, expected=identity)
            with suppress(OSError):
                self.directory.remove_child(self._temporary_name, self._stage_identity)
        finally:
            self._stage.close()

    def _create(self, name):
        output = self._stage.open(name, "xb")
        try:
            self._owned_files[name] = os.fstat(output.fileno())
        except BaseException:
            try:
                output.close()
            finally:
                with suppress(OSError):
                    self._stage.unlink(name)
            raise
        return output

    def add(self, table: pa.Table) -> None:
        check_cancelled()
        if not isinstance(table, pa.Table):
            raise TypeError("Streaming adapters must yield Arrow tables")
        if not table.column_names or len(set(table.column_names)) != len(table.column_names):
            raise ValueError("Query must return uniquely named columns")
        if self.schema is not None and table.column_names != self.schema.names:
            raise ValueError("Query column names changed between batches")
        if table.nbytes > self.limits.max_batch_bytes:
            raise ValueError("Query batch exceeds max_batch_bytes; reduce the configured batch_rows")
        if self.rows + table.num_rows > self.limits.max_rows:
            raise ValueError("Query exceeds max_rows; export smaller query ranges")
        if self.bytes + table.nbytes > self.limits.max_bytes:
            raise ValueError("Query exceeds max_bytes; export smaller query ranges")
        self.schema = (
            table.schema
            if self.schema is None
            else pa.unify_schemas([self.schema, table.schema], promote_options="permissive")
        )
        path = f"{len(self.paths):06d}.arrow"
        with self._create(path) as output:
            with pa.ipc.new_file(output, table.schema) as writer:
                writer.write_table(table)
            output.flush()
            size = os.fstat(output.fileno()).st_size
        self.paths.append(path)
        self.rows += table.num_rows
        self.bytes += max(table.nbytes, size)
        if self.bytes > self.limits.max_bytes:
            raise ValueError("Query staging exceeds max_bytes")

    def tables(self):
        for path in self.paths:
            check_cancelled()
            with self._stage.open(path, "rb") as source:
                if not _same_file(os.fstat(source.fileno()), self._owned_files[path]):
                    raise ValueError("Query staging file changed during execution")
                yield pa.ipc.open_file(source).read_all().cast(self.schema)

    def publish(self, name: str, metadata: dict) -> None:
        """Publish a complete file without overwriting an existing user file."""
        import pyarrow.csv as csv
        import pyarrow.parquet as parquet

        if self.schema is None:
            raise ValueError("Streaming adapter did not return result metadata")
        self.directory.verify_current()
        if self.directory.exists(name):
            raise FileExistsError("Output file already exists; choose another output_file")
        sidecar = name + ".meta.json"
        if self.directory.exists(sidecar):
            raise FileExistsError("Output metadata already exists; choose another output_file")
        suffix = Path(name).suffix.lower()
        partial = "result" + suffix
        if suffix == ".csv":
            # CSV has no type metadata; null-only columns are emitted as empty cells.
            schema = pa.schema([(f.name, pa.string() if pa.types.is_null(f.type) else f.type) for f in self.schema])
            with self._create(partial) as output:
                with csv.CSVWriter(output, schema) as writer:
                    for table in self.tables():
                        writer.write_table(table.cast(schema))
                        if output.tell() > self.limits.max_bytes:
                            raise ValueError("Export file exceeds max_bytes")
                output.flush()
                identity = os.fstat(output.fileno())
        elif suffix == ".parquet":
            with self._create(partial) as output:
                with parquet.ParquetWriter(output, self.schema) as writer:
                    for table in self.tables():
                        writer.write_table(table)
                        if output.tell() > self.limits.max_bytes:
                            raise ValueError("Export file exceeds max_bytes")
                output.flush()
                identity = os.fstat(output.fileno())
        else:
            raise ValueError("output_file must end in .csv or .parquet")
        check_cancelled()
        if identity.st_size > self.limits.max_bytes:
            raise ValueError("Export file exceeds max_bytes")
        metadata.update(row_count=self.rows, byte_count=identity.st_size, complete=True)
        # Hard-link publication is atomic and refuses overwrite, unlike rename.
        # The staging directory is created in the destination's filesystem.
        self.directory.verify_current()
        self.directory.link_from(self._stage, partial, name)
        sidecar_identity = None
        try:
            if not _same_file(self.directory.stat(name), identity):
                raise ValueError("Export file changed during publication")
            check_cancelled()
            with self.directory.open(sidecar, "xb") as raw:
                sidecar_identity = os.fstat(raw.fileno())
                with io.TextIOWrapper(raw, encoding="utf-8") as output:
                    json.dump(metadata, output, ensure_ascii=False)
            self.directory.verify_current()
            check_cancelled()
        except BaseException:
            self.directory.unlink(name, expected=identity)
            if sidecar_identity is not None:
                self.directory.unlink(sidecar, expected=sidecar_identity)
            raise
