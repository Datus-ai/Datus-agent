# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Legacy SQLite → DuckDB companion bridge for engine execution.

SQLite is not a SQL dialect in Dosi. Current engine builds accept it as a
storage connection and execute DuckDB SQL against the file directly. This
module remains as a compatibility fallback for older bindings: it builds a
cached DuckDB companion database whose views expose every SQLite table through
``sqlite_scan``, so those engines execute against a plain DuckDB connection
while the data stays in the original SQLite file.

The companion contains views only — no copied data — so it is cheap to
rebuild and is regenerated whenever the SQLite file (or its WAL sidecar)
moves past it. Reading the views requires DuckDB's ``sqlite`` extension,
which current DuckDB builds autoinstall/autoload on first use.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

from datus.tools.semantic_tools.exceptions import SemanticCoreException

_BRIDGE_DIR_NAME = "dosi-sqlite-bridge"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def normalize_sqlite_source(source: str) -> str:
    """Resolve a Datus SQLite datasource reference to an absolute file path.

    Datus normalizes single-file datasource URIs to ``sqlite:///{expanded
    path}`` (``agent_config._parse_single_file_db``), so the value reaching
    the runtime usually carries a scheme: ``sqlite:////abs/path`` or
    ``sqlite:///abs/path``. Bare paths and ``~`` are accepted too.
    """
    text = str(source).strip()
    if text.lower().startswith("sqlite:"):
        text = text[len("sqlite:") :]
        # ``sqlite:`` URIs stack slashes (``sqlite:////abs/path``); collapse
        # the authority/root slashes down to a single filesystem root.
        while text.startswith("//"):
            text = text[1:]
    return os.path.abspath(os.path.expanduser(text))


def sqlite_source_mtime(sqlite_path: str) -> float:
    """Latest mtime across the SQLite file and its WAL/SHM sidecars.

    WAL-mode commits (including schema changes) land in ``<db>-wal`` without
    necessarily touching the main file's mtime, so freshness checks must
    consider the sidecars too.
    """
    newest = os.path.getmtime(sqlite_path)
    for suffix in ("-wal", "-shm"):
        sidecar = sqlite_path + suffix
        if os.path.exists(sidecar):
            newest = max(newest, os.path.getmtime(sidecar))
    return newest


def _sqlite_tables(sqlite_path: str) -> list[str]:
    try:
        with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
    except sqlite3.Error as exc:
        raise SemanticCoreException(f"cannot read SQLite datasource {sqlite_path!r}: {exc}") from exc
    return [row[0] for row in rows]


def _companion_path(sqlite_path: str) -> Path:
    digest = hashlib.sha256(sqlite_path.encode("utf-8")).hexdigest()[:16]
    stem = Path(sqlite_path).stem or "db"
    bridge_dir = Path(tempfile.gettempdir()) / _BRIDGE_DIR_NAME
    bridge_dir.mkdir(parents=True, exist_ok=True)
    return bridge_dir / f"{stem}-{digest}.duckdb"


def duckdb_companion_for_sqlite(sqlite_source: str) -> str:
    """Return a DuckDB database path exposing the SQLite file's tables as views.

    The companion is cached per absolute SQLite path and rebuilt when it is
    missing or older than the SQLite file (WAL/SHM sidecars included).
    Raises ``SemanticCoreException`` with an actionable message when the
    SQLite file cannot be read or the ``duckdb`` package is unavailable.
    """
    sqlite_path = normalize_sqlite_source(sqlite_source)
    if not os.path.isfile(sqlite_path):
        raise SemanticCoreException(f"SQLite datasource file not found: {sqlite_path!r} (from {str(sqlite_source)!r})")

    companion = _companion_path(sqlite_path)
    if companion.exists() and companion.stat().st_mtime >= sqlite_source_mtime(sqlite_path):
        return str(companion)

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - guarded by dependency
        raise SemanticCoreException(
            "The `duckdb` package is required to execute Dosi queries against "
            "a SQLite datasource (it bridges SQLite into the engine's DuckDB "
            "connector); install it with `pip install duckdb`."
        ) from exc

    tables = _sqlite_tables(sqlite_path)
    if not tables:
        raise SemanticCoreException(f"SQLite datasource {sqlite_path!r} contains no tables to expose")

    # Build into a sibling temp file, then atomically replace, so a concurrent
    # reader never observes a half-written companion. The scratch name is
    # unique per build: several runtime instances in one process may race on
    # a cold cache, and each needs its own scratch file.
    scratch = companion.with_name(companion.name + f".tmp-{uuid.uuid4().hex}")
    conn = duckdb.connect(str(scratch))
    try:
        for table in tables:
            conn.execute(
                f"CREATE VIEW {_quote_ident(table)} AS "
                f"SELECT * FROM sqlite_scan({_quote_literal(sqlite_path)}, {_quote_literal(table)})"
            )
    finally:
        conn.close()
    os.replace(scratch, companion)
    return str(companion)
