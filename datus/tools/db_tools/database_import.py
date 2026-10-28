# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Import a whole DuckDB file into the datasource the agent is already connected to.

Why this exists: the datasource's DuckDB file is held open by the agent process, so any
second writer is refused with

    IOException: Could not set lock on file "...": Conflicting lock is held in python (PID N)

which makes "generate a database and hand it over" impossible from the outside. Running the
copy through the connection that already owns the lock sidesteps it entirely: ATTACH the
source read-only, replay each table's declared DDL, INSERT the rows, CHECKPOINT, DETACH.

Replaying the DDL (rather than ``CREATE TABLE AS SELECT``) is what preserves PRIMARY KEY,
UNIQUE, NOT NULL and FOREIGN KEY. DuckDB has no ``ALTER TABLE ADD CONSTRAINT``, so the
constraints must be present at creation time, and tables must be created and filled
parents-first.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabaseImportError(Exception):
    """Raised when the source file cannot be imported."""


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _rewrite_create(stmt: str, table: str) -> str:
    """Point a source CREATE TABLE statement at the target catalog.

    DuckDB emits the statement unqualified (``CREATE TABLE customers(...)``), which already
    lands in the connection's default catalog. Any explicit ``CREATE TABLE src.main.t`` form
    is stripped back to the bare table name so the copy never writes into the attached
    read-only source.
    """
    return re.sub(
        r"^\s*CREATE\s+(OR\s+REPLACE\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\w.\"`]+",
        f"CREATE TABLE {_quote(table)}",
        stmt,
        count=1,
        flags=re.IGNORECASE,
    )


def _dependency_order(tables: Sequence[str], fks: Dict[str, Set[str]]) -> List[str]:
    """Parents before children, so an FK target exists when the child is created and filled.

    A cycle (or a self-reference) cannot be ordered; those tables are appended at the end and
    will simply fall back to a constraint-free copy when their creation is rejected.
    """
    remaining, ordered, placed = list(tables), [], set()
    in_scope = set(tables)
    while remaining:
        # Only dependencies that are themselves being imported can be waited on; a subset import
        # whose parent is out of scope would otherwise never become ready and fall into the cycle
        # branch, silently losing the parents-first guarantee for the tables that do have one.
        ready = [t for t in remaining if not ((fks.get(t, set()) & in_scope) - placed - {t})]
        if not ready:  # cycle - emit the rest in the original order
            ordered.extend(remaining)
            break
        for t in ready:
            ordered.append(t)
            placed.add(t)
        remaining = [t for t in remaining if t not in placed]
    return ordered


def _source_metadata(
    con: Any, alias: str
) -> Tuple[List[str], Dict[str, str], Dict[str, Set[str]], Dict[str, str], Dict[Tuple[str, str], str]]:
    rows = con.execute(
        "SELECT table_name, sql, comment FROM duckdb_tables() "
        f"WHERE database_name = {_sql_literal(alias)} ORDER BY table_name"
    ).fetchall()
    tables = [r[0] for r in rows]
    create_sql = {r[0]: r[1] for r in rows if r[1]}
    table_comments = {r[0]: r[2] for r in rows if r[2]}

    fks: Dict[str, Set[str]] = {}
    try:
        for t, ctype, text in con.execute(
            "SELECT table_name, constraint_type, constraint_text FROM duckdb_constraints() "
            f"WHERE database_name = {_sql_literal(alias)}"
        ).fetchall():
            if ctype != "FOREIGN KEY":
                continue
            m = re.search(r"REFERENCES\s+[\"`]?([\w.]+)[\"`]?", text or "", re.IGNORECASE)
            if m:
                fks.setdefault(t, set()).add(m.group(1).split(".")[-1])
    except Exception as e:  # noqa: BLE001 - constraint introspection is best-effort
        logger.debug("could not read source constraints: %s", e)

    column_comments: Dict[Tuple[str, str], str] = {}
    try:
        for t, c, note in con.execute(
            "SELECT table_name, column_name, comment FROM duckdb_columns() "
            f"WHERE database_name = {_sql_literal(alias)} AND comment IS NOT NULL"
        ).fetchall():
            column_comments[(t, c)] = note
    except Exception as e:  # noqa: BLE001
        logger.debug("could not read source column comments: %s", e)

    return tables, create_sql, fks, table_comments, column_comments


def _external_dependents(con: Any, importing: Set[str]) -> Dict[str, Set[str]]:
    """Target tables in `importing` that something outside `importing` depends on.

    Only foreign keys held by tables this import does not itself replace can block a DROP; the
    ones inside the set are dropped children-first. Returns {blocked table: {referencing tables}}.
    """
    blocked: Dict[str, Set[str]] = {}
    try:
        rows = con.execute(
            "SELECT table_name, constraint_type, constraint_text FROM duckdb_constraints() "
            "WHERE database_name = current_database()"
        ).fetchall()
    except Exception as e:  # noqa: BLE001 - without introspection DuckDB still refuses the DROP itself
        logger.debug("could not read target constraints: %s", e)
        return blocked
    for child, ctype, text in rows:
        if ctype != "FOREIGN KEY" or child in importing:
            continue
        m = re.search(r"REFERENCES\s+[\"`]?([\w.]+)[\"`]?", text or "", re.IGNORECASE)
        if m:
            parent = m.group(1).split(".")[-1]
            if parent in importing:
                blocked.setdefault(parent, set()).add(child)
    return blocked


def _refuse_importing_the_target_itself(con: Any, source_path: Path) -> None:
    """Stop an import whose source IS the datasource's own file.

    DuckDB answers this with ``Binder Error: Unique file handle conflict: Cannot attach
    "datus_import_..." - the database file "..." is already attached by database "datasource"``,
    which names neither the cause nor the fix. A production run hit it after generating the
    "canonical artifact" straight onto the live datasource path, and the same overwrite left a
    19 MB ``.nfs*`` orphan behind because the file was unlinked while this process held it open.

    Best-effort: a connection that cannot answer the question is left to ATTACH and fail as before.
    """
    try:
        rows = con.execute("SELECT path FROM duckdb_databases() WHERE database_name = current_database()").fetchall()
    except Exception as e:  # noqa: BLE001 - this is a better error message, never a requirement
        logger.debug("could not resolve the current database file: %s", e)
        return

    current = next((r[0] for r in rows if r and r[0]), None)
    if not current:
        return
    try:
        same = Path(current).resolve() == source_path.resolve()
    except OSError:
        return
    if same:
        raise DatabaseImportError(
            f"{source_path.name} IS the file this datasource is open on, so there is nothing to "
            "import - the rows are already served. Generate to a separate path (the skill uses "
            "data/_build/) and import that, and never write the datasource file directly: this "
            "process holds it open, so overwriting it corrupts the handle and can strand the old "
            "copy on disk."
        )


def import_duckdb_file(
    con: Any,
    source_path: Path,
    *,
    mode: str = "replace",
    tables: Optional[Sequence[str]] = None,
    keep_constraints: bool = True,
    copy_comments: bool = True,
) -> Dict[str, Any]:
    """Copy every table of ``source_path`` into the database ``con`` is connected to.

    Args:
        con: An open DuckDB connection to the *target*, held by whoever owns the file lock.
        source_path: The DuckDB file to read. Must exist and be readable by this process.
        mode: ``replace`` drops a same-named target table first; ``skip_existing`` leaves it
            alone and reports it as skipped.
        tables: Restrict the import to these source tables. Default: all of them.
        keep_constraints: Replay the source DDL so keys survive. When False (or when a
            statement is rejected) the table is copied with CREATE TABLE AS SELECT and
            arrives without constraints.
        copy_comments: Carry table and column comments across. The agent reads them to
            understand the schema, so this is on by default.

    Returns:
        A dict with ``imported`` (per-table row counts), ``skipped``, ``degraded``
        (tables that lost their constraints, with the reason) and ``table_count``.

    Raises:
        DatabaseImportError: The file is missing, holds no table, or the requested tables
            are not in it.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise DatabaseImportError(f"Source database not found: {source_path}")
    if not source_path.is_file():
        raise DatabaseImportError(f"Source path is not a file: {source_path}")
    if mode not in ("replace", "skip_existing"):
        raise DatabaseImportError(f"Unknown mode {mode!r}; expected 'replace' or 'skip_existing'")

    alias = f"datus_import_{uuid.uuid4().hex[:8]}"
    imported: Dict[str, int] = {}
    skipped: List[str] = []
    degraded: List[str] = []
    refused: List[str] = []

    _refuse_importing_the_target_itself(con, source_path)
    try:
        con.execute(f"ATTACH {_sql_literal(str(source_path.resolve()))} AS {alias} (READ_ONLY)")
    except Exception as e:  # noqa: BLE001 - turn a configuration refusal into an actionable message
        msg = str(e)
        if "enable_external_access" in msg or "file system operations are disabled" in msg:
            raise DatabaseImportError(
                "This datasource runs with enable_external_access=false, so DuckDB refuses to "
                "ATTACH any file and the database cannot be imported. Ask the operator to allow "
                f"external access for this datasource, or load the tables another way. ({msg.splitlines()[0]})"
            ) from e
        raise DatabaseImportError(f"Cannot attach {source_path.name}: {msg.splitlines()[0]}") from e
    try:
        all_tables, create_sql, fks, table_comments, column_comments = _source_metadata(con, alias)
        if not all_tables:
            raise DatabaseImportError(f"No table found in {source_path.name}")

        if tables:
            unknown = [t for t in tables if t not in all_tables]
            if unknown:
                raise DatabaseImportError(
                    f"Not in {source_path.name}: {', '.join(unknown)}. Available: {', '.join(all_tables)}"
                )
            wanted = [t for t in all_tables if t in set(tables)]
        else:
            wanted = all_tables

        existing = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE database_name = current_database()"
            ).fetchall()
        }

        ordered = _dependency_order(wanted, fks)

        if mode == "replace":
            # A target table outside this import may hold a foreign key to one we are about to
            # replace, and DuckDB then refuses the DROP. Detect that before touching anything:
            # discovering it mid-import leaves the datasource half-replaced, and the CREATE OR
            # REPLACE fallback cannot recover because it is blocked by the same dependency.
            blockers = _external_dependents(con, set(ordered))
            if blockers:
                raise DatabaseImportError(
                    "Cannot replace "
                    + ", ".join(sorted(blockers))
                    + ": "
                    + "; ".join(f"{t} is referenced by {', '.join(sorted(d))}" for t, d in sorted(blockers.items()))
                    + ". Drop the referencing table(s) first, or import with mode='skip_existing'."
                )
            # Drop children before parents. A second import of the same dataset would otherwise fail
            # on "Could not drop the table because this table is main key table of ...": the previous
            # run's child table still holds a foreign key to the parent being replaced.
            for t in reversed(ordered):
                if _IDENT_RE.match(t) and t in existing:
                    try:
                        con.execute(f"DROP TABLE IF EXISTS {_quote(t)}")
                    except Exception as e:  # noqa: BLE001 - a table we cannot drop is handled below
                        logger.debug("pre-drop of %s failed: %s", t, e)

        for t in ordered:
            if not _IDENT_RE.match(t):
                # Everything below interpolates the name; refuse anything that is not a plain
                # identifier rather than building SQL out of it. This table is not imported at
                # all, so it is reported separately from `degraded` (imported, constraints lost).
                refused.append(f"{t}: not a plain identifier")
                continue
            if t in existing and mode == "skip_existing":
                skipped.append(t)
                continue

            src = f"{alias}.{_quote(t)}"
            stmt = create_sql.get(t) if keep_constraints else None
            done = False
            if stmt:
                try:
                    con.execute(f"DROP TABLE IF EXISTS {_quote(t)}")
                    con.execute(_rewrite_create(stmt, t))
                    con.execute(f"INSERT INTO {_quote(t)} SELECT * FROM {src}")
                    done = True
                except Exception as e:  # noqa: BLE001 - fall back rather than abort the import
                    reason = str(e).splitlines()[0][:160]
                    degraded.append(f"{t}: {reason}")
                    logger.info("constraint-preserving import failed for %s: %s", t, reason)
                    try:
                        con.execute(f"DROP TABLE IF EXISTS {_quote(t)}")
                    except Exception as drop_err:  # noqa: BLE001 - CREATE OR REPLACE below still works
                        logger.debug("cleanup drop of %s failed: %s", t, drop_err)
            if not done:
                con.execute(f"CREATE OR REPLACE TABLE {_quote(t)} AS SELECT * FROM {src}")

            imported[t] = con.execute(f"SELECT count(*) FROM {_quote(t)}").fetchone()[0]

        if copy_comments:
            for t, note in table_comments.items():
                if t in imported:
                    try:
                        con.execute(f"COMMENT ON TABLE {_quote(t)} IS {_sql_literal(note)}")
                    except Exception as e:  # noqa: BLE001
                        logger.debug("table comment failed for %s: %s", t, e)
            for (t, c), note in column_comments.items():
                if t in imported:
                    try:
                        con.execute(f"COMMENT ON COLUMN {_quote(t)}.{_quote(c)} IS {_sql_literal(note)}")
                    except Exception as e:  # noqa: BLE001
                        logger.debug("column comment failed for %s.%s: %s", t, c, e)

        # Flush to the file so the data survives a reconnect, not just this session.
        try:
            con.execute("CHECKPOINT")
        except Exception as e:  # noqa: BLE001 - a checkpoint refusal is not a failed import
            logger.debug("checkpoint after import failed: %s", e)
    finally:
        try:
            con.execute(f"DETACH {alias}")
        except Exception as e:  # noqa: BLE001
            logger.debug("detach %s failed: %s", alias, e)

    return {
        "imported": imported,
        "table_count": len(imported),
        "total_rows": sum(imported.values()),
        "skipped": skipped,
        "degraded": degraded,
        "refused": refused,
    }


def read_generator_meta(source_path: Path) -> Optional[Dict[str, Any]]:
    """Load the structural metadata the generator drops next to a database.

    ``gen-datasource`` writes ``.<stem>.meta.json`` (table roles, declared keys, strict-DDL
    flag) alongside the database. The quality check reuses it instead of inferring a second,
    conflicting view of the schema. Returns None when it is absent or unreadable.

    Accepts either the database path (the sidecar is derived from it) or the metadata file
    itself - the tool documents the latter, and deriving a sidecar from a sidecar silently
    yielded nothing.
    """
    import json

    p = Path(source_path).resolve()
    f = p if p.name.endswith(".meta.json") else p.parent / f".{p.stem}.meta.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except Exception as e:  # noqa: BLE001 - metadata is an optimisation, never a requirement
        logger.debug("could not read generator metadata %s: %s", f, e)
        return None
