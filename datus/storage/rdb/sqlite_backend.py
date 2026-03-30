# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Default SQLite implementation of the three-level RDB abstraction."""

import dataclasses
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Type

from datus_storage_base.rdb.base import (
    BaseRdbBackend,
    ColumnDef,
    IntegrityError,
    RdbDatabase,
    RdbTable,
    T,
    TableDefinition,
    UniqueViolationError,
    WhereClause,
    WhereOp,
    _normalize_where,
)

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _safe_ident(name: str) -> str:
    """Validate and quote a SQL identifier to prevent injection."""
    if not _IDENTIFIER_RE.fullmatch(name):
        raise DatusException(
            ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
            message=f"Invalid SQL identifier: {name}",
        )
    return name


def _safe_path_segment(value: str, field_name: str) -> str:
    """Validate a filesystem path segment to prevent directory traversal."""
    if not value:
        return value
    if not _SEGMENT_RE.fullmatch(value):
        raise DatusException(
            ErrorCode.STORAGE_FAILED,
            message=f"Invalid {field_name}: {value!r}. Only alphanumeric, underscore, dot, and hyphen are allowed.",
        )
    return value


_SQLITE_TYPE_MAP: Dict[str, str] = {
    "INTEGER": "INTEGER",
    "TEXT": "TEXT",
    "TIMESTAMP": "TEXT",
    "BOOLEAN": "INTEGER",
    "REAL": "REAL",
    "BLOB": "BLOB",
}


def _sqlite_map_type(col_type: str) -> str:
    """Map a generic column type to a SQLite-specific type."""
    return _SQLITE_TYPE_MAP.get(col_type.upper(), col_type)


def _sqlite_col_ddl(col: ColumnDef) -> str:
    """Generate DDL fragment for a single column (SQLite dialect)."""
    parts: List[str] = [col.name]

    if col.primary_key and col.autoincrement:
        parts.append("INTEGER PRIMARY KEY AUTOINCREMENT")
    else:
        parts.append(_sqlite_map_type(col.col_type))
        if col.primary_key:
            parts.append("PRIMARY KEY")
        if col.unique:
            parts.append("UNIQUE")
        if not col.nullable:
            parts.append("NOT NULL")
        if col.default is not None:
            if isinstance(col.default, str):
                parts.append(f"DEFAULT '{col.default}'")
            else:
                parts.append(f"DEFAULT {col.default}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Level 3: Table
# ---------------------------------------------------------------------------


class SqliteRdbTable(RdbTable):
    """SQLite implementation of table-level CRUD."""

    def __init__(self, database: "SqliteRdbDatabase", table_name: str) -> None:
        self._database = database
        self._table_name = table_name

    @property
    def table_name(self) -> str:
        return self._table_name

    def insert(self, record: Any) -> int:
        return self._database._insert(self._table_name, record)

    def query(
        self,
        model: Type[T],
        where: Optional[WhereClause] = None,
        columns: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
    ) -> List[T]:
        return self._database._query(self._table_name, model, where=where, columns=columns, order_by=order_by)

    def update(self, data: Dict[str, Any], where: Optional[WhereClause] = None) -> int:
        return self._database._update(self._table_name, data, where=where)

    def delete(self, where: Optional[WhereClause] = None) -> int:
        return self._database._delete(self._table_name, where=where)

    def upsert(self, record: Any, conflict_columns: List[str]) -> None:
        return self._database._upsert(self._table_name, record, conflict_columns=conflict_columns)


# ---------------------------------------------------------------------------
# Level 2: Database
# ---------------------------------------------------------------------------


class SqliteRdbDatabase(RdbDatabase):
    """SQLite implementation of database-level handle."""

    def __init__(self, db_file: str, isolation: str = "physical", datasource_id: str = "") -> None:
        self._db_file = db_file
        self._local = threading.local()
        self._isolation = isolation
        self._datasource_id = datasource_id
        os.makedirs(os.path.dirname(self._db_file) or ".", exist_ok=True)

    # ========== Internal helpers ==========

    @contextmanager
    def _auto_conn(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection: reuse transaction conn or open a fresh auto-commit one."""
        txn_conn = getattr(self._local, "txn_conn", None)
        if txn_conn is not None:
            yield txn_conn
        else:
            conn = self._open_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        """Open a new SQLite connection with row_factory set."""
        try:
            conn = sqlite3.connect(self._db_file)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Database connection error: {str(e)}") from e

    def _build_where(self, where: Optional[WhereClause]) -> tuple:
        """Build WHERE clause SQL and params from WhereClause."""
        conditions = _normalize_where(where)
        if not conditions:
            return "", []

        parts = []
        params = []
        for col, op, val in conditions:
            safe_col = _safe_ident(col)
            if op in (WhereOp.IS_NULL, WhereOp.IS_NOT_NULL):
                parts.append(f"{safe_col} {op.value}")
            else:
                parts.append(f"{safe_col} {op.value} ?")
                params.append(val)

        return " WHERE " + " AND ".join(parts), params

    def _build_order_by(self, order_by: Optional[List[str]]) -> str:
        """Build ORDER BY clause. '-col' means DESC."""
        if not order_by:
            return ""
        parts = []
        for item in order_by:
            if item.startswith("-"):
                parts.append(f"{_safe_ident(item[1:])} DESC")
            else:
                parts.append(f"{_safe_ident(item)} ASC")
        return " ORDER BY " + ", ".join(parts)

    def _generate_ddl(self, table_def: TableDefinition) -> List[str]:
        """Generate CREATE TABLE and CREATE INDEX DDL statements for SQLite."""
        statements: List[str] = []

        col_parts = [_sqlite_col_ddl(col) for col in table_def.columns]
        col_parts.extend(table_def.constraints)

        create_table = (
            f"CREATE TABLE IF NOT EXISTS {table_def.table_name} (\n" + ",\n".join(f"    {p}" for p in col_parts) + "\n)"
        )
        statements.append(create_table)

        for idx in table_def.indices:
            unique = "UNIQUE " if idx.unique else ""
            cols = ", ".join(idx.columns)
            statements.append(f"CREATE {unique}INDEX IF NOT EXISTS {idx.name} ON {table_def.table_name}({cols})")

        return statements

    # ========== RdbDatabase interface ==========

    def ensure_table(self, table_def: TableDefinition) -> SqliteRdbTable:
        ddl_statements = self._generate_ddl(table_def)
        try:
            with self._auto_conn() as conn:
                # Migrate missing columns on existing tables before running DDL
                self._migrate_missing_columns(conn, table_def)
                for stmt in ddl_statements:
                    conn.execute(stmt)
        except DatusException:
            raise
        except Exception as e:
            ddl_text = "\n".join(ddl_statements)
            logger.error(f"Auto-create table '{table_def.table_name}' failed: {e}")
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message=f"Failed to create table '{table_def.table_name}'. Please create it manually:\n\n{ddl_text}",
            ) from e
        return SqliteRdbTable(self, table_def.table_name)

    @staticmethod
    def _migrate_missing_columns(conn, table_def: TableDefinition) -> None:
        """Add columns that exist in the definition but not in the live table."""
        safe_table = _safe_ident(table_def.table_name)
        cursor = conn.execute(f"PRAGMA table_info({safe_table})")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if not existing_cols:
            return  # table doesn't exist yet — CREATE TABLE will handle it
        for col in table_def.columns:
            if col.name not in existing_cols:
                col_ddl = _sqlite_col_ddl(col)
                conn.execute(f"ALTER TABLE {safe_table} ADD COLUMN {col_ddl}")
                logger.info(f"Migrated: ALTER TABLE {safe_table} ADD COLUMN {col_ddl}")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        conn = self._open_connection()
        self._local.txn_conn = conn
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._local.txn_conn = None
            conn.close()

    def close(self) -> None:
        pass  # SQLite connections are opened/closed per operation

    # ========== Logical isolation helpers ==========

    def _inject_datasource_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Force-set datasource_id into record dict for LOGICAL mode."""
        if self._isolation == "logical" and self._datasource_id:
            data["datasource_id"] = self._datasource_id
        return data

    def _inject_datasource_where(self, where: Optional[WhereClause]) -> Optional[WhereClause]:
        """Prepend datasource_id condition to WHERE clause for LOGICAL mode."""
        if self._isolation != "logical" or not self._datasource_id:
            return where
        ds_cond = ("datasource_id", WhereOp.EQ, self._datasource_id)
        conditions = _normalize_where(where) if where else []
        return [ds_cond] + list(conditions)

    # ========== Internal CRUD (called by SqliteRdbTable) ==========

    def _insert(self, table: str, record: Any) -> int:
        data = {k: v for k, v in dataclasses.asdict(record).items() if v is not None}
        data = self._inject_datasource_record(data)
        columns = list(data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)
        sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
        try:
            with self._auto_conn() as conn:
                cursor = conn.execute(sql, tuple(data.values()))
                return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise UniqueViolationError(str(e)) from e
            raise IntegrityError(str(e)) from e

    def _query(
        self,
        table: str,
        model: Type[T],
        where: Optional[WhereClause] = None,
        columns: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
    ) -> List[T]:
        where = self._inject_datasource_where(where)
        col_str = ", ".join(columns) if columns else "*"
        where_sql, params = self._build_where(where)
        order_sql = self._build_order_by(order_by)
        sql = f"SELECT {col_str} FROM {table}{where_sql}{order_sql}"
        with self._auto_conn() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [model(**dict(row)) for row in rows]

    def _update(self, table: str, data: Dict[str, Any], where: Optional[WhereClause] = None) -> int:
        if not data:
            return 0
        where = self._inject_datasource_where(where)
        set_parts = [f"{col} = ?" for col in data.keys()]
        set_sql = ", ".join(set_parts)
        where_sql, where_params = self._build_where(where)
        sql = f"UPDATE {table} SET {set_sql}{where_sql}"
        params = list(data.values()) + where_params
        try:
            with self._auto_conn() as conn:
                cursor = conn.execute(sql, params)
                return cursor.rowcount
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise UniqueViolationError(str(e)) from e
            raise IntegrityError(str(e)) from e

    def _delete(self, table: str, where: Optional[WhereClause] = None) -> int:
        where = self._inject_datasource_where(where)
        where_sql, params = self._build_where(where)
        sql = f"DELETE FROM {table}{where_sql}"
        with self._auto_conn() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def _upsert(self, table: str, record: Any, conflict_columns: List[str]) -> None:
        data = {k: v for k, v in dataclasses.asdict(record).items() if v is not None}
        data = self._inject_datasource_record(data)
        columns = list(data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)
        sql = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"
        try:
            with self._auto_conn() as conn:
                conn.execute(sql, tuple(data.values()))
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise UniqueViolationError(str(e)) from e
            raise IntegrityError(str(e)) from e

    @property
    def db_file(self) -> str:
        return self._db_file


# ---------------------------------------------------------------------------
# Level 1: Backend
# ---------------------------------------------------------------------------


class SqliteRdbBackend(BaseRdbBackend):
    """SQLite backend — reusable singleton that produces ``SqliteRdbDatabase`` instances."""

    def __init__(self):
        self._data_dir: str = ""
        self._isolation: str = "physical"

    def initialize(self, config: Dict[str, Any]) -> None:
        self._data_dir = config.get("data_dir", "")
        self._isolation = config.get("isolation", "physical")

    def connect(self, namespace: str, store_db_name: str) -> SqliteRdbDatabase:
        safe_store = _safe_path_segment(store_db_name, "store_db_name")
        if self._isolation == "logical":
            # LOGICAL: all namespaces share datus_db/, isolation via datasource_id column
            base_path = os.path.join(self._data_dir, "datus_db")
        else:
            # PHYSICAL: each namespace gets its own directory
            db_name = f"datus_db_{namespace}" if namespace else "datus_db"
            base_path = os.path.join(self._data_dir, db_name)
        db_file = os.path.join(base_path, f"{safe_store}.db")
        return SqliteRdbDatabase(db_file, isolation=self._isolation, datasource_id=namespace)

    def close(self) -> None:
        pass  # SQLite connections are opened/closed per operation
