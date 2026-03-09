# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Abstract base classes for the relational database backend.

Three-level abstraction:

    BaseRdbBackend          (lifecycle: initialize, connect, close)
        │
        └─ connect(ns, db) → RdbDatabase   (DDL + transaction)
                                │
                                └─ ensure_table(def) → RdbTable(ABC)  (table-level CRUD)
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, TypeVar, Union


class WhereOp(Enum):
    """Supported comparison operators for WHERE clauses."""

    EQ = "="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


WhereCondition = Tuple[str, WhereOp, Any]
WhereClause = Union[List[WhereCondition], Dict[str, Any]]


class IntegrityError(Exception):
    """Backend-agnostic constraint violation (wraps sqlite3.IntegrityError etc.)."""


class UniqueViolationError(IntegrityError):
    """Raised when a UNIQUE or PRIMARY KEY constraint is violated."""


T = TypeVar("T")


@dataclass
class ColumnDef:
    """Dialect-neutral column definition."""

    name: str
    col_type: str = "TEXT"  # "INTEGER", "TEXT", "TIMESTAMP", "BOOLEAN"
    primary_key: bool = False
    autoincrement: bool = False
    nullable: bool = True
    default: Any = None
    unique: bool = False


@dataclass
class IndexDef:
    """Index definition."""

    name: str
    columns: List[str]
    unique: bool = False


@dataclass
class TableDefinition:
    """Dialect-neutral table definition."""

    table_name: str
    columns: List[ColumnDef]
    indices: List[IndexDef] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


def _normalize_where(where: Optional[WhereClause]) -> List[WhereCondition]:
    """Convert a WhereClause (dict or tuple list) into a list of WhereConditions."""
    if where is None:
        return []
    if isinstance(where, dict):
        return [(col, WhereOp.EQ, val) for col, val in where.items()]
    return list(where)


class RdbTable(ABC):
    """Abstract table-level CRUD handle.

    Returned by ``RdbDatabase.ensure_table()`` so that callers no longer
    need to pass the table name through every CRUD call.
    """

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the bound table name."""

    @abstractmethod
    def insert(self, record: Any) -> int:
        """Insert a dataclass record and return lastrowid.

        Raises:
            UniqueViolationError: When a UNIQUE constraint is violated.
            IntegrityError: On other integrity constraint violations.
        """

    @abstractmethod
    def query(
        self,
        model: Type[T],
        where: Optional[WhereClause] = None,
        columns: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
    ) -> List[T]:
        """Query rows and return typed model instances."""

    @abstractmethod
    def update(self, data: Dict[str, Any], where: Optional[WhereClause] = None) -> int:
        """Update rows and return affected count.

        Raises:
            UniqueViolationError: When a UNIQUE constraint is violated.
            IntegrityError: On other integrity constraint violations.
        """

    @abstractmethod
    def delete(self, where: Optional[WhereClause] = None) -> int:
        """Delete rows and return affected count."""

    @abstractmethod
    def upsert(self, record: Any, conflict_columns: List[str]) -> None:
        """Insert or replace a dataclass record."""


class RdbDatabase(ABC):
    """Abstract database-level handle providing DDL and transaction support.

    Obtained via ``BaseRdbBackend.connect()``.
    """

    @abstractmethod
    def ensure_table(self, table_def: TableDefinition) -> RdbTable:
        """Auto-create table and return an ``RdbTable`` handle."""

    @abstractmethod
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for explicit transactions.

        Individual CRUD calls auto-commit. Within a transaction block,
        commit happens on successful exit; rollback on exception.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this database connection."""


class BaseRdbBackend(ABC):
    """Abstract base for relational database backends (lifecycle only).

    A backend is a reusable singleton that can produce ``RdbDatabase``
    connections via ``connect()``.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the backend with configuration."""

    @abstractmethod
    def connect(self, namespace: str, store_db_name: str) -> RdbDatabase:
        """Create or return a database-level handle for the given namespace/store."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this backend."""
