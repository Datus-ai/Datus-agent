# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""RDB abstraction layer with pluggable backends."""

from datus_storage_base.rdb.base import (
    BaseRdbBackend,
    ColumnDef,
    IndexDef,
    IntegrityError,
    RdbDatabase,
    RdbTable,
    TableDefinition,
    UniqueViolationError,
    WhereOp,
)
from datus_storage_base.rdb.registry import RdbRegistry

from datus.storage.rdb.sqlite_backend import SqliteRdbBackend

# Register built-in SQLite backend
RdbRegistry.register("sqlite", SqliteRdbBackend)

# Discover external adapters via entry points
RdbRegistry.discover_adapters()

__all__ = [
    "BaseRdbBackend",
    "ColumnDef",
    "IndexDef",
    "IntegrityError",
    "UniqueViolationError",
    "RdbDatabase",
    "RdbTable",
    "TableDefinition",
    "RdbRegistry",
    "SqliteRdbBackend",
    "WhereOp",
]
