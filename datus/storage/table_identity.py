# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared semantic-object identity helpers."""

from __future__ import annotations

from typing import Any, Mapping

from datus.tools.db_tools import connector_registry
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode


def _dialect_value(db_type: Any) -> str:
    raw = getattr(db_type, "value", db_type)
    return str(raw or "").strip()


def build_semantic_table_identity(values: Mapping[str, Any], db_type: Any) -> str:
    """Build the table identity used by semantic table/column storage ids."""

    table_name = str(values.get("table_name") or "").strip()
    dialect = _dialect_value(db_type)
    if not dialect:
        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT,
            message_args={"error_message": "db_type is required to build semantic table identity"},
        )

    sqlite_value = _dialect_value(DBType.SQLITE)
    parts = [
        values.get("catalog_name", "") if connector_registry.support_catalog(dialect) else "",
        values.get("database_name", "")
        if connector_registry.support_database(dialect) or dialect == sqlite_value
        else "",
        values.get("schema_name", "") if connector_registry.support_schema(dialect) else "",
        table_name,
    ]
    return ".".join(str(part).strip() for part in parts if str(part).strip()) or table_name
