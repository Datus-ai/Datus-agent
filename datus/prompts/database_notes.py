# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Resolve adapter-provided SQL generation guidance with legacy fallbacks."""

from typing import Any

from datus.tools.db_tools import connector_registry


def get_database_notes(database_type: str) -> str:
    getter = getattr(connector_registry, "get_sql_generation_notes", None)
    notes: Any = getter(database_type) if callable(getter) else None
    if callable(notes):
        notes = notes()
    if notes:
        return f"\n{str(notes).strip()}"
    if (database_type or "").lower() == "snowflake":
        return (
            "\nEnclose all column names in double quotes to comply with Snowflake syntax requirements and avoid errors. "
            "When referencing table names in Snowflake SQL, include both the database_name and schema_name."
        )
    return ""
