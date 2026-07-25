# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

from datus_db_core import connector_registry

from datus.prompts.database_notes import get_database_notes


def test_adapter_sql_generation_notes_are_used():
    saved_connectors = connector_registry._connectors.copy()
    saved_metadata = connector_registry._metadata.copy()
    try:
        connector_registry.register(
            "flexdb",
            object,
            sql_generation_notes="Use project.table or project.schema.table.",
        )
        assert "project.schema.table" in get_database_notes("flexdb")
    finally:
        connector_registry._connectors = saved_connectors
        connector_registry._metadata = saved_metadata


def test_snowflake_legacy_notes_are_preserved():
    notes = get_database_notes("snowflake")
    assert "double quotes" in notes
    assert "database_name and schema_name" in notes
