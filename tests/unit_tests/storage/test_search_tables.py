# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SchemaWithValueRAG.search_tables table-name parsing logic."""

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from datus.tools.db_tools import connector_registry


@pytest.fixture(autouse=True)
def _register_test_capabilities():
    """Register dialect capabilities for testing, with snapshot/restore for isolation."""
    from datus_db_core import ConnectorRegistry

    attrs = ("_capabilities", "_uri_builders", "_context_resolvers")
    snapshots = {a: getattr(ConnectorRegistry, a).copy() for a in attrs}
    connector_registry.register_handlers("starrocks", capabilities={"catalog", "database"})
    connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
    connector_registry.register_handlers("mysql", capabilities={"database"})
    yield
    for a, snap in snapshots.items():
        setattr(ConnectorRegistry, a, snap)


def _make_mock_rag():
    """Create a mock SchemaWithValueRAG with mocked internal stores."""
    rag = MagicMock()
    rag.schema_store = MagicMock()
    rag.value_store = MagicMock()
    rag.schema_store._apply_scope_filter = lambda x: x
    rag.value_store._apply_scope_filter = lambda x: x

    # Mock the search chain to return empty arrow tables
    empty_schema = pa.table(
        {
            "identifier": pa.array([], type=pa.string()),
            "catalog_name": pa.array([], type=pa.string()),
            "database_name": pa.array([], type=pa.string()),
            "schema_name": pa.array([], type=pa.string()),
            "table_name": pa.array([], type=pa.string()),
            "table_type": pa.array([], type=pa.string()),
            "definition": pa.array([], type=pa.string()),
        }
    )
    empty_value = pa.table(
        {
            "identifier": pa.array([], type=pa.string()),
            "catalog_name": pa.array([], type=pa.string()),
            "database_name": pa.array([], type=pa.string()),
            "schema_name": pa.array([], type=pa.string()),
            "table_name": pa.array([], type=pa.string()),
            "table_type": pa.array([], type=pa.string()),
            "sample_rows": pa.array([], type=pa.string()),
        }
    )

    # Mock query_with_filter to return proper arrow tables (used by search_tables)
    rag.schema_store.query_with_filter.return_value = empty_schema
    rag.value_store.query_with_filter.return_value = empty_value

    mock_search = MagicMock()
    mock_search.where.return_value = mock_search
    mock_search.select.return_value = mock_search
    mock_search.limit.return_value = mock_search
    mock_search.to_arrow.side_effect = [empty_schema, empty_value]

    rag.schema_store.table.search.return_value = mock_search
    rag.value_store.table.search.return_value = mock_search
    return rag


def test_search_tables_3part_catalog_no_schema():
    """Cover line 393: 3-part name with catalog+database dialect (e.g., StarRocks)."""
    from datus.storage.schema_metadata.store import SchemaWithValueRAG

    rag = _make_mock_rag()
    # Call the real method on the mock object
    SchemaWithValueRAG.search_tables(rag, tables=["cat.db.tbl"], dialect="starrocks")
    # The key is that line 393 was executed (catalog supported, schema not)
    # Verify it ran without error
    rag.schema_store._ensure_table_ready.assert_called_once()


def test_search_tables_3part_with_schema():
    """Cover line 393 else branch: 3-part name with schema dialect (e.g., PostgreSQL)."""
    rag = _make_mock_rag()
    from datus.storage.schema_metadata.store import SchemaWithValueRAG

    SchemaWithValueRAG.search_tables(
        rag, tables=["db.schema.tbl"], catalog_name="", database_name="", schema_name="", dialect="postgresql"
    )
    rag.schema_store._ensure_table_ready.assert_called_once()


def test_search_tables_2part_no_schema():
    """Cover line 410: 2-part name with no-schema dialect (e.g., MySQL)."""
    rag = _make_mock_rag()
    from datus.storage.schema_metadata.store import SchemaWithValueRAG

    SchemaWithValueRAG.search_tables(
        rag, tables=["db.tbl"], catalog_name="cat", database_name="", schema_name="", dialect="mysql"
    )
    rag.schema_store._ensure_table_ready.assert_called_once()


def test_search_tables_2part_with_schema():
    """Cover line 410 else branch: 2-part name with schema dialect (e.g., PostgreSQL)."""
    rag = _make_mock_rag()
    from datus.storage.schema_metadata.store import SchemaWithValueRAG

    SchemaWithValueRAG.search_tables(
        rag,
        tables=["schema.tbl"],
        catalog_name="",
        database_name="mydb",
        schema_name="",
        dialect="postgresql",
    )
    rag.schema_store._ensure_table_ready.assert_called_once()
