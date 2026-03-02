# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.schema_metadata.local_init."""

from unittest.mock import MagicMock

import pytest

from datus.storage.schema_metadata.local_init import _fill_sample_rows, store_tables

# ---------------------------------------------------------------------------
# store_tables
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestStoreTables:
    """Tests for the store_tables function."""

    def test_empty_tables_no_store(self):
        """No tables means no store_batch call."""
        mock_store = MagicMock()
        mock_connector = MagicMock()

        store_tables(
            table_lineage_store=mock_store,
            database_name="test_db",
            exists_tables={},
            exists_values=set(),
            tables=[],
            table_type="table",
            connector=mock_connector,
        )

        mock_store.store_batch.assert_not_called()

    def test_new_table_is_stored(self):
        """A table not in exists_tables is added."""
        mock_store = MagicMock()
        mock_connector = MagicMock()
        mock_connector.identifier.return_value = "db.schema.orders"
        mock_connector.get_sample_rows.return_value = []

        tables = [
            {
                "catalog_name": "",
                "database_name": "test_db",
                "schema_name": "public",
                "table_name": "orders",
                "definition": "CREATE TABLE orders (id INT)",
            }
        ]

        store_tables(
            table_lineage_store=mock_store,
            database_name="test_db",
            exists_tables={},
            exists_values=set(),
            tables=tables,
            table_type="table",
            connector=mock_connector,
        )

        mock_store.store_batch.assert_called_once()
        new_tables, new_values = mock_store.store_batch.call_args[0]
        assert len(new_tables) == 1

    def test_existing_table_same_definition_skipped(self):
        """Table with same definition in exists_tables is not re-stored."""
        mock_store = MagicMock()
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = []

        tables = [
            {
                "identifier": "db.orders",
                "catalog_name": "",
                "database_name": "test_db",
                "schema_name": "public",
                "table_name": "orders",
                "definition": "CREATE TABLE orders (id INT)",
            }
        ]

        store_tables(
            table_lineage_store=mock_store,
            database_name="test_db",
            exists_tables={"db.orders": "CREATE TABLE orders (id INT)"},
            exists_values={"db.orders"},
            tables=tables,
            table_type="table",
            connector=mock_connector,
        )

        # Both table and values already exist with same definition - no store
        mock_store.store_batch.assert_not_called()

    def test_existing_table_different_definition_updated(self):
        """Table with changed definition is updated (remove + re-add)."""
        mock_store = MagicMock()
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = []

        tables = [
            {
                "identifier": "db.orders",
                "catalog_name": "",
                "database_name": "test_db",
                "schema_name": "public",
                "table_name": "orders",
                "definition": "CREATE TABLE orders (id INT, name VARCHAR)",
            }
        ]

        store_tables(
            table_lineage_store=mock_store,
            database_name="test_db",
            exists_tables={"db.orders": "CREATE TABLE orders (id INT)"},
            exists_values={"db.orders"},
            tables=tables,
            table_type="table",
            connector=mock_connector,
        )

        mock_store.remove_data.assert_called_once()
        mock_store.store_batch.assert_called_once()
        new_tables, new_values = mock_store.store_batch.call_args[0]
        assert len(new_tables) == 1

    def test_missing_identifier_is_generated(self):
        """When identifier is missing, connector.identifier is called."""
        mock_store = MagicMock()
        mock_connector = MagicMock()
        mock_connector.identifier.return_value = "cat.db.schema.tbl"
        mock_connector.get_sample_rows.return_value = []

        tables = [
            {
                "catalog_name": "cat",
                "database_name": "db",
                "schema_name": "schema",
                "table_name": "tbl",
                "definition": "CREATE TABLE tbl (id INT)",
            }
        ]

        store_tables(
            table_lineage_store=mock_store,
            database_name="db",
            exists_tables={},
            exists_values=set(),
            tables=tables,
            table_type="table",
            connector=mock_connector,
        )

        mock_connector.identifier.assert_called_once_with(
            catalog_name="cat",
            database_name="db",
            schema_name="schema",
            table_name="tbl",
        )
        mock_store.store_batch.assert_called_once()

    def test_missing_database_name_filled(self):
        """When table has no database_name, it is filled from parameter."""
        mock_store = MagicMock()
        mock_connector = MagicMock()
        mock_connector.identifier.return_value = "db.tbl"
        mock_connector.get_sample_rows.return_value = []

        tables = [
            {
                "catalog_name": "",
                "schema_name": "",
                "table_name": "tbl",
                "definition": "CREATE TABLE tbl (id INT)",
            }
        ]

        store_tables(
            table_lineage_store=mock_store,
            database_name="my_db",
            exists_tables={},
            exists_values=set(),
            tables=tables,
            table_type="table",
            connector=mock_connector,
        )

        stored_tables = mock_store.store_batch.call_args[0][0]
        assert stored_tables[0]["database_name"] == "my_db"

    def test_existing_table_missing_value_adds_value_only(self):
        """Table exists with same definition but no value: only add sample rows."""
        mock_store = MagicMock()
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = [
            {"identifier": "db.tbl", "column": "id", "value": "1"},
        ]

        tables = [
            {
                "identifier": "db.tbl",
                "catalog_name": "",
                "database_name": "db",
                "schema_name": "",
                "table_name": "tbl",
                "definition": "CREATE TABLE tbl (id INT)",
            }
        ]

        store_tables(
            table_lineage_store=mock_store,
            database_name="db",
            exists_tables={"db.tbl": "CREATE TABLE tbl (id INT)"},
            exists_values=set(),  # No values exist
            tables=tables,
            table_type="table",
            connector=mock_connector,
        )

        # store_batch should be called with empty new_tables but non-empty new_values
        mock_store.store_batch.assert_called_once()
        new_tables, new_values = mock_store.store_batch.call_args[0]
        assert len(new_tables) == 0
        assert len(new_values) >= 1

    def test_table_type_set_on_values(self):
        """table_type is set on each value item before storing."""
        mock_store = MagicMock()
        mock_connector = MagicMock()
        mock_connector.identifier.return_value = "db.tbl"
        mock_connector.get_sample_rows.return_value = [
            {"column": "id", "value": "42"},
        ]

        tables = [
            {
                "catalog_name": "",
                "database_name": "db",
                "schema_name": "",
                "table_name": "tbl",
                "definition": "CREATE TABLE tbl (id INT)",
            }
        ]

        store_tables(
            table_lineage_store=mock_store,
            database_name="db",
            exists_tables={},
            exists_values=set(),
            tables=tables,
            table_type="view",
            connector=mock_connector,
        )

        _, new_values = mock_store.store_batch.call_args[0]
        for val in new_values:
            assert val["table_type"] == "view"


# ---------------------------------------------------------------------------
# _fill_sample_rows
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestFillSampleRows:
    """Tests for the _fill_sample_rows helper."""

    def test_fills_sample_rows(self):
        """Sample rows from connector are appended to new_values."""
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = [
            {"column": "id", "value": "1"},
            {"column": "name", "value": "Alice"},
        ]

        new_values = []
        table_data = {
            "table_name": "users",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
        }

        _fill_sample_rows(new_values, "db.users", table_data, mock_connector)

        assert len(new_values) == 2
        for val in new_values:
            assert val["identifier"] == "db.users"

    def test_empty_sample_rows(self):
        """No sample rows means nothing is added."""
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = []

        new_values = []
        table_data = {
            "table_name": "empty_table",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
        }

        _fill_sample_rows(new_values, "db.empty_table", table_data, mock_connector)

        assert len(new_values) == 0

    def test_none_sample_rows(self):
        """None from get_sample_rows means nothing is added."""
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = None

        new_values = []
        table_data = {
            "table_name": "tbl",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
        }

        _fill_sample_rows(new_values, "db.tbl", table_data, mock_connector)

        assert len(new_values) == 0

    def test_exception_is_swallowed(self):
        """Exception from get_sample_rows is caught, not raised."""
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.side_effect = RuntimeError("connection lost")

        new_values = []
        table_data = {
            "table_name": "tbl",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
        }

        # Should not raise
        _fill_sample_rows(new_values, "db.tbl", table_data, mock_connector)
        assert len(new_values) == 0

    def test_identifier_set_on_rows_without_it(self):
        """Sample rows without identifier get it set."""
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = [
            {"column": "id", "value": "1"},
        ]

        new_values = []
        table_data = {
            "table_name": "users",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
        }

        _fill_sample_rows(new_values, "db.users", table_data, mock_connector)

        assert new_values[0]["identifier"] == "db.users"

    def test_identifier_preserved_if_present(self):
        """Sample rows that already have identifier keep it."""
        mock_connector = MagicMock()
        mock_connector.get_sample_rows.return_value = [
            {"identifier": "existing.id", "column": "id", "value": "1"},
        ]

        new_values = []
        table_data = {
            "table_name": "users",
            "catalog_name": "",
            "database_name": "db",
            "schema_name": "",
        }

        _fill_sample_rows(new_values, "db.users", table_data, mock_connector)

        assert new_values[0]["identifier"] == "existing.id"
