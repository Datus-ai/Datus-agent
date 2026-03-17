# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.schema_metadata.local_init."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.schema_metadata.local_init import _fill_sample_rows, store_tables
from datus.utils.constants import DBType

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


pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(namespace="test_ns", db_type=DBType.SQLITE, db_name="mydb"):
    config = MagicMock()
    config.current_namespace = namespace
    db_config = MagicMock()
    db_config.type = db_type
    db_config.database = db_name
    db_config.schema = ""
    db_config.catalog = ""
    # namespaces[namespace] can be a dict or a single DbConfig
    config.namespaces = {namespace: {db_name: db_config}}
    return config, db_config


def _make_db_manager(identifier_return="test_ns.mydb.users", sample_rows=None):
    db_manager = MagicMock()
    conn = MagicMock()
    conn.identifier.return_value = identifier_return
    conn.get_sample_rows.return_value = sample_rows or []
    conn.get_tables_with_ddl.return_value = []
    db_manager.get_conn.return_value = conn
    return db_manager, conn


# ---------------------------------------------------------------------------
# init_sqlite_schema
# ---------------------------------------------------------------------------


class TestInitSqliteSchema:
    def test_calls_get_tables_with_ddl(self):
        from datus.storage.schema_metadata.local_init import init_sqlite_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.SQLITE)
        db_manager, conn = _make_db_manager()

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_sqlite_schema(mock_store, agent_config, db_config, db_manager, table_type="table")

        conn.get_tables_with_ddl.assert_called_once()

    def test_skips_views_when_table_type_is_table(self):
        from datus.storage.schema_metadata.local_init import init_sqlite_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.SQLITE)
        db_manager, conn = _make_db_manager()

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_sqlite_schema(mock_store, agent_config, db_config, db_manager, table_type="table")

        # get_views_with_ddl should not be called when table_type="table"
        conn.get_views_with_ddl.assert_not_called()

    def test_calls_get_views_when_table_type_is_view(self):
        from datus.storage.schema_metadata.local_init import init_sqlite_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.SQLITE)
        db_manager, conn = _make_db_manager()
        conn.get_views_with_ddl.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_sqlite_schema(mock_store, agent_config, db_config, db_manager, table_type="view")

        conn.get_views_with_ddl.assert_called_once()

    def test_full_table_type_calls_both_tables_and_views(self):
        from datus.storage.schema_metadata.local_init import init_sqlite_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.SQLITE)
        db_manager, conn = _make_db_manager()
        conn.get_views_with_ddl.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_sqlite_schema(mock_store, agent_config, db_config, db_manager, table_type="full")

        conn.get_tables_with_ddl.assert_called_once()
        conn.get_views_with_ddl.assert_called_once()

    def test_database_name_set_on_tables(self):
        from datus.storage.schema_metadata.local_init import init_sqlite_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.SQLITE, db_name="testdb")
        db_manager, conn = _make_db_manager()
        conn.get_tables_with_ddl.return_value = [
            {
                "catalog_name": "",
                "schema_name": "",
                "table_name": "users",
                "definition": "CREATE TABLE users (id INT)",
                "identifier": "testdb.users",
            }
        ]
        conn.get_sample_rows.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_sqlite_schema(mock_store, agent_config, db_config, db_manager, table_type="table")

        # store_batch should be called with tables that have database_name set
        mock_store.store_batch.assert_called_once()
        stored_tables = mock_store.store_batch.call_args[0][0]
        assert stored_tables[0]["database_name"] == "testdb"


# ---------------------------------------------------------------------------
# init_duckdb_schema
# ---------------------------------------------------------------------------


class TestInitDuckdbSchema:
    def test_calls_get_tables_with_ddl(self):
        from datus.storage.schema_metadata.local_init import init_duckdb_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.DUCKDB)
        db_manager, conn = _make_db_manager()

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_duckdb_schema(
                mock_store,
                agent_config,
                db_config,
                db_manager,
                database_name="mydb",
                schema_name="",
                table_type="table",
            )

        conn.get_tables_with_ddl.assert_called_once()

    def test_uses_db_config_database_when_no_override(self):
        from datus.storage.schema_metadata.local_init import init_duckdb_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.DUCKDB, db_name="duckdb_main")
        db_config.schema = "main"
        db_manager, conn = _make_db_manager()

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ) as mock_exists:
            init_duckdb_schema(
                mock_store, agent_config, db_config, db_manager, database_name="", schema_name="", table_type="table"
            )

        # exists_table_value should be called with database_name from db_config
        mock_exists.assert_called_once()
        kwargs = mock_exists.call_args
        assert "duckdb_main" in str(kwargs)

    def test_full_type_processes_tables_and_views(self):
        from datus.storage.schema_metadata.local_init import init_duckdb_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.DUCKDB)
        db_manager, conn = _make_db_manager()
        conn.get_views_with_ddl.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_duckdb_schema(
                mock_store, agent_config, db_config, db_manager, database_name="mydb", schema_name="", table_type="full"
            )

        conn.get_tables_with_ddl.assert_called_once()
        conn.get_views_with_ddl.assert_called_once()

    def test_tables_get_database_name_set(self):
        from datus.storage.schema_metadata.local_init import init_duckdb_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type=DBType.DUCKDB, db_name="quack")
        db_manager, conn = _make_db_manager()
        conn.get_tables_with_ddl.return_value = [
            {
                "catalog_name": "",
                "schema_name": "",
                "table_name": "events",
                "definition": "CREATE TABLE events (id INT)",
                "identifier": "quack.events",
            }
        ]
        conn.get_sample_rows.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_duckdb_schema(
                mock_store,
                agent_config,
                db_config,
                db_manager,
                database_name="quack",
                schema_name="",
                table_type="table",
            )

        mock_store.store_batch.assert_called_once()
        stored_tables = mock_store.store_batch.call_args[0][0]
        assert stored_tables[0]["database_name"] == "quack"


# ---------------------------------------------------------------------------
# init_other_three_level_schema
# ---------------------------------------------------------------------------


class TestInitOtherThreeLevelSchema:
    def test_calls_get_tables_with_ddl(self):
        from datus.storage.schema_metadata.local_init import init_other_three_level_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type="starrocks")
        db_config.type = "starrocks"
        db_config.database = "mydb"
        db_config.schema = ""
        db_config.catalog = ""
        db_manager, conn = _make_db_manager()
        conn.get_tables_with_ddl.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ), patch("datus.storage.schema_metadata.local_init.connector_registry") as mock_registry:
            mock_registry.support_schema.return_value = False
            init_other_three_level_schema(
                mock_store,
                agent_config,
                db_config,
                db_manager,
                catalog_name="",
                database_name="mydb",
                table_type="table",
            )

        conn.get_tables_with_ddl.assert_called_once()

    def test_fallback_get_tables_when_no_ddl_method(self):
        from datus.storage.schema_metadata.local_init import init_other_three_level_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type="starrocks")
        db_config.type = "starrocks"
        db_config.database = "mydb"
        db_config.schema = ""
        db_config.catalog = ""
        db_manager, conn = _make_db_manager()

        # Remove get_tables_with_ddl to trigger fallback
        del conn.get_tables_with_ddl
        conn.get_tables.return_value = ["users", "orders"]
        conn.identifier.return_value = "mydb.users"

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ), patch("datus.storage.schema_metadata.local_init.connector_registry") as mock_registry:
            mock_registry.support_schema.return_value = False
            init_other_three_level_schema(
                mock_store,
                agent_config,
                db_config,
                db_manager,
                catalog_name="",
                database_name="mydb",
                table_type="table",
            )

        conn.get_tables.assert_called_once()

    def test_views_processed_when_full_type(self):
        from datus.storage.schema_metadata.local_init import init_other_three_level_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type="mysql")
        db_config.type = "mysql"
        db_config.database = "mydb"
        db_config.schema = ""
        db_config.catalog = ""
        db_manager, conn = _make_db_manager()
        conn.get_tables_with_ddl.return_value = []
        conn.get_views_with_ddl.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ), patch("datus.storage.schema_metadata.local_init.connector_registry") as mock_registry:
            mock_registry.support_schema.return_value = False
            init_other_three_level_schema(
                mock_store,
                agent_config,
                db_config,
                db_manager,
                catalog_name="",
                database_name="mydb",
                table_type="full",
            )

        conn.get_views_with_ddl.assert_called_once()

    def test_materialized_views_processed_when_full_type(self):
        from datus.storage.schema_metadata.local_init import init_other_three_level_schema

        mock_store = MagicMock()
        agent_config, db_config = _make_agent_config(db_type="starrocks")
        db_config.type = "starrocks"
        db_config.database = "mydb"
        db_config.schema = ""
        db_config.catalog = ""
        db_manager, conn = _make_db_manager()
        conn.get_tables_with_ddl.return_value = []
        conn.get_views_with_ddl.return_value = []
        conn.get_materialized_views_with_ddl.return_value = []

        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ), patch("datus.storage.schema_metadata.local_init.connector_registry") as mock_registry:
            mock_registry.support_schema.return_value = False
            init_other_three_level_schema(
                mock_store,
                agent_config,
                db_config,
                db_manager,
                catalog_name="",
                database_name="mydb",
                table_type="full",
            )

        conn.get_materialized_views_with_ddl.assert_called_once()


# ---------------------------------------------------------------------------
# init_local_schema - dispatch
# ---------------------------------------------------------------------------


class TestInitLocalSchema:
    def _make_real_agent_config(self, db_type, db_name="mydb"):
        """Build agent_config with a real DbConfig so isinstance(db_config, DbConfig) works."""
        from datus.configuration.agent_config import DbConfig

        db_config = DbConfig(type=db_type, database=db_name)
        agent_config = MagicMock()
        agent_config.current_namespace = "test_ns"
        agent_config.namespaces = {"test_ns": {db_name: db_config}}
        return agent_config, db_config

    def test_sqlite_dispatched(self):
        from datus.storage.schema_metadata.local_init import init_local_schema

        mock_store = MagicMock()
        agent_config, db_config = self._make_real_agent_config(db_type=DBType.SQLITE)
        db_manager, conn = _make_db_manager()

        with patch("datus.storage.schema_metadata.local_init.init_sqlite_schema") as mock_init_sqlite:
            init_local_schema(mock_store, agent_config, db_manager)

        mock_init_sqlite.assert_called_once()
        mock_store.after_init.assert_called_once()

    def test_duckdb_dispatched(self):
        from datus.storage.schema_metadata.local_init import init_local_schema

        mock_store = MagicMock()
        agent_config, db_config = self._make_real_agent_config(db_type=DBType.DUCKDB)
        db_manager, conn = _make_db_manager()

        with patch("datus.storage.schema_metadata.local_init.init_duckdb_schema") as mock_init_duckdb:
            init_local_schema(mock_store, agent_config, db_manager)

        mock_init_duckdb.assert_called_once()
        mock_store.after_init.assert_called_once()

    def test_other_db_dispatched(self):
        from datus.storage.schema_metadata.local_init import init_local_schema

        mock_store = MagicMock()
        agent_config, db_config = self._make_real_agent_config(db_type="mysql")
        db_manager, conn = _make_db_manager()

        with patch("datus.storage.schema_metadata.local_init.init_other_three_level_schema") as mock_init_other:
            init_local_schema(mock_store, agent_config, db_manager)

        mock_init_other.assert_called_once()
        mock_store.after_init.assert_called_once()

    def test_multiple_db_configs_iterates_all(self):
        from datus.storage.schema_metadata.local_init import init_local_schema

        mock_store = MagicMock()
        agent_config = MagicMock()
        agent_config.current_namespace = "test_ns"

        db_config_a = MagicMock()
        db_config_a.type = DBType.SQLITE
        db_config_b = MagicMock()
        db_config_b.type = DBType.SQLITE
        # Multiple db configs
        agent_config.namespaces = {"test_ns": {"db_a": db_config_a, "db_b": db_config_b}}
        db_manager, conn = _make_db_manager()

        with patch("datus.storage.schema_metadata.local_init.init_sqlite_schema") as mock_init_sqlite, patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_local_schema(mock_store, agent_config, db_manager)

        assert mock_init_sqlite.call_count == 2
        mock_store.after_init.assert_called_once()

    def test_multiple_db_with_filter_skips_others(self):
        from datus.storage.schema_metadata.local_init import init_local_schema

        mock_store = MagicMock()
        agent_config = MagicMock()
        agent_config.current_namespace = "test_ns"

        db_config_a = MagicMock()
        db_config_a.type = DBType.SQLITE
        db_config_b = MagicMock()
        db_config_b.type = DBType.SQLITE
        agent_config.namespaces = {"test_ns": {"db_a": db_config_a, "db_b": db_config_b}}
        db_manager, conn = _make_db_manager()

        with patch("datus.storage.schema_metadata.local_init.init_sqlite_schema") as mock_init_sqlite, patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_local_schema(mock_store, agent_config, db_manager, init_database_name="db_a")  # only process db_a

        # Only db_a should be initialized
        assert mock_init_sqlite.call_count == 1

    def test_empty_multiple_db_configs_returns_early(self):
        from datus.storage.schema_metadata.local_init import init_local_schema

        mock_store = MagicMock()
        agent_config = MagicMock()
        agent_config.current_namespace = "test_ns"
        agent_config.namespaces = {"test_ns": {}}  # empty
        db_manager = MagicMock()

        # Should return early without error and without calling after_init
        init_local_schema(mock_store, agent_config, db_manager)

        mock_store.after_init.assert_not_called()

    def test_unsupported_db_type_in_multi_warns(self):
        from datus.storage.schema_metadata.local_init import init_local_schema

        mock_store = MagicMock()
        agent_config = MagicMock()
        agent_config.current_namespace = "test_ns"

        db_config = MagicMock()
        db_config.type = "oracle"  # not SQLITE or DUCKDB in multi-db mode
        agent_config.namespaces = {"test_ns": {"oradb": db_config}}
        db_manager = MagicMock()

        # Should not raise, just log warning
        with patch(
            "datus.storage.schema_metadata.local_init.exists_table_value",
            return_value=({}, set()),
        ):
            init_local_schema(mock_store, agent_config, db_manager)

        mock_store.after_init.assert_called_once()
