# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException


class MockSession:
    """Mock ClickZetta session for testing."""

    def __init__(self):
        self.sql_results = {}
        self.file = Mock()
        self.close = Mock()

    def sql(self, query: str):
        # Return mock results based on query
        if "SELECT 1" in query:
            return self._make_result(pd.DataFrame({"1": [1]}))
        elif "SHOW CATALOGS" in query:
            return self._make_result(pd.DataFrame({"catalog_name": ["test_catalog"]}))
        elif "information_schema.tables" in query:
            return self._make_result(
                pd.DataFrame(
                    {
                        "table_name": ["test_table", "test_view"],
                        "table_type": ["MANAGED_TABLE", "VIEW"],
                        "comment": ["", ""],
                    }
                )
            )
        elif "information_schema.columns" in query:
            return self._make_result(
                pd.DataFrame(
                    {
                        "table_name": ["test_table"],
                        "column_name": ["id", "name"],
                        "data_type": ["INTEGER", "STRING"],
                        "comment": ["", ""],
                    }
                )
            )
        elif "SELECT * FROM" in query and "LIMIT" in query:
            return self._make_result(pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}))
        elif "INSERT" in query or "UPDATE" in query or "DELETE" in query:
            return self._make_result(pd.DataFrame({"rows_affected": [1]}))
        elif "CREATE" in query or "DROP" in query:
            return self._make_result(pd.DataFrame())
        else:
            return self._make_result(pd.DataFrame())

    def _make_result(self, df: pd.DataFrame):
        result = Mock()
        result.to_pandas.return_value = df
        return result


class MockSessionBuilder:
    """Mock session builder for testing."""

    def __init__(self):
        self._config: dict = {}

    # Make configs a fluent method as in real SDK: builder.configs({...}).create()
    def configs(self, cfg: dict | None = None):
        if cfg:
            self._config.update(cfg)
        return self

    def create(self):
        return MockSession()


@pytest.fixture
def mock_session():
    """Fixture providing a mock ClickZetta session."""
    with patch("datus.tools.db_tools.clickzetta_connector.Session") as mock_session_class:
        mock_session_class.builder = MockSessionBuilder()
        yield mock_session_class


@pytest.fixture
def connector(mock_session):
    """Fixture providing a ClickzettaConnector instance."""
    return ClickzettaConnector(
        service="test-service",
        username="test_user",
        password="test_pass",
        instance="test_instance",
        workspace="test_workspace",
        schema="test_schema",
        vcluster="test_vcluster",
    )


class TestClickzettaConnector:
    """Test cases for ClickzettaConnector."""

    def test_init_success(self, mock_session):
        """Test successful connector initialization."""
        connector = ClickzettaConnector(
            service="test-service",
            username="test_user",
            password="test_pass",
            instance="test_instance",
            workspace="test_workspace",
        )

        assert connector.service == "test-service"
        assert connector.user == "test_user"
        assert connector.instance == "test_instance"
        assert connector._workspace == "test_workspace"
        assert connector.schema_name == "PUBLIC"  # default
        assert connector.vcluster == "DEFAULT_AP"  # default
        assert connector.dialect == DBType.CLICKZETTA

    def test_init_missing_fields(self, mock_session):
        """Test initialization with missing required fields."""
        with pytest.raises(DatusException) as exc_info:
            ClickzettaConnector(
                service="",
                username="test_user",
                password="test_pass",
                instance="test_instance",
                workspace="test_workspace",
            )
        assert "Missing ClickZetta connection fields" in str(exc_info.value)

    def test_init_missing_dependency(self):
        """Test initialization when ClickZetta package is not available."""
        with patch("datus.tools.db_tools.clickzetta_connector.Session", None):
            with patch("datus.tools.db_tools.clickzetta_connector._CLICKZETTA_IMPORT_ERROR", ImportError("test")):
                with pytest.raises(DatusException) as exc_info:
                    ClickzettaConnector(
                        service="test-service",
                        username="test_user",
                        password="test_pass",
                        instance="test_instance",
                        workspace="test_workspace",
                    )
                assert "clickzetta-connector-python" in str(exc_info.value)

    def test_connect(self, connector):
        """Test database connection."""
        connector.connect()
        assert connector._session is not None
        assert connector.connection is not None

    def test_close(self, connector):
        """Test database connection closure."""
        connector.connect()
        connector.close()
        assert connector._session is None
        assert connector.connection is None

    def test_execute_query_csv(self, connector):
        """Test CSV query execution."""
        result = connector.execute_csv("SELECT 1")
        assert result.success is True
        assert "1" in result.sql_return
        assert result.row_count == 1

    def test_execute_query_pandas(self, connector):
        """Test pandas query execution."""
        result = connector.execute_pandas("SELECT 1")
        assert result.success is True
        assert isinstance(result.sql_return, pd.DataFrame)
        assert result.row_count == 1

    def test_execute_insert(self, connector):
        """Test INSERT execution."""
        result = connector.execute_insert("INSERT INTO test VALUES (1, 'test')")
        assert result.success is True
        assert result.row_count == 1

    def test_execute_update(self, connector):
        """Test UPDATE execution."""
        result = connector.execute_update("UPDATE test SET name='updated' WHERE id=1")
        assert result.success is True
        assert result.row_count == 1

    def test_execute_delete(self, connector):
        """Test DELETE execution."""
        result = connector.execute_delete("DELETE FROM test WHERE id=1")
        assert result.success is True
        assert result.row_count == 1

    def test_execute_ddl(self, connector):
        """Test DDL execution."""
        result = connector.execute_ddl("CREATE TABLE test (id INT)")
        assert result.success is True
        assert result.sql_return == "Successful"

    def test_get_catalogs(self, connector):
        """Test catalog retrieval."""
        catalogs = connector.get_catalogs()
        assert "test_catalog" in catalogs

    def test_get_databases(self, connector):
        """Test database retrieval."""
        databases = connector.get_databases()
        assert "test_workspace" in databases

    def test_get_schemas(self, connector):
        """Test schema retrieval."""
        with patch.object(connector, "_run_query") as mock_run:
            mock_run.return_value = pd.DataFrame({"table_schema": ["PUBLIC", "TEST_SCHEMA"]})
            schemas = connector.get_schemas()
            assert "PUBLIC" in schemas
            assert "TEST_SCHEMA" in schemas

    def test_get_tables(self, connector):
        """Test table retrieval."""
        with patch.object(connector, "_run_query") as mock_run:
            mock_run.return_value = pd.DataFrame({"table_name": ["test_table"], "table_type": ["MANAGED_TABLE"]})
            tables = connector.get_tables(database_name="test_workspace", schema_name="PUBLIC")
            assert "test_table" in tables

    def test_get_views(self, connector):
        """Test view retrieval."""
        with patch.object(connector, "_run_query") as mock_run:
            mock_run.return_value = pd.DataFrame({"table_name": ["test_view"], "table_type": ["VIEW"]})
            views = connector.get_views(database_name="test_workspace", schema_name="PUBLIC")
            assert "test_view" in views

    def test_get_materialized_views(self, connector):
        """Test materialized view retrieval."""
        with patch.object(connector, "_run_query") as mock_run:
            mock_run.return_value = pd.DataFrame({"table_name": ["test_mv"], "table_type": ["MATERIALIZED_VIEW"]})
            mvs = connector.get_materialized_views(database_name="test_workspace", schema_name="PUBLIC")
            assert "test_mv" in mvs

    def test_get_schema(self, connector):
        """Test table schema retrieval."""
        with patch.object(connector, "_run_query") as mock_run:
            mock_run.return_value = pd.DataFrame(
                {"column_name": ["id", "name"], "data_type": ["INTEGER", "STRING"], "comment": ["", ""]}
            )
            schema = connector.get_schema(database_name="test_workspace", schema_name="PUBLIC", table_name="test_table")
            assert len(schema) == 2
            assert schema[0]["name"] == "id"
            assert schema[1]["name"] == "name"

    def test_get_sample_rows(self, connector):
        """Test sample data retrieval."""
        with patch.object(connector, "get_tables") as mock_get_tables:
            with patch.object(connector, "_run_query") as mock_run:
                mock_get_tables.return_value = ["test_table"]
                mock_run.return_value = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
                samples = connector.get_sample_rows(database_name="test_workspace", schema_name="PUBLIC")
                assert len(samples) == 1
                assert samples[0]["table_name"] == "test_table"
                assert "Alice" in samples[0]["sample_rows"]

    def test_read_volume_file(self, connector):
        """Test volume file reading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.txt"
            test_file.write_text("test content")

            # Ensure session exists before patching file.get
            connector.connect()
            with patch.object(connector._session.file, "get"):
                with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                    mock_tmpdir.return_value.__enter__.return_value = tmp_dir
                    content = connector.read_volume_file("volume:test", "test.txt")
                    assert content == "test content"

    def test_list_volume_files(self, connector):
        """Test volume file listing."""
        # Ensure a valid session exists so we can patch its sql method safely
        connector.connect()
        with patch.object(connector._session, "sql") as mock_sql:  # type: ignore[arg-type]
            mock_result = Mock()
            mock_result.to_pandas.return_value = pd.DataFrame(
                {"relative_path": ["test1.yml", "test2.yaml", "test3.txt"]}
            )
            mock_sql.return_value = mock_result

            files = connector.list_volume_files("volume:test")
            assert "test1.yml" in files
            assert "test2.yaml" in files
            assert "test3.txt" not in files  # filtered by suffix

    def test_full_name(self, connector):
        """Test full table name generation."""
        full_name = connector.full_name(database_name="test_db", schema_name="test_schema", table_name="test_table")
        assert full_name == "test_db.test_schema.test_table"

    def test_identifier(self, connector):
        """Test table identifier generation."""
        identifier = connector.identifier(database_name="test_db", schema_name="test_schema", table_name="test_table")
        assert "test_db" in identifier
        assert "test_schema" in identifier
        assert "test_table" in identifier

    def test_error_handling(self, connector):
        """Test error handling in query execution."""
        with patch.object(connector, "_run_query") as mock_run:
            mock_run.side_effect = Exception("Test error")
            result = connector.execute_csv("SELECT 1")
            assert result.success is False
            assert "Test error" in result.error

    def test_test_connection(self, connector):
        """Test connection testing."""
        # Should not raise an exception
        connector.test_connection()

    def test_do_switch_context_schema_success(self, connector):
        """Test successful schema switching."""
        # Should not raise an exception when switching schema
        connector.do_switch_context(schema_name="new_schema")

    def test_do_switch_context_workspace_rejected(self, connector):
        """Test that workspace switching is rejected."""
        with pytest.raises(DatusException) as exc_info:
            connector.do_switch_context(database_name="different_workspace")

        error_message = str(exc_info.value)
        assert "Workspace switching" in error_message
        assert "not supported" in error_message
        assert "separate namespaces" in error_message

    def test_do_switch_context_same_workspace_allowed(self, connector):
        """Test that switching to the same workspace is allowed."""
        # Should not raise an exception when "switching" to the same workspace
        connector.do_switch_context(database_name=connector.database_name)
