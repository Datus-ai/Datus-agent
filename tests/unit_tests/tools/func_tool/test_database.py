"""
Test cases for DBFuncTool compressor model_name initialization and execute_ddl.
"""

from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool


class TestDBFuncToolCompressorModelName:
    """Verify that DBFuncTool uses agent_config's model name for DataCompressor."""

    def test_compressor_uses_agent_config_model(self):
        """When agent_config is provided, compressor should use its active model name."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        mock_config = Mock()
        mock_config.active_model.return_value.model = "claude-sonnet-4"

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_connector, agent_config=mock_config)

        assert tool.compressor.model_name == "claude-sonnet-4"

    def test_compressor_defaults_without_agent_config(self):
        """When agent_config is None, compressor should fall back to gpt-3.5-turbo."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG"),
            patch("datus.tools.func_tool.database.SemanticModelRAG"),
        ):
            tool = DBFuncTool(mock_connector)

        assert tool.compressor.model_name == "gpt-3.5-turbo"


class TestDBFuncToolExecuteDDL:
    """Tests for DBFuncTool.execute_ddl method."""

    def _make_tool(self, connector):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector)

    def test_execute_ddl_success(self):
        """Test successful DDL execution."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock()
        ddl_result.success = True
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 1
        assert result.result["message"] == "DDL executed successfully"
        assert result.result["sql"] == "CREATE TABLE test (id INT)"

    def test_execute_ddl_failure(self):
        """Test DDL execution returning error."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock()
        ddl_result.success = False
        ddl_result.error = "table already exists"
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 0
        assert "table already exists" in result.error

    def test_execute_ddl_unsupported(self):
        """Test DDL on connector without execute_ddl support."""
        mock_connector = Mock(spec=[])  # No attributes at all
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases = Mock(return_value=[])

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 0
        assert "does not support DDL" in result.error

    def test_execute_ddl_not_in_available_tools(self):
        """Verify that execute_ddl is NOT in the default available_tools() list."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        tool = self._make_tool(mock_connector)
        tool_names = [t.name for t in tool.available_tools()]

        assert "execute_ddl" not in tool_names

    def test_execute_ddl_exception_handling(self):
        """Test DDL execution when connector raises an exception."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_ddl.side_effect = RuntimeError("connection lost")

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE test (id INT)")

        assert result.success == 0
        assert "connection lost" in result.error


class TestExecuteDDLStatementValidation:
    """Tests for execute_ddl SQL statement type validation."""

    def _make_tool(self, connector=None):
        if connector is None:
            connector = Mock()
            connector.dialect = "sqlite"
            connector.get_databases.return_value = []
            ddl_result = Mock()
            ddl_result.success = True
            connector.execute_ddl.return_value = ddl_result
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector)

    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE TABLE test (id INT)",
            "CREATE TABLE IF NOT EXISTS test (id INT)",
            "CREATE TABLE test AS SELECT * FROM other",
            "CREATE SCHEMA staging",
            "CREATE SCHEMA IF NOT EXISTS staging",
            "  CREATE TABLE test (id INT)",
            "ALTER TABLE test ADD COLUMN name TEXT",
            "DROP TABLE test",
            "DROP TABLE IF EXISTS test",
            "CREATE VIEW v AS SELECT 1",
            "DROP VIEW v",
            "CREATE OR REPLACE VIEW v AS SELECT 1",
            "CREATE TEMPORARY TABLE tmp AS SELECT 1",
            "CREATE TEMP TABLE tmp (id INT)",
        ],
    )
    def test_allowed_ddl_statements(self, sql):
        """Allowed DDL statement types should pass validation."""
        tool = self._make_tool()
        result = tool.execute_ddl(sql)
        assert result.success == 1

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM users",
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE users SET name='x'",
            "DELETE FROM users",
            "TRUNCATE TABLE users",
            "GRANT ALL ON users TO public",
            "CREATE OR REPLACE FUNCTION test() RETURNS void",
            "CREATE PROCEDURE test() BEGIN END",
        ],
    )
    def test_rejected_non_ddl_statements(self, sql):
        """Non-DDL statements should be rejected."""
        tool = self._make_tool()
        result = tool.execute_ddl(sql)
        assert result.success == 0
        assert "Only DDL statements are allowed" in result.error

    def test_rejected_multi_statement(self):
        """Multi-statement SQL should be rejected."""
        tool = self._make_tool()
        result = tool.execute_ddl("CREATE TABLE t1 (id INT); DROP TABLE users")
        assert result.success == 0
        assert "Multi-statement" in result.error

    def test_rejected_empty_sql(self):
        """Empty SQL should be rejected."""
        tool = self._make_tool()
        result = tool.execute_ddl("   ")
        assert result.success == 0
        assert "Empty SQL" in result.error

    def test_sql_comments_stripped(self):
        """SQL comments should be stripped before validation."""
        tool = self._make_tool()
        result = tool.execute_ddl("-- comment\nCREATE TABLE test (id INT)")
        assert result.success == 1


class TestDBFuncToolExecuteWrite:
    """Tests for DBFuncTool.execute_write method."""

    def _make_tool(self, connector=None):
        if connector is None:
            connector = Mock()
            connector.dialect = "sqlite"
            connector.get_databases.return_value = []
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector)

    def test_execute_write_insert_success(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        write_result = Mock(success=True, row_count=2)
        mock_connector.execute_insert.return_value = write_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_write("INSERT INTO users VALUES (1), (2)")

        assert result.success == 1
        assert result.result["sql_type"] == "insert"
        assert result.result["row_count"] == 2

    def test_execute_write_update_success(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        write_result = Mock(success=True, row_count=3)
        mock_connector.execute_update.return_value = write_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_write("UPDATE users SET active = 1")

        assert result.success == 1
        assert result.result["sql_type"] == "update"
        assert result.result["row_count"] == 3

    def test_execute_write_delete_success(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        write_result = Mock(success=True, row_count=1)
        mock_connector.execute_delete.return_value = write_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_write("DELETE FROM users WHERE id = 1")

        assert result.success == 1
        assert result.result["sql_type"] == "delete"
        assert result.result["row_count"] == 1

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM users",
            "CREATE TABLE users (id INT)",
            "ALTER TABLE users ADD COLUMN email TEXT",
        ],
    )
    def test_execute_write_rejects_non_dml(self, sql):
        tool = self._make_tool()
        result = tool.execute_write(sql)

        assert result.success == 0
        assert "Only single-statement writes" in result.error

    def test_execute_write_rejects_merge_for_now(self):
        tool = self._make_tool()
        result = tool.execute_write(
            "MERGE INTO target t USING source s ON t.id = s.id WHEN MATCHED THEN UPDATE SET name = s.name"
        )

        assert result.success == 0
        assert "MERGE statements are not supported" in result.error

    def test_execute_write_rejects_multi_statement(self):
        tool = self._make_tool()
        result = tool.execute_write("INSERT INTO users VALUES (1); DELETE FROM users")

        assert result.success == 0
        assert "Multi-statement" in result.error

    def test_execute_write_rejects_empty_sql(self):
        tool = self._make_tool()
        result = tool.execute_write("   ")

        assert result.success == 0
        assert "Empty SQL" in result.error

    def test_execute_write_supports_sql_file_path(self, tmp_path):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        write_result = Mock(success=True, row_count=1)
        mock_connector.execute_insert.return_value = write_result

        sql_file = tmp_path / "insert.sql"
        sql_file.write_text("INSERT INTO users VALUES (1)", encoding="utf-8")

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.storage.workspace_root = str(tmp_path)

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_connector, agent_config=mock_config)

        result = tool.execute_write("insert.sql")

        assert result.success == 1
        assert result.result["sql"] == "INSERT INTO users VALUES (1)"

    def test_execute_write_honors_min_rows(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        write_result = Mock(success=True, row_count=1)
        mock_connector.execute_update.return_value = write_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_write("UPDATE users SET active = 1", min_rows=2)

        assert result.success == 0
        assert "below min_rows=2" in result.error

    def test_execute_write_honors_max_rows(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        write_result = Mock(success=True, row_count=5)
        mock_connector.execute_delete.return_value = write_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_write("DELETE FROM users", max_rows=3)

        assert result.success == 0
        assert "above max_rows=3" in result.error

    def test_execute_write_connector_failure(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        write_result = Mock(success=False, error="constraint violation")
        mock_connector.execute_insert.return_value = write_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_write("INSERT INTO users VALUES (1)")

        assert result.success == 0
        assert "constraint violation" in result.error

    def test_execute_write_dry_run_not_supported_yet(self):
        tool = self._make_tool()
        result = tool.execute_write("INSERT INTO users VALUES (1)", dry_run=True)

        assert result.success == 0
        assert "dry_run is not supported yet" in result.error

    def test_execute_write_not_in_available_tools(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        tool = self._make_tool(mock_connector)
        tool_names = [t.name for t in tool.available_tools()]

        assert "execute_write" not in tool_names


class TestDescribeTableDuckDBSchemaPrefix:
    """Verify that describe_table correctly splits 'schema.table' for DuckDB."""

    def _make_duckdb_tool(self):
        mock_connector = Mock()
        mock_connector.dialect = "duckdb"
        mock_connector.get_databases.return_value = []
        # DuckDB get_schema returns column dicts
        mock_connector.get_schema.return_value = [
            {"name": "stage_id", "type": "INTEGER", "comment": ""},
            {"name": "name", "type": "VARCHAR", "comment": ""},
        ]
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(mock_connector), mock_connector

    def test_describe_table_dotted_name_splits_schema_and_table(self):
        """describe_table('raw.stage') must call get_schema with schema_name='raw', table_name='stage'."""
        tool, mock_connector = self._make_duckdb_tool()
        result = tool.describe_table(table_name="raw.stage")

        assert result.success == 1, f"Expected success but got error: {result.error}"
        assert len(result.result.get("columns", [])) == 2

        call_kwargs = mock_connector.get_schema.call_args
        assert call_kwargs is not None, "get_schema was not called"
        # Accept both positional and keyword invocation
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        # Reconstruct as keyword map (get_schema signature: catalog, database, schema_name, table_name)
        param_names = ["catalog_name", "database_name", "schema_name", "table_name"]
        effective = dict(zip(param_names, args))
        effective.update(kwargs)

        assert effective.get("schema_name") == "raw", (
            f"Expected schema_name='raw', got {effective.get('schema_name')!r}"
        )
        assert effective.get("table_name") == "stage", (
            f"Expected table_name='stage', got {effective.get('table_name')!r}"
        )

    def test_describe_table_plain_name_uses_default_schema(self):
        """describe_table('stage') must call get_schema with table_name='stage' (no schema split)."""
        tool, mock_connector = self._make_duckdb_tool()
        result = tool.describe_table(table_name="stage")

        assert result.success == 1, f"Expected success but got error: {result.error}"

        call_kwargs = mock_connector.get_schema.call_args
        assert call_kwargs is not None, "get_schema was not called"
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        param_names = ["catalog_name", "database_name", "schema_name", "table_name"]
        effective = dict(zip(param_names, args))
        effective.update(kwargs)

        assert effective.get("table_name") == "stage", (
            f"Expected table_name='stage', got {effective.get('table_name')!r}"
        )

    def test_describe_table_explicit_schema_name_overrides(self):
        """describe_table('stage', schema_name='raw') must use schema_name='raw'."""
        tool, mock_connector = self._make_duckdb_tool()
        result = tool.describe_table(table_name="stage", schema_name="raw")

        assert result.success == 1, f"Expected success but got error: {result.error}"

        call_kwargs = mock_connector.get_schema.call_args
        assert call_kwargs is not None, "get_schema was not called"
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        param_names = ["catalog_name", "database_name", "schema_name", "table_name"]
        effective = dict(zip(param_names, args))
        effective.update(kwargs)

        assert effective.get("schema_name") == "raw"
        assert effective.get("table_name") == "stage"
