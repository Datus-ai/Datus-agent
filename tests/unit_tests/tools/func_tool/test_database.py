"""
Test cases for DBFuncTool compressor model_name initialization and execute_ddl.
"""

from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool
from datus.utils.exceptions import DatusException


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
            "DROP SCHEMA staging",
            "DROP SCHEMA IF EXISTS staging",
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
        mock_config.project_root = str(tmp_path)

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
        assert "already been committed" in result.error

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
        assert "already been committed" in result.error

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

    def test_execute_write_missing_method(self):
        """Connector that doesn't support the write method should return error."""
        mock_connector = Mock(spec=["dialect", "get_databases"])  # no execute_insert/update/delete
        mock_connector.dialect = "generic"
        mock_connector.get_databases.return_value = []
        tool = self._make_tool(mock_connector)
        result = tool.execute_write("INSERT INTO t VALUES (1)")
        assert result.success == 0
        assert "not support" in result.error.lower() or "does not support" in result.error.lower()

    def test_execute_write_exception_during_execution(self):
        """Connector that raises during execution should return error."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_insert.side_effect = RuntimeError("connection reset")

        tool = self._make_tool(mock_connector)
        result = tool.execute_write("INSERT INTO t VALUES (1)")
        assert result.success == 0
        assert "failed" in result.error.lower()


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


class TestExecuteDDLDatabaseParam:
    """Tests for execute_ddl with the database parameter for multi-connector routing."""

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

    def test_execute_ddl_with_database_routes_to_connector(self):
        """execute_ddl(database='greenplum') should call _get_connector('greenplum')."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock(success=True)
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        with patch.object(tool, "_get_connector", return_value=mock_connector) as mock_get:
            tool.execute_ddl("CREATE TABLE t (id INT)", datasource="greenplum")
            mock_get.assert_called_once_with("greenplum")

    def test_execute_ddl_without_database_uses_default(self):
        """execute_ddl() without database should call _get_connector with empty string."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock(success=True)
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        with patch.object(tool, "_get_connector", return_value=mock_connector) as mock_get:
            tool.execute_ddl("CREATE TABLE t (id INT)")
            mock_get.assert_called_once_with("")

    def test_execute_ddl_returns_database_in_result(self):
        """Successful execute_ddl should include database name in result."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock(success=True)
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE t (id INT)", datasource="greenplum")
        assert result.success == 1
        assert "datasource" in result.result


class TestGetConnectorRouting:
    """Tests for _get_connector routing in single vs multi connector mode.

    Verifies that single-connector mode ignores the database parameter
    (always returning the primary connector), while multi-connector mode
    correctly routes to different connectors by logical name.
    """

    def _make_single_mode_tool(self, connector):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector)

    def test_single_connector_ignores_database_param(self):
        """In single-connector mode, _get_connector always returns the same connector."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        tool = self._make_single_mode_tool(mock_connector)

        conn_default = tool._get_connector()
        conn_named = tool._get_connector("greenplum")
        conn_other = tool._get_connector("starrocks")

        # All three should be the exact same object
        assert conn_default is conn_named
        assert conn_named is conn_other
        assert conn_default is mock_connector

    def test_multi_connector_routes_by_database_name(self):
        """In multi-connector mode, _get_connector returns different connectors."""
        from datus.tools.db_tools.db_manager import DBManager

        mock_source = Mock()
        mock_source.dialect = "duckdb"
        mock_source.get_databases.return_value = []
        mock_target = Mock()
        mock_target.dialect = "greenplum"
        mock_target.get_databases.return_value = []

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.get_conn.side_effect = lambda ns, name: mock_target if name == "greenplum" else mock_source
        mock_db_manager.first_conn.return_value = mock_source

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "duckdb"
        # Must have >1 database so DBFuncTool enters true multi-connector mode
        mock_config.current_db_configs.return_value = {"duckdb": Mock(), "greenplum": Mock()}

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(
                mock_db_manager,
                agent_config=mock_config,
                default_datasource="duckdb",
            )

        # Verify multi-connector mode is active
        assert tool._is_multi_connector is True

        conn_source = tool._get_connector("duckdb")
        conn_target = tool._get_connector("greenplum")

        assert conn_source is mock_source
        assert conn_target is mock_target
        assert conn_source is not conn_target

    def test_multi_connector_defaults_coordinate_database_to_physical_name(self):
        """When database is omitted, table coordinates should use the connector's physical database name, not the datasource."""
        from datus.tools.db_tools.db_manager import DBManager

        mock_source = Mock()
        mock_source.dialect = "duckdb"
        mock_source.database_name = "PHYSICAL_DB"
        mock_source.get_databases.return_value = []

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.get_conn.return_value = mock_source
        mock_db_manager.first_conn.return_value = mock_source

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "duckdb"
        mock_config.current_db_configs.return_value = {"duckdb": Mock(), "greenplum": Mock()}

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(
                mock_db_manager,
                agent_config=mock_config,
                default_datasource="duckdb",
            )

        coordinate = tool._build_table_coordinate("orders")

        assert tool._is_multi_connector is True
        assert coordinate.database == "PHYSICAL_DB"

    def test_explicit_database_raises_when_not_configured(self):
        """When caller explicitly passes a database that doesn't exist, raise DatusException (no silent fallback)."""
        from datus.tools.db_tools.db_manager import DBManager

        mock_connector = Mock()
        mock_connector.dialect = "duckdb"
        mock_connector.get_databases.return_value = []

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.get_conn.side_effect = KeyError("not found")
        mock_db_manager.first_conn.return_value = mock_connector

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "default_db"
        mock_config.current_db_configs.return_value = {"default_db": Mock(), "other_db": Mock()}

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_db_manager, agent_config=mock_config, default_datasource="default_db")

        with pytest.raises(DatusException, match="not configured"):
            tool._get_connector("unknown_db")

    def test_default_datasource_fallback(self):
        """When using empty datasource, falls back to _default_datasource and looks up via db_manager."""
        from datus.tools.db_tools.db_manager import DBManager

        mock_connector = Mock()
        mock_connector.dialect = "duckdb"
        mock_connector.get_databases.return_value = []

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.get_conn.return_value = mock_connector
        mock_db_manager.first_conn.return_value = mock_connector

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "default_db"
        mock_config.current_db_configs.return_value = {"default_db": Mock(), "other_db": Mock()}

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_db_manager, agent_config=mock_config, default_datasource="default_db")

        # Empty string = use default datasource
        conn = tool._get_connector("")
        assert conn is mock_connector
        mock_db_manager.get_conn.assert_called_with("default_db", "default_db")

    def test_list_databases_multi_connector_returns_real_databases(self):
        """In multi-connector mode, list_databases should query the connector for real databases."""
        from datus.tools.db_tools.db_manager import DBManager

        mock_source = Mock()
        mock_source.dialect = "duckdb"
        mock_source.get_databases.return_value = ["analytics", "staging"]

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.first_conn.return_value = mock_source
        mock_db_manager.get_conn.return_value = mock_source

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "source_db"
        databases = {"source_db": Mock(), "other_db": Mock()}
        mock_config.current_db_configs.return_value = databases

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_db_manager, agent_config=mock_config, default_datasource=list(databases.keys())[0])
        assert tool._is_multi_connector is True

        result = tool.list_databases()

        assert result.success == 1
        assert result.result == ["analytics", "staging"]

    def test_list_databases_multi_connector_error(self):
        """In multi-connector mode, connector failure returns error result."""
        from datus.tools.db_tools.db_manager import DBManager

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.first_conn.return_value = Mock(dialect="duckdb")
        mock_db_manager.get_conn.side_effect = ConnectionError("adapter not installed")

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "source_db"
        databases = {"source_db": Mock(), "broken_db": Mock()}
        mock_config.current_db_configs.return_value = databases

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_db_manager, agent_config=mock_config, default_datasource=list(databases.keys())[0])
        assert tool._is_multi_connector is True

        result = tool.list_databases()

        assert result.success == 0
        assert "adapter not installed" in result.error


class TestTransferQueryResult:
    """Tests for DBFuncTool.transfer_query_result method."""

    def _make_multi_tool(self, source_connector, target_connector, default_db="source_db"):
        """Create a DBFuncTool with mocked _get_connector for multi-db routing."""
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(source_connector)

        def get_connector(datasource=None):
            if datasource == "target_db":
                return target_connector
            return source_connector

        tool._get_connector = Mock(side_effect=get_connector)
        tool._default_datasource = default_db
        return tool

    def _make_source_connector(self, df):
        """Create a mock source connector that returns a pandas DataFrame."""

        connector = Mock()
        connector.dialect = "duckdb"
        connector.get_databases.return_value = []

        exec_result = Mock()
        exec_result.success = True
        exec_result.sql_return = df
        exec_result.row_count = len(df)
        connector.execute_pandas.return_value = exec_result
        return connector

    def _make_target_connector(self):
        """Create a mock target connector with execute_insert support."""
        connector = Mock()
        connector.dialect = "postgresql"
        connector.get_databases.return_value = []

        # Mock DDL execution (for TRUNCATE)
        ddl_result = Mock(success=True)
        connector.execute_ddl.return_value = ddl_result

        # Mock execute_insert for batch INSERT
        insert_result = Mock(success=True, row_count=0)
        connector.execute_insert.return_value = insert_result
        return connector, connector.execute_insert

    def test_transfer_replace_mode_success(self):
        import pandas as pd

        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        source = self._make_source_connector(df)
        target, cursor = self._make_target_connector()

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT * FROM users",
            source_datasource="source_db",
            target_table="tgt.users",
            target_datasource="target_db",
            mode="replace",
            batch_size=5000,
        )

        assert result.success == 1
        assert result.result["rows_transferred"] == 3
        assert result.result["mode"] == "replace"
        # TRUNCATE should be called in replace mode
        target.execute_ddl.assert_called_once()
        assert "TRUNCATE" in target.execute_ddl.call_args[0][0].upper()

    def test_transfer_append_mode_no_truncate(self):
        import pandas as pd

        df = pd.DataFrame({"id": [1, 2]})
        source = self._make_source_connector(df)
        target, cursor = self._make_target_connector()

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT * FROM users",
            source_datasource="source_db",
            target_table="tgt.users",
            target_datasource="target_db",
            mode="append",
        )

        assert result.success == 1
        assert result.result["rows_transferred"] == 2
        # TRUNCATE should NOT be called in append mode
        target.execute_ddl.assert_not_called()

    def test_transfer_empty_result_set(self):
        import pandas as pd

        df = pd.DataFrame(columns=["id", "name"])
        source = self._make_source_connector(df)
        target, cursor = self._make_target_connector()

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT * FROM empty_table",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
            mode="replace",
        )

        assert result.success == 1
        assert result.result["rows_transferred"] == 0

    def test_transfer_source_query_failure(self):
        source = Mock()
        source.dialect = "duckdb"
        source.get_databases.return_value = []
        exec_result = Mock(success=False, error="syntax error in SQL")
        source.execute_pandas.return_value = exec_result

        target, _ = self._make_target_connector()
        tool = self._make_multi_tool(source, target)

        result = tool.transfer_query_result(
            source_sql="SELECT bad syntax",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
        )

        assert result.success == 0
        assert "syntax error" in result.error

    def test_transfer_exceeds_row_limit(self):

        source = Mock()
        source.dialect = "duckdb"
        source.get_databases.return_value = []

        # Create a mock DataFrame that reports >1M rows via len()
        large_df = Mock()
        large_df.__len__ = Mock(return_value=1_000_001)
        large_df.columns = ["id"]

        exec_result = Mock()
        exec_result.success = True
        exec_result.sql_return = large_df
        source.execute_pandas.return_value = exec_result

        target, _ = self._make_target_connector()
        tool = self._make_multi_tool(source, target)

        result = tool.transfer_query_result(
            source_sql="SELECT * FROM huge",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
        )

        assert result.success == 0
        assert "1,000,000" in result.error

    def test_transfer_invalid_mode(self):
        import pandas as pd

        df = pd.DataFrame({"id": [1]})
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT 1",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
            mode="upsert",
        )

        assert result.success == 0
        assert "mode" in result.error.lower()

    def test_transfer_uses_correct_connectors(self):
        import pandas as pd

        df = pd.DataFrame({"id": [1]})
        source = self._make_source_connector(df)
        target, cursor = self._make_target_connector()

        tool = self._make_multi_tool(source, target)
        tool.transfer_query_result(
            source_sql="SELECT * FROM t",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
            mode="append",
        )

        # Verify _get_connector was called with both databases
        calls = [c[0][0] for c in tool._get_connector.call_args_list]
        assert "source_db" in calls
        assert "target_db" in calls

    def test_transfer_batch_partial_failure(self):
        import pandas as pd

        # Create a df that will need multiple batches
        df = pd.DataFrame({"id": range(10), "name": [f"n{i}" for i in range(10)]})
        source = self._make_source_connector(df)
        target, execute_insert = self._make_target_connector()

        # Make execute_insert fail on the second call
        execute_insert.side_effect = [Mock(success=True), RuntimeError("disk full")]

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT * FROM t",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
            mode="append",
            batch_size=5,  # Force 2 batches
        )

        assert result.success == 0
        assert "disk full" in result.error

    def test_transfer_truncate_failure_in_replace_mode(self):
        """Replace mode should report error when TRUNCATE fails."""
        import pandas as pd

        df = pd.DataFrame({"id": [1]})
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()
        # Make TRUNCATE fail
        target.execute_ddl.return_value = Mock(success=False, error="permission denied")

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT 1",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
            mode="replace",
        )

        assert result.success == 0
        assert "permission denied" in result.error

    def test_transfer_source_without_execute_pandas(self):
        """Source connector without execute_pandas should report clear error."""
        source = Mock(spec=["dialect", "get_databases"])
        source.dialect = "sqlite"
        source.get_databases.return_value = []
        target, _ = self._make_target_connector()

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT 1",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
        )

        assert result.success == 0
        assert "pandas" in result.error.lower()

    def test_transfer_batch_size_zero(self):
        """batch_size <= 0 should be rejected."""
        import pandas as pd

        df = pd.DataFrame({"id": [1]})
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT 1",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
            batch_size=0,
        )

        assert result.success == 0
        assert "batch_size" in result.error

    def test_transfer_invalid_target_table(self):
        """SQL-injection-style target_table should be rejected."""
        import pandas as pd

        df = pd.DataFrame({"id": [1]})
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()

        tool = self._make_multi_tool(source, target)

        for bad_name in ["users; DROP TABLE x", "123bad", "table name with spaces"]:
            result = tool.transfer_query_result(
                source_sql="SELECT 1",
                source_datasource="source_db",
                target_table=bad_name,
                target_datasource="target_db",
            )
            assert result.success == 0, f"Expected rejection for target_table='{bad_name}'"
            assert "invalid" in result.error.lower() or "identifier" in result.error.lower()

    def test_transfer_target_connector_raises_returns_error(self):
        """When _get_connector raises for target_datasource, should return success=0."""
        source = Mock()
        source.dialect = "duckdb"
        source.get_databases.return_value = []

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(source)

        def get_connector(datasource=None):
            if datasource == "target_db":
                raise ConnectionError("target adapter not installed")
            return source

        tool._get_connector = Mock(side_effect=get_connector)
        tool._default_datasource = "source_db"

        result = tool.transfer_query_result(
            source_sql="SELECT 1",
            source_datasource="source_db",
            target_table="tgt.t",
            target_datasource="target_db",
        )

        assert result.success == 0
        assert "target" in result.error.lower()


class TestPathTraversalGuard:
    """Tests for _read_sql_from_file path traversal prevention."""

    def _make_tool(self):
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

    def test_rejects_absolute_path(self):
        """Absolute paths must be rejected to prevent sandbox escape."""
        from datus.utils.exceptions import DatusException

        tool = self._make_tool()
        with pytest.raises(DatusException):
            tool._read_sql_from_file("/etc/passwd")

    def test_rejects_dotdot_traversal(self):
        """Paths with .. must be rejected."""
        from datus.utils.exceptions import DatusException

        tool = self._make_tool()
        with pytest.raises(DatusException):
            tool._read_sql_from_file("../../../etc/passwd")

    def test_execute_write_rejects_absolute_sql_file(self):
        """execute_write must reject absolute .sql file paths."""
        tool = self._make_tool()
        result = tool.execute_write("/etc/passwd.sql")
        assert result.success == 0
        assert "failed" in result.error.lower()
