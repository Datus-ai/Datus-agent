"""
Test cases for DBFuncTool compressor model_name initialization and execute_ddl.
"""

from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool
from datus.utils.exceptions import DatusException, ErrorCode


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

    def test_table_semantic_profile_store_disabled_when_size_probe_fails(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-4o"
        mock_config.project_name = "project"

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
            patch("datus.tools.func_tool.database.TableSemanticProfileRAG") as mock_profile,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            mock_profile.return_value.get_size.side_effect = RuntimeError("storage unavailable")

            tool = DBFuncTool(mock_connector, agent_config=mock_config)

        assert tool.has_table_semantic_profiles is False
        assert tool._table_semantic_profiles is None


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
            "CREATE DATABASE blockchain",
            "CREATE DATABASE IF NOT EXISTS blockchain",
            "DROP DATABASE blockchain",
            "DROP DATABASE IF EXISTS blockchain",
            "  CREATE TABLE test (id INT)",
            "ALTER TABLE test ADD COLUMN name TEXT",
            "DROP TABLE test",
            "DROP TABLE IF EXISTS test",
            "CREATE VIEW v AS SELECT 1",
            "DROP VIEW v",
            "CREATE OR REPLACE VIEW v AS SELECT 1",
            "CREATE TEMPORARY TABLE tmp AS SELECT 1",
            "CREATE TEMP TABLE tmp (id INT)",
            # Non-read, non-DML statements no longer pre-rejected — permission
            # gates them, the tool executes whatever the engine accepts.
            "TRUNCATE TABLE users",
            "GRANT ALL ON users TO public",
            "CREATE INDEX idx ON users (id)",
            "MERGE INTO target t USING source s ON t.id = s.id WHEN MATCHED THEN UPDATE SET name = s.name",
        ],
    )
    def test_allowed_ddl_statements(self, sql):
        """Non-read, non-DML statements pass validation and execute."""
        tool = self._make_tool()
        result = tool.execute_ddl(sql)
        assert result.success == 1

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            ("SELECT * FROM users", "read path"),
            ("INSERT INTO users VALUES (1, 'test')", "write path"),
            ("UPDATE users SET name='x'", "write path"),
            ("DELETE FROM users", "write path"),
        ],
    )
    def test_read_and_dml_rejected_from_ddl_path(self, sql, expected):
        """Read-only and DML statements have dedicated paths and are refused here."""
        tool = self._make_tool()
        result = tool.execute_ddl(sql)
        assert result.success == 0
        assert expected in result.error

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
        assert "does not support INSERT operations" in result.error

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

        mock_connector.get_schema.assert_called_once()
        call_kwargs = mock_connector.get_schema.call_args
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

        mock_connector.get_schema.assert_called_once()
        call_kwargs = mock_connector.get_schema.call_args
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

        mock_connector.get_schema.assert_called_once()
        call_kwargs = mock_connector.get_schema.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        param_names = ["catalog_name", "database_name", "schema_name", "table_name"]
        effective = dict(zip(param_names, args))
        effective.update(kwargs)

        assert effective.get("schema_name") == "raw"
        assert effective.get("table_name") == "stage"


class TestDescribeTableConstraintPassthrough:
    """describe_table must surface connector-reported constraint facts.

    Connectors return ``pk`` / ``nullable`` / ``default_value`` per column
    (see e.g. the PostgreSQL connector); describe_table must pass the
    informative values through instead of dropping them. ``pk=False`` and
    ``nullable=True`` are NOT emitted: several connectors hardcode those
    when the engine exposes no constraint metadata, so they mean "unknown".
    """

    def _make_tool(self, schema_rows):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.get_schema.return_value = schema_rows
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(mock_connector)

    def test_pk_and_not_null_surfaced_from_connector_rows(self):
        """pk=True and nullable=False pass through; pk=False / nullable=True are omitted as unknown."""
        # Row shape mirrors real connector get_schema output (cid/name/type/nullable/default_value/pk/comment)
        tool = self._make_tool(
            [
                {
                    "cid": 0,
                    "name": "etl_dt",
                    "type": "date",
                    "nullable": False,
                    "default_value": None,
                    "pk": True,
                    "comment": "snapshot date",
                },
                {
                    "cid": 1,
                    "name": "org_no",
                    "type": "character varying",
                    "nullable": True,
                    "default_value": None,
                    "pk": False,
                    "comment": None,
                },
            ]
        )
        result = tool.describe_table("t")

        assert result.success == 1, f"Expected success but got error: {result.error}"
        cols = {c["name"]: c for c in result.result["columns"]}
        assert cols["etl_dt"]["pk"] is True
        assert cols["etl_dt"]["nullable"] is False
        assert "pk" not in cols["org_no"], "pk=False must be omitted (means unknown, not verified absent)"
        assert "nullable" not in cols["org_no"], "nullable=True is uninformative and must be omitted"

    def test_sqlite_integer_pk_positions_normalized_to_true(self):
        """SQLite PRAGMA reports pk as 1-based composite-key position; any positive value means key."""
        tool = self._make_tool(
            [
                {"cid": 0, "name": "etl_dt", "type": "DATE", "nullable": False, "default_value": None, "pk": 1},
                {"cid": 1, "name": "org_no", "type": "TEXT", "nullable": False, "default_value": None, "pk": 2},
                {"cid": 2, "name": "amount", "type": "REAL", "nullable": True, "default_value": None, "pk": 0},
            ]
        )
        result = tool.describe_table("t")

        assert result.success == 1, f"Expected success but got error: {result.error}"
        cols = {c["name"]: c for c in result.result["columns"]}
        assert cols["etl_dt"]["pk"] is True
        assert cols["org_no"]["pk"] is True
        assert "pk" not in cols["amount"]

    def test_default_value_surfaced_only_when_defined(self):
        """Non-empty default_value passes through as str; None and empty string are omitted."""
        tool = self._make_tool(
            [
                {"name": "status", "type": "TEXT", "comment": "", "default_value": "'new'"},
                {"name": "qty", "type": "INTEGER", "comment": "", "default_value": 0},
                {"name": "note", "type": "TEXT", "comment": "", "default_value": None},
                {"name": "tag", "type": "TEXT", "comment": "", "default_value": ""},
            ]
        )
        result = tool.describe_table("t")

        assert result.success == 1, f"Expected success but got error: {result.error}"
        cols = {c["name"]: c for c in result.result["columns"]}
        assert cols["status"]["default_value"] == "'new'"
        assert cols["qty"]["default_value"] == "0"
        assert "default_value" not in cols["note"]
        assert "default_value" not in cols["tag"]

    def test_legacy_connector_rows_produce_unchanged_shape(self):
        """Connectors returning only name/type/comment must yield exactly the pre-existing column shape."""
        tool = self._make_tool([{"name": "a", "type": "INTEGER", "comment": "ident"}])
        result = tool.describe_table("t")

        assert result.success == 1, f"Expected success but got error: {result.error}"
        assert result.result["columns"] == [{"name": "a", "type": "INTEGER", "comment": "ident"}]

    def test_constraint_facts_coexist_with_semantic_enrichment(self):
        """pk passthrough and semantic-model enrichment must land on the same column dict."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.get_schema.return_value = [
            {"name": "order_id", "type": "INTEGER", "comment": "", "pk": True, "nullable": False},
            {"name": "amount", "type": "DOUBLE", "comment": "", "pk": False, "nullable": True},
        ]

        tool = DBFuncTool(mock_connector)
        tool._table_semantic_profiles = Mock()
        tool.has_semantic_models = True
        tool._semantic_storage = Mock()
        tool._table_semantic_profiles.get_profile.return_value = {
            "format": "osi",
            "physical_table_fq_name": "main.orders",
            "semantic_model_name": "shop",
            "dataset_name": "orders",
            "data_source_name": "",
            "description": "Orders dataset",
            "ai_context_json": "",
            "columns_json": (
                "["
                '{"name":"order_id","expr":"order_id","role":"primary_key","description":"Order key"},'
                '{"name":"amount","expr":"amount","role":"measure","description":"Order amount"}'
                "]"
            ),
            "relationships_json": "[]",
            "custom_extensions_json": "",
            "yaml_path": "/tmp/orders.yml",
        }

        result = tool.describe_table("orders")

        assert result.success == 1, f"Expected success but got error: {result.error}"
        cols = {c["name"]: c for c in result.result["columns"]}
        assert cols["order_id"]["pk"] is True
        assert cols["order_id"]["nullable"] is False
        assert cols["order_id"]["semantic_role"] == "primary_key"
        assert "pk" not in cols["amount"]
        assert cols["amount"]["semantic_role"] == "measure"


class TestDescribeTableSemanticProfile:
    def test_describe_table_enriches_from_table_semantic_profile(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.get_schema.return_value = [
            {"name": "order_id", "type": "INTEGER", "comment": ""},
            {"name": "order_date", "type": "DATE", "comment": ""},
            {"name": "amount", "type": "DOUBLE", "comment": ""},
        ]

        tool = DBFuncTool(mock_connector)
        tool._table_semantic_profiles = Mock()
        tool.has_semantic_models = True
        tool._semantic_storage = Mock()
        tool._table_semantic_profiles.get_profile.return_value = {
            "format": "osi",
            "physical_table_fq_name": "main.orders",
            "semantic_model_name": "shop",
            "dataset_name": "orders",
            "data_source_name": "",
            "description": "Orders dataset",
            "ai_context_json": '{"synonyms": ["purchases"]}',
            "columns_json": (
                "["
                '{"name":"order_id","expr":"order_id","role":"primary_key","description":"Order key"},'
                '{"name":"order_date","expr":"order_date","role":"time_dimension","description":"Order date"},'
                '{"name":"amount","expr":"amount","role":"measure","description":"Order amount"}'
                "]"
            ),
            "relationships_json": '[{"name":"orders_to_customers","to_dataset":"customers"}]',
            "custom_extensions_json": "",
            "yaml_path": "/tmp/orders.yml",
        }

        result = tool.describe_table("orders")

        assert result.success == 1
        tool._semantic_storage.get_semantic_model.assert_not_called()
        assert result.result["table"] == {
            "name": "orders",
            "description": "Orders dataset",
            "ai_context": {"synonyms": ["purchases"]},
        }
        assert result.result["semantic"]["relationships"][0]["name"] == "orders_to_customers"
        assert "filters" not in result.result["semantic"]
        assert "format" not in result.result["semantic"]
        assert "semantic_model_name" not in result.result["semantic"]
        assert "dataset_name" not in result.result["semantic"]
        assert "data_source_name" not in result.result["semantic"]
        assert "physical_table" not in result.result["semantic"]
        assert "custom_extensions" not in result.result["semantic"]
        assert "yaml_path" not in result.result["semantic"]
        columns = {col["name"]: col for col in result.result["columns"]}
        assert columns["order_id"]["semantic_role"] == "primary_key"
        assert "is_entity_key" not in columns["order_id"]
        assert columns["order_date"]["is_dimension"] is True
        assert columns["amount"]["semantic_role"] == "measure"
        assert "is_measure" not in columns["amount"]
        assert columns["amount"]["comment"] == "Order amount"

    def test_describe_table_keeps_metricflow_profile_enrichment(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.get_schema.return_value = [
            {"name": "order_id", "type": "INTEGER", "comment": ""},
            {"name": "order_date", "type": "DATE", "comment": ""},
            {"name": "amount", "type": "DOUBLE", "comment": ""},
        ]

        tool = DBFuncTool(mock_connector)
        tool._table_semantic_profiles = Mock()
        tool._table_semantic_profiles.get_profile.return_value = {
            "table_name": "orders",
            "semantic_model_name": "orders_source",
            "dataset_name": "",
            "data_source_name": "orders_source",
            "description": "Orders data source",
            "ai_context_json": '{"synonyms": ["sales orders"]}',
            "columns_json": (
                "["
                '{"name":"order_id","expr":"order_id","role":"primary_key","description":"Order key"},'
                '{"name":"order_date","expr":"order_date","role":"time_dimension","description":"Order date"},'
                '{"name":"amount","expr":"amount","role":"measure","description":"Order amount","agg":"sum"}'
                "]"
            ),
            "relationships_json": '[{"name":"orders_to_customers","to_dataset":"customers"}]',
        }

        result = tool.describe_table("orders")

        assert result.success == 1
        assert result.result["table"] == {
            "name": "orders_source",
            "description": "Orders data source",
            "ai_context": {"synonyms": ["sales orders"]},
        }
        assert result.result["semantic"] == {
            "relationships": [{"name": "orders_to_customers", "to_dataset": "customers"}],
        }
        columns = {col["name"]: col for col in result.result["columns"]}
        assert columns["order_date"]["semantic_role"] == "time_dimension"
        assert columns["order_date"]["is_dimension"] is True
        assert columns["amount"]["semantic_role"] == "measure"
        assert columns["amount"]["comment"] == "Order amount"
        assert "is_measure" not in columns["amount"]
        assert "is_entity_key" not in columns["order_id"]


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
            mock_get.assert_called_once_with("greenplum", "")

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
            mock_get.assert_called_once_with("", "")

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

    def test_execute_ddl_postgresql_dialect_does_not_fail_after_execution(self):
        """Config/adapter dialect is ``postgresql`` but sqlglot's dialect name
        is ``postgres``. A successful connector DDL must not be reported as
        failed while building the validation target."""
        mock_connector = Mock()
        mock_connector.dialect = "postgresql"
        mock_connector.database = "analytics"
        mock_connector.get_databases.return_value = []
        ddl_result = Mock(success=True)
        mock_connector.execute_ddl.return_value = ddl_result

        tool = self._make_tool(mock_connector)
        result = tool.execute_ddl("CREATE TABLE public.t (id INT)", datasource="superset")

        assert result.success == 1
        assert result.result["datasource"] == "superset"
        assert result.result["deliverable_target"]["schema"] == "public"
        assert result.result["deliverable_target"]["table"] == "t"


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
        mock_db_manager.get_conn.side_effect = lambda ns, db="": mock_target if ns == "greenplum" else mock_source
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

        def _get_conn(ns, db=""):
            if ns == "unknown_db":
                raise KeyError("not found")
            return mock_connector

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.get_conn.side_effect = _get_conn
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
        mock_db_manager.get_conn.assert_called_with("default_db", "")

    def test_default_database_routes_connector_by_database(self):
        """DBFuncTool(default_database=X) binds its connector via get_conn(datasource, X)."""
        from datus.tools.db_tools.db_manager import DBManager

        conn = Mock()
        conn.dialect = "postgresql"
        conn.get_databases.return_value = []

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.get_conn.return_value = conn
        mock_db_manager.first_conn.return_value = conn

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "pg"
        mock_config.current_db_configs.return_value = {"pg": Mock(), "other": Mock()}

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(
                mock_db_manager, agent_config=mock_config, default_datasource="pg", default_database="target_db"
            )

        assert tool._default_database == "target_db"
        # Primary connector bound via get_conn(datasource, target_db).
        mock_db_manager.get_conn.assert_any_call("pg", "target_db")

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
        # Not a glob datasource → list_databases asks the connector.
        mock_config.current_db_config.return_value.path_pattern = ""
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

        # First get_conn (primary connector at construction) succeeds; the list-databases
        # call then fails, exercising the error path.
        _state = {"first": True}

        def _gc(ns, db=""):
            if _state["first"]:
                _state["first"] = False
                return Mock(dialect="duckdb", database_name="source_db", get_databases=Mock(return_value=[]))
            raise ConnectionError("adapter not installed")

        mock_db_manager = Mock(spec=DBManager)
        mock_db_manager.first_conn.return_value = Mock(dialect="duckdb")
        mock_db_manager.get_conn.side_effect = _gc

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-5.4"
        mock_config.current_datasource = "source_db"
        mock_config.current_db_config.return_value.path_pattern = ""
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

    def test_transfer_helpers_honor_new_adapter_dialects(self, monkeypatch):
        import pandas as pd

        from datus.tools.db_tools import connector_registry

        monkeypatch.setattr(
            connector_registry,
            "get_parser_dialect",
            lambda dialect: "postgres" if dialect == "hologres" else None,
            raising=False,
        )

        assert DBFuncTool._identifier_quote_char("doris") == "`"
        assert DBFuncTool._infer_transfer_column_type(pd.Series([1.5]), "hologres") == "DOUBLE PRECISION"

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

    def test_transfer_replace_creates_missing_target_table(self):
        import pandas as pd

        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "is_open": [True, False]})
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()
        target.execute_ddl.side_effect = [
            Mock(success=False, error='relation "tgt.users" does not exist'),
            Mock(success=True),
        ]

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT id, name, is_open FROM users",
            source_datasource="source_db",
            target_table="tgt.users",
            target_datasource="target_db",
            mode="replace",
        )

        assert result.success == 1
        assert result.result["rows_transferred"] == 2
        assert result.result["target_table_created"] is True
        ddl_calls = [call.args[0] for call in target.execute_ddl.call_args_list]
        assert ddl_calls[0] == "TRUNCATE TABLE tgt.users"
        assert ddl_calls[1].startswith("CREATE TABLE tgt.users")
        assert '"id" BIGINT' in ddl_calls[1]
        assert '"name" TEXT' in ddl_calls[1]
        assert '"is_open" BOOLEAN' in ddl_calls[1]
        assert "INSERT INTO tgt.users" in target.execute_insert.call_args.args[0]

    def test_transfer_dispatches_registered_dialect_operations(self):
        import pandas as pd

        df = pd.DataFrame({"id": [1, 2], "is_open": [True, False]})
        source = self._make_source_connector(df)
        source.execute_query.return_value = Mock(success=True, sql_return=[(2,)])
        target, _ = self._make_target_connector()
        target.execute_ddl.side_effect = [
            Mock(success=False, error="ORA-00942: table or view does not exist"),
            Mock(success=True),
        ]

        source_operations = Mock()
        source_operations.render_count.return_value = (
            "SELECT COUNT(*) AS __datus_count FROM (SELECT id, is_open FROM users) __datus_src"
        )
        target_operations = Mock()
        target_operations.quote_identifier.side_effect = lambda name: f'"{name.upper()}"'
        target_operations.infer_transfer_type.side_effect = lambda _series: "NUMBER"
        target_operations.write_dataframe.return_value = 2

        def resolve_operations(*, connector=None, dialect=""):
            del dialect
            return source_operations if connector is source else target_operations

        tool = self._make_multi_tool(source, target)
        with patch("datus.tools.func_tool.database.get_dialect_operations", side_effect=resolve_operations):
            result = tool.transfer_query_result(
                source_sql="SELECT id, is_open FROM users",
                source_datasource="source_db",
                target_table="tgt.users",
                target_datasource="target_db",
                mode="replace",
                batch_size=100,
            )

        assert result.success == 1
        source_operations.render_count.assert_called_once_with(
            "SELECT id, is_open FROM users",
            "__datus_src",
        )
        source.execute_query.assert_called_once_with(source_operations.render_count.return_value)
        create_sql = target.execute_ddl.call_args_list[1].args[0]
        assert '"ID" NUMBER' in create_sql
        assert '"IS_OPEN" NUMBER' in create_sql
        target_operations.write_dataframe.assert_called_once()
        assert target_operations.write_dataframe.call_args.args[0] is target
        assert target_operations.write_dataframe.call_args.args[1] == "tgt.users"
        assert target_operations.write_dataframe.call_args.args[3] == 100
        target.execute_insert.assert_not_called()

    def test_transfer_replace_creates_missing_target_table_for_mysql_contraction(self):
        import pandas as pd

        df = pd.DataFrame({"id": [1], "name": ["a"]})
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()
        target.execute_ddl.side_effect = [
            Mock(success=False, error="Table 'tgt.users' doesn't exist"),
            Mock(success=True),
        ]

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT id, name FROM users",
            source_datasource="source_db",
            target_table="tgt.users",
            target_datasource="target_db",
            mode="replace",
        )

        assert result.success == 1
        assert result.result["target_table_created"] is True
        assert target.execute_ddl.call_args_list[1].args[0].startswith("CREATE TABLE tgt.users")

    def test_transfer_create_target_keeps_complex_values_text_compatible(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "payload": [{"a": 1}],
                "raw_bytes": [b"abc"],
            }
        )
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()
        target.execute_ddl.side_effect = [
            Mock(success=False, error='relation "tgt.events" does not exist'),
            Mock(success=True),
        ]

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT payload, raw_bytes FROM events",
            source_datasource="source_db",
            target_table="tgt.events",
            target_datasource="target_db",
            mode="replace",
        )

        assert result.success == 1
        create_sql = target.execute_ddl.call_args_list[1].args[0]
        assert '"payload" TEXT' in create_sql
        assert '"raw_bytes" TEXT' in create_sql
        assert "JSONB" not in create_sql
        assert "BYTEA" not in create_sql

    def test_transfer_replace_creates_missing_target_for_empty_result(self):
        import pandas as pd

        df = pd.DataFrame(columns=["id", "name"])
        source = self._make_source_connector(df)
        target, _ = self._make_target_connector()
        target.execute_ddl.side_effect = [
            Mock(success=False, error="no such table: tgt.users"),
            Mock(success=True),
        ]

        tool = self._make_multi_tool(source, target)
        result = tool.transfer_query_result(
            source_sql="SELECT id, name FROM users WHERE 1 = 0",
            source_datasource="source_db",
            target_table="tgt.users",
            target_datasource="target_db",
            mode="replace",
        )

        assert result.success == 1
        assert result.result["rows_transferred"] == 0
        assert result.result["target_table_created"] is True
        assert result.result["target_table_create_sql"].startswith("CREATE TABLE tgt.users")
        target.execute_insert.assert_not_called()

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
            assert f"Invalid target_table identifier: '{bad_name}'" in result.error

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


class TestDBFuncToolExecuteSql:
    """Tests for the unified ``execute_sql`` dispatch entry point."""

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

    def test_select_routes_to_read_query(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_query.return_value = Mock(success=True, sql_return=[{"a": 1}])

        tool = self._make_tool(mock_connector)
        tool.compressor.compress = Mock(return_value={"original_rows": 1, "compressed_data": "a\n1"})

        result = tool.execute_sql("SELECT * FROM users")

        assert result.success == 1
        mock_connector.execute_query.assert_called_once()
        # Read results are the compressor payload (carries compressed_data).
        assert result.result == {"original_rows": 1, "compressed_data": "a\n1"}

    def test_insert_routes_to_execute_write(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_insert.return_value = Mock(success=True, row_count=2)

        tool = self._make_tool(mock_connector)
        result = tool.execute_sql("INSERT INTO users VALUES (1), (2)")

        assert result.success == 1
        assert result.result["sql_type"] == "insert"
        assert result.result["row_count"] == 2
        mock_connector.execute_insert.assert_called_once()

    def test_create_table_routes_to_execute_ddl(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_ddl.return_value = Mock(success=True)

        tool = self._make_tool(mock_connector)
        result = tool.execute_sql("CREATE TABLE test (id INT)")

        assert result.success == 1
        assert result.result["message"] == "DDL executed successfully"
        mock_connector.execute_ddl.assert_called_once()

    def test_read_only_allows_select(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_query.return_value = Mock(success=True, sql_return=[{"a": 1}])

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_connector, read_only=True)
        tool.compressor.compress = Mock(return_value={"original_rows": 1, "compressed_data": "a\n1"})

        result = tool.execute_sql("SELECT * FROM users")

        assert result.success == 1
        mock_connector.execute_query.assert_called_once()

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO users VALUES (1)",
            "UPDATE users SET a = 1",
            "DELETE FROM users",
            "CREATE TABLE t (id INT)",
            "DROP TABLE users",
            "TRUNCATE TABLE users",
        ],
    )
    def test_read_only_rejects_non_read(self, sql):
        """A read-only DBFuncTool hard-rejects every non-read statement at the
        tool layer, independent of PermissionHooks."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_connector, read_only=True)

        result = tool.execute_sql(sql)

        assert result.success == 0
        assert "read-only" in (result.error or "")
        mock_connector.execute_insert.assert_not_called()
        mock_connector.execute_ddl.assert_not_called()

    def test_min_max_rows_forwarded_to_write(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_insert.return_value = Mock(success=True, row_count=5)

        tool = self._make_tool(mock_connector)
        # Affected 5 rows but max_rows=1 → DML safety bound violated.
        result = tool.execute_sql("INSERT INTO users VALUES (1)", max_rows=1)

        assert result.success == 0
        assert "above max_rows" in result.error

    @pytest.mark.parametrize(
        "sql",
        [
            "MERGE INTO target t USING source s ON t.id = s.id WHEN MATCHED THEN UPDATE SET name = s.name",
            "GRANT SELECT ON users TO bob",
            "TRUNCATE TABLE users",
            "CREATE DATABASE blockchain",
        ],
    )
    def test_non_read_non_dml_routes_to_generic_execute(self, sql):
        """MERGE / GRANT / TRUNCATE / CREATE DATABASE are not pre-rejected — the
        permission layer gates them and the tool executes them generically."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_ddl.return_value = Mock(success=True)

        tool = self._make_tool(mock_connector)
        result = tool.execute_sql(sql)

        assert result.success == 1
        mock_connector.execute_ddl.assert_called_once()

    def test_rejects_multi_statement(self):
        """Multi-statement scripts are still refused — one statement per call."""
        tool = self._make_tool()
        result = tool.execute_sql("INSERT INTO a VALUES (1); DELETE FROM a")

        assert result.success == 0

    def test_sql_file_path_routes_by_content(self, tmp_path):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.execute_insert.return_value = Mock(success=True, row_count=1)

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

        result = tool.execute_sql("insert.sql")

        assert result.success == 1
        assert result.result["sql_type"] == "insert"

    def test_available_tools_exposes_execute_sql_only(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        tool = self._make_tool(mock_connector)
        tool_names = {t.name for t in tool.available_tools()}

        assert "execute_sql" in tool_names
        # The legacy split tools are internal-only now.
        assert "read_query" not in tool_names
        assert "execute_ddl" not in tool_names
        assert "execute_write" not in tool_names


class TestDBFuncToolExecuteReadEnforced:
    """execute_read_enforced: the single policy-enforced raw-read path shared by
    read_query and the report/dashboard artifact query executors. It must reject
    multi-statement / non-read SQL and apply the SQL policy before the statement
    reaches the engine — the guard the artifact save paths previously bypassed."""

    def _make_tool(self, connector, agent_config=None):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector, agent_config=agent_config)

    def _connector(self):
        connector = Mock()
        connector.dialect = "sqlite"
        connector.get_databases.return_value = []
        exec_result = Mock()
        exec_result.success = True
        exec_result.sql_return = [{"n": 1}]
        connector.execute_query.return_value = exec_result
        return connector

    def test_readonly_select_executes_and_returns_raw_result(self):
        connector = self._connector()
        tool = self._make_tool(connector)

        result = tool.execute_read_enforced("SELECT 1 AS n", connector)

        assert result.success is True
        connector.execute_query.assert_called_once()
        args, kwargs = connector.execute_query.call_args
        assert args[0] == "SELECT 1 AS n"
        assert kwargs["result_format"] == "list"

    def test_multi_statement_rejected_without_touching_connector(self):
        connector = self._connector()
        tool = self._make_tool(connector)

        result = tool.execute_read_enforced("SELECT 1; DROP TABLE t", connector)

        assert result.success is False
        assert "Multi-statement" in result.error
        connector.execute_query.assert_not_called()

    def test_non_read_statement_rejected(self):
        connector = self._connector()
        tool = self._make_tool(connector)

        result = tool.execute_read_enforced("INSERT INTO t VALUES (1)", connector)

        assert result.success is False
        connector.execute_query.assert_not_called()

    def test_policy_rewrite_is_executed_verbatim(self):
        connector = self._connector()
        tool = self._make_tool(connector)

        # A policy that injects a row cap; the rewrite is still a single
        # read-only statement, so it must reach the connector verbatim.
        with patch.object(tool, "_enforce_sql_policy", return_value="SELECT 1 AS n LIMIT 100"):
            result = tool.execute_read_enforced("SELECT 1 AS n", connector)

        assert result.success is True
        assert connector.execute_query.call_args[0][0] == "SELECT 1 AS n LIMIT 100"

    def test_policy_denial_surfaces_as_failure_without_executing(self):
        connector = self._connector()
        tool = self._make_tool(connector)

        with patch.object(
            tool,
            "_enforce_sql_policy",
            side_effect=DatusException(ErrorCode.TOOL_INVALID_INPUT, message="denied by policy"),
        ):
            result = tool.execute_read_enforced("SELECT 1 AS n", connector)

        assert result.success is False
        assert "denied by policy" in result.error
        connector.execute_query.assert_not_called()

    def test_policy_rewrite_to_non_read_is_rejected(self):
        connector = self._connector()
        tool = self._make_tool(connector)

        # A buggy/hostile policy rewrite that turns a read into a mutation must
        # be caught by the post-rewrite re-validation, not forwarded to the DB.
        with patch.object(tool, "_enforce_sql_policy", return_value="DROP TABLE t"):
            result = tool.execute_read_enforced("SELECT 1 AS n", connector)

        assert result.success is False
        connector.execute_query.assert_not_called()


class TestDBFuncToolGuardEstimatedRows:
    """guard_estimated_rows: EXPLAIN-based pre-flight row-count guard. Blocks a
    query whose optimizer estimate exceeds the ceiling before it executes;
    fail-open on anything it can't measure."""

    def _make_tool(self, connector):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector)

    def _connector_with_explain(self, explain_rows, dialect="starrocks", success=True):
        connector = Mock()
        connector.dialect = dialect
        connector.get_databases.return_value = []
        explain_result = Mock()
        explain_result.success = success
        explain_result.sql_return = explain_rows
        connector.execute_query.return_value = explain_result
        return connector

    def test_oversize_estimate_is_rejected_with_actionable_message(self):
        connector = self._connector_with_explain([{"plan": "CROSS JOIN cardinality: 9006993124"}])
        tool = self._make_tool(connector)

        result = tool.guard_estimated_rows("SELECT * FROM a, b", connector)

        assert result.success == 0
        assert "9,006,993,124" in result.error
        # EXPLAIN must be planning-only — never the bare statement.
        assert connector.execute_query.call_args[0][0].startswith("EXPLAIN ")

    def test_multi_statement_rejected_before_explain_runs(self):
        # A driver that splits statements would run the DROP as part of
        # ``EXPLAIN SELECT 1; DROP TABLE t`` — reject before EXPLAIN is issued.
        connector = self._connector_with_explain([{"plan": "cardinality: 10"}])
        tool = self._make_tool(connector)

        result = tool.guard_estimated_rows("SELECT 1; DROP TABLE t", connector)

        assert result.success == 0
        connector.execute_query.assert_not_called()

    def test_estimate_under_ceiling_allows_query(self):
        connector = self._connector_with_explain([{"plan": "cardinality: 42"}])
        tool = self._make_tool(connector)

        assert tool.guard_estimated_rows("SELECT 1", connector) is None

    def test_boundary_around_the_ceiling(self):
        from datus.tools.sql_guard import MAX_ESTIMATED_ROWS

        # At the ceiling → allowed; one row over → rejected.
        at_ceiling = self._connector_with_explain([{"plan": f"cardinality: {MAX_ESTIMATED_ROWS}"}])
        assert self._make_tool(at_ceiling).guard_estimated_rows("SELECT 1", at_ceiling) is None

        over = self._connector_with_explain([{"plan": f"cardinality: {MAX_ESTIMATED_ROWS + 1}"}])
        assert self._make_tool(over).guard_estimated_rows("SELECT 1", over).success == 0

    def test_explain_raising_fails_open(self):
        connector = Mock()
        connector.dialect = "starrocks"
        connector.get_databases.return_value = []
        connector.execute_query.side_effect = RuntimeError("EXPLAIN unsupported")
        tool = self._make_tool(connector)

        assert tool.guard_estimated_rows("SELECT 1", connector) is None

    def test_explain_unsuccessful_fails_open(self):
        connector = self._connector_with_explain([], success=False)
        tool = self._make_tool(connector)

        assert tool.guard_estimated_rows("SELECT 1", connector) is None

    def test_unparseable_dialect_fails_open(self):
        connector = self._connector_with_explain([{"detail": "SCAN t"}], dialect="sqlite")
        tool = self._make_tool(connector)

        assert tool.guard_estimated_rows("SELECT 1", connector) is None
<<<<<<< HEAD
=======


class TestExecuteSqlWriteLaundering:
    """A write that carries a read must not run while a policy context exists.

    `execute_sql` routes reads through `execute_read_enforced`, where the plugin
    rewrites them — but DML and DDL went straight to the connector. So a caller
    restricted to two stores could ask chat for
    `CREATE TABLE mine AS SELECT * FROM orders` and end up owning every row the
    policy withholds, in a table no policy covers. Found end-to-end, not by any
    unit test: the console path had the same hole and was fixed on its own.
    """

    def _tool(self, connector, policy_context):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as sem,
        ):
            rag.return_value.schema_store.table_size.return_value = 0
            sem.return_value.get_size.return_value = 0
            # `_mock_agent_config`, not a bare `Mock()`: a bare Mock answers
            # every attribute with a truthy Mock, so `sql_read_only` would read
            # as on and these write-path assertions would fail for a reason
            # unrelated to what they test.
            config = _mock_agent_config()
            config.active_model.return_value.model = "gpt-4o"
            config.policy_context = policy_context
            tool = DBFuncTool(connector, agent_config=config)
        return tool

    def _connector(self):
        c = Mock()
        c.dialect = "postgresql"
        c.get_databases.return_value = []
        c.execute_ddl.return_value = Mock(success=True)
        c.execute_insert.return_value = Mock(success=True, row_count=1)
        return c

    SCOPED = {"row_filter": {"access_mode": "scoped", "store_ids": ["S001"]}}

    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE TABLE mine AS SELECT * FROM orders",
            "INSERT INTO mine SELECT * FROM orders",
            # sqlglot cannot parse this and returns an opaque Command.
            "CREATE TABLE mine AS TABLE orders",
            "COPY orders TO '/tmp/orders.csv'",
        ],
    )
    def test_a_write_that_reads_is_refused(self, sql):
        connector = self._connector()
        tool = self._tool(connector, self.SCOPED)

        result = tool.execute_sql(sql)

        assert result.success == 0
        assert "row-level policies" in result.error
        connector.execute_ddl.assert_not_called()
        connector.execute_insert.assert_not_called()

    def test_a_plain_write_still_runs(self):
        """Enabling a policy must not turn the agent read-only."""
        connector = self._connector()
        tool = self._tool(connector, self.SCOPED)

        result = tool.execute_sql("CREATE TABLE plain_t (id int)")

        assert result.success == 1
        connector.execute_ddl.assert_called_once()

    def test_without_policies_the_same_write_is_allowed(self):
        connector = self._connector()
        tool = self._tool(connector, {})

        result = tool.execute_sql("CREATE TABLE mine AS SELECT * FROM orders")

        assert result.success == 1
        connector.execute_ddl.assert_called_once()


class TestMcpFactoriesHonorDeploymentReadOnly:
    """``create_dynamic`` / ``create_static`` are the MCP server's construction
    path (``mcp_server._create_context`` and ``_init_tools`` loop over the tool
    registry and call them generically). Neither can pass ``read_only``, so this
    is the path the deployment switch has to cover on its own — testing
    ``__init__`` alone would miss it entirely.
    """

    @staticmethod
    def _config(**attrs):
        config = _mock_agent_config(**attrs)
        config.project_name = "proj"
        return config

    def _build(self, factory, config):
        manager = Mock()
        manager.get_conn.return_value = Mock(dialect="sqlite")
        with (
            patch("datus.tools.func_tool.database.db_manager_instance", return_value=manager),
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG"),
            patch("datus.tools.func_tool.database.metadata_fts_enabled", return_value=False),
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return factory(config)

    def test_create_dynamic_inherits_deployment_read_only(self):
        """``.read_only`` reports the posture the write paths will enforce, not
        the constructor argument — the factory passes no ``read_only`` at all,
        so anything reading the attribute would otherwise be told this MCP tool
        is writable on a hardened deployment."""
        tool = self._build(DBFuncTool.create_dynamic, self._config(sql_read_only=True))

        assert tool.read_only is True
        assert tool._read_only is False  # nothing passed it; the config supplied it

    def test_create_static_inherits_deployment_read_only(self):
        tool = self._build(DBFuncTool.create_static, self._config(sql_read_only=True))

        assert tool.read_only is True

    def test_create_dynamic_rejects_writes_end_to_end(self):
        """The property is only meaningful if execute_sql actually refuses."""
        tool = self._build(DBFuncTool.create_dynamic, self._config(sql_read_only=True))
        connector = Mock()
        connector.dialect = "sqlite"
        tool._get_connector = Mock(return_value=connector)

        result = tool.execute_sql("INSERT INTO users VALUES (1)")

        assert result.success == 0
        assert "read-only" in (result.error or "")
        connector.execute_insert.assert_not_called()

    def test_create_dynamic_stays_writable_by_default(self):
        tool = self._build(DBFuncTool.create_dynamic, self._config(sql_read_only=False))

        assert tool.read_only is False


class TestUploadsCatalogSqlGuard:
    """``execute_sql`` must not let model-authored SQL read arbitrary files.

    The uploads catalog is a DuckDB datasource with external file access on —
    that is what makes a lazy view over a spreadsheet work. Reached through a
    hand-written ``read_csv_auto('/etc/passwd')`` the same capability reads
    anything the process can see, routing around the filesystem path policy that
    is the agent's only containment boundary.
    """

    def _make_tool(self, *, datasources=("local_files",), default="local_files", registered=("jeffshop_q3",)):
        import contextlib

        from datus.tools.db_tools.db_manager import DBManager

        catalog = Mock()
        catalog.execute.return_value.fetchall.return_value = [(name, None) for name in registered]

        connector = Mock()
        connector.dialect = "duckdb"
        connector.get_databases.return_value = []
        connector.exclusive_connection = lambda: contextlib.nullcontext(catalog)
        manager = Mock(spec=DBManager)
        manager.get_conn.return_value = connector
        config = _mock_agent_config()
        config.current_datasource = default
        config.current_db_configs.return_value = {name: Mock(type="duckdb") for name in datasources}
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(manager, agent_config=config), connector

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM read_csv_auto('/etc/passwd')",
            "SELECT * FROM read_parquet('/data/secret.parquet')",
            "SELECT * FROM read_xlsx('/tmp/other.xlsx')",
            "SELECT read_text('/etc/hosts')",
            "SELECT * FROM glob('/**')",
            "SELECT * FROM (SELECT * FROM read_csv_auto('/etc/passwd')) t",
            # SQL is case-insensitive and tolerates space before the paren, so a
            # gate that is not would be trivially sidestepped.
            "SELECT * FROM READ_CSV_AUTO('/etc/passwd')",
            "SELECT * FROM Read_Csv_Auto('/etc/passwd')",
            "SELECT * FROM read_csv_auto ('/etc/passwd')",
        ],
    )
    def test_rejects_file_reading_sql_on_the_uploads_datasource(self, sql):
        tool, connector = self._make_tool()

        result = tool.execute_sql(sql, datasource="local_files")

        assert result.success == 0
        assert "load_file_as_table" in result.error
        # Rejected before touching the database at all.
        connector.execute_query.assert_not_called()

    def test_catalog_is_read_while_the_connector_lock_is_held(self):
        """The whole read, not just the handle, has to be inside the lock.

        Regression: the guard took the connection out of ``exclusive_connection``
        and queried it afterwards. ``DuckDBPyConnection`` is not thread-safe, and
        an LLM emitting parallel tool calls makes concurrent reads the normal
        case — observed in a real session as batches of N parallel ``execute_sql``
        where exactly N-1 were refused as "not a table registered" for a table
        that was registered, and under sustained load as a segfault.
        """
        import contextlib

        from datus.tools.db_tools.db_manager import DBManager

        held = []
        # Recorded rather than asserted inside the mock: the guard wraps the read
        # in ``except Exception``, so an assert raised in here would be swallowed
        # and the test would pass against the very bug it exists to catch.
        locked_during_read = []
        catalog = Mock()

        def execute(*args, **kwargs):
            locked_during_read.append(bool(held))
            result = Mock()
            result.fetchall.return_value = [("jeffshop_q3", None)]
            return result

        catalog.execute.side_effect = execute

        @contextlib.contextmanager
        def exclusive():
            held.append(True)
            try:
                yield catalog
            finally:
                held.pop()

        connector = Mock()
        connector.dialect = "duckdb"
        connector.get_databases.return_value = []
        connector.exclusive_connection = exclusive
        manager = Mock(spec=DBManager)
        manager.get_conn.return_value = connector
        config = _mock_agent_config()
        config.current_datasource = "local_files"
        config.current_db_configs.return_value = {"local_files": Mock(type="duckdb")}
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(manager, agent_config=config)

        result = tool.execute_sql("SELECT * FROM jeffshop_q3", datasource="local_files")

        assert locked_during_read, "the guard never read the catalog"
        assert all(locked_during_read), "catalog read happened outside exclusive_connection()"
        assert "not a table registered" not in (result.error or "")

    def test_unreadable_catalog_is_not_reported_as_an_unregistered_table(self):
        """A failed catalog read must not read as "your table does not exist".

        The two are opposite instructions to the model: one says retry, the other
        says the registration it just did was a no-op — so it re-registers, re-lists,
        and re-runs the same query. Distinguishable only if the read is allowed to
        fail rather than returning an empty catalog.
        """
        tool, _ = self._make_tool()
        tool._get_connector = Mock(side_effect=RuntimeError("database is locked"))

        result = tool.execute_sql("SELECT * FROM jeffshop_q3", datasource="local_files")

        assert result.success == 0
        assert "unavailable" in result.error
        assert "not a table registered" not in result.error

    def test_allows_ordinary_reads_of_registered_tables(self):
        """Asserting only "no rejection message" would also pass if the query
        never ran, so check it reached the connector."""
        tool, connector = self._make_tool()

        result = tool.execute_sql("SELECT region, sum(amount) FROM jeffshop_q3 GROUP BY 1", datasource="local_files")

        assert "not a table registered" not in (result.error or "")
        connector.execute_query.assert_called()

    def test_a_column_named_like_a_reader_is_not_mistaken_for_one(self):
        """The function gate matches ``name(``; a bare identifier must not trip it."""
        tool, connector = self._make_tool()

        result = tool.execute_sql("SELECT read_csv_auto FROM jeffshop_q3", datasource="local_files")

        assert "cannot be called directly" not in (result.error or "")
        connector.execute_query.assert_called()

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM '/data/tenants/other/proj/x.parquet'",
            "SELECT * FROM '/data/tenants/*/**/*.csv'",
            "SELECT a.k FROM jeffshop_q3 a JOIN '/tmp/y.parquet' b ON a.k = b.k",
        ],
    )
    def test_rejects_a_bare_path_used_as_a_table(self, sql):
        """DuckDB's replacement scan reads the file with no function call to catch
        and a statement that parses as a plain SELECT — the statement-class gate
        and a function-name check both pass it."""
        tool, connector = self._make_tool()

        result = tool.execute_sql(sql, datasource="local_files")

        assert result.success == 0
        assert "not a table registered" in result.error
        connector.execute_query.assert_not_called()

    def test_rejects_an_unregistered_table_name(self):
        tool, _ = self._make_tool()
        result = tool.execute_sql("SELECT * FROM someone_elses_table", datasource="local_files")
        assert result.success == 0

    def test_fails_closed_when_the_catalog_cannot_be_read(self):
        """Without the catalog there is no way to authorise a reference, and this
        gate is the boundary."""
        tool, connector = self._make_tool()
        connector.exclusive_connection = Mock(side_effect=RuntimeError("catalog gone"))

        result = tool.execute_sql("SELECT * FROM jeffshop_q3", datasource="local_files")

        assert result.success == 0
        assert "unavailable" in result.error

    @pytest.mark.parametrize(
        "sql",
        [
            "ATTACH '/data/tenants/other/x.duckdb' AS other",
            "COPY (SELECT 1) TO '/tmp/exfil.csv'",
            "CREATE TABLE t AS SELECT 1",
            "DROP VIEW jeffshop_q3",
            "INSTALL httpfs",
            "EXPORT DATABASE '/tmp/dump'",
        ],
    )
    def test_rejects_non_read_statements_on_the_uploads_datasource(self, sql):
        """A function-name check alone is not a boundary: ATTACH and COPY TO are
        not function calls, and both reach outside the project — ATTACH onto the
        shared tenant volume, COPY TO writing anywhere the process can."""
        tool, connector = self._make_tool()

        result = tool.execute_sql(sql, datasource="local_files")

        assert result.success == 0
        assert "Only read queries are allowed" in result.error
        connector.execute_ddl.assert_not_called()
        connector.execute_query.assert_not_called()

    def test_unparseable_sql_fails_closed_on_the_uploads_datasource(self):
        tool, _ = self._make_tool()
        result = tool.execute_sql("this is not sql at all ((", datasource="local_files")
        assert result.success == 0

    def test_writes_still_allowed_on_other_datasources(self):
        """The tightening is scoped to the uploads catalog; a project's own
        datasource keeps whatever posture it had."""
        tool, connector = self._make_tool(datasources=("warehouse", "local_files"), default="warehouse")

        result = tool.execute_sql("CREATE TABLE t AS SELECT 1", datasource="warehouse")

        assert "Only read queries are allowed" not in (result.error or "")
        connector.execute_ddl.assert_called()

    def test_guard_applies_when_the_uploads_catalog_is_the_default(self):
        """Reached via the default route, not just an explicit datasource argument."""
        tool, connector = self._make_tool()
        result = tool.execute_sql("SELECT * FROM read_csv_auto('/etc/passwd')")
        assert result.success == 0
        assert "load_file_as_table" in result.error
        connector.execute_query.assert_not_called()

    def test_other_datasources_are_left_alone(self):
        """A project's own DuckDB datasource had this reach before uploads existed;
        narrowing it here would be an unrelated behaviour change."""
        tool, _ = self._make_tool(datasources=("warehouse", "local_files"), default="warehouse")
        result = tool.execute_sql("SELECT * FROM read_csv_auto('/data/lake/x.csv')", datasource="warehouse")
        assert "load_file_as_table" not in (result.error or "")


class TestLoadFileAsTableIsExposed:
    """Two different surfaces, and only one of them is what the model sees.

    ``all_tools_name()`` feeds VALID_TOOL_METHODS and the permission registry;
    ``available_tools()`` is the list actually handed to the LLM and is
    hand-curated, not derived from the former. A tool present in the first and
    absent from the second is registered, permissioned, catalogued — and
    uncallable.
    """

    def _make_tool(self, datasources):
        from datus.tools.db_tools.db_manager import DBManager

        connector = Mock()
        connector.dialect = "postgresql"
        connector.get_databases.return_value = []
        manager = Mock(spec=DBManager)
        manager.get_conn.return_value = connector
        config = _mock_agent_config()
        config.current_datasource = datasources[0]
        config.current_db_configs.return_value = {name: Mock(type="postgresql") for name in datasources}
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(manager, agent_config=config)

    def test_registered_on_the_agent_tool_surface(self):
        """VALID_TOOL_METHODS and the permission registry both derive from this."""
        assert "load_file_as_table" in DBFuncTool.all_tools_name()

    def test_not_treated_as_an_internal_dispatch_helper(self):
        assert "load_file_as_table" not in DBFuncTool._INTERNAL_SQL_METHODS

    def test_mounted_for_the_llm_when_the_uploads_catalog_exists(self):
        tool = self._make_tool(["warehouse", "local_files"])
        assert "load_file_as_table" in [item.name for item in tool.available_tools()]

    def test_not_mounted_without_an_uploads_catalog(self):
        """On a CLI install there is no catalog to load into, so the tool would
        exist only to return "datasource not configured"."""
        tool = self._make_tool(["warehouse"])
        assert "load_file_as_table" not in [item.name for item in tool.available_tools()]

    def test_single_connector_mode_still_builds_its_tool_list(self):
        """Legacy single-connector mode names no datasources at all; the mount
        check must not assume the attribute exists."""
        connector = Mock()
        connector.dialect = "sqlite"
        connector.get_databases.return_value = []
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(connector)

        names = [item.name for item in tool.available_tools()]
        assert "execute_sql" in names
        assert "load_file_as_table" not in names


class TestVscodeSessionsAreRefused:
    """A vscode session's workspace lives on the client, so there is no
    server-side path to resolve — ``_resolve_workspace_root`` reports "." there
    rather than leaking the daemon CWD. Resolving against that would produce a
    "File not found" nobody can explain.
    """

    def _make_tool(self, client_source):
        import contextlib

        from datus.tools.db_tools.db_manager import DBManager

        connector = Mock()
        connector.dialect = "duckdb"
        connector.get_databases.return_value = []
        connector.exclusive_connection = lambda: contextlib.nullcontext(Mock())
        manager = Mock(spec=DBManager)
        manager.get_conn.return_value = connector
        config = _mock_agent_config()
        config._client_source = client_source
        config.current_datasource = "local_files"
        config.current_db_configs.return_value = {"local_files": Mock(type="duckdb")}
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(manager, agent_config=config)

    def test_vscode_gets_an_explanation_not_a_missing_file(self):
        tool = self._make_tool("vscode")

        result = tool.load_file_as_table(path="book.xlsx")

        assert result.success == 0
        assert "vscode" in result.error
        assert "not found" not in result.error.lower()

    def test_web_sessions_are_unaffected(self):
        tool = self._make_tool("web")
        result = tool.load_file_as_table(path="definitely-absent.xlsx")
        assert "vscode" not in (result.error or "")


class TestFilesystemRootPlumbing:
    """``load_file_as_table`` resolves relative paths against this anchor, and it
    must be the same one the node's filesystem tools use — otherwise a path the
    model just got back from ``glob`` lands somewhere else.
    """

    def test_explicit_root_wins_over_project_root(self, tmp_path):
        from datus.tools.db_tools.db_manager import DBManager

        (tmp_path / "here.csv").write_text("a\n1\n")
        manager = Mock(spec=DBManager)
        manager.get_conn.return_value = Mock(dialect="duckdb", get_databases=Mock(return_value=[]))
        config = _mock_agent_config()
        config.project_root = str(tmp_path / "elsewhere")
        config.filesystem_allowlist = None
        config.path_manager = None
        config.current_datasource = "local_files"
        config.current_db_configs.return_value = {"local_files": Mock(type="duckdb")}
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(manager, agent_config=config, filesystem_root=str(tmp_path))

        resolved = tool._resolve_data_file("here.csv")
        assert resolved.resolved == (tmp_path / "here.csv")

    def test_every_node_construction_site_passes_the_anchor(self):
        """A site that forgets it is the whole failure mode, and there are
        nineteen of them across eleven files — so it is asserted rather than left
        to review. Reads the sources instead of booting every node: what
        regresses is a new construction site, and that is a diff between two
        lists.

        Deliberately not solved by a shared factory: several node tests patch
        ``<module>.DBFuncTool`` as their seam, and routing construction elsewhere
        breaks that seam for reasons unrelated to what they test.
        """
        import pathlib
        import re

        node_dir = pathlib.Path(__file__).resolve().parents[4] / "datus" / "agent" / "node"
        offenders = []
        for path in sorted(node_dir.glob("*.py")):
            text = path.read_text()
            for match in re.finditer(r"DBFuncTool\(", text):
                # Slice out this call's argument list by paren balance.
                start_index = match.end()
                depth, index = 1, start_index
                while index < len(text) and depth:
                    depth += (text[index] == "(") - (text[index] == ")")
                    index += 1
                if "filesystem_root" not in text[start_index:index]:
                    line = text.count("\n", 0, match.start()) + 1
                    offenders.append(f"{path.name}:{line}")
        assert offenders == [], f"DBFuncTool built without filesystem_root: {offenders}"


class TestHeaderRowNormalisation:
    """``header_row`` is the one optional the tool schema types as a number, so
    the "unspecified" value a model sends is 0 rather than an omitted key."""

    def _tool(self):
        import contextlib

        from datus.tools.db_tools.db_manager import DBManager

        connector = Mock()
        connector.dialect = "duckdb"
        connector.db_path = "/tmp/local_files.duckdb"
        connector.get_databases.return_value = []
        connector.exclusive_connection = lambda: contextlib.nullcontext(Mock())
        manager = Mock(spec=DBManager)
        manager.get_conn.return_value = connector
        config = _mock_agent_config()
        config._client_source = "web"
        config.current_datasource = "local_files"
        config.current_db_configs.return_value = {"local_files": Mock(type="duckdb")}
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(manager, agent_config=config)

    def _capture(self, tmp_path, **kwargs):
        from types import SimpleNamespace

        tool = self._tool()
        book = tmp_path / "book.xlsx"
        book.write_bytes(b"")
        tool._resolve_data_file = lambda path: SimpleNamespace(resolved=book, display=path)
        captured = {}

        def load_file(*_args, **call_kwargs):
            captured.update(call_kwargs)
            return [], []

        with (
            patch("datus.tools.func_tool.database.load_file", side_effect=load_file),
            patch("datus.tools.func_tool.database.registered_objects", return_value={}),
        ):
            tool.load_file_as_table(path="book.xlsx", **kwargs)
        return captured

    def test_zero_means_unspecified(self, tmp_path):
        """Rows are 1-based, so 0 taken literally fails every sheet with
        "header_row must be 1 or greater" and the whole file reads as unloadable.
        Observed in real traffic: the session's inspect call carried
        ``header_row: 0``, where it happened to be ignored."""
        assert self._capture(tmp_path, header_row=0)["header_row"] is None

    def test_a_real_header_row_is_passed_through(self, tmp_path):
        assert self._capture(tmp_path, header_row=3)["header_row"] == 3

    def test_a_negative_row_is_still_a_mistake(self, tmp_path):
        """Not folded into "unspecified": nothing sends -1 by accident, and
        silently auto-detecting would hide a caller bug."""
        assert self._capture(tmp_path, header_row=-1)["header_row"] == -1
>>>>>>> bb84920 ([BugFix] Serialise the uploads catalog read so parallel tool calls stop being refused (#1340))
