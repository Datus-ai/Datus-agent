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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG"),
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_profile,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            mock_profile.return_value.get_size.side_effect = RuntimeError("storage unavailable")

            tool = DBFuncTool(mock_connector, agent_config=mock_config)

        assert tool.has_semantic_datasets is False
        assert tool._semantic_datasets is None


class TestDBFuncToolExecuteDDL:
    """Tests for DBFuncTool.execute_ddl method."""

    def _make_tool(self, connector):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
        tool._semantic_datasets = Mock()
        tool.has_semantic_models = True
        tool._semantic_storage = Mock()
        tool._semantic_datasets.get_table_projection.return_value = {
            "semantic_model_name": "shop",
            "dataset_name": "orders",
            "source_table": "orders",
            "description": "Orders dataset",
            "ai_context_json": "",
            "fields": [
                {
                    "name": "order_id",
                    "expr": "order_id",
                    "is_primary_key": True,
                    "description": "Order key",
                },
                {"name": "amount", "expr": "amount", "description": "Order amount"},
            ],
            "relationships": [],
            "alternatives": [],
            "yaml_path": "/tmp/orders.yml",
        }

        result = tool.describe_table("orders")

        assert result.success == 1, f"Expected success but got error: {result.error}"
        cols = {c["name"]: c for c in result.result["columns"]}
        assert cols["order_id"]["pk"] is True
        assert cols["order_id"]["nullable"] is False
        assert cols["order_id"]["semantic_role"] == "primary_key"
        assert "pk" not in cols["amount"]
        assert cols["amount"]["semantic_role"] == "field"


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
        tool._semantic_datasets = Mock()
        tool.has_semantic_models = True
        tool._semantic_storage = Mock()
        tool._semantic_datasets.get_table_projection.return_value = {
            "semantic_model_name": "shop",
            "dataset_name": "orders",
            "source_table": "orders",
            "description": "Orders dataset",
            "ai_context_json": '{"synonyms": ["purchases"]}',
            "fields": [
                {"name": "order_id", "expr": "order_id", "is_primary_key": True, "description": "Order key"},
                {
                    "name": "order_date",
                    "expr": "order_date",
                    "is_dimension": True,
                    "is_time": True,
                    "description": "Order date",
                },
                {"name": "amount", "expr": "amount", "description": "Order amount"},
            ],
            "relationships": [{"name": "orders_to_customers", "to_dataset": "customers"}],
            "alternatives": [],
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
        assert "semantic_model_name" not in result.result["semantic"]
        assert "dataset_name" not in result.result["semantic"]
        assert "yaml_path" not in result.result["semantic"]
        columns = {col["name"]: col for col in result.result["columns"]}
        assert columns["order_id"]["semantic_role"] == "primary_key"
        assert "is_entity_key" not in columns["order_id"]
        assert columns["order_date"]["is_dimension"] is True
        assert columns["order_date"]["semantic_role"] == "time_dimension"
        assert columns["amount"]["semantic_role"] == "field"
        assert "is_measure" not in columns["amount"]
        assert columns["amount"]["comment"] == "Order amount"

    def test_describe_table_falls_back_to_the_dataset_name_when_unnamed(self):
        """A dataset row always names the table; the model name is the last resort."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.get_schema.return_value = [{"name": "order_id", "type": "INTEGER", "comment": ""}]

        tool = DBFuncTool(mock_connector)
        tool._semantic_datasets = Mock()
        tool._semantic_datasets.get_table_projection.return_value = {
            "semantic_model_name": "orders_analytics",
            "dataset_name": "",
            "source_table": "orders",
            "description": "Orders",
            "fields": [],
            "relationships": [],
            "alternatives": [],
        }

        result = tool.describe_table("orders")

        assert result.success == 1
        assert result.result["table"]["name"] == "orders_analytics"


class TestDescribeTableMultipleDatasets:
    """One physical table may be modelled by several semantic models.

    The primary dataset alone supplies the meaning; the rest are navigation.
    Merging them would assert a join graph that exists in neither model, since
    OSI relationships reference dataset names local to their own model.
    """

    def _make_tool(self, projection):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.get_schema.return_value = [
            {"name": "order_id", "type": "INTEGER", "comment": ""},
            {"name": "amount", "type": "DOUBLE", "comment": ""},
        ]
        tool = DBFuncTool(mock_connector)
        tool._semantic_datasets = Mock()
        tool._semantic_datasets.get_table_projection.return_value = projection
        return tool

    @staticmethod
    def _dataset_row(model, *, description):
        return {
            "semantic_model_name": model,
            "dataset_name": "orders",
            "source_table": "orders",
            "description": description,
            "ai_context_json": "",
            "yaml_path": f"/tmp/{model}.yml",
        }

    def _projection(self, *, with_alternative: bool):
        return {
            **self._dataset_row("fulfillment", description="Orders being shipped"),
            "fields": [{"name": "amount", "expr": "amount", "description": "Shipped amount"}],
            "relationships": [{"name": "orders_to_shipments", "from_dataset": "orders", "to_dataset": "shipments"}],
            "alternatives": [self._dataset_row("sales", description="Orders booked")] if with_alternative else [],
        }

    def test_single_dataset_output_carries_no_disambiguation_keys(self):
        """The common case must stay byte-identical for existing consumers."""
        tool = self._make_tool(self._projection(with_alternative=False))

        table = tool.describe_table("orders").result["table"]

        assert "alternatives" not in table
        assert "semantic_model" not in table

    def test_primary_dataset_alone_supplies_meaning(self):
        tool = self._make_tool(self._projection(with_alternative=True))

        result = tool.describe_table("orders").result

        assert result["table"]["description"] == "Orders being shipped"
        assert result["table"]["semantic_model"] == "fulfillment"
        columns = {col["name"]: col for col in result["columns"]}
        assert columns["amount"]["semantic_description"] == "Shipped amount"

    def test_relationships_are_never_merged_across_models(self):
        tool = self._make_tool(self._projection(with_alternative=True))

        relationships = tool.describe_table("orders").result["semantic"]["relationships"]

        assert relationships == [{"name": "orders_to_shipments", "from_dataset": "orders", "to_dataset": "shipments"}]

    def test_other_models_are_surfaced_as_navigation(self):
        tool = self._make_tool(self._projection(with_alternative=True))

        alternatives = tool.describe_table("orders").result["table"]["alternatives"]

        assert alternatives == [
            {
                "semantic_model": "sales",
                "dataset": "orders",
                "description": "Orders booked",
                "yaml_path": "/tmp/sales.yml",
            }
        ]

    def test_semantic_model_argument_is_passed_down_to_storage(self):
        tool = self._make_tool(self._projection(with_alternative=False))

        tool.describe_table("orders", semantic_model="sales")

        assert tool._semantic_datasets.get_table_projection.call_args.kwargs["semantic_model"] == "sales"

    def test_a_model_that_does_not_describe_the_table_is_an_error(self):
        """Degrading to physical columns would read as "this table has no
        semantic model", which sends the caller off to guess at raw columns
        when the name is merely misspelled."""
        tool = self._make_tool(self._projection(with_alternative=True))
        tool._semantic_datasets.get_table_projection.return_value = None
        tool._semantic_datasets.list_datasets.return_value = [
            {"semantic_model_name": "fulfillment"},
            {"semantic_model_name": "sales"},
        ]

        result = tool.describe_table("orders", semantic_model="salez")

        assert result.success == 0
        assert "salez" in result.error
        assert "fulfillment" in result.error and "sales" in result.error

    def test_an_unmodelled_table_still_degrades_to_physical_columns(self):
        """No model names to offer means the table simply is not modelled, so
        the physical schema is the honest answer rather than an error."""
        tool = self._make_tool(self._projection(with_alternative=True))
        tool._semantic_datasets.get_table_projection.return_value = None
        tool._semantic_datasets.list_datasets.return_value = []

        result = tool.describe_table("orders", semantic_model="sales")

        assert result.success == 1
        assert "table" not in result.result


class TestDescribeTableWithoutProjection:
    """After an upgrade the projection starts empty until it is re-synced.

    describe_table must degrade to the physical schema rather than fail, and
    must say how to restore the semantics instead of going silent.
    """

    def _make_tool(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        mock_connector.get_schema.return_value = [{"name": "order_id", "type": "INTEGER", "comment": ""}]
        tool = DBFuncTool(mock_connector)
        tool._semantic_datasets = Mock()
        tool._semantic_datasets.get_table_projection.return_value = None
        return tool

    def test_empty_projection_degrades_to_physical_schema(self):
        result = self._make_tool().describe_table("orders")

        assert result.success == 1
        assert result.result["columns"] == [{"name": "order_id", "type": "INTEGER", "comment": ""}]
        assert "table" not in result.result
        assert "semantic" not in result.result

    def test_stale_projection_is_reported_once_per_project(self):
        from datus.tools.func_tool import database as database_module

        database_module._STALE_PROJECTION_WARNED.clear()
        config = Mock()
        config.project_name = "shop"

        with (
            patch("datus.storage.semantic_dataset.store.semantic_projection_is_stale", return_value=True),
            patch.object(database_module, "logger") as mock_logger,
        ):
            database_module._warn_once_if_projection_is_stale(config)
            database_module._warn_once_if_projection_is_stale(config)

        assert mock_logger.warning.call_count == 1
        assert "sync-yaml" in mock_logger.warning.call_args.args[0]
        database_module._STALE_PROJECTION_WARNED.clear()


class TestSearchTableWithSharedTable:
    """A table modelled by two semantic models must not break table discovery.

    The pre-profile lookup raises in exactly that case, and search_table had no
    inner guard, so a single shared table failed the entire search.
    """

    def _make_tool(self):
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []
        tool = DBFuncTool(mock_connector)
        tool.has_schema = True
        tool.schema_rag = Mock()
        tool.schema_rag.search_similar.return_value = (Mock(num_rows=1), None)
        tool._metadata_search_rows = Mock(
            return_value=[
                {
                    "catalog_name": "",
                    "database_name": "shop",
                    "schema_name": "",
                    "table_name": "orders",
                    "identifier": "shop.orders",
                    "description": "raw metadata description",
                }
            ]
        )
        tool._sample_rows_by_identifier = Mock(return_value={})
        tool._search_table_result_row = lambda row, _samples: row
        return tool

    def test_search_survives_a_table_owned_by_two_models(self):
        tool = self._make_tool()
        tool._semantic_datasets = Mock()
        tool._semantic_datasets.list_datasets.return_value = [
            {"semantic_model_name": "fulfillment", "dataset_name": "orders", "description": "Orders being shipped"},
            {"semantic_model_name": "sales", "dataset_name": "orders", "description": "Orders booked"},
        ]

        result = tool.search_table("orders")

        assert result.success == 1
        assert result.result["metadata"][0]["description"] == "Orders being shipped"

    def test_raising_legacy_lookup_does_not_fail_the_search(self):
        tool = self._make_tool()
        tool._semantic_datasets = None
        tool.has_semantic_models = True
        tool._semantic_storage = Mock()
        tool._semantic_storage.get_semantic_model.side_effect = RuntimeError(
            "Table `orders` belongs to multiple semantic models"
        )

        result = tool.search_table("orders")

        assert result.success == 1
        assert result.result["metadata"][0]["description"] == "raw metadata description"


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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_connector, read_only=True)

        result = tool.execute_sql(sql)

        assert result.success == 0
        assert "read-only" in (result.error or "")
        mock_connector.execute_insert.assert_not_called()
        mock_connector.execute_ddl.assert_not_called()

<<<<<<< HEAD
=======
    # ------------------------------------------------------------------ #
    # Deployment-wide ``agent.sql_read_only``                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build(connector, **kwargs):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector, **kwargs)

    @staticmethod
    def _connector():
        connector = Mock()
        connector.dialect = "sqlite"
        connector.get_databases.return_value = []
        connector.execute_insert.return_value = Mock(success=True, row_count=1)
        return connector

    def test_config_read_only_rejects_writes_when_caller_did_not_ask(self):
        """The deployment switch must reach callers that never pass read_only —
        which is 19 of the 24 DBFuncTool construction sites, plus the MCP server.
        """
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))

        result = tool.execute_sql("INSERT INTO users VALUES (1)")

        assert result.success == 0
        assert "read-only" in (result.error or "")
        connector.execute_insert.assert_not_called()

    def test_config_read_only_still_allows_select(self):
        connector = self._connector()
        connector.execute_query.return_value = Mock(success=True, sql_return=[{"a": 1}])
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))
        tool.compressor.compress = Mock(return_value={"original_rows": 1, "compressed_data": "a\n1"})

        assert tool.execute_sql("SELECT * FROM users").success == 1

    def test_instance_flag_wins_when_config_is_writable(self):
        """OR semantics, direction 1: the config may not relax a caller that
        already asked for read-only (Explore, ask_report, the validator copy).
        """
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=False), read_only=True)

        assert tool.read_only is True
        assert tool.execute_sql("INSERT INTO users VALUES (1)").success == 0
        connector.execute_insert.assert_not_called()

    def test_config_flag_wins_when_instance_is_writable(self):
        """OR semantics, direction 2."""
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True), read_only=False)

        assert tool.read_only is True

    def test_no_agent_config_is_not_read_only(self):
        connector = self._connector()
        tool = self._build(connector)

        assert tool.read_only is False
        assert tool.execute_sql("INSERT INTO users VALUES (1)").success == 1

    def test_config_without_the_attribute_is_not_read_only(self):
        """Old / duck-typed config objects predating the switch must not break."""

        class LegacyConfig:
            def active_model(self):
                return Mock(model="gpt-4")

        connector = self._connector()
        tool = self._build(connector, agent_config=LegacyConfig())

        assert tool.read_only is False

    def test_string_false_is_not_read_only(self):
        """Proves ``_coerce_bool``, not ``bool()``: ``bool("false")`` is True and
        would wedge a duck-typed config into permanent read-only.
        """
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only="false"))

        assert tool.read_only is False

    def test_config_hardened_after_construction_is_honoured(self):
        """The API hands nodes a per-request config clone, so the flag is
        resolved per access rather than snapshotted in __init__.
        """
        connector = self._connector()
        config = _mock_agent_config(sql_read_only=False)
        tool = self._build(connector, agent_config=config)
        assert tool.read_only is False

        config.sql_read_only = True

        assert tool.read_only is True
        assert tool.execute_sql("INSERT INTO users VALUES (1)").success == 0

    def test_refusal_is_logged_with_its_source(self, caplog):
        """A refusal is the event an operator wants to see -- on a deployment
        running third-party content it means that content just tried to write.
        Successful reads are already logged in read_query, so a silent refusal
        would record the benign path and drop the notable one.
        """
        import logging

        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))

        with caplog.at_level(logging.WARNING):
            tool.execute_sql("INSERT INTO users VALUES (1)")

        assert "rejected by read-only policy" in caplog.text
        # `source` separates a hardened deployment from a read-only agent doing
        # its job -- "investigate this" vs "working as intended".
        assert "deployment" in caplog.text

    def test_refusal_log_attributes_agent_read_only_separately(self, caplog):
        import logging

        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=False), read_only=True)

        with caplog.at_level(logging.WARNING):
            tool.execute_sql("INSERT INTO users VALUES (1)")

        assert "rejected by read-only policy" in caplog.text
        assert "agent" in caplog.text

    def test_ddl_refusal_logs_the_specific_keyword(self, caplog):
        """`ddl` alone cannot tell an operator whether third-party content tried
        to CREATE a table or DROP one, and those warrant different responses.
        """
        import logging

        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))

        with caplog.at_level(logging.WARNING):
            tool.execute_sql("DROP TABLE users")
        assert logged_fields(caplog.text)["sql_type"] == "drop"

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            tool.execute_sql("CREATE TABLE t (id INT)")
        assert logged_fields(caplog.text)["sql_type"] == "create"

    def test_multi_statement_refusal_is_logged(self, caplog):
        """The sneakiest input in the set must not be the one with no audit
        trail. ``parse_sql_type`` reads only the first statement, so this
        arrives as a SELECT and is refused by the multi-statement rule rather
        than by the read-only gate -- a different code path, which is exactly
        how it went unlogged.
        """
        import logging

        connector = self._connector()
        connector.execute_query.return_value = Mock(success=True, sql_return=[])
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))

        with caplog.at_level(logging.WARNING):
            result = tool.execute_sql("SELECT 1; DROP TABLE users")

        assert result.success == 0
        assert "statement-shape rules" in caplog.text
        # The violation CODE, not the prose: `validate_read_only_sql` returns a
        # code so each caller words its own error, which also gives the log a
        # stable value to aggregate on.
        assert logged_fields(caplog.text)["rule"] == "multi_statement"
        # And the caller's own wording still reaches the model.
        assert "Multi-statement" in (result.error or "")

    def test_shallow_copy_can_still_force_read_only(self):
        """Pins the contract datus.validation.llm_runner depends on: it copies a
        write-capable tool and flips ``.read_only`` to True on the copy.
        """
        import copy as _copy

        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=False))
        read_only_tool = _copy.copy(tool)
        read_only_tool.read_only = True

        assert read_only_tool.read_only is True
        assert tool.read_only is False

    def test_assignment_can_tighten_but_not_relax(self):
        """The setter mirrors ``AgentConfig.harden_sql_read_only``: a caller may
        harden an instance, but may not hand a write-capable view of a read-only
        deployment to anything downstream.
        """
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=False))

        tool.read_only = True
        assert tool.read_only is True

        tool.read_only = False
        assert tool.read_only is True

    def test_deployment_read_only_cannot_be_relaxed_by_assignment(self):
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))

        tool.read_only = False

        assert tool.read_only is True
        assert tool.execute_sql("INSERT INTO users VALUES (1)").success == 0

>>>>>>> cfac0a4 ([Refactor] Key the semantic projection on datasets so a table can be modelled more than once (#1327))
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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


<<<<<<< HEAD
=======
class TestDBFuncToolWritePathsHonorReadOnly:
    """Every write entry point, not just ``execute_sql``.

    ``execute_sql`` dispatches to ``execute_write``/``execute_ddl`` and gates
    before it does, but all three are callable directly, and gen_job mounts
    ``transfer_query_result`` as a tool of its own. A deployment-wide read-only
    posture that only covered the dispatcher would leave those reachable.
    """

    @staticmethod
    def _build(connector, **kwargs):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
        ):
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            return DBFuncTool(connector, **kwargs)

    @staticmethod
    def _connector():
        connector = Mock()
        connector.dialect = "sqlite"
        connector.get_databases.return_value = []
        connector.execute_insert.return_value = Mock(success=True, row_count=1)
        connector.execute_ddl.return_value = Mock(success=True)
        return connector

    def test_execute_ddl_is_refused(self):
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))

        result = tool.execute_ddl("CREATE TABLE t (a int)")

        assert result.success == 0
        assert "read-only" in (result.error or "")
        connector.execute_ddl.assert_not_called()

    def test_execute_write_is_refused(self):
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))

        result = tool.execute_write("INSERT INTO users VALUES (1)")

        assert result.success == 0
        assert "read-only" in (result.error or "")
        connector.execute_insert.assert_not_called()

    def test_transfer_query_result_is_refused(self):
        """``source_sql`` is read-only, but the transfer WRITES to the target
        datasource — CREATE TABLE / TRUNCATE / INSERT — without ever going
        through ``execute_sql``.
        """
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=True))
        tool._get_connector = Mock(side_effect=AssertionError("must refuse before touching a connector"))

        result = tool.transfer_query_result(
            source_sql="SELECT * FROM users",
            target_table="copy_of_users",
            target_datasource="warehouse",
        )

        assert result.success == 0
        assert "read-only" in (result.error or "")

    def test_read_only_agent_is_refused_too(self):
        """The gate is the effective posture, so a read-only agent on a writable
        deployment is covered by the same check.
        """
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=False), read_only=True)

        assert tool.execute_ddl("CREATE TABLE t (a int)").success == 0
        assert tool.execute_write("INSERT INTO users VALUES (1)").success == 0

    def test_write_paths_still_work_by_default(self):
        """The gate is opt-in — the default posture is unchanged."""
        connector = self._connector()
        tool = self._build(connector, agent_config=_mock_agent_config(sql_read_only=False))

        assert tool.execute_ddl("CREATE TABLE t (a int)").success == 1
        connector.execute_ddl.assert_called_once()


>>>>>>> cfac0a4 ([Refactor] Key the semantic projection on datasets so a table can be modelled more than once (#1327))
class TestDBFuncToolExecuteReadEnforced:
    """execute_read_enforced: the single policy-enforced raw-read path shared by
    read_query and the report/dashboard artifact query executors. It must reject
    multi-statement / non-read SQL and apply the SQL policy before the statement
    reaches the engine — the guard the artifact save paths previously bypassed."""

    def _make_tool(self, connector, agent_config=None):
        with (
            patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
            patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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
>>>>>>> cfac0a4 ([Refactor] Key the semantic projection on datasets so a table can be modelled more than once (#1327))
