"""
Test cases for DBFuncTool class in datus/tools/tools.py
"""

from unittest.mock import Mock

import pytest

from datus.tools.db_tools import BaseSqlConnector
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.constants import SUPPORT_DATABASE_DIALECTS, SUPPORT_SCHEMA_DIALECTS, DBType


class FakeRecordBatch:
    """Minimal Arrow-like table for select/to_pylist behavior in tests."""

    def __init__(self, rows):
        self._rows = rows

    def select(self, fields):
        selected = [{field: row.get(field) for field in fields} for row in self._rows]
        return FakeRecordBatch(selected)

    def to_pylist(self):
        return list(self._rows)


@pytest.fixture
def mock_connector():
    """Create a mock database connector."""
    connector = Mock(spec=BaseSqlConnector)
    connector.dialect = "postgresql"
    connector.catalog_name = ""
    connector.database_name = "db1"
    connector.schema_name = "schema1"

    # Setup mock return values
    connector.get_catalogs.return_value = ["catalog1", "catalog2"]
    connector.get_databases.return_value = ["db1", "db2"]
    connector.get_schemas.return_value = ["schema1", "schema2"]
    connector.get_tables.return_value = ["users", "orders"]
    connector.get_views.return_value = ["user_view", "order_view"]
    connector.get_materialized_views.return_value = ["sales_mv"]
    connector.get_schema.return_value = [
        {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
        {"column_name": "name", "data_type": "varchar", "is_nullable": "YES"},
    ]

    # Mock execute_query result
    mock_query_result = Mock()
    mock_query_result.success = True
    mock_query_result.sql_return = [{"id": 1, "name": "test"}]
    connector.execute_query.return_value = mock_query_result

    return connector


@pytest.fixture
def db_func_tool(mock_connector):
    """Create a DBFuncTool instance with mocked connector."""
    return DBFuncTool(mock_connector)


@pytest.fixture
def scoped_db_func_tool(mock_connector):
    """Create a DBFuncTool instance with scoped tables."""
    return DBFuncTool(
        mock_connector,
        scoped_tables={"db1.schema1.orders", "db1.schema1.user_view", "db2.*.orders", "*.schema1.*"},
    )


class TestDBFuncTool:
    """Test cases for DBFuncTool class."""

    def test_initialization(self, db_func_tool, mock_connector):
        """Test that DBFuncTool initializes correctly."""
        assert db_func_tool.connector == mock_connector
        assert hasattr(db_func_tool, "compressor")

    def test_available_tools(self, db_func_tool, mock_connector):
        """Test that available_tools returns correct tools based on dialect."""
        tools = db_func_tool.available_tools()

        # Should have base tools plus dialect-specific tools
        expected_tool_count = 4  # list_tables, describe_table, read_query, get_table_ddl
        if mock_connector.dialect in SUPPORT_DATABASE_DIALECTS:
            expected_tool_count += 1
        if mock_connector.dialect in SUPPORT_SCHEMA_DIALECTS:
            expected_tool_count += 1

        assert len(tools) == expected_tool_count

        # Verify tool names
        tool_names = {tool.name for tool in tools}
        expected_base_tools = {"list_tables", "describe_table", "read_query", "get_table_ddl"}

        assert expected_base_tools.issubset(tool_names)
        if mock_connector.dialect in SUPPORT_DATABASE_DIALECTS:
            assert "list_databases" in tool_names
        if mock_connector.dialect in SUPPORT_SCHEMA_DIALECTS:
            assert "list_schemas" in tool_names
        assert "search_table" not in tool_names

    def test_list_databases_success(self, db_func_tool, mock_connector):
        """Test successful list_databases execution."""
        result = db_func_tool.list_databases(catalog="test_catalog", include_sys=True)

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert result.result == ["db1", "db2"]

        mock_connector.get_databases.assert_called_once_with("test_catalog", include_sys=True)

    def test_list_databases_default_params(self, db_func_tool, mock_connector):
        """Test list_databases with default parameters."""
        result = db_func_tool.list_databases()

        assert result.success == 1
        mock_connector.get_databases.assert_called_once_with("", include_sys=False)

    def test_list_databases_failure(self, db_func_tool, mock_connector):
        """Test list_databases with exception."""
        mock_connector.get_databases.side_effect = Exception("Database retrieval failed")

        result = db_func_tool.list_databases()

        assert result.success == 0
        assert "Database retrieval failed" in result.error

    def test_list_schemas_success(self, db_func_tool, mock_connector):
        """Test successful list_schemas execution."""
        result = db_func_tool.list_schemas(catalog="test_catalog", database="test_db", include_sys=True)

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert result.result == ["schema1", "schema2"]

        mock_connector.get_schemas.assert_called_once_with("test_catalog", "test_db", include_sys=True)

    def test_list_schemas_default_params(self, db_func_tool, mock_connector):
        """Test list_schemas with default parameters."""
        result = db_func_tool.list_schemas()

        assert result.success == 1
        mock_connector.get_schemas.assert_called_once_with("", "", include_sys=False)

    def test_list_schemas_failure(self, db_func_tool, mock_connector):
        """Test list_schemas with exception."""
        mock_connector.get_schemas.side_effect = Exception("Schema retrieval failed")

        result = db_func_tool.list_schemas()

        assert result.success == 0
        assert "Schema retrieval failed" in result.error

    def test_list_tables_success_with_views(self, db_func_tool, mock_connector):
        """Test successful list_tables execution including views."""
        result = db_func_tool.list_tables(
            catalog="test_catalog", database="test_db", schema_name="test_schema", include_views=True
        )

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None

        # Should include tables, views, and materialized views
        expected_result = [
            {"type": "table", "name": "users"},
            {"type": "table", "name": "orders"},
            {"type": "view", "name": "user_view"},
            {"type": "view", "name": "order_view"},
            {"type": "materialized_view", "name": "sales_mv"},
        ]

        assert len(result.result) == len(expected_result)

        # Verify tables were called
        mock_connector.get_tables.assert_called_once_with("test_catalog", "test_db", "test_schema")
        mock_connector.get_views.assert_called_once_with("test_catalog", "test_db", "test_schema")
        mock_connector.get_materialized_views.assert_called_once_with("test_catalog", "test_db", "test_schema")

    def test_list_tables_with_scope(self, scoped_db_func_tool):
        """Scoped tables should filter combined list of tables and views."""
        result = scoped_db_func_tool.list_tables(include_views=True)

        assert result.success == 1
        assert [item["name"] for item in result.result] == [
            "users",
            "orders",
            "user_view",
            "order_view",
            "sales_mv",
        ]

    def test_list_databases_respects_scope(self, scoped_db_func_tool, mock_connector):
        """list_databases should honor scoped table restrictions."""
        result = scoped_db_func_tool.list_databases()

        assert result.success == 1
        assert result.result == ["db1", "db2"]
        mock_connector.get_databases.assert_called_once_with("", include_sys=False)

    def test_list_schemas_respects_scope(self, scoped_db_func_tool, mock_connector):
        """list_schemas should honor scoped table restrictions."""
        result = scoped_db_func_tool.list_schemas(database="db1")

        assert result.success == 1
        assert result.result == ["schema1"]
        mock_connector.get_schemas.assert_called_once_with("", "db1", include_sys=False)

    def test_list_schemas_wildcard_allows_other_database(self, scoped_db_func_tool, mock_connector):
        """Wildcard entries should allow schemas for additional databases."""
        mock_connector.get_schemas.reset_mock()
        mock_connector.get_schemas.return_value = ["schema1", "schema2"]

        result = scoped_db_func_tool.list_schemas(database="db2")

        assert result.success == 1
        assert result.result == ["schema1", "schema2"]
        mock_connector.get_schemas.assert_called_once_with("", "db2", include_sys=False)

    def test_list_databases_wildcard_scope_allows_multiple(self):
        """Wildcard scopes should permit every matching database."""
        connector = Mock(spec=BaseSqlConnector)
        connector.dialect = "postgresql"
        connector.catalog_name = ""
        connector.database_name = "db1"
        connector.schema_name = "schema1"
        connector.get_databases.return_value = ["db1", "analytics"]

        tool = DBFuncTool(connector, scoped_tables={"*.schema1.orders"})

        result = tool.list_databases()
        assert result.success == 1
        assert result.result == ["db1", "analytics"]
        connector.get_databases.assert_called_once_with("", include_sys=False)

    def test_list_databases_wildcard_scope_filters_non_matching(self):
        """Wildcard scopes should drop databases that do not match."""
        connector = Mock(spec=BaseSqlConnector)
        connector.dialect = "postgresql"
        connector.catalog_name = ""
        connector.database_name = "db1"
        connector.schema_name = "schema1"
        connector.get_databases.return_value = ["db1", "sales", "analytics"]

        tool = DBFuncTool(connector, scoped_tables={"db*.schema1.orders"})

        result = tool.list_databases()
        assert result.success == 1
        assert result.result == ["db1"]
        connector.get_databases.assert_called_once_with("", include_sys=False)

    def test_list_schemas_wildcard_scope(self):
        """Wildcard scopes should retain every schema for the matched database."""
        connector = Mock(spec=BaseSqlConnector)
        connector.dialect = "postgresql"
        connector.catalog_name = ""
        connector.database_name = "db1"
        connector.schema_name = "schema1"

        def fake_schemas(_catalog, database, include_sys=False):
            return ["schema1", "schema2"] if database == "db1" else ["other"]

        connector.get_schemas.side_effect = fake_schemas

        tool = DBFuncTool(connector, scoped_tables={"db1.*.orders"})

        result = tool.list_schemas(database="db1")
        assert result.success == 1
        assert result.result == ["schema1", "schema2"]
        connector.get_schemas.assert_called_once_with("", "db1", include_sys=False)

    def test_list_schemas_wildcard_scope_filters_non_matching(self):
        """Wildcard scopes should return empty when the database is out of scope."""
        connector = Mock(spec=BaseSqlConnector)
        connector.dialect = "postgresql"
        connector.catalog_name = ""
        connector.database_name = "db1"
        connector.schema_name = "schema1"
        connector.get_schemas.return_value = ["schema3"]

        tool = DBFuncTool(connector, scoped_tables={"db1.schema*.orders"})

        result = tool.list_schemas(database="reports")
        assert result.success == 1
        assert result.result == []
        connector.get_schemas.assert_not_called()

    def test_list_tables_wildcard_scope_filters_by_schema(self):
        """Wildcard scopes should allow matching schemas and exclude others."""
        connector = Mock(spec=BaseSqlConnector)
        connector.dialect = "postgresql"
        connector.catalog_name = ""
        connector.database_name = "db1"
        connector.schema_name = "schema1"

        def fake_tables(_catalog, _database, schema):
            return ["orders", "users"] if schema == "schema1" else ["misc"]

        def fake_views(_catalog, _database, schema):
            return ["view1"] if schema == "schema1" else ["view_misc"]

        def fake_materialized_views(_catalog, _database, schema):
            return ["mv1"] if schema == "schema1" else ["mv_misc"]

        connector.get_tables.side_effect = fake_tables
        connector.get_views.side_effect = fake_views
        connector.get_materialized_views.side_effect = fake_materialized_views

        tool = DBFuncTool(connector, scoped_tables={"*.schema1.*"})

        allowed = tool.list_tables(database="analytics", schema_name="schema1", include_views=True)
        assert allowed.success == 1
        assert [entry["name"] for entry in allowed.result] == ["orders", "users", "view1", "mv1"]
        connector.get_tables.assert_called_with("", "analytics", "schema1")
        connector.get_views.assert_called_with("", "analytics", "schema1")
        connector.get_materialized_views.assert_called_with("", "analytics", "schema1")

        connector.get_tables.reset_mock()
        connector.get_views.reset_mock()
        connector.get_materialized_views.reset_mock()

        blocked = tool.list_tables(database="analytics", schema_name="schema2", include_views=True)
        assert blocked.success == 1
        assert blocked.result == []
        connector.get_tables.assert_called_once_with("", "analytics", "schema2")
        connector.get_views.assert_called_once_with("", "analytics", "schema2")
        connector.get_materialized_views.assert_called_once_with("", "analytics", "schema2")

    def test_list_tables_wildcard_scope_filters_non_matching_database(self):
        """Wildcard scopes should filter out tables from databases outside scope."""
        connector = Mock(spec=BaseSqlConnector)
        connector.dialect = "postgresql"
        connector.catalog_name = ""
        connector.database_name = "db1"
        connector.schema_name = "schema1"
        connector.get_tables.return_value = ["orders"]
        connector.get_views.return_value = []
        connector.get_materialized_views.return_value = []

        tool = DBFuncTool(connector, scoped_tables={"db1.*.orders"})

        result = tool.list_tables(database="inventory", schema_name="schema1")
        assert result.success == 1
        assert result.result == []
        connector.get_tables.assert_called_once_with("", "inventory", "schema1")
        connector.get_views.assert_called_once_with("", "inventory", "schema1")
        connector.get_materialized_views.assert_called_once_with("", "inventory", "schema1")

    def test_list_tables_without_views(self, db_func_tool, mock_connector):
        """Test list_tables execution excluding views."""
        result = db_func_tool.list_tables(include_views=False)

        assert result.success == 1

        # Should only include tables, no views
        table_results = [item for item in result.result if item["type"] == "table"]
        view_results = [item for item in result.result if item["type"] != "table"]

        assert len(table_results) == 2  # users, orders
        assert len(view_results) == 0

        # Views methods should not be called
        mock_connector.get_views.assert_not_called()
        mock_connector.get_materialized_views.assert_not_called()

    def test_list_tables_view_methods_not_implemented(self, db_func_tool, mock_connector):
        """Test list_tables when view methods are not implemented."""
        # Make view methods raise NotImplementedError
        mock_connector.get_views.side_effect = NotImplementedError("Views not supported")
        mock_connector.get_materialized_views.side_effect = AttributeError("Method not available")

        result = db_func_tool.list_tables(include_views=True)

        # Should still succeed with just tables
        assert result.success == 1
        assert len(result.result) == 2  # Only tables

        # Both view methods should have been attempted
        mock_connector.get_views.assert_called()
        mock_connector.get_materialized_views.assert_called()

    def test_list_tables_failure(self, db_func_tool, mock_connector):
        """Test list_tables with exception."""
        mock_connector.get_tables.side_effect = Exception("Table retrieval failed")

        result = db_func_tool.list_tables()

        assert result.success == 0
        assert "Table retrieval failed" in result.error

    def test_search_table_without_schema_rag(self, db_func_tool):
        """search_table should report unavailable when schema storage is missing."""
        result = db_func_tool.search_table("customer data")

        assert result.success == 0
        assert "unavailable" in (result.error or "")

    def test_describe_table_success(self, db_func_tool, mock_connector):
        """Test successful describe_table execution."""
        result = db_func_tool.describe_table(
            table_name="users", catalog="test_catalog", database="test_db", schema_name="test_schema"
        )

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert isinstance(result.result, dict)
        assert "columns" in result.result
        assert "table_info" in result.result
        assert len(result.result["columns"]) == 2  # Two columns
        assert result.result["columns"][0]["column_name"] == "id"
        assert result.result["table_info"] == {}

        mock_connector.get_schema.assert_called_once_with(
            catalog_name="test_catalog", database_name="test_db", schema_name="test_schema", table_name="users"
        )

    def test_describe_table_scope_validation(self, mock_connector):
        """describe_table should block tables outside scoped set."""
        tool = DBFuncTool(mock_connector, scoped_tables={"db1.schema1.orders"})

        allowed = tool.describe_table(table_name="orders")
        assert allowed.success == 1
        mock_connector.get_schema.assert_called_once_with(
            catalog_name="", database_name="", schema_name="", table_name="orders"
        )

        mock_connector.get_schema.reset_mock()

        denied = tool.describe_table(table_name="users")
        assert denied.success == 0
        assert "users" in (denied.error or "")
        mock_connector.get_schema.assert_not_called()

    @pytest.mark.parametrize(
        "scoped_entry, kwargs",
        [
            ("db1.schema1.orders", {}),
            ('db1."schema1"."orders"', {}),
            ("*.*.orders", {}),
            ("db1.*.orders", {}),
            ("*.schema1.orders", {}),
        ],
    )
    def test_describe_table_scope_variants_allow(self, mock_connector, scoped_entry, kwargs):
        """Different scoped table formats should still authorize describe_table calls."""
        mock_connector.get_schema.reset_mock()
        tool = DBFuncTool(mock_connector, scoped_tables={scoped_entry})

        result = tool.describe_table(table_name="orders", **kwargs)

        assert result.success == 1
        mock_connector.get_schema.assert_called_once()

    def test_describe_table_default_params(self, db_func_tool, mock_connector):
        """Test describe_table with default parameters."""
        result = db_func_tool.describe_table(table_name="users")

        assert result.success == 1
        mock_connector.get_schema.assert_called_once_with(
            catalog_name="", database_name="", schema_name="", table_name="users"
        )

    def test_describe_table_failure(self, db_func_tool, mock_connector):
        """Test describe_table with exception."""
        mock_connector.get_schema.side_effect = Exception("Schema retrieval failed")

        result = db_func_tool.describe_table(table_name="nonexistent")

        assert result.success == 0
        assert "Schema retrieval failed" in result.error

    def test_describe_table_includes_semantic_details(self, db_func_tool, mock_connector):
        """Semantic model details should enrich describe_table output."""
        db_func_tool.has_semantic_models = True
        db_func_tool._get_semantic_model = Mock(
            return_value={
                "description": "Orders semantic model",
                "dimensions": ["customer_id"],
                "measures": ["total_sales"],
            }
        )

        result = db_func_tool.describe_table(table_name="orders")

        assert result.success == 1
        table_info = result.result["table_info"]
        assert table_info["description"] == "Orders semantic model"
        assert table_info["dimensions"] == ["customer_id"]
        assert table_info["measures"] == ["total_sales"]
        mock_connector.get_schema.assert_called_once()

    def test_read_query_success(self, db_func_tool, mock_connector):
        """Test successful read_query execution."""
        result = db_func_tool.read_query("SELECT * FROM users")

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert result.result is not None  # Should be compressed data

        mock_connector.execute_query.assert_called_once_with("SELECT * FROM users", result_format="list")

    def test_read_query_snowflake_uses_arrow(self, db_func_tool, mock_connector):
        """Snowflake dialect should request Arrow results."""
        mock_connector.dialect = DBType.SNOWFLAKE
        mock_query_result = Mock()
        mock_query_result.success = True
        mock_query_result.sql_return = [{"id": 1}]
        mock_connector.execute_query.return_value = mock_query_result

        result = db_func_tool.read_query("SELECT * FROM snowflake_table")

        assert result.success == 1
        mock_connector.execute_query.assert_called_once_with("SELECT * FROM snowflake_table", result_format="arrow")

    def test_read_query_query_failure(self, db_func_tool, mock_connector):
        """Test read_query when query execution fails."""
        mock_query_result = Mock()
        mock_query_result.success = False
        mock_query_result.error = "Syntax error"
        mock_connector.execute_query.return_value = mock_query_result

        result = db_func_tool.read_query("SELECT * FROM")

        assert result.success == 0
        assert "Syntax error" in result.error
        assert result.result is None
        mock_connector.execute_query.assert_called_once_with("SELECT * FROM", result_format="list")

    def test_read_query_execution_failure(self, db_func_tool, mock_connector):
        """Test read_query with execution exception."""
        mock_connector.execute_query.side_effect = Exception("Connection failed")

        result = db_func_tool.read_query("SELECT * FROM users")

        assert result.success == 0
        assert "Connection failed" in result.error
        assert result.result is None
        mock_connector.execute_query.assert_called_once_with("SELECT * FROM users", result_format="list")

    def test_get_table_ddl_success(self, db_func_tool, mock_connector):
        """get_table_ddl should return connector DDL info."""
        mock_connector.get_tables_with_ddl.return_value = [
            {"identifier": "db1.schema1.orders", "definition": "CREATE TABLE orders (...)"},
        ]

        result = db_func_tool.get_table_ddl("orders")

        assert result.success == 1
        assert result.result["identifier"] == "db1.schema1.orders"
        mock_connector.get_tables_with_ddl.assert_called_once_with(
            catalog_name="", database_name="", schema_name="", tables=["orders"]
        )

    def test_get_table_ddl_not_found(self, db_func_tool, mock_connector):
        """Empty DDL responses should surface as errors."""
        mock_connector.get_tables_with_ddl.return_value = []

        result = db_func_tool.get_table_ddl("missing")

        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    def test_get_table_ddl_scope_violation(self, mock_connector):
        """Scoped tables should block DDL retrieval outside the allowed set."""
        mock_connector.get_tables_with_ddl.return_value = [
            {"identifier": "db1.schema1.orders", "definition": "CREATE TABLE orders (...)"}
        ]
        tool = DBFuncTool(mock_connector, scoped_tables={"db1.schema1.orders"})

        denied = tool.get_table_ddl("users")

        assert denied.success == 0
        assert "outside the scoped context" in (denied.error or "")
        mock_connector.get_tables_with_ddl.assert_not_called()

    def test_get_table_ddl_failure(self, db_func_tool, mock_connector):
        """Connector exceptions should propagate as tool errors."""
        mock_connector.get_tables_with_ddl.side_effect = Exception("DDL fetch failed")

        result = db_func_tool.get_table_ddl("orders")

        assert result.success == 0
        assert "DDL fetch failed" in (result.error or "")

    def test_catalog_scoped_tables_filter_results(self):
        """Catalog-qualified scopes should restrict databases, schemas, and tables."""
        connector = Mock(spec=BaseSqlConnector)
        connector.dialect = DBType.SNOWFLAKE
        connector.catalog_name = "cat1"
        connector.database_name = "analytics"
        connector.schema_name = "public"

        connector.get_databases.return_value = ["analytics", "sales"]
        connector.get_schemas.return_value = ["public", "marketing"]

        def fake_tables(_catalog, _database, schema):
            if schema == "public":
                return ["orders", "users"]
            return ["history"]

        connector.get_tables.side_effect = fake_tables
        connector.get_views.return_value = []
        connector.get_materialized_views.return_value = []

        tool = DBFuncTool(connector, scoped_tables={"cat1.analytics.public.orders"})

        allowed_db = tool.list_databases()
        assert allowed_db.result == ["analytics"]

        connector.get_databases.assert_called_with("", include_sys=False)
        connector.get_databases.reset_mock()

        blocked_db = tool.list_databases(catalog="cat2")
        assert blocked_db.result == []
        connector.get_databases.assert_called_once_with("cat2", include_sys=False)

        allowed_schema = tool.list_schemas(catalog="cat1", database="analytics")
        assert allowed_schema.result == ["public"]

        blocked_schema = tool.list_schemas(catalog="cat1", database="sales")
        assert blocked_schema.result == []

        allowed_tables = tool.list_tables(catalog="cat1", database="analytics", schema_name="public")
        assert [entry["name"] for entry in allowed_tables.result] == ["orders"]

        blocked_tables = tool.list_tables(catalog="cat1", database="analytics", schema_name="marketing")
        assert blocked_tables.result == []

    def test_scoped_tables_loaded_from_agent_config(self, monkeypatch, mock_connector):
        """When scoped tables come from AgentConfig, describe_table should respect them."""

        class StubSchemaStore:
            def table_size(self):
                return 0

        class StubSchemaRAG:
            def __init__(self, *args, **kwargs):
                self.schema_store = StubSchemaStore()

        class StubSemanticRAG:
            def __init__(self, *args, **kwargs):
                pass

            def get_size(self):
                return 0

        monkeypatch.setattr("datus.tools.func_tool.database.SchemaWithValueRAG", StubSchemaRAG)
        monkeypatch.setattr("datus.tools.func_tool.database.SemanticModelRAG", StubSemanticRAG)

        class DummyAgentConfig:
            def __init__(self):
                self.agentic_nodes = {
                    "sales": {
                        "system_prompt": "Sales agent",
                        "scoped_context": {"tables": "db1.schema1.orders, db1.schema1.customers"},
                    }
                }

            def sub_agent_config(self, name: str):
                return self.agentic_nodes.get(name, {})

        tool = DBFuncTool(mock_connector, agent_config=DummyAgentConfig(), sub_agent_name="sales")

        mock_connector.get_schema.reset_mock()
        allowed = tool.describe_table("orders")
        assert allowed.success == 1
        mock_connector.get_schema.assert_called_once()

        mock_connector.get_schema.reset_mock()
        denied = tool.describe_table("users")
        assert denied.success == 0
        mock_connector.get_schema.assert_not_called()


class TestDBFuncToolEdgeCases:
    """Test edge cases for DBFuncTool."""

    def test_empty_results(self, db_func_tool, mock_connector):
        """Test methods with empty results."""
        # Setup empty returns
        mock_connector.get_catalogs.return_value = []
        mock_connector.get_databases.return_value = []
        mock_connector.get_schemas.return_value = []
        mock_connector.get_tables.return_value = []

        # Test each method
        methods = [
            db_func_tool.list_databases,
            db_func_tool.list_schemas,
            lambda: db_func_tool.list_tables(include_views=False),
        ]

        for method in methods:
            result = method()
            assert result.success == 1
            assert result.result == []

    def test_different_dialects(self):
        """Test available_tools with different database dialects."""
        dialects = [
            DBType.POSTGRES,
            DBType.MYSQL,
            DBType.STARROCKS,
            DBType.DUCKDB,
            DBType.SQLITE,
            DBType.SNOWFLAKE,
        ]

        for dialect in dialects:
            mock_connector = Mock()
            mock_connector.dialect = dialect

            tool = DBFuncTool(mock_connector)
            tools = tool.available_tools()

            expected_tool_count = 4
            if dialect in SUPPORT_DATABASE_DIALECTS:
                expected_tool_count += 1
            if dialect in SUPPORT_SCHEMA_DIALECTS:
                expected_tool_count += 1

            assert len(tools) == expected_tool_count, f"Failed for dialect {dialect}"

    def test_error_handling_different_exceptions(self, db_func_tool, mock_connector):
        """Test that different exception types are handled properly."""
        test_cases = [
            (ValueError("Invalid parameter"), "Invalid parameter"),
            (RuntimeError("Connection failed"), "Connection failed"),
            (Exception("Generic error"), "Generic error"),
        ]

        for exception, expected_error in test_cases:
            mock_connector.get_tables.side_effect = exception

            result = db_func_tool.list_tables()

            assert result.success == 0
            assert expected_error in result.error

    def test_method_return_types(self, db_func_tool):
        """Test that all methods return FuncToolResult instances."""
        methods_to_test = [
            db_func_tool.list_databases,
            db_func_tool.list_schemas,
            lambda: db_func_tool.list_tables(),
            lambda: db_func_tool.describe_table("test"),
            lambda: db_func_tool.read_query("SELECT 1"),
        ]

        for method in methods_to_test:
            result = method()
            assert isinstance(result, FuncToolResult)


class TestDBFuncToolIntegration:
    """Integration-style tests for DBFuncTool."""

    def _build_metadata_batch(self):
        return FakeRecordBatch(
            [
                {
                    "catalog_name": "",
                    "database_name": "db1",
                    "schema_name": "public",
                    "table_name": "orders",
                    "table_type": "table",
                    "definition": "CREATE TABLE orders (...);",
                    "identifier": "db1.public.orders",
                    "_distance": 0.05,
                }
            ]
        )

    def _build_sample_batch(self):
        return FakeRecordBatch(
            [
                {
                    "identifier": "db1.public.orders",
                    "table_type": "table",
                    "sample_rows": [{"id": 1, "total": 10}],
                    "_distance": 0.07,
                }
            ]
        )

    def test_search_table_returns_metadata_and_samples(self, db_func_tool):
        """search_table should emit metadata and sample rows when available."""
        db_func_tool.has_schema = True
        db_func_tool.schema_rag = Mock()
        db_func_tool.schema_rag.search_similar.return_value = (
            self._build_metadata_batch(),
            self._build_sample_batch(),
        )

        result = db_func_tool.search_table("orders table")

        assert result.success == 1
        metadata = result.result["metadata"]
        samples = result.result["sample_data"]
        assert metadata[0]["table_name"] == "orders"
        assert samples[0]["sample_rows"] == [{"id": 1, "total": 10}]

    def test_search_table_enriches_semantic_model(self, db_func_tool):
        """When semantic models exist, metadata rows should include enriched context."""
        db_func_tool.has_schema = True
        db_func_tool.schema_rag = Mock()
        db_func_tool.schema_rag.search_similar.return_value = (self._build_metadata_batch(), self._build_sample_batch())
        db_func_tool.has_semantic_models = True
        db_func_tool._get_semantic_model = Mock(
            return_value={
                "description": "Orders summary",
                "dimensions": ["order_id"],
                "measures": ["total_amount"],
            }
        )

        result = db_func_tool.search_table("orders table")

        assert result.success == 1
        metadata = result.result["metadata"]
        assert metadata[0]["description"] == "Orders summary"
        assert metadata[0]["dimensions"] == ["order_id"]
        assert metadata[0]["measures"] == ["total_amount"]
        assert result.result["sample_data"] == []
        db_func_tool._get_semantic_model.assert_called_once()

    def test_tool_transformation_integration(self, db_func_tool):
        """Test that tools can be transformed properly."""
        from datus.tools.func_tool import trans_to_function_tool

        # Test that a method can be transformed
        tool = trans_to_function_tool(db_func_tool.list_tables)

        assert tool is not None
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "params_json_schema")

        # Verify the schema doesn't contain 'self'
        schema = tool.params_json_schema
        if isinstance(schema, dict):
            assert "self" not in schema.get("properties", {})
            if "required" in schema:
                assert "self" not in schema["required"]

    def test_compression_integration(self, db_func_tool, mock_connector):
        """Test that read_query properly uses compression."""

        # Mock query result data
        test_data = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        mock_query_result = Mock()
        mock_query_result.success = True
        mock_query_result.sql_return = test_data
        mock_connector.execute_query.return_value = mock_query_result

        result = db_func_tool.read_query("SELECT * FROM users")

        assert result.success == 1
        assert result.result is not None
        assert result.result["is_compressed"] is False
