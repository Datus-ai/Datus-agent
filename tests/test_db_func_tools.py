"""
Test cases for DBFuncTool class in datus/tools/tools.py
"""

from unittest.mock import Mock

import pytest

from datus.tools.db_tools import BaseSqlConnector
from datus.tools.tools import DBFuncTool, FuncToolResult
from datus.utils.constants import SUPPORT_CATALOG_DIALECTS, SUPPORT_DATABASE_DIALECTS, SUPPORT_SCHEMA_DIALECTS, DBType


@pytest.fixture
def mock_connector():
    """Create a mock database connector."""
    connector = Mock(spec=BaseSqlConnector)
    connector.dialect = "postgresql"

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
        expected_tool_count = 3  # list_tables, describe_table, read_query

        if mock_connector.dialect in SUPPORT_CATALOG_DIALECTS:
            expected_tool_count += 1
        if mock_connector.dialect in SUPPORT_DATABASE_DIALECTS:
            expected_tool_count += 1
        if mock_connector.dialect in SUPPORT_SCHEMA_DIALECTS:
            expected_tool_count += 1

        assert len(tools) == expected_tool_count

        # Verify tool names
        tool_names = [tool.name for tool in tools]
        expected_base_tools = ["list_tables", "describe_table", "read_query"]

        for expected_tool in expected_base_tools:
            assert any(expected_tool in name for name in tool_names)

    def test_list_catalogs_success(self, db_func_tool, mock_connector):
        """Test successful list_catalogs execution."""
        result = db_func_tool.list_catalogs()

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert result.result == ["catalog1", "catalog2"]

        mock_connector.get_catalogs.assert_called_once()

    def test_list_catalogs_failure(self, db_func_tool, mock_connector):
        """Test list_catalogs with exception."""
        mock_connector.get_catalogs.side_effect = Exception("Catalog retrieval failed")

        result = db_func_tool.list_catalogs()

        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert "Catalog retrieval failed" in result.error
        assert result.result is None

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

    def test_describe_table_success(self, db_func_tool, mock_connector):
        """Test successful describe_table execution."""
        result = db_func_tool.describe_table(
            table_name="users", catalog="test_catalog", database="test_db", schema_name="test_schema"
        )

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert len(result.result) == 2  # Two columns
        assert result.result[0]["column_name"] == "id"

        mock_connector.get_schema.assert_called_once_with(
            catalog_name="test_catalog", database_name="test_db", schema_name="test_schema", table_name="users"
        )

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

    def test_read_query_success(self, db_func_tool, mock_connector):
        """Test successful read_query execution."""
        result = db_func_tool.read_query("SELECT * FROM users")

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert result.result is not None  # Should be compressed data

        mock_connector.execute_query.assert_called_once_with("SELECT * FROM users")

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

    def test_read_query_execution_failure(self, db_func_tool, mock_connector):
        """Test read_query with execution exception."""
        mock_connector.execute_query.side_effect = Exception("Connection failed")

        result = db_func_tool.read_query("SELECT * FROM users")

        assert result.success == 0
        assert "Connection failed" in result.error
        assert result.result is None


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
            db_func_tool.list_catalogs,
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
        test_cases = [
            (DBType.POSTGRES, 5),  # Supports catalogs, databases, schemas
            (DBType.MYSQL, 4),  # Supports databases, schemas
            (DBType.STARROCKS, 5),  # Supports catalogs, databases
            (DBType.DUCKDB, 5),  # Supports databases, schemas
            (DBType.SQLITE, 3),  # Only base tools
            (DBType.SNOWFLAKE, 6),  # Supports catalogs, databases, schemas
        ]

        for dialect, expected_tool_count in test_cases:
            mock_connector = Mock()
            mock_connector.dialect = dialect

            tool = DBFuncTool(mock_connector)
            tools = tool.available_tools()

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
            db_func_tool.list_catalogs,
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

    def test_tool_transformation_integration(self, db_func_tool):
        """Test that tools can be transformed properly."""
        from datus.tools.tools import trans_to_function_tool

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
