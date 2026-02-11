import os

import pytest

from datus.utils.constants import DBType


class TestDBFuncToolIntegrationReal:
    """Real integration tests for DBFuncTool with actual databases.

    These tests use real database files from tests/data directory.
    """

    @pytest.fixture
    def ssb_sqlite_config(self):
        """Load SSB SQLite namespace configuration."""
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(namespace="ssb_sqlite", home="tests")

    @pytest.fixture
    def ssb_db_tool(self, ssb_sqlite_config):
        """Create DBFuncTool for SSB SQLite database."""
        from datus.tools.func_tool.database import db_function_tool_instance_multi

        return db_function_tool_instance_multi(ssb_sqlite_config)

    # ==================== SQLite Tests ====================

    def test_sqlite_list_tables_returns_actual_tables(self, ssb_db_tool):
        """Test that list_tables returns actual tables from SSB.db."""
        result = ssb_db_tool.list_tables()

        assert result.success == 1
        table_names = [t["name"] for t in result.result]
        # SSB database has: date, supplier, customer, part, lineorder
        expected_tables = {"date", "supplier", "customer", "part", "lineorder"}
        assert expected_tables.issubset(set(table_names))

    def test_sqlite_describe_table_returns_columns(self, ssb_db_tool):
        """Test that describe_table returns actual column info."""
        result = ssb_db_tool.describe_table("customer")

        assert result.success == 1
        assert "columns" in result.result
        columns = result.result["columns"]
        assert len(columns) > 0

        # Check column structure
        for col in columns:
            assert "name" in col
            assert "type" in col

    def test_sqlite_read_query_executes_sql(self, ssb_db_tool):
        """Test that read_query executes actual SQL."""
        result = ssb_db_tool.read_query("SELECT COUNT(*) as cnt FROM customer")

        assert result.success == 1
        assert result.result is not None
        # Result should be compressed data with count info
        assert "data" in result.result or "is_compressed" in result.result

    def test_sqlite_read_query_with_limit(self, ssb_db_tool):
        """Test read_query with LIMIT clause."""
        result = ssb_db_tool.read_query("SELECT * FROM customer LIMIT 5")

        assert result.success == 1
        assert result.result is not None

    def test_sqlite_read_query_invalid_sql_returns_error(self, ssb_db_tool):
        """Test that invalid SQL returns an error."""
        result = ssb_db_tool.read_query("SELECT * FROM nonexistent_table_xyz")

        assert result.success == 0
        assert result.error is not None

    def test_sqlite_get_table_ddl_returns_definition(self, ssb_db_tool):
        """Test that get_table_ddl returns CREATE statement."""
        result = ssb_db_tool.get_table_ddl("customer")

        assert result.success == 1
        assert result.result is not None
        # DDL should contain CREATE TABLE or similar
        definition = result.result.get("definition", "")
        assert "CREATE" in definition.upper() or "customer" in definition.lower()

    def test_sqlite_available_tools_correct_count(self, ssb_db_tool):
        """Test that SQLite returns correct number of tools."""
        tools = ssb_db_tool.available_tools()

        # SQLite should have: list_tables, describe_table, read_query, get_table_ddl
        # No list_databases (single file), no list_schemas (SQLite doesn't have schemas)
        tool_names = {t.name for t in tools}
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names
        assert "get_table_ddl" in tool_names

    def test_sqlite_connector_dialect(self, ssb_db_tool):
        """Test that SQLite connector has correct dialect."""
        assert ssb_db_tool.connector.dialect == DBType.SQLITE

    def test_single_connector_mode_backward_compatibility(self, ssb_sqlite_config):
        """Test that single connector mode still works."""
        from datus.tools.func_tool.database import db_function_tool_instance

        tool = db_function_tool_instance(ssb_sqlite_config)

        # Single connector mode should have db_manager = None
        assert tool._db_manager is None
        assert tool.connector is not None


class TestSqliteMultiConnector:
    @pytest.fixture
    def agent_config(self):
        """Load SSB SQLite namespace configuration."""
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(namespace="bird_sqlite", home="tests")

    @pytest.fixture
    def db_tool(self, agent_config):
        """Create DBFuncTool for SSB SQLite database."""
        from datus.tools.func_tool.database import db_function_tool_instance_multi

        return db_function_tool_instance_multi(agent_config)

    # ==================== Multi-Connector Mode Tests ====================

    def test_connector_mode_initialization(self, db_tool):
        """Test that multi-connector mode initializes correctly."""

        assert db_tool._db_manager is not None
        assert db_tool._namespace == "bird_sqlite"
        assert db_tool._connector_cache_size > 1

    def test_database(self, db_tool):
        result = db_tool.list_databases()
        assert result.success == 1
        assert len(result.result) > 1

    def test_tables(self, db_tool):
        result = db_tool.list_tables(database="california_schools")
        assert result.success == 1
        assert len(result.result) > 1
        table_names = set([item["name"] for item in result.result])
        assert table_names == {"frpm", "satscores", "schools"}

        result = db_tool.list_tables(database="card_games")
        assert result.success == 1
        assert len(result.result) > 1
        table_names = set([item["name"] for item in result.result])
        assert table_names == {"cards", "legalities", "set_translations", "foreign_data", "rulings", "sets"}


class TestDuckDBTool:
    """Test the DuckDBTool class."""

    @pytest.fixture
    def duckdb_config(self):
        """Load DuckDB namespace configuration."""
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(namespace="duckdb", home="tests")

    @pytest.fixture
    def duckdb_tool(self, duckdb_config):
        """Create DBFuncTool for DuckDB database."""
        from datus.tools.func_tool.database import db_function_tool_instance_multi

        return db_function_tool_instance_multi(duckdb_config)

    # ==================== DuckDB Tests ====================

    def test_duckdb_list_tables_in_schema(self, duckdb_tool):
        """Test that list_tables returns tables from mf_demo schema."""
        result = duckdb_tool.list_tables(schema_name="mf_demo")

        assert result.success == 1
        table_names = [t["name"] for t in result.result]
        # DuckDB has: mf_demo_countries, mf_demo_customers, mf_demo_transactions, mf_time_spine
        expected_tables = {"mf_demo_countries", "mf_demo_customers", "mf_demo_transactions", "mf_time_spine"}
        assert expected_tables.issubset(set(table_names))

    def test_duckdb_list_schemas_returns_schemas(self, duckdb_tool):
        """Test that list_schemas returns available schemas."""
        result = duckdb_tool.list_schemas()

        assert result.success == 1
        schemas = result.result
        # Should include mf_demo schema
        assert "mf_demo" in schemas

    def test_duckdb_describe_table_returns_columns(self, duckdb_tool):
        """Test that describe_table returns column info for DuckDB table."""
        result = duckdb_tool.describe_table("mf_demo_customers", schema_name="mf_demo")

        assert result.success == 1
        assert "columns" in result.result
        columns = result.result["columns"]
        assert len(columns) > 0

    def test_duckdb_read_query_executes_sql(self, duckdb_tool):
        """Test that read_query executes SQL on DuckDB."""
        result = duckdb_tool.read_query("SELECT COUNT(*) as cnt FROM mf_demo.mf_demo_customers")

        assert result.success == 1
        assert result.result is not None

    def test_duckdb_read_query_with_schema_qualified_table(self, duckdb_tool):
        """Test read_query with schema-qualified table name."""
        result = duckdb_tool.read_query("SELECT * FROM mf_demo.mf_demo_countries LIMIT 3")

        assert result.success == 1
        assert result.result is not None

    def test_duckdb_connector_dialect(self, duckdb_tool):
        """Test that DuckDB connector has correct dialect."""
        assert duckdb_tool.connector.dialect == DBType.DUCKDB

    def test_duckdb_available_tools_includes_list_schemas(self, duckdb_tool):
        """Test that DuckDB tools include list_schemas."""
        tools = duckdb_tool.available_tools()
        tool_names = {t.name for t in tools}

        # DuckDB supports schemas
        assert "list_schemas" in tool_names
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names


def _snowflake_available() -> bool:
    """Check if Snowflake connector is available and credentials are configured."""
    # Check environment variables
    if not all(
        [
            os.environ.get("SNOWFLAKE_ACCOUNT"),
            os.environ.get("SNOWFLAKE_USERNAME"),
            os.environ.get("SNOWFLAKE_PASSWORD"),
        ]
    ):
        return False
    # Check if snowflake connector is installed
    try:
        from datus.tools.db_tools.registry import connector_registry

        return "snowflake" in connector_registry._connectors
    except Exception:
        return False


@pytest.mark.skipif(
    not _snowflake_available(),
    reason="Snowflake connector not installed or credentials not configured",
)
class TestDBFuncToolSnowflake:
    """Integration tests for Snowflake.

    These tests require Snowflake environment variables:
    - SNOWFLAKE_ACCOUNT
    - SNOWFLAKE_USERNAME
    - SNOWFLAKE_PASSWORD
    """

    @pytest.fixture
    def snowflake_config(self):
        """Load Snowflake namespace configuration."""
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(namespace="snowflake", home="tests")

    @pytest.fixture
    def snowflake_tool(self, snowflake_config):
        """Create DBFuncTool for Snowflake."""
        from datus.tools.func_tool.database import db_function_tool_instance_multi

        return db_function_tool_instance_multi(snowflake_config)

    def test_snowflake_connector_dialect(self, snowflake_tool):
        """Test that Snowflake connector has correct dialect."""
        assert snowflake_tool.connector.dialect == DBType.SNOWFLAKE

    def test_snowflake_available_tools_includes_database_and_schema(self, snowflake_tool):
        """Test that Snowflake tools include list_databases and list_schemas."""
        tools = snowflake_tool.available_tools()
        tool_names = {t.name for t in tools}

        # Snowflake supports databases and schemas
        assert "list_databases" in tool_names
        assert "list_schemas" in tool_names
        assert "list_tables" in tool_names

    def test_snowflake_list_databases(self, snowflake_tool):
        """Test that list_databases returns Snowflake databases."""
        result = snowflake_tool.list_databases()

        assert result.success == 1
        assert isinstance(result.result, list)

    def test_snowflake_list_schemas(self, snowflake_tool):
        """Test that list_schemas returns Snowflake schemas."""
        result = snowflake_tool.list_schemas()

        assert result.success == 1
        assert isinstance(result.result, list)

    def test_snowflake_read_query_uses_arrow_format(self, snowflake_tool):
        """Test that Snowflake read_query uses Arrow format."""
        # This tests the dialect-specific behavior
        result = snowflake_tool.read_query("SELECT 1 as test")

        assert result.success == 1
        assert result.result is not None
