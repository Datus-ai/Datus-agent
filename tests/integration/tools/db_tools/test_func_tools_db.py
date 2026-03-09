import os

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import DBFuncTool
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
        assert snowflake_tool.connector.dialect == "snowflake"

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


# =============================================================================
# Nightly: DB error scenarios, scoped tables, search table, context search
# =============================================================================


@pytest.mark.nightly
class TestDBFuncToolErrors:
    """N11-07: read_query failure scenarios with real SSB SQLite database."""

    @pytest.fixture
    def ssb_config(self):
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(namespace="ssb_sqlite", home="tests")

    @pytest.fixture
    def ssb_db_tool(self, ssb_config):
        from datus.tools.func_tool.database import db_function_tool_instance_multi

        return db_function_tool_instance_multi(ssb_config)

    def test_read_query_nonexistent_table(self, ssb_db_tool):
        """N11-07a: read_query with nonexistent table returns meaningful error."""
        result = ssb_db_tool.read_query("SELECT * FROM nonexistent_xyz_table")

        assert result.success == 0, "Should fail for nonexistent table"
        assert result.error is not None, "Error message should not be None"
        assert len(result.error) > 10, f"Error message should be descriptive, got: {result.error}"

    def test_read_query_invalid_sql_syntax(self, ssb_db_tool):
        """N11-07b: read_query with completely invalid SQL returns error."""
        result = ssb_db_tool.read_query("COMPLETELY INVALID SQL STATEMENT")

        assert result.success == 0, "Should fail for invalid SQL"
        assert result.error is not None, "Error message should not be None"
        assert len(result.error) > 0, "Error message should not be empty"


@pytest.mark.nightly
class TestScopedTables:
    """N11-09: Scoped tables filtering with real SSB SQLite database."""

    @pytest.fixture
    def ssb_config(self):
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(namespace="ssb_sqlite", home="tests")

    @pytest.fixture
    def scoped_db_tool(self, ssb_config):
        """Create DBFuncTool with scoped_tables limited to customer and lineorder."""
        db_manager = db_manager_instance(ssb_config.namespaces)
        return DBFuncTool(
            db_manager,
            agent_config=ssb_config,
            default_database=ssb_config.current_database,
            scoped_tables=["customer", "lineorder"],
        )

    def test_list_tables_respects_scope(self, scoped_db_tool):
        """N11-09a: list_tables only returns tables within scoped_tables."""
        result = scoped_db_tool.list_tables()

        assert result.success == 1, f"list_tables should succeed, got error: {result.error}"
        table_names = [t["name"] for t in result.result]

        assert "customer" in table_names, "customer should be in scoped results"
        assert "lineorder" in table_names, "lineorder should be in scoped results"
        assert "supplier" not in table_names, "supplier should be filtered out by scope"
        assert "part" not in table_names, "part should be filtered out by scope"
        assert "date" not in table_names, "date should be filtered out by scope"

    def test_describe_table_blocked_by_scope(self, ssb_config):
        """N11-09b: describe_table rejects tables outside scoped_tables."""
        db_manager = db_manager_instance(ssb_config.namespaces)
        scoped_tool = DBFuncTool(
            db_manager,
            agent_config=ssb_config,
            default_database=ssb_config.current_database,
            scoped_tables=["customer"],
        )

        # Allowed table should work
        allowed_result = scoped_tool.describe_table("customer")
        assert (
            allowed_result.success == 1
        ), f"describe_table for scoped table should succeed, got: {allowed_result.error}"
        assert "columns" in allowed_result.result, "Should have columns in result"

        # Blocked table should fail
        blocked_result = scoped_tool.describe_table("supplier")
        assert blocked_result.success == 0, "describe_table for out-of-scope table should fail"
        assert blocked_result.error is not None, "Should have error message"


@pytest.mark.nightly
class TestSearchTable:
    """N11-08: search_table RAG functionality."""

    def test_search_table_available_tools(self, agent_config: AgentConfig):
        """N11-08: Verify search_table presence in available_tools depends on has_schema."""
        from datus.tools.func_tool.database import db_function_tool_instance_multi

        db_tool = db_function_tool_instance_multi(agent_config)

        tools = db_tool.available_tools()
        tool_names = {tool.name for tool in tools}

        # search_table should be in available_tools only if schema RAG exists
        if hasattr(db_tool, "has_schema") and db_tool.has_schema:
            assert "search_table" in tool_names, "search_table should be available when schema RAG exists"
        else:
            assert "search_table" not in tool_names, "search_table should not be available without schema RAG"


@pytest.mark.nightly
class TestContextSearchTools:
    """N11-13 to N11-16: ContextSearchTools with bird_school configuration."""

    @pytest.fixture
    def ctx_tools(self, agent_config: AgentConfig):
        return ContextSearchTools(agent_config)

    def test_search_metrics(self, ctx_tools):
        """N11-13: search_metrics returns structured results."""
        assert ctx_tools.has_metrics is True, "bird_school should have metrics data"

        result = ctx_tools.search_metrics("school")

        assert result.success == 1, f"search_metrics should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) > 0, "Should find at least one metric matching 'school'"

        # Verify result structure
        first = result.result[0]
        assert "name" in first, "Each metric should have a 'name' field"

    def test_get_metrics(self, ctx_tools):
        """N11-13b: get_metrics retrieves specific metric details."""
        assert ctx_tools.has_metrics is True, "bird_school should have metrics data"

        # First search to get a valid subject_path and name
        search_result = ctx_tools.search_metrics("school")
        assert search_result.success == 1 and len(search_result.result) > 0, "Need search results to test get_metrics"

        first = search_result.result[0]
        subject_path = first.get("subject_path", [])
        name = first.get("name", "")

        assert subject_path and name, f"Search result should have subject_path and name, got: {first}"

        get_result = ctx_tools.get_metrics(subject_path=subject_path, name=name)

        assert get_result.success == 1, f"get_metrics should succeed, got error: {get_result.error}"
        assert get_result.result is not None, "Should return metric details"

    def test_search_reference_sql(self, ctx_tools):
        """N11-14: search_reference_sql returns list of SQL queries."""
        assert ctx_tools.has_reference_sql is True, "bird_school should have reference SQL data"

        result = ctx_tools.search_reference_sql("school")

        assert result.success == 1, f"search_reference_sql should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) > 0, "Should find at least one reference SQL matching 'school'"

        # Verify structure
        first = result.result[0]
        assert (
            "name" in first or "sql" in first
        ), f"Each result should have 'name' or 'sql', got keys: {list(first.keys())}"

    def test_get_reference_sql(self, ctx_tools):
        """N11-15: get_reference_sql retrieves specific SQL details."""
        assert ctx_tools.has_reference_sql is True, "bird_school should have reference SQL data"

        search_result = ctx_tools.search_reference_sql("school")
        assert (
            search_result.success == 1 and len(search_result.result) > 0
        ), "Need search results to test get_reference_sql"

        first = search_result.result[0]
        subject_path = first.get("subject_path", [])
        name = first.get("name", "")

        assert subject_path and name, f"Search result should have subject_path and name, got: {first}"

        get_result = ctx_tools.get_reference_sql(subject_path=subject_path, name=name)

        assert get_result.success == 1, f"get_reference_sql should succeed, got error: {get_result.error}"
        assert get_result.result is not None, "Should return SQL details"

    def test_search_semantic_objects_availability(self, ctx_tools):
        """N11-16: Verify search_semantic_objects availability and behavior."""
        # Test that the flag is a boolean
        assert isinstance(ctx_tools.has_semantic_objects, bool), "has_semantic_objects should be a boolean"

        if ctx_tools.has_semantic_objects:
            result = ctx_tools.search_semantic_objects("school")
            assert result.success == 1, f"search_semantic_objects should succeed when data exists, got: {result.error}"
            assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        else:
            # Verify it's correctly not in available tools
            tool_names = {t.name for t in ctx_tools.available_tools()}
            assert (
                "search_semantic_objects" not in tool_names
            ), "search_semantic_objects should not be available without data"
