import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool.database import DBFuncTool
from datus.utils.constants import DBType


@pytest.mark.acceptance
class TestDBFuncToolIntegrationReal:
    """Real integration tests for DBFuncTool with actual databases.

    These tests use real database files from tests/data directory.
    """

    @pytest.fixture
    def ssb_sqlite_config(self):
        """Load SSB SQLite datasource configuration."""
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(datasource="ssb_sqlite", home="tests")

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
        """Test that db_function_tool_instance still works (auto-creates DBManager)."""
        from datus.tools.func_tool.database import db_function_tool_instance

        tool = db_function_tool_instance(ssb_sqlite_config)

        assert tool._db_manager is not None
        assert tool.connector is not None


@pytest.mark.acceptance
class TestSqliteMultiConnector:
    @pytest.fixture
    def agent_config(self):
        """Load acceptance config using a valid current database."""
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(datasource="california_schools", home="tests")

    @pytest.fixture
    def db_tool(self, agent_config):
        """Create DBFuncTool for SSB SQLite database."""
        from datus.tools.func_tool.database import db_function_tool_instance_multi

        return db_function_tool_instance_multi(agent_config)

    # ==================== Multi-Connector Mode Tests ====================

    def test_connector_mode_initialization(self, db_tool):
        """Test that multi-connector mode initializes correctly."""

        assert db_tool._db_manager is not None
        assert db_tool._default_datasource == "california_schools"
        assert db_tool._connector_cache_size > 1

    def test_database(self, db_tool):
        result = db_tool.list_databases()
        assert result.success == 1
        available_names = set(result.result)
        assert "main" in available_names

    def test_tables(self, db_tool):
        result = db_tool.list_tables(datasource="california_schools")
        assert result.success == 1
        assert len(result.result) > 1
        table_names = set([item["name"] for item in result.result])
        assert table_names == {"frpm", "satscores", "schools"}

        result = db_tool.list_tables(datasource="card_games")
        assert result.success == 1
        assert len(result.result) > 1
        table_names = set([item["name"] for item in result.result])
        assert table_names == {"cards", "legalities", "set_translations", "foreign_data", "rulings", "sets"}


@pytest.mark.acceptance
class TestDuckDBTool:
    """Test the DuckDBTool class."""

    @pytest.fixture
    def duckdb_config(self):
        """Load DuckDB datasource configuration."""
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(datasource="duckdb", home="tests")

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


# =============================================================================
# Connector interface tests (merged from test_regression_db.py)
# =============================================================================


@pytest.mark.acceptance
class TestConnectorInterface:
    """Test BaseSqlConnector.test_connection() directly."""

    def test_sqlite_test_connection(self):
        """Test SQLite connector test_connection() health check."""
        from datus_db_core import connector_registry

        from datus.tools.db_tools.config import SQLiteConfig

        config = SQLiteConfig(db_path="tests/data/SSB.db")
        connector = connector_registry.create_connector("sqlite", config)
        try:
            result = connector.test_connection()
            assert result is not None
        finally:
            connector.close()

    def test_duckdb_test_connection(self):
        """Test DuckDB connector test_connection() health check."""
        from datus_db_core import connector_registry

        from datus.tools.db_tools.config import DuckDBConfig

        config = DuckDBConfig(db_path="tests/data/datus_metricflow_db/duck.db")
        connector = connector_registry.create_connector("duckdb", config)
        try:
            result = connector.test_connection()
            assert result is not None
        finally:
            connector.close()


# =============================================================================
# Acceptance: DB error scenarios, scoped tables, search table
# =============================================================================


@pytest.mark.acceptance
class TestDBFuncToolErrors:
    """N11-07: read_query failure scenarios with real SSB SQLite database."""

    @pytest.fixture
    def ssb_config(self):
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(datasource="ssb_sqlite", home="tests")

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


@pytest.mark.acceptance
class TestScopedTables:
    """N11-09: Scoped tables filtering with real SSB SQLite database."""

    @pytest.fixture
    def ssb_config(self):
        from tests.conftest import load_acceptance_config

        return load_acceptance_config(datasource="ssb_sqlite", home="tests")

    @pytest.fixture
    def scoped_db_tool(self, ssb_config):
        """Create DBFuncTool with scoped_tables limited to customer and lineorder."""
        db_manager = db_manager_instance(ssb_config.datasource_configs)
        return DBFuncTool(
            db_manager,
            agent_config=ssb_config,
            default_datasource=ssb_config.current_datasource,
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
        db_manager = db_manager_instance(ssb_config.datasource_configs)
        scoped_tool = DBFuncTool(
            db_manager,
            agent_config=ssb_config,
            default_datasource=ssb_config.current_datasource,
            scoped_tables=["customer"],
        )

        # Allowed table should work
        allowed_result = scoped_tool.describe_table("customer")
        assert allowed_result.success == 1, (
            f"describe_table for scoped table should succeed, got: {allowed_result.error}"
        )
        assert "columns" in allowed_result.result, "Should have columns in result"

        # Blocked table should fail
        blocked_result = scoped_tool.describe_table("supplier")
        assert blocked_result.success == 0, "describe_table for out-of-scope table should fail"
        assert blocked_result.error is not None, "Should have error message"


@pytest.mark.acceptance
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
