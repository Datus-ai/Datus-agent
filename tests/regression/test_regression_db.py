"""
Regression Tests: Built-in Database Connector Compatibility

Tests built-in database connectors through the BaseSqlConnector interface:
- builtin-sqlite: SQLite (tests/data/SSB.db)
- builtin-duckdb: DuckDB (tests/data/datus_metricflow_db/duck.db)
"""
from pathlib import Path

import pytest
from dotenv import load_dotenv

from datus.tools.db_tools.config import DuckDBConfig, SQLiteConfig
from datus.tools.db_tools.registry import connector_registry
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================
# DB Connector Registry
# ============================================================
DB_CONNECTORS = {
    "builtin-sqlite": {
        "type": "sqlite",
        "config": {
            "uri": str(PROJECT_ROOT / "tests" / "data" / "SSB.db"),
        },
        "test_query": "SELECT COUNT(*) FROM customer",
    },
    "builtin-duckdb": {
        "type": "duckdb",
        "config": {
            "uri": str(PROJECT_ROOT / "tests" / "data" / "datus_metricflow_db" / "duck.db"),
        },
        "test_query": "SELECT 1",
    },
}


# ============================================================
# Helpers
# ============================================================
def create_connector(db_spec: dict):
    """Create a connector instance from a DB_CONNECTORS spec."""
    db_type = db_spec["type"]
    db_path = db_spec["config"]["uri"]

    if db_type == "sqlite":
        if db_path.startswith("sqlite:///"):
            db_path = db_path.replace("sqlite:///", "")
        config = SQLiteConfig(db_path=db_path)
        return connector_registry.create_connector("sqlite", config)

    elif db_type == "duckdb":
        if db_path.startswith("duckdb:///"):
            db_path = db_path.replace("duckdb:///", "")
        config = DuckDBConfig(db_path=db_path)
        return connector_registry.create_connector("duckdb", config)

    else:
        raise ValueError(f"Unsupported db type: {db_type}")


def _build_db_params():
    """Build pytest.param list from DB_CONNECTORS."""
    params = []
    for case_id, spec in DB_CONNECTORS.items():
        params.append(pytest.param(case_id, spec, id=case_id))
    return params


DB_CONNECTOR_PARAMS = _build_db_params()


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module", autouse=True)
def setup_env():
    load_dotenv()


# ============================================================
# DB Connector Regression Tests
# ============================================================
@pytest.mark.regression
class TestRegressionDBConnector:
    """Test BaseSqlConnector interface for built-in SQLite and DuckDB."""

    @pytest.mark.parametrize("db_type,db_spec", DB_CONNECTOR_PARAMS)
    def test_connection(self, db_type, db_spec):
        """Test connection establishment via test_connection()."""
        connector = create_connector(db_spec)
        try:
            result = connector.test_connection()
            assert result is not None
        finally:
            connector.close()

    @pytest.mark.parametrize("db_type,db_spec", DB_CONNECTOR_PARAMS)
    def test_get_tables(self, db_type, db_spec):
        """Test table listing."""
        connector = create_connector(db_spec)
        try:
            connector.connect()
            tables = connector.get_tables()
            assert isinstance(tables, list)
            assert len(tables) > 0, "Should have at least one table"
        finally:
            connector.close()

    @pytest.mark.parametrize("db_type,db_spec", DB_CONNECTOR_PARAMS)
    def test_get_databases(self, db_type, db_spec):
        """Test database listing."""
        connector = create_connector(db_spec)
        try:
            connector.connect()
            databases = connector.get_databases()
            assert isinstance(databases, list)
            assert len(databases) > 0, "Should have at least one database"
        finally:
            connector.close()

    @pytest.mark.parametrize("db_type,db_spec", DB_CONNECTOR_PARAMS)
    def test_get_schema(self, db_type, db_spec):
        """Test table schema retrieval (columns and types)."""
        connector = create_connector(db_spec)
        try:
            connector.connect()
            tables = connector.get_tables()
            if not tables:
                pytest.skip(f"{db_type}: no tables found for schema test")
            # DuckDB requires schema_name for table lookup
            schema_kwargs = {"table_name": tables[0]}
            if db_spec["type"] == "duckdb":
                schema_kwargs["schema_name"] = "mf_demo"
            schema = connector.get_schema(**schema_kwargs)
            assert isinstance(schema, list)
            assert len(schema) > 0, "Table should have at least one column"
        finally:
            connector.close()

    @pytest.mark.parametrize("db_type,db_spec", DB_CONNECTOR_PARAMS)
    def test_get_tables_with_ddl(self, db_type, db_spec):
        """Test DDL output for tables."""
        connector = create_connector(db_spec)
        try:
            connector.connect()
            result = connector.get_tables_with_ddl()
            assert isinstance(result, list)
            if len(result) > 0:
                item = result[0]
                assert isinstance(item, dict)
                # The connector returns 'identifier' (qualified name) and 'definition' (DDL)
                assert "identifier" in item or "table_name" in item
                assert "definition" in item or "ddl" in item
        finally:
            connector.close()

    @pytest.mark.parametrize("db_type,db_spec", DB_CONNECTOR_PARAMS)
    def test_execute_query(self, db_type, db_spec):
        """Test SELECT query execution."""
        connector = create_connector(db_spec)
        try:
            connector.connect()
            test_sql = db_spec.get("test_query", "SELECT 1")
            result = connector.execute_query(test_sql)
            assert result is not None
            assert result.sql_return is not None
        finally:
            connector.close()
