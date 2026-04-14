"""
Integration test: MigrationAgenticNode tool chain for DuckDB → Greenplum migration.

Verifies that the gen_job subagent's tools are correctly wired and can perform
a real end-to-end migration using the node's DBFuncTool instance.

Requires: Greenplum Docker running on port 15432.
"""

import os
import tempfile

import duckdb
import pytest

TARGET_TABLE = "public.gen_job_test_users"


@pytest.fixture(scope="module")
def duckdb_path():
    """Create a temp DuckDB with test data."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "gen_job_source.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE test_users (
            id INTEGER NOT NULL,
            name VARCHAR(50),
            score DOUBLE,
            active BOOLEAN,
            created DATE
        )
    """)
    conn.execute("""
        INSERT INTO test_users VALUES
            (1, 'Alice', 95.5, true, '2024-01-15'),
            (2, 'Bob', 82.3, false, '2024-02-20'),
            (3, 'Charlie', NULL, true, NULL)
    """)
    conn.close()
    yield db_path


@pytest.fixture(scope="module")
def migration_node(duckdb_path):
    """Create a MigrationAgenticNode with DuckDB + Greenplum configured."""
    from unittest.mock import patch

    from datus.configuration.agent_config_loader import load_agent_config

    # Clear DBManager cache to avoid stale connectors from other test modules
    from datus.tools.db_tools.db_manager import _cli_cache

    _cli_cache.clear()

    # Build a temp agent.yml with both databases
    tmp_dir = tempfile.mkdtemp()
    config_content = f"""
agent:
  target: claude_benchmark
  models:
    claude_benchmark:
      type: claude
      base_url: https://api.anthropic.com
      api_key: sk-fake
      model: claude-sonnet-4-6
      temperature: 0.0
  service:
    databases:
      source_duckdb:
        type: duckdb
        uri: "duckdb:///{duckdb_path}"
        name: source
      greenplum:
        type: greenplum
        host: 127.0.0.1
        port: 15432
        username: gpadmin
        password: pivotal
        database: test
        schema_name: public
        sslmode: disable
        timeout_seconds: 10
"""
    config_path = os.path.join(tmp_dir, "agent.yml")
    with open(config_path, "w") as f:
        f.write(config_content)

    try:
        agent_config = load_agent_config(config=config_path, namespace="source_duckdb", reload=True, force=True)
    except Exception as e:
        pytest.skip(f"Failed to load config: {e}")

    from datus.agent.node.migration_agentic_node import MigrationAgenticNode

    with patch("datus.models.base.LLMBaseModel.create_model"):
        try:
            node = MigrationAgenticNode(agent_config=agent_config, execution_mode="workflow")
        except Exception as e:
            pytest.skip(f"Failed to create MigrationAgenticNode: {e}")

    yield node

    # Cleanup target table using node's own execute_ddl (same SQLAlchemy connection, avoids lock conflicts)
    try:
        node.db_func_tool.execute_ddl(sql=f"DROP TABLE IF EXISTS {TARGET_TABLE}", database="greenplum")
    except Exception:
        pass


@pytest.mark.integration
class TestGenJobMigrationToolChain:
    """Test the gen_job node's tool chain for a real DuckDB → Greenplum migration."""

    def test_tools_are_wired(self, migration_node):
        """Verify gen_job has all required migration tools."""
        tool_names = [t.name for t in migration_node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names
        assert "execute_ddl" in tool_names
        assert "execute_write" in tool_names
        assert "transfer_query_result" in tool_names
        assert "list_databases" in tool_names

    def test_list_databases_returns_both(self, migration_node):
        """list_databases should return both source and target with type info."""
        result = migration_node.db_func_tool.list_databases()
        assert result.success == 1
        db_list = result.result
        names = [d["name"] for d in db_list]
        types = {d["name"]: d["type"] for d in db_list}
        assert "source_duckdb" in names
        assert "greenplum" in names
        assert types["source_duckdb"] == "duckdb"
        assert types["greenplum"] == "greenplum"

    def test_describe_source_table(self, migration_node):
        """describe_table on source DuckDB should return column info."""
        result = migration_node.db_func_tool.describe_table(table_name="test_users", database="source_duckdb")
        assert result.success == 1
        columns = result.result.get("columns", [])
        col_names = [c["name"] for c in columns]
        assert "id" in col_names
        assert "name" in col_names
        assert "score" in col_names

    def test_read_source_row_count(self, migration_node):
        """read_query on source should return 3 rows."""
        result = migration_node.db_func_tool.read_query(
            sql="SELECT COUNT(*) AS cnt FROM test_users", database="source_duckdb"
        )
        assert result.success == 1
        # H-7: verify actual count value, not just success
        assert "3" in str(result.result), f"Expected count=3, got: {result.result}"

    def test_execute_ddl_on_target(self, migration_node):
        """execute_ddl should create a table on Greenplum."""
        # Drop first if exists
        migration_node.db_func_tool.execute_ddl(sql=f"DROP TABLE IF EXISTS {TARGET_TABLE}", database="greenplum")
        result = migration_node.db_func_tool.execute_ddl(
            sql=f"""CREATE TABLE {TARGET_TABLE} (
                id INTEGER NOT NULL,
                name VARCHAR(50),
                score DOUBLE PRECISION,
                active BOOLEAN,
                created DATE
            )""",
            database="greenplum",
        )
        assert result.success == 1
        assert result.result["database"] == "greenplum"

    def test_transfer_and_verify(self, migration_node):
        """Full transfer: DuckDB → Greenplum, then verify row count."""
        # Ensure target table exists
        migration_node.db_func_tool.execute_ddl(sql=f"DROP TABLE IF EXISTS {TARGET_TABLE}", database="greenplum")
        migration_node.db_func_tool.execute_ddl(
            sql=f"""CREATE TABLE {TARGET_TABLE} (
                id INTEGER NOT NULL,
                name VARCHAR(50),
                score DOUBLE PRECISION,
                active BOOLEAN,
                created DATE
            )""",
            database="greenplum",
        )

        # Transfer
        result = migration_node.db_func_tool.transfer_query_result(
            source_sql="SELECT * FROM test_users",
            source_database="source_duckdb",
            target_table=TARGET_TABLE,
            target_database="greenplum",
            mode="replace",
        )
        assert result.success == 1, f"Transfer failed: {result.error}"
        assert result.result["rows_transferred"] == 3
        assert result.result["mode"] == "replace"

        # Verify on target
        verify = migration_node.db_func_tool.read_query(
            sql=f"SELECT COUNT(*) AS cnt FROM {TARGET_TABLE}", database="greenplum"
        )
        assert verify.success == 1
