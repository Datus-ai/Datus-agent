"""
Contract tests: PostgreSQL adapter via DBFuncTool.

Opt-in (both required):
  * install the adapter:       `uv pip install datus-postgresql`
  * start the docker service:  `cd datus-db-adapters/datus-postgresql && docker compose up -d`
  * set env var:                ADAPTERS_PG=1

Env for overriding defaults (defaults match the adapter's docker-compose.yml):
  POSTGRESQL_HOST=localhost  POSTGRESQL_PORT=5432
  POSTGRESQL_USER=test_user  POSTGRESQL_PASSWORD=test_password
  POSTGRESQL_DATABASE=test   POSTGRESQL_SCHEMA=public

See `tests/integration/adapters/README.md` for the full workflow.
"""

import os
from typing import Generator

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

# Opt-in gate MUST run before the optional-package import. Otherwise, when the
# user hasn't set ADAPTERS_PG=1 and hasn't installed the adapter, the tests
# skip with a misleading "not installed" reason instead of "opt-in not set".
require_opt_in_env("ADAPTERS_PG", "tests/integration/adapters/README.md")

# datus-postgresql is NOT a hard dep of Datus-agent; audit allows importorskip here
# because the package is not in [project.dependencies].
datus_postgresql = import_required(
    "datus_postgresql",
    reason="datus-postgresql not installed; run `uv pip install datus-postgresql`",
)

PostgreSQLConfig = datus_postgresql.PostgreSQLConfig
PostgreSQLConnector = datus_postgresql.PostgreSQLConnector

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.nightly]


SCHEMA = os.getenv("POSTGRESQL_SCHEMA", "public")
REGION_TABLE = "datus_adapter_region"
NATION_TABLE = "datus_adapter_nation"

REGION_DDL = f"""
CREATE TABLE IF NOT EXISTS "{SCHEMA}"."{REGION_TABLE}" (
    r_regionkey INTEGER PRIMARY KEY,
    r_name VARCHAR(25) NOT NULL,
    r_comment VARCHAR(152)
)
"""
NATION_DDL = f"""
CREATE TABLE IF NOT EXISTS "{SCHEMA}"."{NATION_TABLE}" (
    n_nationkey INTEGER PRIMARY KEY,
    n_name VARCHAR(25) NOT NULL,
    n_regionkey INTEGER NOT NULL,
    n_comment VARCHAR(152)
)
"""
REGION_ROWS = [
    (0, "AFRICA", "lar deposits."),
    (1, "AMERICA", "hs use ironic requests."),
    (2, "ASIA", "ges. pinto beans."),
]
NATION_ROWS = [
    (0, "ALGERIA", 0, "haggle."),
    (1, "ARGENTINA", 1, "foxes promise."),
    (2, "BRAZIL", 1, "of pending deposits."),
]


def _escape_sql_value(v: object) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


@pytest.fixture(scope="module")
def pg_config() -> PostgreSQLConfig:
    return PostgreSQLConfig(
        host=os.getenv("POSTGRESQL_HOST", "localhost"),
        port=int(os.getenv("POSTGRESQL_PORT", "5432")),
        username=os.getenv("POSTGRESQL_USER", "test_user"),
        password=os.getenv("POSTGRESQL_PASSWORD", "test_password"),
        database=os.getenv("POSTGRESQL_DATABASE", "test"),
        schema_name=SCHEMA,
    )


@pytest.fixture(scope="module")
def pg_connector(pg_config: PostgreSQLConfig) -> Generator[PostgreSQLConnector, None, None]:
    """
    Connect to the docker PostgreSQL instance.

    ADAPTERS_PG=1 is a user-confirmed opt-in: if the container is unreachable
    at this point, that's a config/setup error and should fail loudly rather
    than silently skip.
    """
    conn = PostgreSQLConnector(pg_config)
    try:
        if not conn.test_connection():
            pytest.fail(
                "PostgreSQL container unreachable despite ADAPTERS_PG=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-postgresql?"
            )
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_connector(pg_connector: PostgreSQLConnector) -> Generator[PostgreSQLConnector, None, None]:
    """Create seeded tables, yield the connector, drop tables on teardown."""

    def _exec(sql: str) -> None:
        result = pg_connector.execute({"sql_query": sql})
        assert result.success == 1, f"seed SQL failed: {sql[:120]} -> {result.error}"

    _exec(f'DROP TABLE IF EXISTS "{SCHEMA}"."{NATION_TABLE}" CASCADE')
    _exec(f'DROP TABLE IF EXISTS "{SCHEMA}"."{REGION_TABLE}" CASCADE')
    _exec(REGION_DDL)
    _exec(NATION_DDL)
    for row in REGION_ROWS:
        values = ", ".join(_escape_sql_value(v) for v in row)
        _exec(f'INSERT INTO "{SCHEMA}"."{REGION_TABLE}" VALUES ({values})')
    for row in NATION_ROWS:
        values = ", ".join(_escape_sql_value(v) for v in row)
        _exec(f'INSERT INTO "{SCHEMA}"."{NATION_TABLE}" VALUES ({values})')

    try:
        yield pg_connector
    finally:
        pg_connector.execute({"sql_query": f'DROP TABLE IF EXISTS "{SCHEMA}"."{NATION_TABLE}" CASCADE'})
        pg_connector.execute({"sql_query": f'DROP TABLE IF EXISTS "{SCHEMA}"."{REGION_TABLE}" CASCADE'})


@pytest.fixture(scope="module")
def db_tool(seeded_connector: PostgreSQLConnector) -> DBFuncTool:
    """DBFuncTool in legacy single-connector mode (no AgentConfig needed)."""
    return DBFuncTool(seeded_connector)


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_list_tables_returns_seeded_tables(db_tool: DBFuncTool) -> None:
    result = db_tool.list_tables()
    assert result.success == 1, f"list_tables failed: {result.error}"
    names = {entry["name"] for entry in result.result}
    assert REGION_TABLE in names, f"{REGION_TABLE} missing from {sorted(names)}"
    assert NATION_TABLE in names, f"{NATION_TABLE} missing from {sorted(names)}"


def test_describe_table_returns_expected_columns(db_tool: DBFuncTool) -> None:
    result = db_tool.describe_table(REGION_TABLE)
    assert result.success == 1, f"describe_table failed: {result.error}"
    assert isinstance(result.result, dict), f"expected dict, got {type(result.result).__name__}"
    columns = result.result.get("columns") or []
    col_names = [c["name"] for c in columns]
    assert col_names == ["r_regionkey", "r_name", "r_comment"], f"unexpected columns: {col_names}"


def test_get_table_ddl_returns_create_statement(db_tool: DBFuncTool) -> None:
    result = db_tool.get_table_ddl(REGION_TABLE)
    assert result.success == 1, f"get_table_ddl failed: {result.error}"
    payload = result.result
    assert isinstance(payload, dict), f"DDL payload must be dict, got {type(payload).__name__}"
    assert payload.get("table_name") == REGION_TABLE, f"table_name mismatch: {payload}"
    definition = payload.get("definition") or ""
    assert "create" in definition.lower(), f"definition missing CREATE: {definition!r}"
    assert REGION_TABLE in definition, f"definition missing {REGION_TABLE}: {definition!r}"


def test_read_query_executes_select(db_tool: DBFuncTool) -> None:
    result = db_tool.read_query(f"SELECT COUNT(*) AS cnt FROM {REGION_TABLE}")
    assert result.success == 1, f"read_query failed: {result.error}"
    # DBFuncTool.read_query compresses rows — payload is a dict with CSV.
    payload = result.result
    assert isinstance(payload, dict), f"compressed payload must be dict, got {type(payload).__name__}"
    assert payload.get("original_rows") == 1, f"expected 1 row, got {payload.get('original_rows')}"
    assert "3" in (payload.get("compressed_data") or ""), f"count(*) should be 3; payload={payload}"


def test_read_query_rejects_dml(db_tool: DBFuncTool) -> None:
    """DBFuncTool.read_query must reject INSERT — read-only guard."""
    result = db_tool.read_query(f"INSERT INTO {REGION_TABLE} VALUES (99, 'X', '')")
    assert result.success == 0, "INSERT via read_query should have been rejected"
    assert "read-only" in (result.error or "").lower(), f"unexpected error: {result.error}"


def test_read_query_rejects_multi_statement(db_tool: DBFuncTool) -> None:
    """DBFuncTool.read_query must reject `SELECT; DELETE` injection."""
    result = db_tool.read_query(f"SELECT 1; DELETE FROM {REGION_TABLE}")
    assert result.success == 0, "multi-statement SQL should have been rejected"
    assert "multi-statement" in (result.error or "").lower(), f"unexpected error: {result.error}"
