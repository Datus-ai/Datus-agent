"""
Contract tests: Greenplum adapter via DBFuncTool.

Greenplum is PostgreSQL wire-protocol compatible; the adapter `GreenplumConnector`
inherits from `PostgreSQLConnector`. We still test it as a separate adapter to
catch GP-specific regressions (DDL generation with distribution policy, system
database/schema filters).

Opt-in (all required):
  * install:    `uv pip install datus-greenplum`
  * start:      `cd datus-db-adapters/datus-greenplum && docker compose up -d`
  * env:         ADAPTERS_GP=1

Env overrides (defaults match the adapter's docker-compose.yml):
  GREENPLUM_HOST=localhost  GREENPLUM_PORT=15432
  GREENPLUM_USER=gpadmin    GREENPLUM_PASSWORD=pivotal
  GREENPLUM_DATABASE=test   GREENPLUM_SCHEMA=public

See `tests/integration/adapters/README.md`.
"""

import os
from typing import Generator

import pytest

if os.getenv("ADAPTERS_GP") != "1":
    pytest.skip(
        "ADAPTERS_GP=1 not set; see tests/integration/adapters/README.md",
        allow_module_level=True,
    )

pytest.importorskip(
    "datus_greenplum",
    reason="datus-greenplum not installed; run `uv pip install datus-greenplum`",
)

from datus_greenplum import GreenplumConfig, GreenplumConnector  # noqa: E402

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration]


SCHEMA = os.getenv("GREENPLUM_SCHEMA", "public")
REGION_TABLE = "datus_adapter_region"
NATION_TABLE = "datus_adapter_nation"

# Greenplum 6.x rejects `CREATE TABLE IF NOT EXISTS`; use plain CREATE after DROP.
REGION_DDL = f"""
CREATE TABLE "{SCHEMA}"."{REGION_TABLE}" (
    r_regionkey INTEGER PRIMARY KEY,
    r_name VARCHAR(25) NOT NULL,
    r_comment VARCHAR(152)
)
"""
NATION_DDL = f"""
CREATE TABLE "{SCHEMA}"."{NATION_TABLE}" (
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


def _escape(v: object) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


@pytest.fixture(scope="module")
def gp_config() -> GreenplumConfig:
    return GreenplumConfig(
        host=os.getenv("GREENPLUM_HOST", "localhost"),
        port=int(os.getenv("GREENPLUM_PORT", "15432")),
        username=os.getenv("GREENPLUM_USER", "gpadmin"),
        password=os.getenv("GREENPLUM_PASSWORD", "pivotal"),
        database=os.getenv("GREENPLUM_DATABASE", "test"),
        schema_name=SCHEMA,
    )


@pytest.fixture(scope="module")
def gp_connector(gp_config: GreenplumConfig) -> Generator[GreenplumConnector, None, None]:
    conn = GreenplumConnector(gp_config)
    try:
        if not conn.test_connection():
            pytest.fail(
                "Greenplum container unreachable despite ADAPTERS_GP=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-greenplum?"
            )
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_connector(gp_connector: GreenplumConnector) -> Generator[GreenplumConnector, None, None]:
    def _exec(sql: str) -> None:
        result = gp_connector.execute({"sql_query": sql})
        assert result.success == 1, f"seed SQL failed: {sql[:120]} -> {result.error}"

    _exec(f'DROP TABLE IF EXISTS "{SCHEMA}"."{NATION_TABLE}" CASCADE')
    _exec(f'DROP TABLE IF EXISTS "{SCHEMA}"."{REGION_TABLE}" CASCADE')
    _exec(REGION_DDL)
    _exec(NATION_DDL)
    for row in REGION_ROWS:
        values = ", ".join(_escape(v) for v in row)
        _exec(f'INSERT INTO "{SCHEMA}"."{REGION_TABLE}" VALUES ({values})')
    for row in NATION_ROWS:
        values = ", ".join(_escape(v) for v in row)
        _exec(f'INSERT INTO "{SCHEMA}"."{NATION_TABLE}" VALUES ({values})')

    try:
        yield gp_connector
    finally:
        gp_connector.execute({"sql_query": f'DROP TABLE IF EXISTS "{SCHEMA}"."{NATION_TABLE}" CASCADE'})
        gp_connector.execute({"sql_query": f'DROP TABLE IF EXISTS "{SCHEMA}"."{REGION_TABLE}" CASCADE'})


@pytest.fixture(scope="module")
def db_tool(seeded_connector: GreenplumConnector) -> DBFuncTool:
    return DBFuncTool(seeded_connector)


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
    payload = result.result
    assert isinstance(payload, dict), f"compressed payload must be dict, got {type(payload).__name__}"
    assert payload.get("original_rows") == 1, f"expected 1 row, got {payload.get('original_rows')}"
    assert "3" in (payload.get("compressed_data") or ""), f"count(*) should be 3; payload={payload}"


def test_read_query_rejects_dml(db_tool: DBFuncTool) -> None:
    result = db_tool.read_query(f"INSERT INTO {REGION_TABLE} VALUES (99, 'X', '')")
    assert result.success == 0, "INSERT via read_query should have been rejected"
    assert "read-only" in (result.error or "").lower(), f"unexpected error: {result.error}"


def test_read_query_rejects_multi_statement(db_tool: DBFuncTool) -> None:
    result = db_tool.read_query(f"SELECT 1; DELETE FROM {REGION_TABLE}")
    assert result.success == 0, "multi-statement SQL should have been rejected"
    assert "multi-statement" in (result.error or "").lower(), f"unexpected error: {result.error}"
