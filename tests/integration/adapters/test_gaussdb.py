"""
Contract tests: GaussDB adapter via DBFuncTool.

GaussDB / openGauss speaks the PostgreSQL wire protocol; the adapter
`GaussDBConnector` inherits from `PostgreSQLConnector`. We still test it as a
separate adapter to catch GaussDB-specific regressions (A/B/PG compatibility
mode detection, distribution-aware DDL, official-driver connection handling).

Opt-in (all required):
  * install:    `uv pip install datus-gaussdb`
  * start:      `cd datus-db-adapters/datus-gaussdb && docker compose up -d`
  * env:         ADAPTERS_GAUSSDB=1

Env overrides (defaults match the adapter's docker-compose.yml):
  GAUSSDB_HOST=127.0.0.1  GAUSSDB_PORT=25434
  GAUSSDB_USER=datus      GAUSSDB_PASSWORD=Datus@123
  GAUSSDB_DATABASE=postgres  GAUSSDB_SCHEMA=public
  GAUSSDB_DRIVER=pg8000      GAUSSDB_SSLMODE=verify-ca
  GAUSSDB_SSLROOTCERT=/path/to/gaussdb-ca.pem

The official `gaussdb` driver binds a GaussDB-family libpq that is published for
Linux only (release wheels bundle that libpq). macOS uses the pure-Python
`pg8000` path by default.

See `tests/integration/adapters/README.md`.
"""

import os
import sys
from typing import Generator

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_GAUSSDB", "tests/integration/adapters/README.md")

datus_gaussdb = import_required(
    "datus_gaussdb",
    reason="datus-gaussdb not installed; run `uv pip install datus-gaussdb`",
)

GaussDBConfig = datus_gaussdb.GaussDBConfig
GaussDBConnector = datus_gaussdb.GaussDBConnector

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.nightly]


SCHEMA = os.getenv("GAUSSDB_SCHEMA", "public")
REGION_TABLE = "datus_adapter_region"
NATION_TABLE = "datus_adapter_nation"

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
def gaussdb_config() -> GaussDBConfig:
    return GaussDBConfig(
        host=os.getenv("GAUSSDB_HOST", "127.0.0.1"),
        port=int(os.getenv("GAUSSDB_PORT", "25434")),
        username=os.getenv("GAUSSDB_USER", "datus"),
        password=os.getenv("GAUSSDB_PASSWORD", "Datus@123"),
        database=os.getenv("GAUSSDB_DATABASE", "postgres"),
        schema_name=SCHEMA,
        driver=os.getenv("GAUSSDB_DRIVER") or ("pg8000" if sys.platform == "darwin" else "gaussdb"),
        sslmode=os.getenv("GAUSSDB_SSLMODE", "prefer"),
        sslrootcert=os.getenv("GAUSSDB_SSLROOTCERT") or None,
    )


@pytest.fixture(scope="module")
def gaussdb_connector(gaussdb_config: GaussDBConfig) -> Generator[GaussDBConnector, None, None]:
    conn = GaussDBConnector(gaussdb_config)
    try:
        if not conn.test_connection():
            pytest.fail(
                "GaussDB container unreachable despite ADAPTERS_GAUSSDB=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-gaussdb?"
            )
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_connector(gaussdb_connector: GaussDBConnector) -> Generator[GaussDBConnector, None, None]:
    def _exec(sql: str) -> None:
        result = gaussdb_connector.execute({"sql_query": sql})
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
        yield gaussdb_connector
    finally:
        gaussdb_connector.execute({"sql_query": f'DROP TABLE IF EXISTS "{SCHEMA}"."{NATION_TABLE}" CASCADE'})
        gaussdb_connector.execute({"sql_query": f'DROP TABLE IF EXISTS "{SCHEMA}"."{REGION_TABLE}" CASCADE'})


@pytest.fixture(scope="module")
def db_tool(seeded_connector: GaussDBConnector) -> DBFuncTool:
    return DBFuncTool(seeded_connector)


def test_list_tables_returns_seeded_tables(db_tool: DBFuncTool) -> None:
    result = db_tool.list_tables()
    assert result.success == 1, f"list_tables failed: {result.error}"
    names = {entry["qualified_name"].split(".")[-1] for entry in result.result}
    assert REGION_TABLE in names, f"{REGION_TABLE} missing from {sorted(names)}"
    assert NATION_TABLE in names, f"{NATION_TABLE} missing from {sorted(names)}"


def test_describe_table_returns_expected_columns(db_tool: DBFuncTool) -> None:
    result = db_tool.describe_table(REGION_TABLE)
    assert result.success == 1, f"describe_table failed: {result.error}"
    assert isinstance(result.result, dict), f"expected dict, got {type(result.result).__name__}"
    columns = result.result.get("columns") or []
    col_names = [c["name"] for c in columns]
    assert col_names == ["r_regionkey", "r_name", "r_comment"], f"unexpected columns: {col_names}"


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
