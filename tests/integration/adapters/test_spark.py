"""
Contract tests: Spark (Thrift) adapter via DBFuncTool.

Opt-in (all required):
  * install:    `uv pip install datus-spark`
  * start:      `cd datus-db-adapters/datus-spark && docker compose up -d`
  *             (wait ~60s for the Thrift server healthcheck)
  * env:         ADAPTERS_SPARK=1

Env overrides (defaults match the adapter's docker-compose.yml):
  SPARK_HOST=localhost  SPARK_PORT=10000
  SPARK_USER=spark      SPARK_PASSWORD=
  SPARK_DATABASE=default SPARK_AUTH_MECHANISM=NONE

Port 10000 collides with Hive Thrift. Run Hive OR Spark contract tests, not both.

See `tests/integration/adapters/README.md`.
"""

import os
from typing import Generator

import pytest

if os.getenv("ADAPTERS_SPARK") != "1":
    pytest.skip(
        "ADAPTERS_SPARK=1 not set; see tests/integration/adapters/README.md",
        allow_module_level=True,
    )

pytest.importorskip(
    "datus_spark",
    reason="datus-spark not installed; run `uv pip install datus-spark`",
)

from datus_spark import SparkConfig, SparkConnector  # noqa: E402

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration]


REGION_TABLE = "datus_adapter_region"
NATION_TABLE = "datus_adapter_nation"

# Spark SQL: uses backticks; default catalog is the session catalog.
REGION_DDL = f"""
CREATE TABLE IF NOT EXISTS {REGION_TABLE} (
    r_regionkey INT,
    r_name STRING,
    r_comment STRING
) USING PARQUET
"""
NATION_DDL = f"""
CREATE TABLE IF NOT EXISTS {NATION_TABLE} (
    n_nationkey INT,
    n_name STRING,
    n_regionkey INT,
    n_comment STRING
) USING PARQUET
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
def spark_config() -> SparkConfig:
    return SparkConfig(
        host=os.getenv("SPARK_HOST", "localhost"),
        port=int(os.getenv("SPARK_PORT", "10000")),
        username=os.getenv("SPARK_USER", "spark"),
        password=os.getenv("SPARK_PASSWORD", ""),
        database=os.getenv("SPARK_DATABASE", "default"),
        auth_mechanism=os.getenv("SPARK_AUTH_MECHANISM", "NONE"),
    )


@pytest.fixture(scope="module")
def spark_connector(spark_config: SparkConfig) -> Generator[SparkConnector, None, None]:
    conn = SparkConnector(spark_config)
    try:
        if not conn.test_connection():
            pytest.fail(
                "Spark container unreachable despite ADAPTERS_SPARK=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-spark? "
                "Wait ~60s for the Thrift server to become healthy."
            )
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_connector(spark_connector: SparkConnector) -> Generator[SparkConnector, None, None]:
    def _exec(sql: str) -> None:
        result = spark_connector.execute({"sql_query": sql})
        assert result.success == 1, f"seed SQL failed: {sql[:120]} -> {result.error}"

    _exec(f"DROP TABLE IF EXISTS {NATION_TABLE}")
    _exec(f"DROP TABLE IF EXISTS {REGION_TABLE}")
    _exec(REGION_DDL)
    _exec(NATION_DDL)
    values_sql = ", ".join("(" + ", ".join(_escape(v) for v in row) + ")" for row in REGION_ROWS)
    _exec(f"INSERT INTO {REGION_TABLE} VALUES {values_sql}")
    values_sql = ", ".join("(" + ", ".join(_escape(v) for v in row) + ")" for row in NATION_ROWS)
    _exec(f"INSERT INTO {NATION_TABLE} VALUES {values_sql}")

    try:
        yield spark_connector
    finally:
        spark_connector.execute({"sql_query": f"DROP TABLE IF EXISTS {NATION_TABLE}"})
        spark_connector.execute({"sql_query": f"DROP TABLE IF EXISTS {REGION_TABLE}"})


@pytest.fixture(scope="module")
def db_tool(seeded_connector: SparkConnector) -> DBFuncTool:
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
