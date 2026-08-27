"""
Contract tests: TiDB adapter via DBFuncTool.

Opt-in (all required):
  * install:    `uv pip install datus-tidb`
  * start:      `cd datus-db-adapters/datus-tidb && docker compose up -d`
  * env:         ADAPTERS_TIDB=1

Env overrides (defaults match the adapter's docker-compose.yml):
  TIDB_HOST=127.0.0.1  TIDB_PORT=4000
  TIDB_USER=root       TIDB_PASSWORD=
  TIDB_DATABASE=test

See `tests/integration/adapters/README.md`.
"""

import os
from typing import Generator

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_TIDB", "tests/integration/adapters/README.md")

datus_tidb = import_required(
    "datus_tidb",
    reason="datus-tidb not installed; run `uv pip install datus-tidb`",
)

TiDBConfig = datus_tidb.TiDBConfig
TiDBConnector = datus_tidb.TiDBConnector

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.nightly]


REGION_TABLE = "datus_adapter_region"
NATION_TABLE = "datus_adapter_nation"

REGION_DDL = f"""
CREATE TABLE IF NOT EXISTS `{REGION_TABLE}` (
    `r_regionkey` INT NOT NULL,
    `r_name` VARCHAR(25) NOT NULL,
    `r_comment` VARCHAR(152),
    PRIMARY KEY (`r_regionkey`)
)
"""
NATION_DDL = f"""
CREATE TABLE IF NOT EXISTS `{NATION_TABLE}` (
    `n_nationkey` INT NOT NULL,
    `n_name` VARCHAR(25) NOT NULL,
    `n_regionkey` INT NOT NULL,
    `n_comment` VARCHAR(152),
    PRIMARY KEY (`n_nationkey`)
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
def tidb_config() -> TiDBConfig:
    return TiDBConfig(
        host=os.getenv("TIDB_HOST", "127.0.0.1"),
        port=int(os.getenv("TIDB_PORT", "4000")),
        username=os.getenv("TIDB_USER", "root"),
        password=os.getenv("TIDB_PASSWORD", ""),
        database=os.getenv("TIDB_DATABASE", "test"),
    )


@pytest.fixture(scope="module")
def tidb_connector(tidb_config: TiDBConfig) -> Generator[TiDBConnector, None, None]:
    conn = TiDBConnector(tidb_config)
    try:
        if not conn.test_connection():
            pytest.fail(
                "TiDB unreachable despite ADAPTERS_TIDB=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-tidb?"
            )
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_connector(tidb_connector: TiDBConnector) -> Generator[TiDBConnector, None, None]:
    def _exec(sql: str) -> None:
        result = tidb_connector.execute({"sql_query": sql})
        assert result.success == 1, f"seed SQL failed: {sql[:120]} -> {result.error}"

    _exec(f"DROP TABLE IF EXISTS `{NATION_TABLE}`")
    _exec(f"DROP TABLE IF EXISTS `{REGION_TABLE}`")
    _exec(REGION_DDL)
    _exec(NATION_DDL)
    for row in REGION_ROWS:
        values = ", ".join(_escape(v) for v in row)
        _exec(f"INSERT INTO `{REGION_TABLE}` VALUES ({values})")
    for row in NATION_ROWS:
        values = ", ".join(_escape(v) for v in row)
        _exec(f"INSERT INTO `{NATION_TABLE}` VALUES ({values})")

    try:
        yield tidb_connector
    finally:
        tidb_connector.execute({"sql_query": f"DROP TABLE IF EXISTS `{NATION_TABLE}`"})
        tidb_connector.execute({"sql_query": f"DROP TABLE IF EXISTS `{REGION_TABLE}`"})


@pytest.fixture(scope="module")
def db_tool(seeded_connector: TiDBConnector) -> DBFuncTool:
    return DBFuncTool(seeded_connector)


def test_list_tables_returns_seeded_tables(db_tool: DBFuncTool) -> None:
    result = db_tool.list_tables()
    assert result.success == 1, f"list_tables failed: {result.error}"
    names = {entry["qualified_name"].split(".")[-1] for entry in result.result}
    assert REGION_TABLE in names, f"{REGION_TABLE} missing from {sorted(names)}"
    assert NATION_TABLE in names, f"{NATION_TABLE} missing from {sorted(names)}"


def test_list_tables_hides_tidb_system_databases(db_tool: DBFuncTool, seeded_connector: TiDBConnector) -> None:
    """METRICS_SCHEMA is TiDB-only; the inherited MySQL filter does not know it."""
    databases = {name.lower() for name in seeded_connector.get_databases()}

    assert "metrics_schema" not in databases
    assert databases.isdisjoint({"information_schema", "performance_schema", "mysql", "sys"})


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


def test_transfer_quotes_identifiers_with_backticks(db_tool: DBFuncTool, seeded_connector: TiDBConnector) -> None:
    """TiDB rejects double-quoted identifiers under its default sql_mode, so a
    transfer target built with the wrong quote character fails outright."""
    quoted = DBFuncTool._quote_column_identifier("r_name", seeded_connector.dialect)
    assert quoted == "`r_name`", f"TiDB identifiers must use backticks, got {quoted}"

    result = seeded_connector.execute(
        {"sql_query": f"SELECT {quoted} FROM `{REGION_TABLE}` ORDER BY `r_regionkey` LIMIT 1"},
        result_format="list",
    )
    assert result.success == 1, f"backtick-quoted identifier rejected: {result.error}"
    assert result.sql_return[0]["r_name"] == "AFRICA"
