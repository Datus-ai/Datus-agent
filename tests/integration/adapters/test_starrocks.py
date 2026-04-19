"""
Contract tests: StarRocks adapter via DBFuncTool.

Opt-in (all required):
  * install:    `uv pip install datus-starrocks`
  * start:      `cd datus-db-adapters/datus-starrocks && docker compose up -d`
  * env:         ADAPTERS_SR=1

Env overrides (defaults match the adapter's docker-compose.yml):
  STARROCKS_HOST=127.0.0.1  STARROCKS_PORT=9030
  STARROCKS_USER=root       STARROCKS_PASSWORD=
  STARROCKS_CATALOG=default_catalog  STARROCKS_DATABASE=test

See `tests/integration/adapters/README.md`.
"""

import os
from typing import Generator

import pytest

if os.getenv("ADAPTERS_SR") != "1":
    pytest.skip(
        "ADAPTERS_SR=1 not set; see tests/integration/adapters/README.md",
        allow_module_level=True,
    )

pytest.importorskip(
    "datus_starrocks",
    reason="datus-starrocks not installed; run `uv pip install datus-starrocks`",
)

from datus_starrocks import StarRocksConfig, StarRocksConnector  # noqa: E402

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration]


REGION_TABLE = "datus_adapter_region"
NATION_TABLE = "datus_adapter_nation"

# StarRocks DDL: ENGINE=OLAP + PRIMARY KEY + DISTRIBUTED BY HASH + PROPERTIES.
REGION_DDL = f"""
CREATE TABLE IF NOT EXISTS `{REGION_TABLE}` (
    `r_regionkey` INT NOT NULL,
    `r_name` VARCHAR(25) NOT NULL,
    `r_comment` VARCHAR(152)
) ENGINE=OLAP
PRIMARY KEY (`r_regionkey`)
DISTRIBUTED BY HASH(`r_regionkey`) BUCKETS 1
PROPERTIES ("replication_num" = "1")
"""
NATION_DDL = f"""
CREATE TABLE IF NOT EXISTS `{NATION_TABLE}` (
    `n_nationkey` INT NOT NULL,
    `n_name` VARCHAR(25) NOT NULL,
    `n_regionkey` INT NOT NULL,
    `n_comment` VARCHAR(152)
) ENGINE=OLAP
PRIMARY KEY (`n_nationkey`)
DISTRIBUTED BY HASH(`n_nationkey`) BUCKETS 1
PROPERTIES ("replication_num" = "1")
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
def sr_config() -> StarRocksConfig:
    return StarRocksConfig(
        host=os.getenv("STARROCKS_HOST", "127.0.0.1"),
        port=int(os.getenv("STARROCKS_PORT", "9030")),
        username=os.getenv("STARROCKS_USER", "root"),
        password=os.getenv("STARROCKS_PASSWORD", ""),
        catalog=os.getenv("STARROCKS_CATALOG", "default_catalog"),
        database=os.getenv("STARROCKS_DATABASE", "test"),
    )


@pytest.fixture(scope="module")
def sr_connector(sr_config: StarRocksConfig) -> Generator[StarRocksConnector, None, None]:
    # Ensure target database exists (connect without db).
    init_config = StarRocksConfig(
        host=sr_config.host,
        port=sr_config.port,
        username=sr_config.username,
        password=sr_config.password,
        catalog=sr_config.catalog,
        database=None,
    )
    init_conn = StarRocksConnector(init_config)
    try:
        if not init_conn.test_connection():
            pytest.fail(
                "StarRocks container unreachable despite ADAPTERS_SR=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-starrocks?"
            )
        if sr_config.database:
            init_conn.execute_ddl(f"CREATE DATABASE IF NOT EXISTS `{sr_config.database}`")
    finally:
        init_conn.close()

    conn = StarRocksConnector(sr_config)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_connector(sr_connector: StarRocksConnector) -> Generator[StarRocksConnector, None, None]:
    def _exec(sql: str) -> None:
        result = sr_connector.execute({"sql_query": sql})
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
        yield sr_connector
    finally:
        sr_connector.execute({"sql_query": f"DROP TABLE IF EXISTS `{NATION_TABLE}`"})
        sr_connector.execute({"sql_query": f"DROP TABLE IF EXISTS `{REGION_TABLE}`"})


@pytest.fixture(scope="module")
def db_tool(seeded_connector: StarRocksConnector) -> DBFuncTool:
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
