"""
Contract tests: Doris adapter via DBFuncTool.

Opt-in (all required):
  * install:    `uv pip install datus-doris`
  * start:      `cd datus-db-adapters/datus-doris && docker compose up -d`
  * set env:    ADAPTERS_DORIS=1

Defaults match the adapter's docker-compose.yml:
  DORIS_HOST=127.0.0.1  DORIS_PORT=9030
  DORIS_USER=root  DORIS_PASSWORD=
  DORIS_CATALOG=internal  DORIS_DATABASE=test
"""

import os
from collections.abc import Generator

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_DORIS", "tests/integration/adapters/README.md")

datus_doris = import_required(
    "datus_doris",
    reason="datus-doris not installed; run `uv pip install datus-doris`",
)

DorisConfig = datus_doris.DorisConfig
DorisConnector = datus_doris.DorisConnector

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.nightly]

REGION_TABLE = "datus_agent_doris_region"
NATION_TABLE = "datus_agent_doris_nation"


def _config(database: str | None = None) -> DorisConfig:
    return DorisConfig(
        host=os.getenv("DORIS_HOST", "127.0.0.1"),
        port=int(os.getenv("DORIS_PORT", "9030")),
        username=os.getenv("DORIS_USER", "root"),
        password=os.getenv("DORIS_PASSWORD", ""),
        catalog=os.getenv("DORIS_CATALOG", "internal"),
        database=database if database is not None else os.getenv("DORIS_DATABASE", "test"),
    )


@pytest.fixture(scope="module")
def doris_connector() -> Generator[DorisConnector, None, None]:
    database = os.getenv("DORIS_DATABASE", "test")
    init_conn = DorisConnector(_config(database="information_schema"))
    try:
        if not init_conn.test_connection():
            pytest.fail(
                "Doris container unreachable despite ADAPTERS_DORIS=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-doris?"
            )
        created = init_conn.execute_ddl(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        assert created.success == 1, created.error
    finally:
        init_conn.close()

    conn = DorisConnector(_config(database=database))
    try:
        for table in (NATION_TABLE, REGION_TABLE):
            conn.execute_ddl(f"DROP TABLE IF EXISTS `{table}`")
        region = conn.execute_ddl(
            f"""
            CREATE TABLE `{REGION_TABLE}` (
                `r_regionkey` INT NOT NULL,
                `r_name` VARCHAR(25) NOT NULL,
                `r_comment` VARCHAR(152)
            ) ENGINE=OLAP
            DUPLICATE KEY(`r_regionkey`)
            DISTRIBUTED BY HASH(`r_regionkey`) BUCKETS 1
            PROPERTIES ("replication_num" = "1")
            """
        )
        assert region.success == 1, region.error
        nation = conn.execute_ddl(
            f"""
            CREATE TABLE `{NATION_TABLE}` (
                `n_nationkey` INT NOT NULL,
                `n_name` VARCHAR(25) NOT NULL,
                `n_regionkey` INT NOT NULL,
                `n_comment` VARCHAR(152)
            ) ENGINE=OLAP
            DUPLICATE KEY(`n_nationkey`)
            DISTRIBUTED BY HASH(`n_nationkey`) BUCKETS 1
            PROPERTIES ("replication_num" = "1")
            """
        )
        assert nation.success == 1, nation.error
        inserted = conn.execute_insert(
            f"INSERT INTO `{REGION_TABLE}` VALUES "
            "(0, 'AFRICA', 'lar deposits.'), "
            "(1, 'AMERICA', 'hs use ironic requests.'), "
            "(2, 'ASIA', 'ges. pinto beans.')"
        )
        assert inserted.success == 1, inserted.error
        yield conn
    finally:
        for table in (NATION_TABLE, REGION_TABLE):
            conn.execute_ddl(f"DROP TABLE IF EXISTS `{table}`")
        conn.close()


@pytest.fixture(scope="module")
def db_tool(doris_connector) -> DBFuncTool:
    return DBFuncTool(doris_connector)


def test_list_tables_returns_seeded_table(db_tool: DBFuncTool) -> None:
    result = db_tool.list_tables()
    assert result.success == 1, result.error
    names = {entry["qualified_name"].split(".")[-1] for entry in result.result}
    assert REGION_TABLE in names
    assert NATION_TABLE in names


def test_describe_table_returns_expected_columns(db_tool: DBFuncTool) -> None:
    result = db_tool.describe_table(REGION_TABLE)
    assert result.success == 1, result.error
    assert [column["name"] for column in result.result["columns"]] == [
        "r_regionkey",
        "r_name",
        "r_comment",
    ]


def test_read_query_executes_select(db_tool: DBFuncTool) -> None:
    result = db_tool.read_query(f"SELECT COUNT(*) AS cnt FROM `{REGION_TABLE}`")
    assert result.success == 1, result.error
    assert result.result["original_rows"] == 1
    assert "3" in result.result["compressed_data"]


def test_read_query_rejects_writes(db_tool: DBFuncTool) -> None:
    dml = db_tool.read_query(f"INSERT INTO `{REGION_TABLE}` VALUES (99, 'X', '')")
    assert dml.success == 0
    assert "read-only" in (dml.error or "").lower()

    multi = db_tool.read_query(f"SELECT 1; DELETE FROM `{REGION_TABLE}`")
    assert multi.success == 0
    assert "multi-statement" in (multi.error or "").lower()
