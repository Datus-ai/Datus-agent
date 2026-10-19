"""
Contract tests: ClickHouse adapter via DBFuncTool.

Opt-in (all required):
  * install:    `uv pip install datus-clickhouse`
  * start:      `cd datus-db-adapters/datus-clickhouse && docker compose up -d`
  * env:         ADAPTERS_CH=1

Env overrides (defaults match the adapter's docker-compose.yml):
  CLICKHOUSE_HOST=localhost  CLICKHOUSE_PORT=8123
  CLICKHOUSE_USER=default_user  CLICKHOUSE_PASSWORD=default_test
  CLICKHOUSE_DATABASE=default_test
  CLICKHOUSE_PROTOCOL=http CLICKHOUSE_VERIFY=true CLICKHOUSE_CA_CERT=<optional PEM file>

See `tests/integration/adapters/README.md`.
"""

import os
from typing import Generator

import pytest
import yaml

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_CH", "tests/integration/adapters/README.md")

datus_clickhouse = import_required(
    "datus_clickhouse",
    reason="datus-clickhouse not installed; run `uv pip install datus-clickhouse`",
)

ClickHouseConfig = datus_clickhouse.ClickHouseConfig
ClickHouseConnector = datus_clickhouse.ClickHouseConnector

from datus.configuration.agent_config import DbConfig  # noqa: E402
from datus.tools.db_tools.db_manager import DBManager  # noqa: E402
from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.nightly]


REGION_TABLE = "datus_adapter_region"
NATION_TABLE = "datus_adapter_nation"

# ClickHouse requires ENGINE + ORDER BY (no PK concept identical to MySQL/PG).
REGION_DDL = f"""
CREATE TABLE IF NOT EXISTS `{REGION_TABLE}` (
    `r_regionkey` Int32,
    `r_name` String,
    `r_comment` String
) ENGINE = MergeTree() ORDER BY r_regionkey
"""
NATION_DDL = f"""
CREATE TABLE IF NOT EXISTS `{NATION_TABLE}` (
    `n_nationkey` Int32,
    `n_name` String,
    `n_regionkey` Int32,
    `n_comment` String
) ENGINE = MergeTree() ORDER BY n_nationkey
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
def ch_config() -> ClickHouseConfig:
    tls = {}
    for key, variable in (
        ("protocol", "CLICKHOUSE_PROTOCOL"),
        ("verify", "CLICKHOUSE_VERIFY"),
        ("ca_cert", "CLICKHOUSE_CA_CERT"),
    ):
        if os.getenv(variable):
            tls[key] = os.environ[variable]
    return ClickHouseConfig(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8443" if tls.get("protocol") == "https" else "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default_user"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "default_test"),
        database=os.getenv("CLICKHOUSE_DATABASE", "default_test"),
        **tls,
    )


@pytest.fixture(scope="module")
def ch_connector(ch_config: ClickHouseConfig) -> Generator[ClickHouseConnector, None, None]:
    # Ensure target database exists (connect without db first).
    init_config = ch_config.model_copy(update={"database": None})
    init_conn = ClickHouseConnector(init_config)
    try:
        if not init_conn.test_connection():
            pytest.fail(
                "ClickHouse container unreachable despite ADAPTERS_CH=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-clickhouse?"
            )
        if ch_config.database:
            init_conn.execute_ddl(f"CREATE DATABASE IF NOT EXISTS `{ch_config.database}`")
    finally:
        init_conn.close()

    conn = ClickHouseConnector(ch_config)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def seeded_connector(ch_connector: ClickHouseConnector) -> Generator[ClickHouseConnector, None, None]:
    def _exec(sql: str) -> None:
        result = ch_connector.execute({"sql_query": sql})
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
        yield ch_connector
    finally:
        ch_connector.execute({"sql_query": f"DROP TABLE IF EXISTS `{NATION_TABLE}`"})
        ch_connector.execute({"sql_query": f"DROP TABLE IF EXISTS `{REGION_TABLE}`"})


@pytest.fixture(scope="module")
def db_tool(seeded_connector: ClickHouseConnector) -> DBFuncTool:
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


def test_yaml_profile_queries_through_dbmanager(ch_config, seeded_connector, tmp_path):
    """Use the real YAML -> DbConfig -> registry path, without a model or driver mock."""
    profile = {
        "agent": {"services": {"datasources": {"clickhouse_contract": ch_config.model_dump() | {"type": "clickhouse"}}}}
    }
    path = tmp_path / "agent.yml"
    path.write_text(yaml.safe_dump(profile))
    parsed = yaml.safe_load(path.read_text())["agent"]["services"]["datasources"]["clickhouse_contract"]
    config = DbConfig.filter_kwargs(DbConfig, parsed)
    manager = DBManager({"clickhouse_contract": config})
    try:
        connector = manager.get_conn("clickhouse_contract")
        assert connector.config.protocol == ch_config.protocol
        assert connector.config.verify == ch_config.verify
        assert connector.config.ca_cert == ch_config.ca_cert
        tool = DBFuncTool(connector)
        result = tool.read_query(f"SELECT count(*) AS cnt FROM {REGION_TABLE}")
        assert result.success == 1, result.error
        assert result.result["original_rows"] == 1
        assert "3" in result.result["compressed_data"]
        metadata = tool.describe_table(REGION_TABLE)
        assert metadata.success == 1, metadata.error
        assert [column["name"] for column in metadata.result["columns"]] == ["r_regionkey", "r_name", "r_comment"]
    finally:
        manager.close()
