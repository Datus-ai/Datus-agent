"""
Contract tests: Trino adapter via DBFuncTool.

Uses Trino's built-in `tpch.tiny` catalog (pre-populated TPC-H data) — no
seeding needed, tables are read-only.

Opt-in (all required):
  * install:    `uv pip install datus-trino`
  * start:      `cd datus-db-adapters/datus-trino && docker compose up -d`
  *             (port 8080 may conflict; use `TRINO_HOST_PORT=8085 docker compose up -d`
  *              and then set TRINO_PORT=8085 here)
  * env:         ADAPTERS_TRINO=1

Env overrides (defaults match the adapter's docker-compose.yml):
  TRINO_HOST=localhost  TRINO_PORT=8080  TRINO_USER=trino

See `tests/integration/adapters/README.md`.
"""

import os
from typing import Generator

import pytest

if os.getenv("ADAPTERS_TRINO") != "1":
    pytest.skip(
        "ADAPTERS_TRINO=1 not set; see tests/integration/adapters/README.md",
        allow_module_level=True,
    )

pytest.importorskip(
    "datus_trino",
    reason="datus-trino not installed; run `uv pip install datus-trino`",
)

from datus_trino import TrinoConfig, TrinoConnector  # noqa: E402

from datus.tools.func_tool.database import DBFuncTool  # noqa: E402

pytestmark = [pytest.mark.integration]


TPCH_TABLE = "region"  # Trino tpch.tiny: columns are unprefixed (regionkey, name, comment)


@pytest.fixture(scope="module")
def trino_config() -> TrinoConfig:
    return TrinoConfig(
        host=os.getenv("TRINO_HOST", "localhost"),
        port=int(os.getenv("TRINO_PORT", "8080")),
        username=os.getenv("TRINO_USER", "trino"),
        password=os.getenv("TRINO_PASSWORD", ""),
        catalog="tpch",
        schema_name="tiny",
        http_scheme=os.getenv("TRINO_HTTP_SCHEME", "http"),
    )


@pytest.fixture(scope="module")
def trino_connector(trino_config: TrinoConfig) -> Generator[TrinoConnector, None, None]:
    conn = TrinoConnector(trino_config)
    try:
        if not conn.test_connection():
            pytest.fail(
                "Trino container unreachable despite ADAPTERS_TRINO=1. "
                "Did you run `docker compose up -d` in datus-db-adapters/datus-trino? "
                "If port 8080 is taken, start with TRINO_HOST_PORT=8085 and set TRINO_PORT=8085."
            )
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def db_tool(trino_connector: TrinoConnector) -> DBFuncTool:
    return DBFuncTool(trino_connector)


def test_list_tables_returns_tpch_tables(db_tool: DBFuncTool) -> None:
    result = db_tool.list_tables()
    assert result.success == 1, f"list_tables failed: {result.error}"
    names = {entry["name"] for entry in result.result}
    # tpch.tiny ships 8 standard TPC-H tables; we check the ones we actually query.
    assert "region" in names, f"region missing from {sorted(names)}"
    assert "nation" in names, f"nation missing from {sorted(names)}"


def test_describe_table_returns_expected_columns(db_tool: DBFuncTool) -> None:
    result = db_tool.describe_table(TPCH_TABLE)
    assert result.success == 1, f"describe_table failed: {result.error}"
    assert isinstance(result.result, dict), f"expected dict, got {type(result.result).__name__}"
    columns = result.result.get("columns") or []
    col_names = {c["name"] for c in columns}
    assert {"regionkey", "name", "comment"}.issubset(col_names), f"missing cols: {col_names}"


def test_get_table_ddl_returns_create_statement(db_tool: DBFuncTool) -> None:
    result = db_tool.get_table_ddl(TPCH_TABLE)
    assert result.success == 1, f"get_table_ddl failed: {result.error}"
    payload = result.result
    assert isinstance(payload, dict), f"DDL payload must be dict, got {type(payload).__name__}"
    assert payload.get("table_name") == TPCH_TABLE, f"table_name mismatch: {payload}"
    definition = payload.get("definition") or ""
    assert "create" in definition.lower(), f"definition missing CREATE: {definition!r}"
    assert TPCH_TABLE in definition.lower(), f"definition missing {TPCH_TABLE}: {definition!r}"


def test_read_query_executes_select(db_tool: DBFuncTool) -> None:
    # tpch.tiny.region has exactly 5 rows (standard TPC-H region count).
    result = db_tool.read_query("SELECT COUNT(*) AS cnt FROM region")
    assert result.success == 1, f"read_query failed: {result.error}"
    payload = result.result
    assert isinstance(payload, dict), f"compressed payload must be dict, got {type(payload).__name__}"
    assert payload.get("original_rows") == 1, f"expected 1 row, got {payload.get('original_rows')}"
    assert "5" in (payload.get("compressed_data") or ""), f"region count should be 5; payload={payload}"


def test_read_query_rejects_dml(db_tool: DBFuncTool) -> None:
    """Guard fires at SQL-type parse stage, before Trino rejects the read-only catalog."""
    result = db_tool.read_query("INSERT INTO region VALUES (99, 'X', '')")
    assert result.success == 0, "INSERT via read_query should have been rejected"
    assert "read-only" in (result.error or "").lower(), f"unexpected error: {result.error}"


def test_read_query_rejects_multi_statement(db_tool: DBFuncTool) -> None:
    result = db_tool.read_query("SELECT 1; DELETE FROM region")
    assert result.success == 0, "multi-statement SQL should have been rejected"
    assert "multi-statement" in (result.error or "").lower(), f"unexpected error: {result.error}"
