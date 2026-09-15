# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json

import duckdb
import pytest

from datus.tools.db_tools.datasource_quality import QualityChecker, summarize


def _status(results, name):
    return next(r["status"] for r in results if r["check"] == name)


def _has(results, name):
    return any(r["check"] == name for r in results)


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE dim_product (product_id BIGINT, product_name VARCHAR, list_price DECIMAL(18, 2));
        CREATE TABLE ods_order (
            order_id BIGINT, product_id BIGINT, stat_dt DATE,
            gmv_amt DECIMAL(18, 2), refund_rate DOUBLE, item_cnt INTEGER
        );
        """
    )
    connection.execute("INSERT INTO dim_product SELECT i, 'Widget ' || i, 10.0 + i FROM range(1, 40) t(i)")
    # 600 days of orders so the span check has more than 13 months to look at.
    connection.execute(
        """
        INSERT INTO ods_order
        SELECT i,
               1 + (i % 39),
               DATE '2024-01-01' + INTERVAL (i % 600) DAY,
               10.0 + (i % 50),
               0.05,
               1 + (i % 3)
        FROM range(1, 4000) t(i)
        """
    )
    yield connection
    connection.close()


@pytest.mark.acceptance
def test_run_returns_named_checks(con):
    results = QualityChecker(con).run()

    assert results
    assert {"check", "status", "detail"} == set(results[0])
    for name in ("layering", "foreign key integrity", "no future-dated rows", "derived quantities sane"):
        assert _has(results, name), f"missing check: {name}"


@pytest.mark.acceptance
def test_ratio_out_of_range_fails(con):
    con.execute("UPDATE ods_order SET refund_rate = 4.2 WHERE order_id = 1")

    results = QualityChecker(con).run()

    assert _status(results, "derived quantities sane") == "FAIL"


@pytest.mark.acceptance
def test_negative_count_fails(con):
    con.execute("UPDATE ods_order SET item_cnt = -3 WHERE order_id = 2")

    results = QualityChecker(con).run()

    assert _status(results, "derived quantities sane") == "FAIL"


@pytest.mark.acceptance
def test_future_dated_rows_fail(con):
    con.execute("UPDATE ods_order SET stat_dt = current_date + 30 WHERE order_id = 3")

    results = QualityChecker(con).run()

    assert _status(results, "no future-dated rows") == "FAIL"


@pytest.mark.acceptance
def test_null_primary_key_is_caught_via_metadata(con):
    """Without this check a table of NULL keys still reports foreign keys at a 100% hit rate."""
    con.execute("UPDATE dim_product SET product_id = NULL WHERE product_name = 'Widget 1'")

    results = QualityChecker(con, meta={"pks": {"dim_product": "product_id"}}).run()

    assert _status(results, "primary key non-null and unique") == "FAIL"


@pytest.mark.acceptance
def test_assertions_from_config(con):
    config = {
        "assertions": [
            {"name": "orders exist", "expect": "nonzero", "sql": "SELECT count(*) FROM ods_order"},
            {"name": "no orphan items", "expect": "zero", "sql": "SELECT count(*) FROM ods_order WHERE gmv_amt < 0"},
            {
                "name": "average in band",
                "expect": {"min": 1, "max": 1000},
                "sql": "SELECT avg(gmv_amt) FROM ods_order",
            },
            {"name": "deliberately broken", "expect": "zero", "sql": "SELECT count(*) FROM ods_order"},
        ]
    }

    results = QualityChecker(con, config).run()

    assert _status(results, "orders exist") == "PASS"
    assert _status(results, "no orphan items") == "PASS"
    assert _status(results, "average in band") == "PASS"
    assert _status(results, "deliberately broken") == "FAIL"


@pytest.mark.acceptance
def test_skip_list_removes_a_check(con):
    results = QualityChecker(con, {"skip": ["layering"]}).run()

    assert not _has(results, "layering")


@pytest.mark.acceptance
def test_head_cap_relaxes_for_small_cardinality():
    """A 21-value dimension cannot be held to the same head share as a 5,000-value one."""
    assert QualityChecker._head_cap(21) > QualityChecker._head_cap(300)
    assert QualityChecker._head_cap(300) > QualityChecker._head_cap(5000)


@pytest.mark.acceptance
def test_summarize_counts_and_ok_flag():
    results = [
        {"check": "a", "status": "PASS", "detail": ""},
        {"check": "b", "status": "WARN", "detail": ""},
        {"check": "c", "status": "FAIL", "detail": ""},
    ]

    out = summarize(results)

    assert out["total"] == 3
    assert (out["passed"], out["warned"], out["failed"]) == (1, 1, 1)
    assert out["ok"] is False
    assert [f["check"] for f in out["failures"]] == ["c"]


@pytest.mark.acceptance
def test_summarize_ok_when_only_warnings():
    out = summarize([{"check": "a", "status": "WARN", "detail": ""}])

    assert out["ok"] is True


class TestQualityToolGate:
    """The tool layer around the checker.

    ``check_datasource_quality`` executes SQL supplied by a workspace JSON file, and it does so on
    the connector's raw connection - which is writable and sees neither the read-only gate nor
    PermissionHooks. The checker itself is a read operation and must stay available to a read-only
    agent, so the gate is on the statements rather than on the tool.
    """

    @pytest.fixture
    def tool(self, tmp_path, monkeypatch):
        from datus.tools.db_tools.config import DuckDBConfig
        from datus.tools.db_tools.duckdb_connector import DuckdbConnector
        from datus.tools.func_tool.database import DBFuncTool

        monkeypatch.chdir(tmp_path)
        connector = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "target.duckdb")))
        with connector.exclusive_connection() as con:
            con.execute("CREATE TABLE keepme (id BIGINT)")
            con.execute("INSERT INTO keepme VALUES (1)")
        return DBFuncTool(connector)

    def _write(self, tmp_path, name, assertions):
        path = tmp_path / name
        path.write_text(json.dumps({"assertions": assertions}), encoding="utf-8")
        return name

    @pytest.mark.acceptance
    def test_write_assertion_is_refused(self, tool, tmp_path):
        cfg = self._write(tmp_path, "bad.json", [{"name": "sneaky", "sql": "DROP TABLE IF EXISTS keepme"}])

        result = tool.check_datasource_quality(config_path=cfg)

        assert result.success == 0
        assert "read-only" in (result.error or "")
        assert "sneaky" in (result.error or "")
        with tool.connector.exclusive_connection() as con:
            assert con.execute("SELECT count(*) FROM keepme").fetchone()[0] == 1

    @pytest.mark.acceptance
    def test_select_assertion_runs(self, tool, tmp_path):
        cfg = self._write(
            tmp_path, "ok.json", [{"name": "rows exist", "expect": "nonzero", "sql": "SELECT count(*) FROM keepme"}]
        )

        result = tool.check_datasource_quality(config_path=cfg)

        assert result.success == 1
        assert result.result["summary"]["total"] > 0
        assert any(c["check"] == "rows exist" and c["status"] == "PASS" for c in result.result["checks"])

    @pytest.mark.acceptance
    def test_missing_config_is_reported(self, tool):
        result = tool.check_datasource_quality(config_path="nope.json")

        assert result.success == 0
        assert "not found" in (result.error or "").lower()
