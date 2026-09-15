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
    def test_multi_statement_assertion_is_refused(self, tool, tmp_path):
        """Classifying the statement kind is not enough on its own.

        parse_sql_type looks at the FIRST statement, so `SELECT 1; DROP TABLE t` reads as a select
        while the driver runs both - the gate has to carry the multi-statement rule too.
        """
        cfg = self._write(tmp_path, "sneak.json", [{"name": "sneaky", "sql": "SELECT 1; DROP TABLE keepme"}])

        result = tool.check_datasource_quality(config_path=cfg)

        assert result.success == 0
        assert "Multi-statement" in (result.error or "")
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

    @pytest.mark.acceptance
    def test_multi_statement_assertion_is_refused_inside_the_checker(self, con):
        """Defense in depth: the tool gates this too, but the checker must not execute a write
        handed to it through any other caller."""
        config = {"assertions": [{"name": "sneaky", "sql": "SELECT 1; DROP TABLE dim_product"}]}

        results = QualityChecker(con, config).run()

        assert _status(results, "sneaky") == "FAIL"
        assert con.execute("SELECT count(*) FROM dim_product").fetchone()[0] > 0


class _FlakyConnection:
    """Wraps a real connection and fails one family of queries, the way an unsupported column
    type or an unaggregatable view does in a live datasource."""

    def __init__(self, con, failing_fragment):
        self._con = con
        self._fragment = failing_fragment

    def execute(self, sql, *args, **kwargs):
        if self._fragment in sql:
            raise RuntimeError("boom")
        return self._con.execute(sql, *args, **kwargs)


@pytest.mark.acceptance
def test_failed_queries_are_reported_not_silently_passed(con):
    """A failed probe returns [] and every caller reads that as "no violations".

    Without surfacing it, a schema the checker cannot address comes back all-PASS.
    """
    checker = QualityChecker(_FlakyConnection(con, "count(DISTINCT"))
    results = checker.run()

    assert checker.query_errors
    assert _status(results, "all checks could run") == "WARN"


@pytest.mark.acceptance
def test_clean_run_has_no_query_health_entry(con):
    results = QualityChecker(con).run()

    assert not _has(results, "all checks could run")


@pytest.mark.acceptance
@pytest.mark.parametrize("bad_sql", [123, ["SELECT 1"], {"a": 1}, "", "   ", None])
def test_non_string_assertion_sql_is_reported_not_raised(con, bad_sql):
    """validate_read_only_sql assumes a string; a number or a list raised out of the loop and
    took every remaining check with it."""
    results = QualityChecker(con, {"assertions": [{"name": "broken", "sql": bad_sql}]}).run()

    assert _status(results, "broken") == "FAIL"
    assert "non-empty string" in next(r["detail"] for r in results if r["check"] == "broken")
    assert _has(results, "layering"), "the rest of the run must still complete"


class TestQualityToolMalformedConfig:
    @pytest.fixture
    def tool(self, tmp_path, monkeypatch):
        from datus.tools.db_tools.config import DuckDBConfig
        from datus.tools.db_tools.duckdb_connector import DuckdbConnector
        from datus.tools.func_tool.database import DBFuncTool

        monkeypatch.chdir(tmp_path)
        connector = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "target.duckdb")))
        with connector.exclusive_connection() as con:
            con.execute("CREATE TABLE keepme (id BIGINT)")
        return DBFuncTool(connector)

    @pytest.mark.acceptance
    def test_non_string_sql_is_a_named_configuration_error(self, tool, tmp_path):
        """It used to reach a validator that assumes a string, and the caller saw an opaque
        "Quality check failed" instead of which assertion is wrong."""
        (tmp_path / "bad.json").write_text(
            json.dumps({"assertions": [{"name": "broken", "sql": 123}]}), encoding="utf-8"
        )

        result = tool.check_datasource_quality(config_path="bad.json")

        assert result.success == 0
        assert "broken" in (result.error or "")
        assert "non-empty string" in (result.error or "")


# ---------------------------------------------------------------------------
# The same database has to produce the same verdict
# ---------------------------------------------------------------------------


@pytest.fixture
def trend_con():
    """A fact table whose headline metric and whose weekday-noisiest metric are different columns."""
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE ods_sales (
            order_id BIGINT, stat_dt DATE,
            gmv_amt DECIMAL(18, 2), discount_amt DECIMAL(18, 2)
        );
        """
    )
    # gmv grows ~4x across the window; the discount column is flat-but-jittery, so whichever
    # column the weekday search happens to like decides the trend verdict if they are coupled.
    connection.execute(
        """
        INSERT INTO ods_sales
        SELECT i,
               DATE '2024-01-01' + INTERVAL (i % 600) DAY,
               100.0 + (i % 600) * 0.9,
               50.0 + (i % 7) * 11
        FROM range(1, 6000) t(i)
        """
    )
    yield connection
    connection.close()


@pytest.mark.acceptance
def test_the_trend_observes_the_headline_metric_not_the_weekday_winner(trend_con):
    """The trend verdict must not depend on which column wiggles most within a week.

    A production run measured the trend on a discount column, then on ad revenue, then on
    discounts again - PASS, FAIL, PASS on substantially the same database - and spent four rounds
    chasing the flip. `_main_daily` already ranks columns by magnitude with preferred names first,
    so the headline series is decided, not searched for.
    """
    results = QualityChecker(trend_con).run()

    trend = next(r["detail"] for r in results if r["check"] == "time trend")
    assert "gmv_amt" in trend, f"the trend must observe the headline metric, got: {trend}"


@pytest.mark.acceptance
def test_repeated_runs_agree(trend_con):
    """Two runs over one unchanged database must return identical verdicts."""
    first = QualityChecker(trend_con).run()
    second = QualityChecker(trend_con).run()

    assert [(r["check"], r["status"]) for r in first] == [(r["check"], r["status"]) for r in second]


@pytest.mark.acceptance
def test_the_weekday_check_still_searches_and_says_what_it_used(trend_con):
    """It is the one check for which hunting is the right behaviour - it measures exactly that."""
    results = QualityChecker(trend_con).run()

    weekday = next(r["detail"] for r in results if r["check"] == "weekday cycle")
    assert weekday.startswith("observing "), weekday


# ---------------------------------------------------------------------------
# A check that verified nothing is not a pass
# ---------------------------------------------------------------------------


@pytest.mark.acceptance
def test_no_foreign_keys_found_is_a_warning_not_a_pass(trend_con):
    """`ok` gates delivery, so a green summary over an unverified structural check is a false green.

    Uses the single-table fixture: the other one does infer a real path, and passing on a path it
    actually checked is correct.
    """
    results = QualityChecker(trend_con).run()

    fk = next(r for r in results if r["check"] == "foreign key integrity")
    assert fk["status"] == "WARN", fk
    assert "nothing was verified" in fk["detail"]


@pytest.mark.acceptance
def test_no_metadata_makes_the_key_check_warn(con):
    results = QualityChecker(con).run()

    pk = next(r for r in results if r["check"] == "primary key non-null and unique")
    assert pk["status"] == "WARN", pk
    assert "no key was verified" in pk["detail"]


@pytest.mark.acceptance
def test_a_close_trend_ratio_is_printed_to_two_decimals(trend_con):
    """`{:.1f}` printed 1.79 as "1.8x" and then failed it against a 1.8 floor."""
    results = QualityChecker(trend_con).run()

    trend = next(r for r in results if r["check"] == "time trend")
    ratio = float(trend["detail"].split("(")[1].split("x")[0])
    decimals = len(trend["detail"].split("(")[1].split("x")[0].split(".")[1])
    assert decimals == 1 if (ratio >= 2 or ratio < 1.5) else decimals == 2, trend["detail"]


class TestGeneratorMetaDiscovery:
    """The check has to find the generator metadata on its own.

    Without it the primary-key and foreign-key checks verify nothing and the role map changes,
    which changes which table the time checks observe. A production run got `ok: true` and then
    `FAIL` on the same database, the only difference being that the later call happened to pass
    `meta_path`. A check whose verdict depends on an optional argument is not reproducible.
    """

    @pytest.fixture
    def tool(self, tmp_path, monkeypatch):
        from datus.tools.db_tools.config import DuckDBConfig
        from datus.tools.db_tools.duckdb_connector import DuckdbConnector
        from datus.tools.func_tool.database import DBFuncTool

        monkeypatch.chdir(tmp_path)
        connector = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "target.duckdb")))
        with connector.exclusive_connection() as con:
            con.execute("CREATE TABLE orders (order_id BIGINT, stat_dt DATE, gmv_amt DECIMAL(18,2))")
            con.execute(
                "INSERT INTO orders SELECT i, DATE '2024-01-01' + INTERVAL (i % 600) DAY, 10.0 + i "
                "FROM range(1, 2000) t(i)"
            )
        return DBFuncTool(connector), tmp_path

    def _write_meta(self, root, where):
        target = root / where
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"generator": "gen-datasource/ddl_engine", "pks": {"orders": "order_id"}, "roles": {}}),
            encoding="utf-8",
        )

    @pytest.mark.parametrize("where", ["data/.datasource.meta.json", "data/_build/.datasource.meta.json"])
    def test_the_metadata_is_found_without_being_named(self, tool, where):
        db_tool, root = tool
        self._write_meta(root, where)

        result = db_tool.check_datasource_quality()

        assert result.success == 1
        assert result.result["generator_meta"] == where
        pk = next(r for r in result.result["checks"] if r["check"] == "primary key non-null and unique")
        assert pk["status"] == "PASS", pk

    def test_an_absent_file_is_reported_rather_than_hidden(self, tool):
        db_tool, _root = tool

        result = db_tool.check_datasource_quality()

        assert result.success == 1
        assert "not found" in result.result["generator_meta"]
        pk = next(r for r in result.result["checks"] if r["check"] == "primary key non-null and unique")
        assert pk["status"] == "WARN", "an unverified key check must not read as a pass"

    def test_naming_the_path_and_leaving_it_out_agree(self, tool):
        """The whole point: both call shapes must produce the same verdicts."""
        db_tool, root = tool
        self._write_meta(root, "data/.datasource.meta.json")

        found = db_tool.check_datasource_quality()
        named = db_tool.check_datasource_quality(meta_path="data/.datasource.meta.json")

        assert [(r["check"], r["status"]) for r in found.result["checks"]] == [
            (r["check"], r["status"]) for r in named.result["checks"]
        ]
