"""Independent quality checks must reject plausible-looking but semantically broken data."""

import copy
import json

import duckdb
import pytest

from datus.tools.db_tools.datasource_quality import QualityChecker, summarize
from datus.tools.db_tools.datasource_semantics import catalog, check_semantic_metadata


@pytest.fixture
def sample():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE entities(id INTEGER, position INTEGER, planned_end DATE);
        CREATE TABLE events(id INTEGER, entity INTEGER, step INTEGER, occurred TIMESTAMP, finished TIMESTAMP, value DOUBLE);
        INSERT INTO entities SELECT i, CASE i%4 WHEN 0 THEN -40 WHEN 1 THEN 40 ELSE 0 END,
            DATE '2030-01-01' FROM range(20) r(i);
        INSERT INTO events
        SELECT d*20+i, i, d, DATE '2025-01-01' + d::INTEGER + INTERVAL '8 hours',
               DATE '2025-01-01' + d::INTEGER + INTERVAL '9 hours',
               100 * pow(i+1, -1.4) * (1+0.2*month(DATE '2025-01-01'+d::INTEGER))
               * CASE WHEN dayofweek(DATE '2025-01-01'+d::INTEGER) IN (0,6) THEN 1 ELSE 2 END
               * CASE WHEN DATE '2025-01-01'+d::INTEGER BETWEEN DATE '2025-03-08' AND DATE '2025-03-14' THEN 2 ELSE 1 END
        FROM range(181) days(d) CROSS JOIN range(20) entities(i);
        COMMENT ON TABLE entities IS 'Service consumers';
        COMMENT ON COLUMN entities.id IS 'Consumer identifier';
        COMMENT ON COLUMN entities.position IS 'Signed position offset';
        COMMENT ON COLUMN entities.planned_end IS 'Planned contract expiry';
        COMMENT ON TABLE events IS 'Daily service usage';
        COMMENT ON COLUMN events.id IS 'Event identifier';
        COMMENT ON COLUMN events.entity IS 'Consumer identifier';
        COMMENT ON COLUMN events.step IS 'Daily event order per consumer';
        COMMENT ON COLUMN events.occurred IS 'Actual start';
        COMMENT ON COLUMN events.finished IS 'Actual completion';
        COMMENT ON COLUMN events.value IS 'Usage units';
    """)
    metric = {"aggregate": "sum", "column": "value", "unit": "usage units"}
    metadata = {
        "quality_contract_version": 1,
        "schema": catalog(con),
        "start_date": "2025-01-01",
        "end_date": "2025-06-30",
        "min_rows": 3600,
        "max_rows": 4000,
        "semantics": {
            "quality_contract_version": 1,
            "tables": {
                "entities": {
                    "role": "dimension",
                    "grain": "one consumer",
                    "logical_key": ["id"],
                    "dates": {"planned_end": "planned"},
                },
                "events": {
                    "role": "event",
                    "grain": "one consumer-day",
                    "logical_key": ["id"],
                    "dates": {"occurred": "actual", "finished": "actual"},
                },
            },
            "relationships": [
                {"table": "events", "columns": ["entity"], "parent": "entities", "parent_columns": ["id"]}
            ],
            "series": [
                {
                    "name": "usage",
                    "table": "events",
                    "date_column": "occurred",
                    "measure": metric,
                    "monthly": "growth",
                    "weekly": "weekday_heavy",
                    "reason": "Expanding B2B usage",
                }
            ],
            "distributions": [
                {
                    "name": "consumer usage",
                    "table": "events",
                    "entity": ["entity"],
                    "measure": metric,
                    "shape": "long_tail",
                    "reason": "Different consumer sizes",
                }
            ],
            "sequences": [
                {
                    "table": "events",
                    "partition_by": ["entity"],
                    "order_by": ["step"],
                    "timestamp": "occurred",
                    "end_timestamp": "finished",
                }
            ],
            "anomalies": [
                {
                    "name": "launch",
                    "series": "usage",
                    "start": "2025-03-08",
                    "end": "2025-03-14",
                    "direction": "up",
                    "explanation": "A declared launch increases usage",
                }
            ],
        },
    }
    yield con, metadata
    con.close()


def verdicts(con, metadata):
    return {c["check"]: c["status"] for c in check_semantic_metadata(con, metadata)}


def test_real_patterns_and_signed_coordinates_pass(sample):
    con, metadata = sample
    assert con.execute("SELECT sum(position), count(DISTINCT position) FROM entities").fetchone() == (0, 3)
    assert set(verdicts(con, metadata).values()) == {"PASS"}


def test_flat_data_cannot_pass_its_own_assertions(sample):
    con, metadata = sample
    con.execute("UPDATE events SET value=10")
    config = {
        "skip": ["usage.monthly", "usage.weekly", "consumer usage.distribution", "launch.anomaly"],
        "assertions": [{"name": "always passes", "sql": "SELECT 0", "expect": "zero"}],
    }
    result = QualityChecker(con, config, metadata).run()
    statuses = {r["check"]: r["status"] for r in result}
    assert not summarize(result)["ok"]
    assert all(statuses[k] == "FAIL" for k in config["skip"])


@pytest.mark.parametrize(
    "sql, check",
    [
        ("UPDATE events SET entity=999 WHERE id=0", "events.relationship(entity)"),
        ("UPDATE events SET entity=NULL WHERE id=0", "events.relationship(entity)"),
        ("UPDATE entities SET id=1 WHERE id=0", "entities.logical_key"),
        (
            "UPDATE events SET occurred=TIMESTAMP '2025-01-03 08:00:00', finished=TIMESTAMP '2025-01-03 09:00:00' WHERE id=0",
            "events.sequence",
        ),
        ("COMMENT ON COLUMN events.value IS NULL", "events.comments"),
        ("COMMENT ON TABLE events IS NULL", "events.comments"),
        ("UPDATE entities SET planned_end=DATE '2030-01-01'", "entities.planned_end.actual_date"),
    ],
)
def test_detects_independent_contract_violations(sample, sql, check):
    con, metadata = sample
    if check.endswith("actual_date"):
        metadata["semantics"]["tables"]["entities"]["dates"]["planned_end"] = "actual"
    con.execute(sql)
    assert verdicts(con, metadata)[check] == "FAIL"


@pytest.mark.parametrize("category", ["series", "distributions", "sequences", "relationships", "anomalies"])
def test_omitted_coverage_fails_closed(sample, category):
    con, metadata = sample
    metadata["semantics"][category] = []
    assert verdicts(con, metadata)["semantic contract executable"] == "FAIL"


def test_exception_is_reported_as_not_assessed(sample):
    con, metadata = sample
    metadata["semantics"]["anomalies"] = []
    metadata["semantics"]["not_applicable"] = {"anomalies": "User explicitly requests steady-state baseline data"}
    result = check_semantic_metadata(con, metadata)
    assert verdicts(con, metadata)["coverage.anomalies"] == "WARN"
    assert "Not assessed" in next(c["detail"] for c in result if c["check"] == "coverage.anomalies")


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_column",
        "invalid_aggregation",
        "version",
        "null_measure",
        "nonfinite",
        "missing_date_role",
        "duplicate_name",
        "empty_series",
    ],
)
def test_invalid_contract_or_unmeasurable_data_never_passes(sample, mutation):
    con, metadata = sample
    series = metadata["semantics"]["series"][0]
    if mutation == "unknown_column":
        series["measure"]["column"] = "missing"
    elif mutation == "invalid_aggregation":
        series["measure"]["aggregate"] = "arbitrary SQL"
    elif mutation == "version":
        metadata["quality_contract_version"] = 2
    elif mutation == "null_measure":
        con.execute("UPDATE events SET value=NULL WHERE id=0")
    elif mutation == "nonfinite":
        con.execute("UPDATE events SET value='NaN'::DOUBLE WHERE id=0")
    elif mutation == "missing_date_role":
        del metadata["semantics"]["tables"]["events"]["dates"]["finished"]
    elif mutation == "duplicate_name":
        metadata["semantics"]["series"].append(copy.deepcopy(series))
    else:
        metadata["start_date"], metadata["end_date"] = "2024-01-01", "2024-06-30"
    assert not summarize(check_semantic_metadata(con, metadata))["ok"]


def test_declared_stable_always_on_business_is_allowed(sample):
    con, metadata = sample
    con.execute("UPDATE events SET value=100*pow(entity+1,-1.4)")
    metadata["semantics"]["series"][0].update(monthly="stable", weekly="flat", reason="Always-on stable workload")
    result = verdicts(con, metadata)
    assert result["usage.monthly"] == result["usage.weekly"] == "PASS"
    assert result["launch.anomaly"] == "FAIL"  # stable declaration cannot excuse an absent promised anomaly


def test_post_import_gate_catches_lost_comments_and_schema(sample, tmp_path):
    from datus.tools.db_tools.database_import import import_duckdb_file

    # Exercise the actual importer, then verify its output using the same quality entry point.
    con, metadata = sample
    source = tmp_path / "source.duckdb"
    con.execute(f"ATTACH '{source}' AS source")
    con.execute("COPY FROM DATABASE memory TO source")
    con.execute("DETACH source")
    with duckdb.connect() as target:
        import_duckdb_file(target, source, mode="replace", keep_constraints=True)
        assert summarize(QualityChecker(target, meta=metadata).run())["ok"]
        target.execute("COMMENT ON COLUMN events.value IS NULL")
        assert not summarize(QualityChecker(target, meta=metadata).run())["ok"]
        target.execute("ALTER TABLE events ADD COLUMN unexpected INTEGER")
        assert verdicts(target, metadata)["events.schema"] == "FAIL"


def test_composite_nullable_logical_relationship(sample):
    con, metadata = sample
    con.execute("UPDATE events SET entity=NULL WHERE id=0")
    rel = metadata["semantics"]["relationships"][0]
    rel.update(columns=["entity", "entity"], parent_columns=["id", "id"], nullable=True)
    # Repeated column tuples are a declaration error, not silently accepted as a composite key.
    assert verdicts(con, metadata)["semantic contract executable"] == "FAIL"
    rel.update(columns=["entity"], parent_columns=["id"])
    assert verdicts(con, metadata)["events.relationship(entity)"] == "PASS"


def test_composite_orphans_are_checked_as_tuples_without_physical_keys():
    with duckdb.connect() as con:
        con.execute("""
            CREATE TABLE parents(a INTEGER,b INTEGER);
            CREATE TABLE children(id INTEGER,a INTEGER,b INTEGER);
            INSERT INTO parents VALUES(1,1),(2,2);
            INSERT INTO children VALUES(1,1,2);
        """)
        for table in ("parents", "children"):
            con.execute(f"COMMENT ON TABLE {table} IS 'Tuple-grain reference data'")
            for col, *_ in con.execute(f"DESCRIBE {table}").fetchall():
                con.execute(f"COMMENT ON COLUMN {table}.{col} IS 'Declared tuple component'")
        semantics = {
            "quality_contract_version": 1,
            "tables": {
                "parents": {"role": "dimension", "grain": "one tuple", "logical_key": ["a", "b"]},
                "children": {"role": "dimension", "grain": "one child", "logical_key": ["id"]},
            },
            "relationships": [
                {"table": "children", "columns": ["a", "b"], "parent": "parents", "parent_columns": ["a", "b"]}
            ],
            "not_applicable": {
                k: "Static reference graph" for k in ("series", "distributions", "sequences", "anomalies")
            },
        }
        meta = {
            "quality_contract_version": 1,
            "schema": catalog(con),
            "start_date": "2025-01-01",
            "end_date": "2025-06-30",
            "min_rows": 3,
            "max_rows": 3,
            "semantics": semantics,
        }
        assert con.execute("SELECT count(*) FROM duckdb_constraints()").fetchone()[0] == 0
        assert verdicts(con, meta)["children.relationship(a,b)"] == "FAIL"
        con.execute("UPDATE children SET b=1")
        assert summarize(check_semantic_metadata(con, meta))["ok"]


def test_scope_observes_the_declared_cohort(sample):
    con, metadata = sample
    metadata["semantics"]["series"][0]["scope"] = {"entity": [0]}
    con.execute("UPDATE events SET value=1 WHERE entity<>0")
    statuses = verdicts(con, metadata)
    assert statuses["launch.anomaly"] == "PASS"
    metadata["semantics"]["series"][0]["scope"] = {"entity": [1]}
    assert verdicts(con, metadata)["launch.anomaly"] == "FAIL"


def test_minimum_root_rows_cannot_be_spent_on_leaf_tables(sample):
    con, metadata = sample
    metadata["semantics"]["tables"]["events"]["min_rows"] = 5000
    assert verdicts(con, metadata)["events.populated"] == "FAIL"


def test_small_balanced_dimension_does_not_have_an_impossible_head_cap(sample):
    con, metadata = sample
    con.execute("UPDATE events SET entity=entity%3, value=10")
    metadata["semantics"]["distributions"][0]["shape"] = "balanced"
    # Three groups necessarily have a head >= 33.3%; balance must not require <30%.
    assert verdicts(con, metadata)["consumer usage.distribution"] == "PASS"
    metadata["semantics"]["distributions"][0]["shape"] = "long_tail"
    assert verdicts(con, metadata)["consumer usage.distribution"] == "FAIL"


def test_actual_tool_finds_sidecar_and_rechecks_imported_database(sample, tmp_path, monkeypatch):
    from datus.tools.db_tools.config import DuckDBConfig
    from datus.tools.db_tools.duckdb_connector import DuckdbConnector
    from datus.tools.func_tool.database import DBFuncTool

    con, metadata = sample
    monkeypatch.chdir(tmp_path)
    build = tmp_path / "data/_build"
    build.mkdir(parents=True)
    source = build / "datasource.duckdb"
    con.execute(f"ATTACH '{source}' AS source")
    con.execute("COPY FROM DATABASE memory TO source")
    con.execute("DETACH source")
    (build / ".datasource.meta.json").write_text(json.dumps(metadata))
    connector = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "target.duckdb")))
    try:
        tool = DBFuncTool(connector)
        imported = tool.import_database_file(path="data/_build/datasource.duckdb", keep_constraints=True)
        assert imported.success and not imported.result.get("degraded")
        result = tool.check_datasource_quality(meta_path="data/_build/.datasource.meta.json")
        assert result.success and result.result["summary"]["ok"]
        automatic = tool.check_datasource_quality()
        assert automatic.result["summary"] == result.result["summary"]
        with connector.exclusive_connection() as target:
            target.execute("UPDATE events SET entity=999 WHERE id=0")
        result = tool.check_datasource_quality(meta_path="data/_build/.datasource.meta.json")
        assert result.success and not result.result["summary"]["ok"]
    finally:
        connector.close()
