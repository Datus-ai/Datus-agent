"""Observable schema/data contracts for the industry-independent generation runner."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import duckdb
import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "datus/resources/skills/gen-datasource-v2/scripts"
SPEC = importlib.util.spec_from_file_location("datasource_v2", SCRIPTS / "datasource.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

DDL = """
CREATE TABLE readings (
  device INTEGER, channel INTEGER, sample INTEGER NOT NULL, reading DOUBLE NOT NULL CHECK (reading >= 0),
  PRIMARY KEY(device,channel,sample), FOREIGN KEY(device,channel) REFERENCES channels(device,channel)
);
CREATE TABLE channels (device INTEGER, channel INTEGER, label VARCHAR NOT NULL, PRIMARY KEY(device,channel));
"""


def test_forward_composite_dependencies_and_constraints():
    schema = runner.read_schema(DDL)
    assert runner.dependency_order(schema) == ["channels", "readings"]
    with duckdb.connect() as con:
        runner.apply_ddl(con, DDL)
        con.execute("INSERT INTO channels VALUES (1,1,'a'),(1,2,'b')")
        con.execute("INSERT INTO readings VALUES (1,1,1,2),(1,2,1,3)")
        report = runner.validate(con, schema, [], 4, 4, require_assertions=False)
        assert report["ok"]
        with pytest.raises(duckdb.ConstraintException):
            con.execute("INSERT INTO readings VALUES (1,3,1,2)")
        with pytest.raises(duckdb.ConstraintException):
            con.execute("INSERT INTO readings VALUES (1,1,1,2)")


def test_audit_detects_stripped_constraints_and_composite_orphans():
    schema = runner.read_schema(DDL)
    with duckdb.connect() as con:
        con.execute("CREATE TABLE channels(device INTEGER,channel INTEGER,label VARCHAR)")
        con.execute("CREATE TABLE readings(device INTEGER,channel INTEGER,sample INTEGER,reading DOUBLE)")
        con.execute("INSERT INTO channels VALUES (1,1,'a'),(2,2,'b')")
        con.execute("INSERT INTO readings VALUES (1,2,1,3),(1,2,1,3)")
        report = runner.validate(con, schema, [], 4, 4, require_assertions=False)
    failures = {c["name"]: c for c in report["checks"] if not c["ok"]}
    assert failures["readings.FOREIGN KEY(device,channel)"]["actual"] == 2
    assert failures["readings.PRIMARY KEY(device,channel,sample)"]["actual"] == 1
    assert "readings.constraints" in failures


@pytest.mark.parametrize(
    "sql", ["SELECT NULL", "SELECT 'NaN'::DOUBLE", "SELECT 1 WHERE false", "SELECT 1 UNION ALL SELECT 2"]
)
def test_invalid_scalar_cannot_pass(sql):
    with duckdb.connect() as con:
        con.execute("CREATE TABLE t(i INTEGER); INSERT INTO t VALUES(1)")
        report = runner.validate(con, runner.catalog(con), [{"name": "bound", "sql": sql, "expect": "zero"}], 1, 1)
    assert not report["ok"]
    assert not report["checks"][-1]["ok"]


def test_null_component_in_optional_fk_uses_sql_match_simple():
    with duckdb.connect() as con:
        con.execute("CREATE TABLE p(a INTEGER,b INTEGER,PRIMARY KEY(a,b))")
        con.execute("CREATE TABLE c(a INTEGER,b INTEGER,FOREIGN KEY(a,b) REFERENCES p(a,b))")
        con.execute("INSERT INTO p VALUES(1,1); INSERT INTO c VALUES(9,NULL)")
        assert runner.validate(con, runner.catalog(con), [], 2, 2, False)["ok"]


def test_repeatable_generation_and_failure_is_not_success(tmp_path):
    ddl = tmp_path / "input.sql"
    ddl.write_text("CREATE TABLE things(id INTEGER PRIMARY KEY,value DOUBLE NOT NULL)")
    out = tmp_path / "data"
    runner.initialize(
        Namespace(ddl=ddl, directory=out, rows=20, min_rows=20, max_rows=20, months=2, seed=17, end_date="2026-09-21")
    )
    semantics = {
        "quality_contract_version": 1,
        "tables": {"things": {"role": "dimension", "grain": "one thing", "logical_key": ["id"]}},
        "not_applicable": {
            k: "Static independent reference catalog"
            for k in ("relationships", "series", "distributions", "sequences", "anomalies")
        },
    }
    (out / "semantics.json").write_text(json.dumps(semantics))
    (out / "generate.sql").write_text("""
        INSERT INTO things SELECT i,u01(i,1) FROM range(20) r(i);
        COMMENT ON TABLE things IS 'Reference catalog';
        COMMENT ON COLUMN things.id IS 'Logical item identifier';
        COMMENT ON COLUMN things.value IS 'Deterministic catalog value';
    """)
    (out / "checks.json").write_text(
        json.dumps(
            {
                "assertions": [
                    {
                        "name": "range",
                        "expect": "zero",
                        "sql": "SELECT count(*) FROM things WHERE value < 0 OR value >= 1",
                    }
                ]
            }
        )
    )
    assert runner.run(out) == 0
    with duckdb.connect(str(out / "_build/datasource.duckdb"), read_only=True) as con:
        first = con.execute("SELECT * FROM things ORDER BY id").fetchall()
    assert runner.run(out) == 0
    with duckdb.connect(str(out / "_build/datasource.duckdb"), read_only=True) as con:
        assert con.execute("SELECT * FROM things ORDER BY id").fetchall() == first
    (out / "checks.json").write_text(
        json.dumps({"assertions": [{"name": "impossible", "sql": "SELECT 1", "expect": "zero"}]})
    )
    assert runner.run(out) == 1
    with duckdb.connect(str(out / "_build/datasource.duckdb"), read_only=True) as con:
        assert con.execute("SELECT * FROM things ORDER BY id").fetchall() == first
    semantics["tables"]["things"]["grain"] = "a corrected grain"
    (out / "semantics.json").write_text(json.dumps(semantics))
    assert runner.run(out) == 1
    assert "contract-change-reason" in json.loads((out / "quality.json").read_text())["error"]
    assert runner.run(out, "Correct the description, retaining the same key") == 1  # assertion still fails
    assert len(json.loads((out / "semantic-changes.json").read_text())) == 1
    (out / "generate.sql").write_text("INSERT INTO things VALUES (1,NULL)")
    assert runner.run(out) == 1
    assert not json.loads((out / "quality.json").read_text())["ok"]
    (out / "quality.json").write_text('{"ok":true}')
    (out / "checks.json").unlink()
    assert runner.run(out) == 1
    assert not json.loads((out / "quality.json").read_text())["ok"]


def test_ddl_statement_parser_handles_semicolon_in_literal():
    schema = runner.read_schema("CREATE TABLE t(i INTEGER, s VARCHAR DEFAULT 'a;b');")
    assert list(schema) == ["t"]


def test_comments_in_original_ddl():
    schema = runner.read_schema("CREATE TABLE t(i INTEGER); COMMENT ON TABLE t IS 'Source data';")
    assert list(schema) == ["t"]
