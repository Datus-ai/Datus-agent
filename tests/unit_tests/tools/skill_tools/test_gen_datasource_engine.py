# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Contract tests for the gen-datasource engine that ships in the skill bundle.

The engine is the skill's payload rather than an importable package, so nothing else in the test
suite exercises it. These pin the parts the skill's instructions promise: the profile overrides an
LLM is told to write, the DDL it is told to normalise, and the two column classes that used to be
filled with nothing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[4] / "datus" / "resources" / "skills" / "gen-datasource"


@pytest.fixture(scope="module")
def engine_module():
    scripts = str(SKILL_DIR / "scripts")
    sys.path.insert(0, scripts)  # ddl_engine imports genlib from its own directory
    try:
        spec = importlib.util.spec_from_file_location("_gd_engine", SKILL_DIR / "scripts" / "ddl_engine.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
    return module


HOSPITAL_DDL = """
CREATE TABLE patients (
    patient_id BIGINT PRIMARY KEY,
    patient_name VARCHAR,
    registered_at TIMESTAMP
);
CREATE TABLE encounters (
    encounter_id BIGINT PRIMARY KEY,
    patient_id BIGINT REFERENCES patients(patient_id),
    admit_time TIMESTAMP,
    encounter_status VARCHAR, -- open / in_treatment / discharged
    total_charge DECIMAL(18, 2),
    insurance_paid DECIMAL(18, 2),
    self_paid DECIMAL(18, 2)
);
"""


def _sem(engine, table, column):
    return next(c["sem"] for c in engine.schema[table] if c["name"] == column)


# --------------------------------------------------------------------------- column semantics


@pytest.mark.acceptance
def test_retail_naming_is_recognised_without_help(engine_module):
    """The regex default has to stay useful, or every run pays to restate the obvious."""
    eng = engine_module.DDLEngine(HOSPITAL_DDL, rows=2000)

    assert _sem(eng, "encounters", "total_charge") == "amount"
    assert _sem(eng, "encounters", "admit_time") == "ts"
    assert _sem(eng, "encounters", "encounter_status") == "enum"


@pytest.mark.acceptance
def test_non_retail_money_columns_need_an_override(engine_module):
    """`insurance_paid` is money, but no naming convention in the table says so.

    This is the gap profile["semantics"] exists to close - pinned so a future rule change that
    happens to cover it does not quietly remove the need for the override contract.
    """
    eng = engine_module.DDLEngine(HOSPITAL_DDL, rows=2000)

    assert _sem(eng, "encounters", "insurance_paid") != "amount"


@pytest.mark.acceptance
def test_semantics_override_is_applied(engine_module):
    eng = engine_module.DDLEngine(
        HOSPITAL_DDL,
        rows=2000,
        profile={"semantics": {"encounters.insurance_paid": "amount", "encounters.self_paid": "amount"}},
    )

    assert _sem(eng, "encounters", "insurance_paid") == "amount"
    assert _sem(eng, "encounters", "self_paid") == "amount"


@pytest.mark.acceptance
def test_semantics_override_survives_inference(engine_module):
    """Inference re-tags declared keys as ids; an explicit override must outrank that."""
    eng = engine_module.DDLEngine(HOSPITAL_DDL, rows=2000, profile={"semantics": {"encounters.encounter_id": "seq"}})

    assert _sem(eng, "encounters", "encounter_id") == "seq"


@pytest.mark.acceptance
def test_precheck_rejects_an_unknown_semantic(engine_module):
    eng = engine_module.DDLEngine(HOSPITAL_DDL, rows=2000, profile={"semantics": {"encounters.total_charge": "money"}})

    with pytest.raises(ValueError, match="not a semantic"):
        eng.precheck()


@pytest.mark.acceptance
def test_precheck_rejects_a_missing_column(engine_module):
    """A typo must fail loudly: left alone the column keeps its wrong guess and the data looks fine."""
    eng = engine_module.DDLEngine(
        HOSPITAL_DDL, rows=2000, profile={"semantics": {"encounters.insurance_pad": "amount"}}
    )

    with pytest.raises(ValueError, match="has no column"):
        eng.precheck()


# --------------------------------------------------------------------------- generation


@pytest.mark.acceptance
def test_measure_columns_on_a_fact_table_are_filled(engine_module, tmp_path):
    """They used to fall through to setdefault("") and the whole column landed NULL."""
    import duckdb

    out = tmp_path / "hosp.duckdb"
    engine_module.DDLEngine(HOSPITAL_DDL, rows=4000).generate(str(out), verbose=False)

    con = duckdb.connect(str(out), read_only=True)
    total, filled = con.execute("SELECT count(*), count(insurance_paid) FROM encounters").fetchone()
    con.close()

    assert total > 0
    assert filled == total, "a column the engine has no dedicated branch for must still be populated"


@pytest.mark.acceptance
def test_declared_constraints_survive_into_the_database(engine_module, tmp_path):
    import duckdb

    out = tmp_path / "hosp.duckdb"
    engine_module.DDLEngine(HOSPITAL_DDL, rows=4000).generate(str(out), verbose=False)

    con = duckdb.connect(str(out), read_only=True)
    found = {(t, k) for t, k in con.execute("SELECT table_name, constraint_type FROM duckdb_constraints()").fetchall()}
    con.close()

    assert ("patients", "PRIMARY KEY") in found
    assert ("encounters", "FOREIGN KEY") in found


# --------------------------------------------------------------------------- DDL normalisation


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "dialect,ddl",
    [
        (
            "mysql",
            "CREATE TABLE `orders` (`order_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, PRIMARY KEY (`order_id`)) ENGINE=InnoDB;",
        ),
        ("postgres", "CREATE TABLE public.orders (order_id BIGSERIAL PRIMARY KEY, meta JSONB);"),
        ("oracle", "CREATE TABLE orders (order_id NUMBER(19) PRIMARY KEY, note VARCHAR2(200));"),
    ],
)
def test_foreign_dialect_fails_with_an_actionable_message(engine_module, dialect, ddl):
    """The engine only speaks DuckDB. A bare parser error left the agent guessing, so the message
    has to name the fix - most users paste DDL exported from their real warehouse."""
    with pytest.raises(ValueError, match="rewrite it to DuckDB"):
        engine_module.DDLEngine(ddl, rows=2000)


@pytest.mark.acceptance
def test_duckdb_dialect_parses(engine_module):
    eng = engine_module.DDLEngine(
        'CREATE TABLE orders ("order_id" BIGINT PRIMARY KEY, paid_amt DECIMAL(18,2));', rows=2000
    )

    assert list(eng.schema) == ["orders"]


# --------------------------------------------------------------------------- enum comments


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "comment,expected",
    [
        ("-- pending / paid / shipped", ["pending", "paid", "shipped"]),
        ("-- order status: pending / paid / shipped", ["pending", "paid", "shipped"]),
        ("-- pending | paid | shipped", ["pending", "paid", "shipped"]),
        ("-- pending / paid / ...", ["pending", "paid"]),
        ("-- the current state of the order", None),
        ("-- pending, paid, shipped", None),
    ],
)
def test_enum_domain_extraction_contract(engine_module, comment, expected):
    """The skill documents this shape so the DDL-normalisation step can target it; the two
    rejected forms are documented as rejected, so they are pinned too."""
    # One column per line: the extractor reads the first identifier of the line, so a comment must
    # sit on the line of the column it describes.
    ddl = f"CREATE TABLE t (\n  id BIGINT PRIMARY KEY,\n  st VARCHAR, {comment}\n  amt DECIMAL(18,2)\n);"
    eng = engine_module.DDLEngine(ddl, rows=1000)

    assert eng.ddl_enums.get("st") == expected


@pytest.mark.acceptance
def test_comment_clause_carries_no_domain(engine_module):
    """MySQL/StarRocks COMMENT 'x' is not a line comment - the skill tells the agent to convert it,
    and this is why."""
    ddl = "CREATE TABLE t (\n  id BIGINT PRIMARY KEY,\n  st VARCHAR,\n  amt DECIMAL(18,2)\n);"
    eng = engine_module.DDLEngine(ddl, rows=1000)

    assert "st" not in eng.ddl_enums


EVENT_DDL = """
CREATE TABLE visits (
    visit_id BIGINT PRIMARY KEY,
    admit_time TIMESTAMP,
    total_charge DECIMAL(18, 2)
);
CREATE TABLE bills (
    bill_id BIGINT PRIMARY KEY,
    visit_id BIGINT REFERENCES visits(visit_id),
    bill_dt DATE,
    bill_amt DECIMAL(18, 2)
);
CREATE TABLE visit_events (
    event_id BIGINT PRIMARY KEY,
    visit_id BIGINT REFERENCES visits(visit_id),
    event_seq INTEGER,
    event_time TIMESTAMP,
    event_type VARCHAR
);
"""


@pytest.mark.acceptance
def test_event_and_downstream_keys_honour_the_declared_type(engine_module, tmp_path):
    """They emitted a prefixed string regardless of the declared type, so a BIGINT key was
    TRY_CAST to NULL and the table then lost every constraint."""
    import duckdb

    out = tmp_path / "ev.duckdb"
    result = engine_module.DDLEngine(EVENT_DDL, rows=20_000).generate(str(out), verbose=False)

    assert result["degraded"] == [], "no table should have to drop its constraints"
    con = duckdb.connect(str(out), read_only=True)
    for table, key in (("visit_events", "event_id"), ("bills", "bill_id")):
        total, filled = con.execute(f"SELECT count(*), count({key}) FROM {table}").fetchone()
        assert total > 0 and filled == total, f"{table}.{key} must be populated"
    con.close()


@pytest.mark.acceptance
def test_event_times_never_pass_the_cut_off(engine_module, tmp_path):
    """The monotonic backstop used to add a second past the cap when the window was used up,
    which produced future-dated rows - a defect the quality check fails on."""
    import duckdb

    out = tmp_path / "ev.duckdb"
    engine_module.DDLEngine(EVENT_DDL, rows=20_000).generate(str(out), verbose=False)

    con = duckdb.connect(str(out), read_only=True)
    future = con.execute("SELECT count(*) FROM visit_events WHERE event_time > now()").fetchone()[0]
    backwards = con.execute(
        "SELECT count(*) FROM (SELECT event_time t, "
        "lag(event_time) OVER (PARTITION BY visit_id ORDER BY event_seq) p FROM visit_events) "
        "WHERE p IS NOT NULL AND t < p"
    ).fetchone()[0]
    con.close()

    assert future == 0
    assert backwards == 0


@pytest.mark.acceptance
def test_event_chain_never_steps_past_the_cap(engine_module):
    """Directly: a chain started beyond its cap has no room, and the cap is the hard invariant."""
    import datetime
    import random
    import sys

    scripts = str(SKILL_DIR / "scripts")
    sys.path.insert(0, scripts)
    try:
        from genlib import EventChain
    finally:
        sys.path.remove(scripts)

    cap = datetime.datetime(2026, 9, 14, 20, 0, 0)
    start = datetime.datetime(2026, 9, 14, 23, 30, 0)
    chain = EventChain(random.Random(1), start)

    stamps = [chain.step(avg_hours=6, cap=cap) for _ in range(4)]

    assert all(t <= start for t in stamps), "the chain must not advance past its starting point"
    assert chain.exhausted is True
