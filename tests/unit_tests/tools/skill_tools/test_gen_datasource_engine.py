# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Contract tests for the gen-datasource engine that ships in the skill bundle.

The engine is the skill's payload rather than an importable package, so nothing else in the test
suite exercises it. These pin the parts the skill's instructions promise: the profile overrides an
LLM is told to write, the DDL it is told to normalise, and the two column classes that used to be
filled with nothing.
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import duckdb
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
def test_a_forward_foreign_key_is_not_a_dialect_problem(engine_module):
    """A DDL that names the fact table first and its dimensions after it is the ordinary shape,
    and DuckDB resolves a REFERENCES target at CREATE time - so the first pass fails on a target
    that appears further down. Reject it and a valid schema comes back as "rewrite the dialect",
    which sends the reader to edit syntax that was never wrong: the same file parsed once the
    statements were reordered."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER REFERENCES customers(id));"
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name VARCHAR);",
        rows=2000,
    )

    assert sorted(eng.schema) == ["customers", "orders"]


@pytest.mark.acceptance
def test_a_reference_chain_resolves_however_deep_it_is(engine_module):
    """One retry pass only resolves a reference ONE level deep. With `a -> b -> c` declared in that
    order, the retry meets `a` again before `b` exists - and the schema is valid. Worse than
    failing, it failed as "no CREATE in this DDL declares that table", sending the reader to hunt a
    typo in a name that is three lines further down. Passes now repeat while any statement lands."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES b(id));"
        "CREATE TABLE b (id INTEGER PRIMARY KEY, c_id INTEGER REFERENCES c(id));"
        "CREATE TABLE c (id INTEGER PRIMARY KEY, name VARCHAR);",
        rows=2000,
    )

    assert sorted(eng.schema) == ["a", "b", "c"]
    assert eng.decl_fk, "the chain must survive as declared foreign keys, not just parse"


@pytest.mark.acceptance
def test_a_reference_to_a_table_nothing_declares_says_so(engine_module):
    """The failure that survives the retry pass. It needs the OPPOSITE fix from a dialect error,
    so it must not borrow that message - and it has to say that ordering is not the cause, or the
    reader's next move is to shuffle statements that are already fine."""
    with pytest.raises(ValueError, match="no CREATE in this DDL declares") as excinfo:
        engine_module.DDLEngine(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER REFERENCES custmers(id));",
            rows=2000,
        )

    assert "rewrite it to DuckDB" not in str(excinfo.value)
    assert "Declaration ORDER is not the problem" in str(excinfo.value)


@pytest.mark.acceptance
def test_the_parse_failure_does_not_hand_back_a_traceback_into_the_engine(engine_module):
    """``from None`` on that raise. With the chain attached, Python prints both DuckDB
    ParserExceptions - each headed by a path into ddl_engine.py - ABOVE the sentence that says
    what to do, and a `| tail -40` then cuts the sentence off. One measured run followed that
    path into the engine's parser and spent the rest of its budget there."""
    with pytest.raises(ValueError) as excinfo:
        engine_module.DDLEngine("CREATE TABLE t (id INT AUTO_INCREMENT) ENGINE=InnoDB;", rows=2000)

    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None or excinfo.value.__suppress_context__


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

    assert all(t <= cap for t in stamps), "the cap is the hard invariant, including on the first step"
    assert stamps[0] == cap, "a start beyond the cap is clamped to it"
    assert chain.exhausted is True


@pytest.mark.acceptance
def test_event_chain_is_unaffected_within_the_cap(engine_module):
    """The clamp must not disturb a chain that fits: still strictly increasing, still capped."""
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
    chain = EventChain(random.Random(1), datetime.datetime(2026, 9, 10, 8, 0, 0))

    stamps = [chain.step(avg_hours=6, cap=cap) for _ in range(5)]

    assert stamps == sorted(stamps)
    assert all(t <= cap for t in stamps)
    assert chain.exhausted is False


@pytest.mark.acceptance
def test_report_prints_the_plan_not_just_the_shape(engine_module, capsys):
    """report() has to answer what the agent would otherwise grep the engine for.

    A measured run spent 66% of its wall clock reading ddl_engine.py through eight greps - VOCAB,
    _joint_plan, _code_val, ROLE_METRIC, _cond_pick, the constructor - trying to predict the output
    before paying for a generation pass. Each line below closes one of those.
    """
    ddl = """
    CREATE TABLE customers (
        customer_id BIGINT PRIMARY KEY,
        customer_name VARCHAR,
        city VARCHAR
    );
    CREATE TABLE orders (
        order_id BIGINT PRIMARY KEY,
        order_no VARCHAR UNIQUE,
        customer_id BIGINT REFERENCES customers(customer_id),
        order_time TIMESTAMP,
        channel VARCHAR,
        paid_amount DECIMAL(18, 2)
    );
    CREATE TABLE daily_channel_metrics (
        metric_id BIGINT PRIMARY KEY,
        metric_date DATE,
        channel VARCHAR,
        impressions BIGINT,
        clicks BIGINT
    );
    """
    profile = {
        "conditional": {"orders.paid_amount": {"__by__": "channel", "social": [10, 50], "__default__": [20, 200]}},
        "derive": {"daily_channel_metrics.clicks": {"from": "impressions", "ratio": (0.01, 0.03)}},
        "joint": {"customers": [{"cols": ["city"], "values": [["Shanghai", 5], ["Shenzhen", 3]]}]},
        # this cut-down schema is too small for the role to be inferred; the grid line is the point
        "roles": {"daily_channel_metrics": "metric_daily"},
    }
    engine_module.DDLEngine(ddl, rows=40_000, profile=profile).report()
    out = capsys.readouterr().out

    assert "seed=" in out and "extra_tables=" in out, "the constructor knobs"
    assert "name samples" in out and "customers.customer_name ->" in out, "what generated names look like"
    assert "orders.order_no" in out, "which columns get a generated business code"
    assert "dimension combos" in out, "how the daily metric grid is sized"
    assert "conditional orders.paid_amount by channel" in out, "which declarative rules resolved"
    assert "derive daily_channel_metrics.clicks from impressions" in out
    assert "joint customers(city) 2 combos" in out


@pytest.mark.acceptance
def test_report_flags_a_conditional_without_a_default(engine_module, capsys):
    """Silent fallback to the engine default is a common and invisible misconfiguration."""
    ddl = "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, channel VARCHAR, amt DECIMAL(18,2));"
    profile = {"conditional": {"orders.amt": {"__by__": "channel", "social": [10, 50]}}}

    engine_module.DDLEngine(ddl, rows=5_000, profile=profile).report()

    assert "NO default" in capsys.readouterr().out


METRIC_DDL = """
CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, amt DECIMAL(18, 2));
CREATE TABLE daily_channel_metrics (
    metric_id BIGINT PRIMARY KEY,
    metric_date DATE,
    channel VARCHAR, -- paid_search / social / organic
    impressions BIGINT,
    clicks BIGINT
);
"""


@pytest.mark.acceptance
def test_report_metric_grid_matches_what_is_generated(engine_module, tmp_path):
    """The row budget is a cap, not the count.

    With three channels and room in the budget for eight combinations, the grid is three wide.
    Reporting the budget would over-state both the combinations and the row count, which is the
    opposite of what report() is for.
    """
    eng = engine_module.DDLEngine(METRIC_DDL, rows=60_000, profile={"roles": {"daily_channel_metrics": "metric_daily"}})
    planned, _ = eng._metric_combos("daily_channel_metrics")
    assert len(planned) == 3, "three channels, whatever the budget allows"

    result = eng.generate(str(tmp_path / "m.duckdb"), verbose=False)

    assert result["tables"]["daily_channel_metrics"] == len(eng.days) * len(planned)


@pytest.mark.acceptance
def test_report_says_when_the_schema_limits_the_grid(engine_module, capsys):
    engine_module.DDLEngine(
        METRIC_DDL, rows=60_000, profile={"roles": {"daily_channel_metrics": "metric_daily"}}
    ).report()

    out = capsys.readouterr().out

    assert "3 dimension combos" in out
    assert "all the schema allows" in out, "say why the grid is smaller than the budget"


@pytest.mark.acceptance
def test_report_knob_line_separates_months_from_days(engine_module, capsys):
    """`months` is a constructor argument; the day count is what it works out to."""
    engine_module.DDLEngine(HOSPITAL_DDL, rows=2000, months=11).report()

    out = capsys.readouterr().out

    assert "months=11 (" in out
    assert "days," in out


CODE_DDL = """
CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    brand VARCHAR,
    product_name VARCHAR,
    sku_code VARCHAR
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    order_no VARCHAR,
    product_id BIGINT REFERENCES products(product_id),
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
def test_report_lists_only_the_columns_that_really_get_a_code(engine_module, tmp_path, capsys):
    """`CODE_COL` is the generators' last branch, not their rule.

    `sku_code` matches the pattern but inference classifies it as an enum, so the enum branch
    fills it and no code is ever generated. Reporting it as a business code sent a production run
    into the engine source to find out which line was lying.
    """
    eng = engine_module.DDLEngine(CODE_DDL, rows=3000, months=3, seed=1)
    eng.report()
    out = capsys.readouterr().out

    codes = next(ln for ln in out.splitlines() if ln.startswith("generated business codes:"))
    assert "orders.order_no" in codes
    assert "sku_code" not in codes, "an enum column is filled by the enum branch, never by a code"

    eng.generate(str(tmp_path / "c.duckdb"), verbose=False)
    import duckdb

    con = duckdb.connect(str(tmp_path / "c.duckdb"))
    try:
        assert con.execute("SELECT order_no FROM orders LIMIT 1").fetchone()[0].startswith("ORD")
        # State the property rather than a forbidden prefix: sku_code must keep drawing from the
        # finite enum domain inference gave it. A prefix test would pass for any wrong-but-not-
        # SKU0 filler, and would fail a legitimate domain that happens to contain "SKU0...".
        domain = set(eng._enum_values("products", "sku_code")[0])
        assert domain, "the column must have an inferred enum domain to be drawn from"
        generated = {row[0] for row in con.execute("SELECT DISTINCT sku_code FROM products").fetchall()}
        assert generated <= domain, f"values outside the inferred domain: {sorted(generated - domain)}"
    finally:
        con.close()


@pytest.mark.acceptance
def test_a_code_column_is_not_also_reported_as_unrecognised(engine_module, capsys):
    """Two adjacent report lines used to contradict each other about the same column."""
    engine_module.DDLEngine(CODE_DDL, rows=3000, months=3, seed=1).report()

    out = capsys.readouterr().out
    unrecognised = [ln for ln in out.splitlines() if "unrecognised" in ln]

    assert not any("order_no" in ln for ln in unrecognised), "it gets a code; the next line says so"


@pytest.mark.acceptance
def test_semantics_can_force_a_code_onto_an_enum_looking_column(engine_module, tmp_path):
    """The documented escape hatch: declare it text and the code fallback takes over."""
    eng = engine_module.DDLEngine(
        CODE_DDL, rows=3000, months=3, seed=1, profile={"semantics": {"products.sku_code": "text"}}
    )
    assert eng._is_code_col("products", "sku_code")

    eng.generate(str(tmp_path / "f.duckdb"), verbose=False)
    import duckdb

    con = duckdb.connect(str(tmp_path / "f.duckdb"))
    try:
        assert con.execute("SELECT sku_code FROM products LIMIT 1").fetchone()[0].startswith("SKU0")
    finally:
        con.close()


@pytest.mark.acceptance
def test_a_primary_key_matching_the_code_pattern_is_not_a_code_column(engine_module):
    eng = engine_module.DDLEngine(
        "CREATE TABLE tickets (ticket_no VARCHAR PRIMARY KEY, note VARCHAR, ref_no VARCHAR);",
        rows=500,
        months=3,
    )

    assert not eng._is_code_col("tickets", "ticket_no"), "the PK is filled by _pk_val"
    assert eng._is_code_col("tickets", "ref_no")


#: The three column names ``date_pk`` matches (``^(date_key|stat_dt|dt)$``). All three are
#: idiomatic for a daily metric table's date column, so all three have to be covered - testing
#: only ``stat_dt`` would leave two thirds of the rule unexercised.
DATE_GRAIN_NAMES = ("stat_dt", "dt", "date_key")


def _daily_metric_ddl(date_col="stat_dt"):
    return f"""
CREATE TABLE daily_channel_metrics (
    {date_col} DATE,
    channel VARCHAR,
    impressions BIGINT,
    clicks BIGINT,
    orders_cnt BIGINT,
    gmv DECIMAL(18, 2)
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
"""


DAILY_METRIC_DDL = _daily_metric_ddl()


@pytest.mark.acceptance
@pytest.mark.parametrize("date_col", DATE_GRAIN_NAMES)
def test_a_daily_metric_table_is_not_mistaken_for_a_date_dimension(engine_module, tmp_path, date_col):
    """`stat_dt` / `dt` / `date_key` are the idiomatic names for a daily metric table's date column.

    All three classify as `date_pk`, and the date-dimension branch matched on that alone - so the
    whole table went through the date-dimension generator, came out one row per day, and left every
    business column NULL. A headline metric table silently emptied is worse than a crash.
    """
    eng = engine_module.DDLEngine(_daily_metric_ddl(date_col), rows=20_000, months=6, seed=42)

    assert eng.roles["daily_channel_metrics"] == "metric_daily"
    assert eng.nrows["daily_channel_metrics"] > len(eng.days), "date x dimension, not one row per day"

    out = tmp_path / f"m_{date_col}.duckdb"
    eng.generate(str(out), verbose=False)
    import duckdb

    con = duckdb.connect(str(out))
    try:
        empty = con.execute(
            "SELECT count(*) FROM daily_channel_metrics "
            "WHERE channel IS NULL OR channel = '' OR gmv IS NULL OR impressions IS NULL"
        ).fetchone()[0]
        assert empty == 0, "every business column must carry a value"
        # The date column itself must still be a real, in-window date on every row.
        nulls, lo, hi = con.execute(
            f"SELECT count(*) - count({date_col}), min({date_col}), max({date_col}) FROM daily_channel_metrics"  # noqa: S608
        ).fetchone()
        assert nulls == 0
        assert eng.start <= lo and hi <= eng.end
    finally:
        con.close()


@pytest.mark.acceptance
def test_a_real_date_dimension_is_still_a_date_dimension(engine_module):
    """The other direction: a genuine calendar table has four measures and two enums of its own.

    Discriminating on the count of measures would demote it; discriminating on whether those
    columns are calendar attributes keeps it.
    """
    cols = ", ".join(f"{name} {dtype}" for name, dtype in engine_module.DATE_DIM_COLS)
    eng = engine_module.DDLEngine(
        f"CREATE TABLE dim_date ({cols});"
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, paid_amount DECIMAL(18,2));",
        rows=5000,
        months=3,
    )

    assert eng.roles["dim_date"] == "date_dim"
    assert eng.nrows["dim_date"] == len(eng.days), "exactly one row per day"


@pytest.mark.acceptance
def test_a_daily_summary_without_a_dimension_column_still_carries_measures(engine_module, tmp_path):
    """The same swallow happened with no enum column at all - date plus measures was enough."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE ads_daily_summary (dt DATE, gmv DECIMAL(18,2), orders_cnt BIGINT, uv BIGINT);"
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, paid_amount DECIMAL(18,2));",
        rows=8000,
        months=3,
        seed=1,
    )

    assert eng.roles["ads_daily_summary"] == "metric_daily"

    eng.generate(str(tmp_path / "s.duckdb"), verbose=False)
    import duckdb

    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    try:
        assert con.execute("SELECT count(*) FROM ads_daily_summary WHERE gmv IS NULL").fetchone()[0] == 0
    finally:
        con.close()


CODE_ON_CHILDREN_DDL = """
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    order_no VARCHAR,
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
CREATE TABLE order_items (
    item_id BIGINT PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id),
    line_no VARCHAR,
    quantity INTEGER,
    item_amount DECIMAL(18, 2)
);
CREATE TABLE order_events (
    event_id BIGINT PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id),
    event_seq INTEGER,
    event_type VARCHAR,
    event_time TIMESTAMP,
    trace_ref VARCHAR
);
"""


@pytest.mark.acceptance
def test_a_code_column_on_a_child_table_is_filled_not_left_null(engine_module, tmp_path):
    """Only `_gen_dim` and `_gen_fact` called `_code_val` directly.

    The detail, downstream, event and metric generators reach a leftover column through
    `_fill_generic`, which had no code branch - so `order_items.line_no` came out NULL on every
    row while `report()` listed it under "generated business codes".
    """
    eng = engine_module.DDLEngine(CODE_ON_CHILDREN_DDL, rows=9000, months=3, seed=1)
    assert eng.roles["order_items"] == "detail"
    assert eng.roles["order_events"] == "event"

    eng.generate(str(tmp_path / "k.duckdb"), verbose=False)
    import duckdb

    con = duckdb.connect(str(tmp_path / "k.duckdb"))
    try:
        for table, column in (("order_items", "line_no"), ("order_events", "trace_ref")):
            total, filled, distinct = con.execute(
                f"SELECT count(*), count({column}), count(DISTINCT {column}) FROM {table}"  # noqa: S608
            ).fetchone()
            assert total > 0
            assert filled == total, f"{table}.{column} left {total - filled} rows NULL"
            # These generators nest loops, so a code keyed on a loop index would repeat.
            assert distinct == total, f"{table}.{column} handed out duplicate codes"
    finally:
        con.close()


@pytest.mark.acceptance
def test_a_joint_column_is_never_reported_as_a_code(engine_module):
    """A joint group is written into the row before the per-column chain runs.

    The membership test therefore has to come before the `id` branch, not after it: an id column
    inside a joint group is claimed by the group whatever its semantic says.
    """
    ddl = (
        "CREATE TABLE sites (site_id BIGINT PRIMARY KEY, region_no VARCHAR, city_ref VARCHAR, site_name VARCHAR);"
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, paid_amount DECIMAL(18,2));"
    )
    profile = {
        "semantics": {"sites.region_no": "text", "sites.city_ref": "id"},
        "joint": {"sites": [{"cols": ["region_no", "city_ref"], "values": [["north", "c1"], ["south", "c2"]]}]},
    }
    eng = engine_module.DDLEngine(ddl, rows=3000, months=3, seed=1, profile=profile)

    assert not eng._is_code_col("sites", "region_no"), "claimed by the joint group"
    assert not eng._is_code_col("sites", "city_ref"), "claimed by the joint group, id semantic or not"


@pytest.mark.acceptance
@pytest.mark.parametrize("date_col", DATE_GRAIN_NAMES)
@pytest.mark.parametrize("extra_tables", ("date_dim", "all"))
def test_a_metric_table_does_not_satisfy_a_request_for_a_date_dimension(engine_module, date_col, extra_tables):
    """`_ensure_date_dim` runs before `_infer`, and had its own, older notion of a date dimension.

    So a caller who explicitly asked for one got none: the metric table's `stat_dt` was read as a
    calendar that already existed. Both sites now share the one test that separates them.
    """
    eng = engine_module.DDLEngine(_daily_metric_ddl(date_col), rows=9000, months=3, seed=1, extra_tables=extra_tables)

    assert "dim_date" in eng.schema, "the caller asked for a date dimension and must get one"
    assert "dim_date" in eng.synthetic
    assert eng.roles["daily_channel_metrics"] == "metric_daily"


@pytest.mark.acceptance
def test_a_real_date_dimension_is_not_duplicated(engine_module):
    """The other direction: a calendar already in the DDL must not get a second one beside it."""
    ddl = (
        "CREATE TABLE dim_date (date_key DATE, year_num INTEGER, quarter_cd VARCHAR, "
        "week_of_year INTEGER, is_weekend INTEGER);" + _daily_metric_ddl()
    )
    eng = engine_module.DDLEngine(ddl, rows=9000, months=3, seed=1, extra_tables="date_dim")

    assert eng.synthetic == set(), "the DDL already has a calendar"
    assert eng.roles["dim_date"] == "date_dim"


@pytest.mark.acceptance
def test_report_says_the_plan_is_pre_calibration(engine_module, capsys):
    """The per-table plan does not add up to the target, and silence about that costs reasoning.

    A production run spent a long stretch of one turn trying to reconcile a plan summing to 88,015
    against a budget of 80,000 before concluding the engine would handle it.
    """
    eng = engine_module.DDLEngine(HOSPITAL_DDL, rows=80_000, months=17, seed=42)
    eng.report()

    out = capsys.readouterr().out
    planned = sum(eng.nrows.get(t, 0) for t in eng.schema)

    assert f"{planned:,}" in out, "print the planned total, not only the per-table rows"
    assert "before calibration" in out
    # A target, not a promise: calibration is capped at three passes and cannot move pinned or
    # role-fixed tables, so a fully pinned schema lands outside the band and generate() says so.
    assert "aims for 80,000" in out
    assert "up to 3 passes" in out
    assert "reports the deviation it reached" in out


SQL_BLOCK_DDL = """
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name VARCHAR
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customers(customer_id),
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2),
    is_first_order BOOLEAN,
    coupon_code VARCHAR
);
"""


@pytest.mark.acceptance
def test_a_column_that_does_not_exist_is_caught_before_generating(engine_module):
    """A mistake in `pre_sql` used to cost a whole generate-import-check cycle to find.

    So a production run hand-verified it instead: 221,000 characters of reasoning in one turn,
    much of it walking DuckDB's type rules by hand. EXPLAIN needs no data, only the schema.
    """
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL, rows=5000, months=3, seed=1, profile={"pre_sql": ["UPDATE orders SET coupon_amount = 0"]}
    )

    errors, _warnings = eng.precheck(strict=False)

    assert any("coupon_amount" in e and e.startswith("pre_sql[1]") for e in errors), errors


@pytest.mark.acceptance
def test_a_syntax_error_is_caught_before_generating(engine_module):
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL, rows=5000, months=3, seed=1, profile={"extra_sql": ["UPDATE orders SET paid_amount = "]}
    )

    errors, _warnings = eng.precheck(strict=False)

    assert any(e.startswith("extra_sql[1]") and "Parser Error" in e for e in errors), errors


#: The shapes a real run writes: a window-function ``UPDATE ... FROM`` and a NULLIF division.
SOUND_STATEMENTS = (
    "UPDATE orders o SET is_first_order = (f.rn = 1) FROM ("
    "SELECT order_id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_time, order_id) AS rn "
    "FROM orders) f WHERE o.order_id = f.order_id",
    "UPDATE orders SET paid_amount = ROUND(paid_amount / NULLIF(1, 0), 2) WHERE coupon_code IS NOT NULL",
)


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(list, id="list"),
        pytest.param(tuple, id="tuple"),
        pytest.param(lambda stmts: ";\n".join(stmts) + ";", id="one-string"),
    ],
)
@pytest.mark.parametrize("key", ["pre_sql", "extra_sql"])
def test_sound_statements_raise_nothing(engine_module, shape, key):
    """Sound SQL must pass in every accepted shape.

    The validator splits a single string with DuckDB's own parser and takes list/tuple elements
    verbatim, so the shapes are separate code paths and each needs its own evidence.
    """
    eng = engine_module.DDLEngine(SQL_BLOCK_DDL, rows=5000, months=3, seed=1, profile={key: shape(SOUND_STATEMENTS)})

    errors, _warnings = eng.precheck(strict=False)

    assert not [e for e in errors if e.startswith(("pre_sql", "extra_sql"))], errors


@pytest.mark.acceptance
def test_one_string_of_several_statements_is_split_and_numbered(engine_module):
    """`pre_sql` also takes a single string; the finding still has to name which statement."""
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL,
        rows=5000,
        months=3,
        seed=1,
        profile={"pre_sql": "UPDATE orders SET paid_amount = 1; UPDATE orders SET nope = 2;"},
    )

    errors, _warnings = eng.precheck(strict=False)

    assert any(e.startswith("pre_sql[2]") and "nope" in e for e in errors), errors


@pytest.mark.acceptance
def test_a_statement_can_use_a_table_an_earlier_statement_created(engine_module):
    """DDL in the block runs for real on the scratch schema, or the rest is planned against a lie."""
    profile = {
        "pre_sql": [
            "CREATE TABLE tmp_first AS SELECT customer_id, min(order_time) AS t FROM orders GROUP BY 1",
            "UPDATE orders o SET is_first_order = (o.order_time = f.t) FROM tmp_first f "
            "WHERE o.customer_id = f.customer_id",
            "DROP TABLE tmp_first",
        ]
    }
    eng = engine_module.DDLEngine(SQL_BLOCK_DDL, rows=5000, months=3, seed=1, profile=profile)

    errors, _warnings = eng.precheck(strict=False)

    assert not [e for e in errors if e.startswith("pre_sql")], errors


@pytest.mark.acceptance
def test_report_plans_the_statements_itself(engine_module, capsys):
    """`report()` must not claim work `precheck()` did, because nothing has called `precheck()`.

    `__init__` stops at `_infer()` and `generate()` is the only other caller, so on the
    `gen.py report` path - the first thing the skill runs - the claim would describe validation
    that had not happened. It plans them itself instead, which is also where the feedback is
    cheapest.
    """
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL, rows=5000, months=3, seed=1, profile={"pre_sql": ["UPDATE orders SET paid_amount = 1"]}
    )
    eng.report()

    out = capsys.readouterr().out

    assert "pre_sql: 1 statement(s) will run" in out
    assert "planned against the schema" in out


@pytest.mark.acceptance
def test_report_shows_the_findings_instead_of_the_claim(engine_module, capsys):
    """A broken statement must surface on the report path, not only inside generate()."""
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL, rows=5000, months=3, seed=1, profile={"pre_sql": ["UPDATE orders SET nope = 1"]}
    )
    eng.report()

    out = capsys.readouterr().out

    assert "configuration pre-check failed" in out
    assert "pre_sql[1]" in out and "nope" in out
    assert "columns and types check out" not in out, "do not claim a clean bill next to a finding"


@pytest.mark.acceptance
@pytest.mark.parametrize("extra_tables", ("summary", "all"))
def test_extra_sql_may_reference_the_auto_summary_layer(engine_module, extra_tables):
    """`extra_sql` runs *after* the summary layer, and post-processing it is its main use.

    Those tables are built during `build_db` and never appear in `self.schema`, so validating
    against the schema alone reported every legitimate reference as "table does not exist" - and
    `generate()` calls `precheck` in strict mode, which made `extra_sql` unusable on these paths.
    """
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL,
        rows=5000,
        months=3,
        seed=1,
        extra_tables=extra_tables,
        profile={"extra_sql": ["UPDATE ads_business_daily SET row_cnt = row_cnt WHERE 1=0"]},
    )

    errors, _warnings = eng.precheck(strict=False)

    assert not [e for e in errors if e.startswith("extra_sql")], errors


@pytest.mark.acceptance
def test_a_bad_column_on_a_summary_table_is_still_caught(engine_module):
    """Staging the layer must not turn into waving it through."""
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL,
        rows=5000,
        months=3,
        seed=1,
        extra_tables="summary",
        profile={"extra_sql": ["UPDATE ads_business_daily SET order_cnt = 1"]},
    )

    errors, _warnings = eng.precheck(strict=False)

    assert any(e.startswith("extra_sql[1]") and "order_cnt" in e for e in errors), errors


@pytest.mark.acceptance
def test_staging_the_summary_layer_leaves_the_engine_untouched(engine_module):
    """`_auto_summary_sql` rewrites `_made`; a pre-check must not leave a trace."""
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL,
        rows=5000,
        months=3,
        seed=1,
        extra_tables="summary",
        profile={"extra_sql": ["UPDATE ads_business_daily SET row_cnt = row_cnt WHERE 1=0"]},
    )
    before = "_made" in eng.__dict__

    eng.precheck(strict=False)

    assert ("_made" in eng.__dict__) == before


@pytest.mark.acceptance
def test_the_summary_path_still_generates_end_to_end(engine_module, tmp_path):
    """The check that matters: strict precheck runs inside generate()."""
    eng = engine_module.DDLEngine(
        SQL_BLOCK_DDL,
        rows=8000,
        months=4,
        seed=1,
        extra_tables="summary",
        profile={"extra_sql": ["UPDATE ads_business_daily SET row_cnt = row_cnt WHERE 1=0"]},
    )

    result = eng.generate(str(tmp_path / "s.duckdb"), verbose=False)

    assert "ads_business_daily" in result["tables"]


BUDGET_DDL = """
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name VARCHAR
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customers(customer_id),
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
CREATE TABLE order_items (
    item_id BIGINT PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id),
    quantity INTEGER,
    item_amount DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
def test_pinning_more_rows_than_the_budget_is_refused(engine_module):
    """Calibration can only scale tables that are neither pinned nor role-fixed.

    Pin past the budget and there is nothing left to scale, so the total can never come back -
    which is worth saying before generating rather than after.
    """
    eng = engine_module.DDLEngine(
        BUDGET_DDL,
        rows=80_000,
        months=6,
        seed=1,
        profile={"table_rows": {"orders": 60_000, "order_items": 90_000}},
    )

    errors, _warnings = eng.precheck(strict=False)

    assert any("table_rows pins 150,000 rows against a budget of 80,000" in e for e in errors), errors


@pytest.mark.acceptance
def test_pinning_inside_the_budget_is_left_alone(engine_module):
    eng = engine_module.DDLEngine(BUDGET_DDL, rows=80_000, months=6, seed=1, profile={"table_rows": {"orders": 20_000}})

    errors, _warnings = eng.precheck(strict=False)

    assert not [e for e in errors if "table_rows pins" in e], errors


@pytest.mark.acceptance
def test_a_deviation_outside_tolerance_is_reported(engine_module, tmp_path, capsys):
    """The warning fired only past 25%, so a run shipped +12.5% in silence."""
    # Pin only the detail table: the fact table stays scalable, so this is a genuine overshoot
    # rather than the "nothing left to calibrate" case, which precheck refuses outright.
    eng = engine_module.DDLEngine(
        BUDGET_DDL,
        rows=4000,
        months=6,
        seed=1,
        profile={"table_rows": {"order_items": 3800}},
    )

    result = eng.generate(str(tmp_path / "b.duckdb"))

    assert abs(result["deviation"]) > 0.06
    out = capsys.readouterr().out
    assert "outside the 6% tolerance" in out, out
    assert "pinned table_rows values leave calibration nothing to scale" in out


@pytest.mark.acceptance
def test_the_calibration_note_says_what_is_reused(engine_module, tmp_path, capsys):
    """ "(reused previous calibration)" read as "the data is cached".

    A production run deleted `data/_build` to fight a cache that does not exist.
    """
    out_path = tmp_path / "c.duckdb"
    engine_module.DDLEngine(BUDGET_DDL, rows=6000, months=3, seed=1).generate(str(out_path))
    capsys.readouterr()

    engine_module.DDLEngine(BUDGET_DDL, rows=6000, months=3, seed=1).generate(str(out_path))

    out = capsys.readouterr().out
    assert "every row was generated fresh" in out, out


@pytest.mark.acceptance
def test_pinning_everything_scalable_is_refused(engine_module):
    """`rows=` stops meaning anything once calibration has no table left to move.

    The production overshoot came from exactly this: four of five tables pinned, one pass, +12.5%.
    """
    eng = engine_module.DDLEngine(
        BUDGET_DDL,
        rows=80_000,
        months=6,
        seed=1,
        profile={"table_rows": {"orders": 20_000, "order_items": 30_000}},
    )

    errors, _warnings = eng.precheck(strict=False)

    assert any("pins every table calibration could scale" in e for e in errors), errors


@pytest.mark.acceptance
def test_leaving_one_table_free_is_allowed(engine_module):
    eng = engine_module.DDLEngine(
        BUDGET_DDL, rows=80_000, months=6, seed=1, profile={"table_rows": {"order_items": 30_000}}
    )

    errors, _warnings = eng.precheck(strict=False)

    assert not [e for e in errors if "pins every table" in e], errors


@pytest.mark.acceptance
def test_report_states_the_amount_identity_the_engine_enforces(engine_module, capsys):
    """A production run restated this identity in `formulas` after deriving it by hand."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, "
        "original_amount DECIMAL(18,2), discount_amount DECIMAL(18,2), shipping_amount DECIMAL(18,2), "
        "tax_amount DECIMAL(18,2), coupon_amount DECIMAL(18,2), paid_amount DECIMAL(18,2));",
        rows=5000,
        months=3,
        seed=1,
    )
    eng.report()

    out = capsys.readouterr().out

    assert "amount identity enforced on orders: paid_amount = original_amount - discount_amount" in out
    assert "coupon is not part of it" in out
    assert "not by restating the identity" in out
    # The roles are what the identity is built from, and a wrong one is the whole failure mode:
    # `scholarship_amount` read as a shipping charge is added to the paid amount, not deducted.
    assert "amount roles on orders: " in out
    assert "shipping_amount=ship" in out
    assert "coupon_amount=coupon" in out


@pytest.mark.acceptance
def test_a_schema_with_nothing_to_calibrate_is_not_blamed_on_the_pin(engine_module):
    """Dimensions and metric tables have nothing calibration can scale, pinned or not.

    Refusing there blames the caller for their DDL, and the message named `table_rows` when only
    `dim_rows` had been set.
    """
    eng = engine_module.DDLEngine(
        "CREATE TABLE customers (customer_id BIGINT PRIMARY KEY, customer_name VARCHAR);"
        "CREATE TABLE daily_metrics (stat_dt DATE, channel VARCHAR, impressions BIGINT, "
        "clicks BIGINT, gmv DECIMAL(18,2));",
        rows=9000,
        months=6,
        seed=1,
        profile={"dim_rows": {"customers": 500}},
    )

    errors, _warnings = eng.precheck(strict=False)

    assert not [e for e in errors if "pins every table" in e], errors


@pytest.mark.acceptance
def test_the_pin_message_names_the_key_that_was_set(engine_module):
    eng = engine_module.DDLEngine(
        BUDGET_DDL,
        rows=80_000,
        months=6,
        seed=1,
        profile={"table_rows": {"orders": 20_000, "order_items": 30_000}},
    )

    errors, _warnings = eng.precheck(strict=False)

    message = next(e for e in errors if "pins every table" in e)
    assert message.startswith("table_rows pins")
    # and names the tables it is talking about
    assert "order_items, orders" in message


@pytest.mark.acceptance
def test_the_profit_identity_names_a_real_column(engine_module, capsys):
    """Every other term on that line is a column; "revenue" was the internal role name."""
    engine_module.DDLEngine(
        "CREATE TABLE order_items (item_id BIGINT PRIMARY KEY, order_time TIMESTAMP, "
        "sales_amount DECIMAL(18,2), total_cost DECIMAL(18,2), gross_profit DECIMAL(18,2));",
        rows=5000,
        months=3,
        seed=1,
    ).report()

    out = capsys.readouterr().out

    assert "gross_profit = sales_amount - total_cost" in out
    assert "revenue -" not in out


@pytest.mark.acceptance
def test_a_metric_table_does_not_claim_an_identity_the_engine_never_enforces(engine_module, capsys):
    """Only `_gen_fact` and `_gen_detail`'s backfill call `_settle_amounts`.

    A metric table fills its amounts independently, so the claimed
    `paid = gross - discount + tax` was off by hundreds per row - the report asserting work the
    engine does not do, which is the failure the line was added to prevent.
    """
    engine_module.DDLEngine(
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, paid_amount DECIMAL(18,2));"
        "CREATE TABLE daily_channel_metrics (stat_dt DATE, channel VARCHAR, impressions BIGINT, "
        "clicks BIGINT, gross_revenue DECIMAL(18,2), discount_amount DECIMAL(18,2), "
        "tax_amount DECIMAL(18,2), paid_revenue DECIMAL(18,2));",
        rows=9000,
        months=6,
        seed=1,
    ).report()

    out = capsys.readouterr().out

    assert "daily_channel_metrics" in out, "the table is still in the plan"
    assert "amount identity enforced" not in out


DOWNSTREAM_DDL = """
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    order_date DATE,
    paid_amount DECIMAL(18, 2)
);
CREATE TABLE order_items (
    item_id BIGINT PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id),
    order_date DATE,
    quantity INTEGER,
    item_amount DECIMAL(18, 2)
);
CREATE TABLE shipments (
    shipment_id BIGINT PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id),
    ship_date DATE,
    freight_amount DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
def test_a_detail_table_is_planned_above_its_parent(engine_module):
    """A detail row is a line of its parent document, so there cannot be fewer of them.

    Independent role shares gave `fact` 0.52 and `detail` 0.28, and a production run was handed
    26,320 order_items under 48,880 orders - 0.54 lines per order. The caller stopped trusting the
    allocation, pinned `table_rows` on all five tables by hand, and spent the turn on arithmetic
    the engine exists to do. SKILL.md section 1.2 has always asked for 1.4-2.2 lines per parent.
    """
    eng = engine_module.DDLEngine(BUDGET_DDL, rows=80_000, months=17, seed=42)

    ratio = eng.nrows["order_items"] / eng.nrows["orders"]
    assert ratio > 1, f"a detail table cannot hold fewer rows than its parent (got {ratio:.2f})"
    assert 1.4 <= ratio <= 2.2, f"lines per parent outside the documented band (got {ratio:.2f})"


@pytest.mark.acceptance
def test_the_planned_detail_count_is_one_the_generator_can_reach(engine_module, tmp_path):
    """The plan `report()` prints has to be the plan `generate()` carries out.

    `_gen_detail` rounds lines-per-parent up to at least one, so a plan below the parent count was
    not merely unrealistic - it was unreachable, and the run silently produced most of a table more
    than the number it had just printed.
    """
    eng = engine_module.DDLEngine(BUDGET_DDL, rows=9_000, months=3, seed=1)
    planned = eng.nrows["order_items"]

    out = tmp_path / "detail.duckdb"
    eng.generate(str(out), verbose=False)
    import duckdb

    con = duckdb.connect(str(out))
    try:
        actual = con.execute("SELECT count(*) FROM order_items").fetchone()[0]
    finally:
        con.close()

    assert abs(actual - planned) / planned < 0.2, f"planned {planned:,}, generated {actual:,}"


@pytest.mark.acceptance
def test_a_downstream_fact_is_planned_below_its_parent(engine_module):
    """A shipment / claim / repayment does not follow every document, so it is not a detail line.

    Both roles used to draw from the same `detail` share; sizing them the same way would have
    swapped one wrong ratio for another.
    """
    eng = engine_module.DDLEngine(DOWNSTREAM_DDL, rows=20_000, months=6, seed=1)
    assert eng.roles["shipments"] == "downstream"

    assert eng.nrows["shipments"] < eng.nrows["orders"]
    assert eng.nrows["order_items"] > eng.nrows["orders"]


NAMING_TPL_DDL = """
CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    product_name VARCHAR,
    category VARCHAR,          -- Electronics / Beauty / Clothing
    list_price DECIMAL(18, 2)
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    product_id BIGINT REFERENCES products(product_id),
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
def test_a_naming_template_naming_a_column_does_not_break_the_report(engine_module, capsys):
    """`report()` previews names before a row exists, and a `tpl` draws from the row being built.

    Passing no row made a perfectly valid `"{brand} {category}"` raise KeyError out of `report()` -
    a traceback in place of the plan, on the first command the skill tells the caller to run. The
    production workaround was a fake `vocab` entry, which then made the preview print names the run
    would never produce.
    """
    engine_module.DDLEngine(
        NAMING_TPL_DDL,
        rows=9_000,
        months=3,
        seed=1,
        profile={"naming": {"products": {"tpl": "{brand} {category}"}}},
    ).report()

    line = next(ln for ln in capsys.readouterr().out.splitlines() if "products.product_name ->" in ln)
    drawn = {v for v in ("Electronics", "Beauty", "Clothing") if v in line}

    assert drawn, f"the preview must show the domain the run will really draw from: {line}"
    assert "CATEG" not in line, f"a fallback code means the preview never saw the DDL domain: {line}"


@pytest.mark.acceptance
def test_a_naming_template_naming_nothing_says_so_instead_of_raising(engine_module, capsys):
    engine_module.DDLEngine(
        NAMING_TPL_DDL,
        rows=9_000,
        months=3,
        seed=1,
        profile={"naming": {"products": {"tpl": "{brand} {nonesuch}"}}},
    ).report()

    out = capsys.readouterr().out

    assert "'nonesuch'" in out
    assert "neither a vocabulary key nor a generated column of products" in out


@pytest.mark.acceptance
def test_report_runs_the_validation_generate_will_run(engine_module, capsys):
    """`gen.py report` is where the caller looks, and it used to validate nothing.

    A production run pinned every table, read a plan with nothing wrong in it, and moved on; the
    "pins every table calibration could scale" error existed the whole time and had no surface to
    appear on until `generate()`, long after the profile was written.
    """
    engine_module.DDLEngine(
        BUDGET_DDL,
        rows=80_000,
        months=6,
        seed=1,
        profile={"table_rows": {"orders": 26_000, "order_items": 40_000}},
    ).report()

    out = capsys.readouterr().out

    assert "pins every table calibration could scale" in out


@pytest.mark.acceptance
def test_a_clean_profile_is_told_it_is_clean(engine_module, capsys):
    """Silence reads as "not checked". It has to say the check ran."""
    engine_module.DDLEngine(BUDGET_DDL, rows=80_000, months=6, seed=1, profile={"trend_mom": 0.03}).report()

    assert "profile validated: no errors, no warnings" in capsys.readouterr().out


@pytest.mark.acceptance
def test_a_bare_ddl_is_not_told_its_profile_validated(engine_module, capsys):
    """`plan_datasource` reports on a DDL alone, before a profile exists.

    Saying "validated" there is a green light for work nobody has done yet.
    """
    engine_module.DDLEngine(BUDGET_DDL, rows=80_000, months=6, seed=1).report()

    assert "profile validated" not in capsys.readouterr().out


@pytest.mark.acceptance
def test_a_failing_sql_block_is_reported_once_not_twice(engine_module, capsys):
    """`_print_plan` listed the SQL problems and then the pre-check listed them again.

    The one report that has to be unambiguous printed the same failure twice, four lines apart.
    """
    engine_module.DDLEngine(
        BUDGET_DDL,
        rows=80_000,
        months=6,
        seed=1,
        profile={"pre_sql": "UPDATE orders SET no_such_col = 1;"},
    ).report()

    out = capsys.readouterr().out

    assert out.count("pre_sql[1]:") == 1, out


@pytest.mark.acceptance
@pytest.mark.parametrize("requested", (1.0, 1.35, 1.84, 2.6, 3.0, 4.0))
def test_lines_per_document_follow_the_plan(engine_module, requested):
    """The line-count draw ignored the plan below two lines per document.

    It returned the raw spread, mean 1.84, whatever `nrows` said - which made a detail table the one
    table calibration could not move, because calibration works by rescaling `nrows` between passes.
    """
    import random

    eng = engine_module.DDLEngine(BUDGET_DDL, rows=20_000, months=6, seed=1)
    rng = random.Random(7)

    mean = sum(eng._lines_for(requested, rng) for _ in range(40_000)) / 40_000

    # An upper clamp used to truncate the tail, and it cost the most exactly where the caller had
    # asked for the most: 2% of the requested mean at three lines per document, 3% at four.
    assert abs(mean - requested) < 0.05, f"asked for {requested} lines per document, drew {mean:.3f}"


@pytest.mark.acceptance
def test_lines_per_document_never_drop_below_one(engine_module):
    """A detail row belongs to a parent document, so a document cannot have zero lines."""
    import random

    eng = engine_module.DDLEngine(BUDGET_DDL, rows=20_000, months=6, seed=1)
    rng = random.Random(7)

    assert min(eng._lines_for(0.4, rng) for _ in range(5_000)) == 1


@pytest.mark.acceptance
def test_calibration_reaches_the_budget_through_the_detail_table(engine_module, tmp_path):
    """Two production runs shipped ~90,000 rows against a requested 80,000.

    Both pinned their other tables and left the detail table free to absorb the difference - which
    it could not do, because the line-count draw was not reading `nrows`. Every one of the three
    passes produced byte-identical row counts, and the run shipped the overshoot.
    """
    eng = engine_module.DDLEngine(
        BUDGET_DDL,
        rows=20_000,
        months=6,
        seed=1,
        profile={"table_rows": {"customers": 1500, "orders": 9000}},
    )

    res = eng.generate(str(tmp_path / "cal.duckdb"), verbose=False)

    assert abs(res["deviation"]) <= 0.06, f"{res['rows']:,} rows, deviation {res['deviation']:+.1%}"
    assert res["tables"]["order_items"] >= res["tables"]["orders"], "still one line per order at the floor"


@pytest.mark.acceptance
def test_a_skipped_funnel_stage_still_gets_a_believable_ratio(engine_module):
    """The ratio belongs to the pair of stages, not to the step's own name.

    Keyed on the name alone, `purchasers` got "purchasers per checkout user" wherever it appeared -
    so a schema that goes straight from sessions to purchasers was handed a 55% site conversion
    rate. The stage levels divide, so a skipped stage narrows by the whole gap instead.
    """
    full = engine_module.DDLEngine._default_ratio("checkout_users", "purchasers")[0]
    skipped = engine_module.DDLEngine._default_ratio("sessions", "purchasers")[0]

    assert 0.4 < full[0] < full[1] < 0.7, full
    assert 0.01 < skipped[0] < skipped[1] < 0.05, skipped


@pytest.mark.acceptance
def test_a_narrowing_funnel_step_never_prefills_above_one(engine_module):
    """A step that loses people cannot gain them on some rows, or the funnel stops being a funnel."""
    for src, dst in (("clicks", "sessions"), ("sessions", "unique_visitors"), ("add_to_carts", "checkout_users")):
        (_lo, hi), known = engine_module.DDLEngine._default_ratio(src, dst)
        assert known, f"{src} -> {dst} was not recognised"
        assert hi < 1.0, f"{src} -> {dst} prefilled {hi}"


@pytest.mark.acceptance
def test_an_unrecognised_step_is_marked_rather_than_guessed_at(engine_module):
    """A default the engine cannot justify has to say so, or it reads as an inferred value."""
    ratio, known = engine_module.DDLEngine._default_ratio("widgets_seen", "sprockets_touched")

    assert not known
    assert ratio == engine_module.DDLEngine.FUNNEL_DEFAULT


@pytest.mark.acceptance
@pytest.mark.parametrize("pin,expected", [({"orders": 12_000}, 12_000), ({"orders": 1_500}, 1_500)])
def test_pinning_the_parent_scales_its_detail_table(engine_module, pin, expected):
    """A pin is a count, not a share, and a detail hangs off its parent's count.

    Pins used to be applied last, as a plain overwrite once the shares were computed, so the detail
    table stayed on the number its share gave it and never looked at the parent again: `orders`
    pinned to 12,000 or to 1,500 both produced 12,085 `order_items` - one line per order or eight.
    """
    eng = engine_module.DDLEngine(BUDGET_DDL, rows=20_000, months=6, seed=1, profile={"table_rows": pin})

    ratio = eng.nrows["order_items"] / eng.nrows["orders"]

    assert eng.nrows["orders"] == expected, "the pin itself is the caller's instruction"
    assert 1.4 <= ratio <= 2.2, f"lines per parent outside the documented band (got {ratio:.2f})"


@pytest.mark.acceptance
def test_pinning_the_detail_table_sizes_its_parent(engine_module):
    """The same relationship read the other way: pinned lines imply how many documents carry them.

    Resolving pins in one direction only left the parent on the 50-row floor - 600 lines per order.
    """
    eng = engine_module.DDLEngine(
        BUDGET_DDL, rows=20_000, months=6, seed=1, profile={"table_rows": {"order_items": 30_000}}
    )

    assert eng.nrows["order_items"] == 30_000
    assert 1.4 <= 30_000 / eng.nrows["orders"] <= 2.2, eng.nrows


@pytest.mark.acceptance
def test_pinning_both_ends_leaves_both_alone(engine_module):
    """An explicit count on both tables is the caller overriding the ratio, which is allowed."""
    eng = engine_module.DDLEngine(
        BUDGET_DDL, rows=20_000, months=6, seed=1, profile={"table_rows": {"orders": 6_000, "order_items": 9_000}}
    )

    assert (eng.nrows["orders"], eng.nrows["order_items"]) == (6_000, 9_000)


@pytest.mark.acceptance
def test_a_wrongly_typed_sql_block_is_reported_not_raised(engine_module, capsys):
    """`_sql_block` raises TypeError, and it did so from inside the report.

    A traceback halfway through the plan, for a mistake the pre-check names in one sentence.
    """
    engine_module.DDLEngine(BUDGET_DDL, rows=5_000, months=3, seed=1, profile={"pre_sql": 123}).report()

    out = capsys.readouterr().out

    assert "pre_sql: must be a str or a list of str, got int" in out
    assert "table" in out, "the rest of the plan still has to print"


NESTED_DETAIL_DDL = """
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
CREATE TABLE order_items (
    item_id BIGINT PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id),
    quantity INTEGER,
    item_amount DECIMAL(18, 2)
);
CREATE TABLE item_serials (
    serial_id BIGINT PRIMARY KEY,
    item_id BIGINT REFERENCES order_items(item_id),
    serial_no VARCHAR
);
"""


@pytest.mark.acceptance
def test_a_detail_table_can_itself_be_a_parent(engine_module, tmp_path):
    """`order_items` has serials; a claim has line items. A detail table is a legitimate parent.

    `_gen_detail` recorded neither `refs` nor `_fact_rows`, so `_parent_of` could not see it and the
    nested table fell through to `_gen_fact` - generated as an independent fact, 10,754 rows with
    the foreign key NULL on every single one. The FK check reports that as resolving, because it
    excludes NULLs from both sides of the ratio.
    """
    import duckdb

    out = tmp_path / "nested.duckdb"
    engine_module.DDLEngine(NESTED_DETAIL_DDL, rows=20_000, months=6, seed=1).generate(str(out), verbose=False)

    con = duckdb.connect(str(out))
    try:
        total, nulls = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE item_id IS NULL) FROM item_serials"
        ).fetchone()
        resolving = con.execute(
            "SELECT count(*) FROM item_serials s JOIN order_items i ON s.item_id = i.item_id"
        ).fetchone()[0]
    finally:
        con.close()

    assert total > 0
    assert nulls == 0, f"{nulls:,} of {total:,} nested-detail rows have no parent key"
    assert resolving == total


REVIEW_METRIC_DDL = """
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
CREATE TABLE daily_seller_metrics (
    stat_dt DATE,
    seller VARCHAR,
    impressions BIGINT,
    clicks BIGINT,
    review_count BIGINT,
    gmv DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
def test_a_stage_word_only_matches_at_a_token_start(engine_module):
    """A bare substring search read `review_count` as a page view, because it contains "view".

    Page views are the one fan-out in the table, so a review column was prefilled to go *up* from
    its predecessor instead of down - the only ratio in the funnel that must never be guessed wrong.
    """
    page_view = dict(engine_module.DDLEngine.FUNNEL_STAGE)[r"view|browse|detail|pv"]

    for name in ("review_count", "review_score", "interview_count"):
        assert engine_module.DDLEngine._stage_level(name) is None, name
    for name in ("view_count", "page_view", "product_detail", "pv_cnt"):
        assert engine_module.DDLEngine._stage_level(name) == page_view, name


@pytest.mark.acceptance
def test_an_unrecognised_stage_is_annotated_in_the_skeleton(engine_module):
    """A default the engine cannot justify has to be marked where the caller will read it."""
    skeleton = engine_module.DDLEngine(REVIEW_METRIC_DDL, rows=20_000, months=6, seed=1).profile_skeleton()

    line = next(ln for ln in skeleton.splitlines() if "review_count" in ln)

    assert "step not recognised" in line, line


CAMPUS_DDL = """
CREATE TABLE students (
    student_id BIGINT PRIMARY KEY,
    student_name VARCHAR,
    gpa DECIMAL(4, 2)
);
CREATE TABLE enrolments (
    enrol_id BIGINT PRIMARY KEY,
    student_id BIGINT REFERENCES students(student_id),
    enrol_time TIMESTAMP,
    scholarship_amount DECIMAL(18, 2),
    tuition_amount DECIMAL(18, 2),
    paid_amount DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "column,role",
    [
        ("scholarship_amount", "gross"),
        ("shipping_amount", "ship"),
        ("shipment_fee", "ship"),
        ("taxonomy_value", "gross"),
        ("tax_amount", "tax"),
        ("taxes_paid", "tax"),
    ],
)
def test_an_amount_role_matches_a_word_not_a_substring(engine_module, column, role):
    """The role words are English business vocabulary and they collide across industries.

    `scholarship_amount` matched the shipping pattern, so a scholarship was settled as a delivery
    charge - *added* to the amount paid instead of deducted from it - and `taxonomy_value` was a tax.
    """
    assert engine_module.DDLEngine._amt_role(column) == role


@pytest.mark.acceptance
def test_the_report_names_the_role_it_gave_each_amount(engine_module, capsys):
    """The identity is built from the roles, so the roles are what has to be checkable.

    The report stated the identity and never said which column it had read as what, which is the
    one thing a caller on a non-retail schema needs to see.
    """
    engine_module.DDLEngine(CAMPUS_DDL, rows=9_000, months=6, seed=1).report()

    out = capsys.readouterr().out

    assert "scholarship_amount=gross" in out
    assert "paid_amount=paid" in out


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "shape,low,high", [("weekend_heavy", 1.15, 1.6), ("weekday_heavy", 0.2, 0.6), ("flat", 0.9, 1.1)]
)
def test_the_weekly_shape_reaches_the_data(engine_module, tmp_path, shape, low, high):
    """`weekend_lift=1.33` was hardcoded: a consumer shop, and nothing else.

    A B2B schema generated that way has its busiest days on the weekend, and the weekday check then
    reports the "B2C shape" it was handed.
    """
    import duckdb

    out = tmp_path / f"{shape}.duckdb"
    engine_module.DDLEngine(BUDGET_DDL, rows=9_000, months=6, seed=1, profile={"weekly_shape": shape}).generate(
        str(out), verbose=False
    )

    con = duckdb.connect(str(out))
    try:
        ratio = con.execute(
            "WITH d AS (SELECT order_time::date dt, count(*) n FROM orders GROUP BY 1) "
            "SELECT avg(n) FILTER (WHERE dayofweek(dt) IN (0, 6)) "
            "     / avg(n) FILTER (WHERE dayofweek(dt) NOT IN (0, 6)) FROM d"
        ).fetchone()[0]
    finally:
        con.close()

    assert low <= ratio <= high, f"{shape} produced a weekend/weekday ratio of {ratio:.2f}"


@pytest.mark.acceptance
def test_the_report_states_which_weekly_shape_is_in_force(engine_module, capsys):
    engine_module.DDLEngine(BUDGET_DDL, rows=9_000, months=6, seed=1).report()

    assert "weekly shape: weekend_heavy" in capsys.readouterr().out


ATTR_DATE_DDL = """
CREATE TABLE parties (
    party_id BIGINT PRIMARY KEY,
    party_name VARCHAR
);
CREATE TABLE movements (
    movement_id BIGINT PRIMARY KEY,
    party_id BIGINT REFERENCES parties(party_id),
    {date_col} TIMESTAMP,
    fee_amount DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
@pytest.mark.parametrize("date_col", ["aggregate_dt", "invalidated_at", "reopened_at", "reentry_time"])
def test_a_business_date_is_not_demoted_by_a_substring(engine_module, date_col):
    """The attribute-date test was an unanchored substring search.

    `aggregate_dt` matched "reg", `invalidated_at` matched "valid", `reopened_at` matched "open"
    and `reentry_time` matched "entry" - so the table's only business date read as an entity
    attribute and the whole table became a dimension, every measure on it generated as a static
    attribute with no time signal.
    """
    eng = engine_module.DDLEngine(ATTR_DATE_DDL.format(date_col=date_col), rows=9_000, months=6, seed=1)

    assert eng.roles["movements"] == "fact"


@pytest.mark.acceptance
def test_a_table_demoted_by_a_date_name_says_so(engine_module, capsys):
    """`opened_at` is an entity attribute on a dimension and the event time on a ticket table.

    The heuristic cannot tell, so when it demotes a table that has measures and a foreign key it
    names the column that decided it and the override, rather than leaving the caller to find out
    from the data.
    """
    ddl = ATTR_DATE_DDL.format(date_col="opened_at")
    eng = engine_module.DDLEngine(ddl, rows=9_000, months=6, seed=1)
    eng.report()

    out = capsys.readouterr().out

    assert eng.roles["movements"] == "dim", "the heuristic still decides; the report just says so"
    assert "`opened_at` reads as an entity attribute date" in out
    assert "'movements': 'fact'" in out

    override = engine_module.DDLEngine(ddl, rows=9_000, months=6, seed=1, profile={"roles": {"movements": "fact"}})
    assert override.roles["movements"] == "fact", "the override the message names has to work"


@pytest.mark.acceptance
def test_measure_columns_arrive_as_skeleton_slots(engine_module):
    """A measure's units cannot be inferred, and the fallback is a 0.1-40 lognormal.

    A production schema got a GPA of 11.79 on a DECIMAL(4,2) because that fallback was hidden
    behind the generator instead of written where the caller would read it.
    """
    skeleton = engine_module.DDLEngine(CAMPUS_DDL, rows=9_000, months=6, seed=1).profile_skeleton()

    assert '"students.gpa": {"range": (0.1, 40)}' in skeleton
    assert "cannot infer a measure" in skeleton


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "column,role",
    [("payment_amount", "paid"), ("duty_free_price", "unit"), ("duty_amount", "tax")],
)
def test_the_role_words_cover_the_obvious_neighbours(engine_module, column, role):
    """`payment_amount` fell to the catch-all because the pattern only had `pay_`, and
    `duty_free_price` is a price rather than a duty."""
    assert engine_module.DDLEngine._amt_role(column) == role


@pytest.mark.acceptance
def test_a_dimension_with_no_date_column_is_not_reported_as_demoted(engine_module, capsys):
    """The demotion message blames a date column, so it must only fire when one decided it.

    A table with measures, a foreign key and no date at all is a dimension for the ordinary reason.
    It was reported as demoted by `` - an empty column name - which pointed the caller at a fact
    table it should not build.
    """
    engine_module.DDLEngine(
        "CREATE TABLE teachers (teacher_id BIGINT PRIMARY KEY, teacher_name VARCHAR);"
        "CREATE TABLE courses (course_id BIGINT PRIMARY KEY, "
        "teacher_id BIGINT REFERENCES teachers(teacher_id), course_name VARCHAR, credits INTEGER);",
        rows=9_000,
        months=6,
        seed=1,
    ).report()

    out = capsys.readouterr().out

    assert "planned as a dimension, because" not in out
    assert "``" not in out


@pytest.mark.acceptance
@pytest.mark.parametrize("shape", ["weekend_heavy", "weekday_heavy", "flat"])
def test_the_declared_weekly_shape_reaches_the_metadata(engine_module, tmp_path, shape):
    """The quality check cannot honour a declaration it cannot see."""
    import json

    out = tmp_path / "m.duckdb"
    engine_module.DDLEngine(BUDGET_DDL, rows=9_000, months=6, seed=1, profile={"weekly_shape": shape}).generate(
        str(out), verbose=False
    )

    meta = json.loads((tmp_path / ".m.meta.json").read_text(encoding="utf-8"))

    assert meta["weekly_shape"] == shape


@pytest.mark.acceptance
def test_measure_columns_left_on_the_fallback_are_named(engine_module):
    """Deleting the skeleton's entry puts the column back on the fallback silently.

    A warning rather than an error: the skeleton has to stay runnable as copied, and 0.1-40 is
    genuinely right for some measures.
    """
    _errors, warnings = engine_module.DDLEngine(CAMPUS_DDL, rows=9_000, months=6, seed=1).precheck(strict=False)

    assert any("students.gpa" in w and "0.1-40 fallback" in w for w in warnings), warnings


@pytest.mark.acceptance
def test_a_corrected_measure_range_clears_the_warning(engine_module):
    _errors, warnings = engine_module.DDLEngine(
        CAMPUS_DDL, rows=9_000, months=6, seed=1, profile={"columns": {"students.gpa": {"range": (0.0, 4.0)}}}
    ).precheck(strict=False)

    assert not [w for w in warnings if "0.1-40 fallback" in w], warnings


@pytest.mark.acceptance
@pytest.mark.parametrize("path", ["dim", "fill_generic"])
def test_a_declared_measure_range_constrains_the_data(engine_module, tmp_path, path):
    """Both measure generators ignored `columns['t.col']['range']`.

    So the slot the skeleton writes and the warning precheck prints asked the caller to set a value
    that changed nothing - a GPA declared (0.0, 4.0) still came out at 49.02. The two also
    disagreed on the fallback, 0.1-50 on a dimension against 0.1-40 elsewhere, so neither matched
    what was documented.
    """
    import duckdb

    table, column, bounds = ("students", "gpa", (0.0, 4.0)) if path == "dim" else ("sessions", "room_area", (5.0, 40.0))
    ddl = (
        "CREATE TABLE students (student_id BIGINT PRIMARY KEY, student_name VARCHAR, gpa DECIMAL(4,2));"
        "CREATE TABLE sessions (sess_id BIGINT PRIMARY KEY, "
        "student_id BIGINT REFERENCES students(student_id), order_time TIMESTAMP, "
        "fee_amount DECIMAL(18,2), room_area DECIMAL(6,2));"
    )
    out = tmp_path / f"{path}.duckdb"
    engine_module.DDLEngine(
        ddl, rows=9_000, months=6, seed=1, profile={"columns": {f"{table}.{column}": {"range": bounds}}}
    ).generate(str(out), verbose=False)

    con = duckdb.connect(str(out))
    try:
        low, high = con.execute(f"SELECT min({column}), max({column}) FROM {table}").fetchone()
    finally:
        con.close()

    assert bounds[0] <= float(low) <= float(high) <= bounds[1], f"{column} spans {low}-{high}, asked for {bounds}"


@pytest.mark.acceptance
def test_a_measure_range_starting_at_zero_is_generable(engine_module, tmp_path):
    """A lower bound of zero is the ordinary case for a bounded measure, and log(0) is not a number.

    The lognormal draw raised `ValueError: math domain error` on the first range a caller would
    write for a score.
    """
    import duckdb

    out = tmp_path / "zero.duckdb"
    engine_module.DDLEngine(
        "CREATE TABLE gauges (gauge_id BIGINT PRIMARY KEY, gauge_name VARCHAR);"
        "CREATE TABLE readings (reading_id BIGINT PRIMARY KEY, "
        "gauge_id BIGINT REFERENCES gauges(gauge_id), order_time TIMESTAMP, "
        "fee_amount DECIMAL(18,2), health_score DECIMAL(5,2));",
        rows=9_000,
        months=3,
        seed=1,
        profile={"columns": {"readings.health_score": {"range": (0.0, 100.0)}}},
    ).generate(str(out), verbose=False)

    con = duckdb.connect(str(out))
    try:
        low, high, below = con.execute(
            "SELECT min(health_score), max(health_score), count(*) FILTER (WHERE health_score < 10) FROM readings"
        ).fetchone()
    finally:
        con.close()

    assert 0.0 <= float(low) <= float(high) <= 100.0, f"{low}-{high}"
    assert below, "a range starting at zero has to be able to produce values near zero"


@pytest.mark.acceptance
def test_an_unsupported_weekly_shape_is_refused(engine_module):
    """`.get(shape, weekend_heavy)` swallowed a typo and the surfaces then disagreed.

    The report and the metadata echoed what was asked for while the data was generated as a
    consumer shop, so the quality check was handed a label the rows did not carry.
    """
    errors, _warnings = engine_module.DDLEngine(
        BUDGET_DDL, rows=9_000, months=6, seed=1, profile={"weekly_shape": "weekday-heavy"}
    ).precheck(strict=False)

    assert any("'weekday-heavy' is not a shape" in e for e in errors), errors


@pytest.mark.acceptance
@pytest.mark.parametrize("date_col", ["regression_dt", "regional_dt"])
def test_reg_is_not_a_prefix_of_every_word_starting_with_reg(engine_module, date_col):
    """Spelled out as `reg_` or `regist*`: a token start alone still caught `regression_dt`."""
    eng = engine_module.DDLEngine(ATTR_DATE_DDL.format(date_col=date_col), rows=9_000, months=6, seed=1)

    assert eng.roles["movements"] == "fact"


@pytest.mark.acceptance
@pytest.mark.parametrize("date_col", ["registered_at", "reg_dt", "registration_date"])
def test_the_real_registration_dates_still_read_as_attributes(engine_module, date_col):
    eng = engine_module.DDLEngine(ATTR_DATE_DDL.format(date_col=date_col), rows=9_000, months=6, seed=1)

    assert eng.roles["movements"] == "dim"


@pytest.mark.acceptance
def test_duty_is_a_tax_only_in_a_monetary_form(engine_module):
    """Anchored at the front alone, `duty_roster_allowance` was a tax."""
    assert engine_module.DDLEngine._amt_role("duty_roster_allowance") == "gross"
    assert engine_module.DDLEngine._amt_role("duty_amount") == "tax"
    assert engine_module.DDLEngine._amt_role("import_duty") == "tax"


@pytest.mark.acceptance
def test_a_table_with_no_foreign_key_also_reports_its_demotion(engine_module, capsys):
    """A table reaches ROLE_DIM with foreign keys and without them; only the first path reported."""
    engine_module.DDLEngine(
        "CREATE TABLE sensors (sensor_id BIGINT PRIMARY KEY, opened_at TIMESTAMP, "
        "reading_score DECIMAL(8,2), calib_score DECIMAL(8,2));",
        rows=9_000,
        months=6,
        seed=1,
    ).report()

    assert "sensors carries measures but is planned as a dimension" in capsys.readouterr().out


@pytest.mark.acceptance
def test_a_staging_warning_says_it_is_advisory(engine_module, capsys, monkeypatch):
    """Every staging failure is already swallowed - the `except` blocks say so in their comments
    ("never block generating on a staging problem"). That fact stayed in the comments: what the
    reader got was an internal step name, a DuckDB error and a `!`, with nothing to say whether it
    mattered. A measured run spent a whole turn hypothesising five different mechanisms for it,
    then went into ddl_engine.py to find out - the first item on SKILL.md's list of ways to lose a
    run. The answer is one clause long and belongs in the line."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, order_dt DATE, amt DOUBLE);",
        rows=2000,
        extra_tables="summary",
    )
    con = eng._scratch_schema()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("Catalog Error: Table with name orders does not exist!")

    monkeypatch.setattr(eng, "_auto_summary_sql", _boom)
    assert eng._stage_summary_layer(con) is False, "a staging failure must never become a hard stop"

    out = capsys.readouterr().out
    assert "could not stage the summary layer" in out, out
    assert "advisory only" in out, out
    assert "generation is unaffected" in out, out


@pytest.mark.acceptance
def test_a_composite_primary_key_is_parsed_but_not_honoured_end_to_end(engine_module, tmp_path):
    """Pins the LIMITATION, so the docs cannot start promising the capability again.

    `_scan_constraints` parses `PRIMARY KEY (a, b)` into `decl_pk` as a list, and that is where it
    stops: `pk_of` returns the first schema column for anything that is not single-column, and
    `_dump_meta` writes `pks` from `pk_of` - so the quality check verifies `a` alone and a table
    whose grain really is two columns reports duplicates however it is declared.

    The second column is a DATE on purpose. An earlier version of this test used a TIMESTAMP and
    passed, which pinned the entropy of a microsecond clock rather than anything the engine does -
    the same DDL at day granularity produced 1,313 duplicate pairs and had its constraint dropped.
    """
    ddl = (
        "CREATE TABLE meters (meter_id VARCHAR PRIMARY KEY, unit VARCHAR);"
        "CREATE TABLE readings ("
        "  meter_id VARCHAR REFERENCES meters(meter_id),"
        "  read_date DATE, kwh DOUBLE, PRIMARY KEY (meter_id, read_date));"
    )
    eng = engine_module.DDLEngine(ddl, rows=5000)

    assert eng.decl_pk["readings"] == ["meter_id", "read_date"], "parsing keeps both columns"
    # ...and every consumer downstream drops the second one.
    assert eng.pk_of("readings") == eng.schema["readings"][0]["name"]

    out = tmp_path / "t.duckdb"
    with contextlib.redirect_stdout(io.StringIO()):
        eng.generate(str(out))
    con = duckdb.connect(str(out), read_only=True)
    dup_pairs = con.execute(
        "SELECT count(*) FROM (SELECT meter_id, read_date FROM readings GROUP BY 1, 2 HAVING count(*) > 1)"
    ).fetchone()[0]

    assert dup_pairs > 0, (
        "if this ever passes, generation learned about composite keys - update profile-spec's "
        "capability table and SKILL.md step 0, which currently tell the reader it cannot"
    )


@pytest.mark.acceptance
def test_a_guessed_enum_domain_is_named(engine_module, capsys):
    """The report already names what it EXTRACTED a domain for and what it did not recognise at
    all. Between them sat the column recognised as an enum whose values nothing declared - filled
    from a built-in vocabulary, silently. It surfaces much later as a quality failure about a
    distribution nobody chose: a measured run wrote an assertion about `flight_sensors.unit`, whose
    domain had been guessed, and got back "sensor series differentiated by unit: actual 0.9994
    (expected 1000~100000)"."""
    # One column per line: the comment scanner anchors the column name at the start of the line,
    # so two columns on one line hand the comment to the first of them.
    eng = engine_module.DDLEngine(
        "CREATE TABLE meters (\n"
        "  meter_id VARCHAR PRIMARY KEY,\n"
        "  unit VARCHAR\n"
        ");\n"
        "CREATE TABLE usage_events (\n"
        "  event_id VARCHAR PRIMARY KEY,\n"
        "  meter_id VARCHAR REFERENCES meters(meter_id),\n"
        "  read_at TIMESTAMP,\n"
        "  state VARCHAR, -- pending / settled / disputed\n"
        "  kwh INTEGER\n"
        ");",
        rows=3000,
    )
    eng.report()
    out = capsys.readouterr().out

    named = out.split("GUESSED", 1)[1].split("\n")[0]
    assert "value domains GUESSED" in out, out
    assert "meters.unit" in named, "the undeclared enum must be named"
    # `state` HAS a DDL comment, so it is extracted rather than guessed - naming it here would
    # send the reader to re-declare something the engine already read.
    assert "usage_events.state" not in named, named


@pytest.mark.acceptance
def test_every_build_says_to_run_the_check_next(engine_module, tmp_path):
    """The instruction exists in SKILL.md too, and SKILL.md is loaded once at the top of a session
    that then runs for dozens of turns. Two measured runs, two models: 79% of the wall clock went
    to the stretch BEFORE the first `import_database_file` - one writing duckdb queries against
    successive builds for 43 minutes, the other reading the engine's source for 33 - and once the
    check finally ran it converged in three rounds either way. This line is printed at the moment
    that choice is made."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE carriers (carrier_id VARCHAR PRIMARY KEY, name VARCHAR);"
        "CREATE TABLE flights (flight_id VARCHAR PRIMARY KEY,"
        "  carrier_id VARCHAR REFERENCES carriers(carrier_id), scheduled_departure TIMESTAMP);",
        rows=3000,
    )
    out = tmp_path / "datasource.duckdb"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng.generate(str(out))
    text = buf.getvalue()

    assert "import_database_file" in text, text
    assert "check_datasource_quality" in text, text
    assert str(out) in text, "the path must be the one just built, so it can be copied verbatim"


@pytest.mark.acceptance
def test_a_broken_reference_downstream_is_the_one_reported(engine_module):
    """`a -> b` and `b -> xxx` fail together, and `a` is first in file order - so reporting the
    first leftover says "Table with name b does not exist" about a table declared on the very next
    line, and never mentions `xxx`, which is the only thing actually wrong. The reader is sent to
    hunt a typo in the wrong statement."""
    with pytest.raises(ValueError) as excinfo:
        engine_module.DDLEngine(
            "CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES b(id));"
            "CREATE TABLE b (id INTEGER PRIMARY KEY, x_id INTEGER REFERENCES xxx(id));",
            rows=2000,
        )

    assert "xxx" in str(excinfo.value), str(excinfo.value)
    assert "Table with name b does not exist" not in str(excinfo.value)


@pytest.mark.acceptance
def test_a_reference_cycle_is_not_called_a_missing_table(engine_module):
    """Both targets ARE declared, so "no CREATE declares that table" would be a false statement -
    and the fix it asks for (check the spelling) does not exist."""
    with pytest.raises(ValueError, match="cycle") as excinfo:
        engine_module.DDLEngine(
            "CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES b(id));"
            "CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER REFERENCES a(id));",
            rows=2000,
        )

    assert "no CREATE in this DDL declares" not in str(excinfo.value)


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "profile",
    [
        {"enums": {"unit": ["kWh", "m3"]}},
        {"columns": {"meters.unit": {"values": ["kWh", "m3"]}}},
    ],
    ids=["profile_enums", "profile_columns_values"],
)
def test_a_domain_the_reader_already_set_is_not_reported_as_guessed(engine_module, capsys, profile):
    """The line tells the reader to set `profile['enums']`. Saying it again after they have is the
    shape this whole branch exists to remove - a hint with no exit, repeating every report.
    `_enum_values_raw` honours `columns[t.c]['values']` above `enums[c]`, so both count."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE meters (meter_id VARCHAR PRIMARY KEY, unit VARCHAR);"
        "CREATE TABLE ev (id VARCHAR PRIMARY KEY,"
        "  meter_id VARCHAR REFERENCES meters(meter_id), read_at TIMESTAMP);",
        rows=3000,
        profile=profile,
    )
    eng.report()

    guessed = [line for line in capsys.readouterr().out.splitlines() if "GUESSED" in line]
    assert not guessed, guessed


@pytest.mark.acceptance
def test_an_unstageable_table_is_named_in_the_warning(engine_module, capsys, monkeypatch):
    """The warning could only quote DuckDB's error before, which names whichever table the failing
    statement REFERENCED - so it pointed at the neighbour rather than at the table that failed."""
    eng = engine_module.DDLEngine(
        "CREATE TABLE carriers (carrier_id VARCHAR PRIMARY KEY, name VARCHAR);"
        "CREATE TABLE flights (flight_id VARCHAR PRIMARY KEY,"
        "  carrier_id VARCHAR REFERENCES carriers(carrier_id));",
        rows=2000,
    )
    # Every CREATE fails, so the retry gives up and both tables reach the warning.
    monkeypatch.setattr(eng, "decl_sql", {t: "CREATE TABLE broken (" for t in eng.schema})
    eng._scratch_schema()

    out = capsys.readouterr().out
    assert "could not stage `flights`" in out, out
    assert "could not stage `carriers`" in out, out
    assert "advisory only" in out
    # The cause claim was retired: ordering is retried until it stops helping, so a failure that
    # survives to this line is not an ordering problem.
    assert "foreign-key order" not in out


@pytest.mark.acceptance
def test_a_problem_hiding_behind_a_dependency_is_reported_not_the_dependency(engine_module):
    """A statement can fail twice for different reasons: first because its FK target does not
    exist yet, then - once everything creatable has been created - because of something else
    entirely, a bad type say. Reporting the first failure then blames a dependency that has since
    been satisfied and never mentions what is actually blocking it.

    Driven through `_blame` directly rather than through DuckDB: which of the two errors a real
    CREATE surfaces first is up to the binder (measured: it reports an unknown TYPE before an
    unknown TABLE), so a DDL fixture would pin DuckDB's internals, not this decision.
    """
    stmts = [
        "CREATE TABLE a (id INTEGER PRIMARY KEY)",
        "CREATE TABLE b (a_id INTEGER REFERENCES a(id), v NUMBER(19))",
    ]
    leftover = [
        (
            stmts[1],
            Exception("Catalog Error: Table with name a does not exist!"),
            Exception("Type with name NUMBER does not exist!"),
        )
    ]

    stmt, failure = engine_module.DDLEngine._blame(leftover, stmts)

    assert stmt == stmts[1]
    assert "NUMBER" in str(failure), "the surviving blocker, not the dependency it hid behind"


@pytest.mark.acceptance
def test_an_empty_enum_placeholder_is_not_a_decision(engine_module, capsys):
    """`profile_skeleton` emits `"col": []` for every domain it could not fill, so a reader who
    pastes the skeleton and fills nothing in has empty lists everywhere. `_enum_values_raw` skips
    those (`if g:`) and falls back to the built-in vocabulary - the domain is still guessed, and
    the line has to keep saying so or it goes quiet for exactly the reader who needs it."""
    ddl = (
        "CREATE TABLE meters (meter_id VARCHAR PRIMARY KEY, unit VARCHAR);"
        "CREATE TABLE ev (id VARCHAR PRIMARY KEY,"
        "  meter_id VARCHAR REFERENCES meters(meter_id), read_at TIMESTAMP);"
    )
    engine_module.DDLEngine(ddl, rows=3000, profile={"enums": {"unit": []}}).report()

    out = capsys.readouterr().out
    assert "meters.unit" in out.split("GUESSED", 1)[1].split("\n")[0], out
