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
    assert "80,000 +/-6%" in out, "and what it will be scaled to"
