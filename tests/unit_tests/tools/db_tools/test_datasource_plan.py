# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""The plan a run needs before it writes a generator.

``report()`` has always printed the engine's whole plan, but it is a method on ``DDLEngine``, so
reaching it meant authoring ``data/gen.py`` first. A measured production run (Datus-saas-dev,
trace e2c917f9b380db89efedb580b28e9369) skipped it, tried to divide the row budget across five
tables by hand, could not make the total come out, opened ``ddl_engine.py`` to find the allocator,
and spent 36 turns and 19,885 output tokens in the source without generating a single row.

These tests pin what that one call must answer, and that a broken or absent bundle degrades to a
message rather than an exception escaping the tool.
"""

import sys

import pytest

from datus.tools.db_tools import datasource_plan
from datus.tools.db_tools.datasource_plan import DatasourcePlanError, plan_from_ddl

DDL = """
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name VARCHAR,
    member_level VARCHAR
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    order_no VARCHAR,
    customer_id BIGINT REFERENCES customers(customer_id),
    order_time TIMESTAMP,
    paid_amount DECIMAL(18, 2)
);
CREATE TABLE daily_channel_metrics (
    stat_dt DATE,
    channel VARCHAR,
    impressions BIGINT,
    clicks BIGINT,
    gmv DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
def test_the_plan_answers_the_question_that_sent_a_run_into_the_source():
    """Row allocation per table, without writing a generator and without reading the engine."""
    plan, _skeleton = plan_from_ddl(DDL, rows=50_000, months=12)

    assert "main fact table orders" in plan
    for table in ("orders", "customers", "daily_channel_metrics"):
        assert table in plan
    # The allocation is the thing the run tried to work out by hand.
    assert "rows" in plan.splitlines()[2]
    assert "column semantics" in plan
    assert "knobs: months=12" in plan


@pytest.mark.acceptance
def test_the_plan_needs_no_profile_and_generates_nothing(tmp_path, monkeypatch):
    """Planning must not touch the filesystem: it runs before there is anything to write."""
    monkeypatch.chdir(tmp_path)

    plan_from_ddl(DDL, rows=20_000, months=6)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.acceptance
def test_end_date_pins_the_window():
    plan, _skeleton = plan_from_ddl(DDL, rows=20_000, months=12, end_date="2026-06-30")

    assert "2025-07-01 ~ 2026-06-30" in plan


@pytest.mark.acceptance
@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"ddl": ""}, "ddl is empty"),
        ({"ddl": "   "}, "ddl is empty"),
        ({"ddl": DDL, "rows": 0}, "rows must be positive"),
        ({"ddl": DDL, "months": 0}, "months must be positive"),
        ({"ddl": DDL, "end_date": "30/06/2026"}, "end_date must be YYYY-MM-DD"),
        ({"ddl": "x" * (datasource_plan.MAX_DDL_CHARS + 1)}, "above the"),
    ],
)
def test_bad_input_is_refused_with_a_reason(kwargs, fragment):
    with pytest.raises(DatasourcePlanError) as excinfo:
        plan_from_ddl(**kwargs)

    assert fragment in str(excinfo.value)


@pytest.mark.acceptance
def test_unparseable_ddl_returns_what_the_engine_managed_first():
    """The partial plan usually names the table the engine choked on, so it is worth keeping."""
    with pytest.raises(DatasourcePlanError) as excinfo:
        plan_from_ddl("CREATE TABLE", rows=1000)

    assert "could not plan this DDL" in str(excinfo.value)


@pytest.mark.acceptance
def test_a_missing_bundle_degrades_to_a_message(monkeypatch, tmp_path):
    """A trimmed install ships no skills. That must not look like a broken datasource."""
    monkeypatch.setattr(datasource_plan, "_engine_module", None)
    monkeypatch.setattr(datasource_plan, "skill_scripts_dir", lambda: tmp_path / "nowhere")

    with pytest.raises(DatasourcePlanError) as excinfo:
        plan_from_ddl(DDL)

    assert "not present in this installation" in str(excinfo.value)


@pytest.mark.acceptance
def test_loading_the_engine_leaves_sys_path_alone(monkeypatch):
    """``ddl_engine`` imports ``genlib`` from its own directory.

    That directory goes on ``sys.path`` for the import and must come off again: leaving it there
    would let any later ``import genlib`` anywhere in the process resolve to the skill bundle.
    """
    # Force a real load: a cached module returns before sys.path is ever touched, which would
    # make this test pass without exercising anything.
    monkeypatch.setattr(datasource_plan, "_engine_module", None)
    before = list(sys.path)

    plan_from_ddl(DDL, rows=5000)

    assert sys.path == before
    assert datasource_plan._engine_module.__name__ == "_datus_gen_datasource_engine"


@pytest.mark.acceptance
def test_concurrent_plans_do_not_interleave():
    """``redirect_stdout`` swaps ``sys.stdout`` for the whole process.

    The agent framework can dispatch tool calls concurrently, so two planning calls have to be
    serialised around the capture or they read each other's output - and an out-of-order restore
    leaves the host's stdout pointed at a dead buffer.
    """
    import threading

    def ddl_for(name):
        return (
            f"CREATE TABLE {name}_customers (customer_id BIGINT PRIMARY KEY, customer_name VARCHAR);"
            f"CREATE TABLE {name}_orders (order_id BIGINT PRIMARY KEY, "
            f"customer_id BIGINT REFERENCES {name}_customers(customer_id), "
            "order_time TIMESTAMP, paid_amount DECIMAL(18,2));"
        )

    names = [f"t{i}" for i in range(8)]
    plans, errors = {}, []
    start = threading.Barrier(len(names))

    def run(name):
        try:
            start.wait(timeout=10)
            plans[name] = plan_from_ddl(ddl_for(name), rows=5000, months=3)[0]
        except Exception as e:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert set(plans) == set(names)
    every_table = {f"{other}_orders" for other in names}
    for name, plan in plans.items():
        # Each plan must name its own fact table and no other thread's: one equality states both
        # "the plan arrived" and "nothing bled into it".
        mentioned = {table for table in every_table if table in plan}
        assert mentioned == {f"{name}_orders"}, f"{name}'s plan mentions {sorted(mentioned)}"


@pytest.mark.acceptance
def test_capture_restores_stdout(capsys):
    """Output written after planning must reach the caller's stdout, not a discarded buffer."""
    plan_from_ddl(DDL, rows=5000, months=3)
    sys.stdout.write("back on the real stdout\n")

    assert capsys.readouterr().out == "back on the real stdout\n"


# ---------------------------------------------------------------------------
# The skeleton: what the engine knows, handed back as something to fill in
# ---------------------------------------------------------------------------


SKELETON_DDL = """
CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name VARCHAR,
    member_level VARCHAR,
    city VARCHAR
);
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customers(customer_id),
    order_time TIMESTAMP,
    order_status VARCHAR,   -- pending / paid / shipped
    channel VARCHAR,
    original_amount DECIMAL(18, 2),
    discount_amount DECIMAL(18, 2),
    paid_amount DECIMAL(18, 2)
);
CREATE TABLE daily_channel_metrics (
    stat_dt DATE,
    channel VARCHAR,
    impressions BIGINT,
    clicks BIGINT,
    sessions BIGINT,
    purchasers BIGINT,
    gmv DECIMAL(18, 2)
);
"""


@pytest.mark.acceptance
def test_the_skeleton_is_valid_python_that_defines_a_profile():
    """It is meant to be copied into `gen.py`, so it has to parse and evaluate as written."""
    import ast

    _plan, skeleton = plan_from_ddl(SKELETON_DDL, rows=80_000, months=17)

    ast.parse(skeleton)
    namespace = {}
    exec(compile(skeleton, "<skeleton>", "exec"), namespace)  # noqa: S102 - the engine wrote it
    assert isinstance(namespace["PROFILE"], dict)
    assert "calendar" in namespace["PROFILE"]


@pytest.mark.acceptance
def _engine_from_skeleton(ddl, rows, months):
    """Build an engine on the skeleton exactly as a caller would: copy it out, run it unedited."""
    import sys

    from datus.tools.db_tools.datasource_plan import load_engine

    _plan, skeleton = plan_from_ddl(ddl, rows=rows, months=months)
    namespace = {}
    exec(compile(skeleton, "<skeleton>", "exec"), namespace)  # noqa: S102 - the engine wrote it

    scripts = str(load_engine().__file__).rsplit("/", 1)[0]
    sys.path.insert(0, scripts)
    try:
        return load_engine().DDLEngine(ddl, rows=rows, months=months, seed=42, profile=namespace["PROFILE"])
    finally:
        sys.path.remove(scripts)


@pytest.mark.acceptance
def test_copying_the_skeleton_unfilled_runs_and_is_believable(tmp_path):
    """An empty slot is a decision, and the skeleton used to open with ten of them.

    Its funnel ratios were `(0.0, 0.0)` placeholders, so the first runnable profile was whatever the
    caller designed against a blank page - 16% of one measured 676-second turn went on deriving
    those ten numbers by hand. Prefilled, the skeleton copied out unedited has to produce a funnel
    that converges, or the prefill is not worth having.
    """
    import duckdb

    engine = _engine_from_skeleton(SKELETON_DDL, rows=30_000, months=6)

    assert not engine.precheck(strict=False)[0], "the skeleton must not need editing to be valid"

    out = tmp_path / "skeleton.duckdb"
    engine.generate(str(out), verbose=False)
    con = duckdb.connect(str(out))
    try:
        rows, ctr, cvr = con.execute(
            "SELECT count(*), 1.0*sum(clicks)/sum(impressions), 1.0*sum(purchasers)/sum(sessions) "
            "FROM daily_channel_metrics"
        ).fetchone()
        broken = con.execute(
            "SELECT count(*) FROM daily_channel_metrics "
            "WHERE clicks > impressions OR sessions > clicks OR purchasers > sessions"
        ).fetchone()[0]
    finally:
        con.close()

    assert broken == 0, f"{broken} of {rows} funnel rows do not narrow"
    assert 0.005 < ctr < 0.12, f"click-through rate {ctr:.4f} is not believable"
    assert 0.002 < cvr < 0.10, f"conversion rate {cvr:.4f} is not believable"


@pytest.mark.acceptance
def test_the_skeleton_leaves_only_the_calendar_blank():
    """Prefilling is for what the engine can know. Promotion windows are a business fact.

    Inventing them would put a Double 11 in a dataset that has no such thing, which is worse than
    an empty list the caller fills in.
    """
    _plan, skeleton = plan_from_ddl(SKELETON_DDL, rows=80_000, months=17)

    assert "(0.0, 0.0)" not in skeleton, "a placeholder ratio is a decision the caller has to make"
    assert '"promos": [' in skeleton, "the one section the engine must not guess stays empty"


@pytest.mark.acceptance
def test_a_hand_written_zero_ratio_is_still_refused():
    """The skeleton no longer emits it, but it multiplies the base to zero wherever it comes from."""
    import sys

    from datus.tools.db_tools.datasource_plan import load_engine

    scripts = str(load_engine().__file__).rsplit("/", 1)[0]
    sys.path.insert(0, scripts)
    try:
        engine = load_engine().DDLEngine(
            SKELETON_DDL,
            rows=80_000,
            months=17,
            profile={"derive": {"daily_channel_metrics.clicks": {"from": "impressions", "ratio": (0.0, 0.0)}}},
        )
    finally:
        sys.path.remove(scripts)

    errors, _warnings = engine.precheck(strict=False)

    assert any("derive[daily_channel_metrics.clicks]" in e and "column of zeros" in e for e in errors), errors


@pytest.mark.acceptance
def test_the_skeleton_carries_what_the_engine_inferred():
    _plan, skeleton = plan_from_ddl(SKELETON_DDL, rows=80_000, months=17)

    # Domains it extracted from the DDL comments, so the caller does not restate them.
    assert "order_status: pending, paid, shipped" in skeleton
    # Enum columns with no domain, which would otherwise be filled with CODE1..CODEn.
    assert '"member_level": []' in skeleton
    assert '"city": []' in skeleton
    # The funnel chain, in column order, with the ratios left to the caller.
    assert '"daily_channel_metrics.clicks": {"from": "impressions"' in skeleton
    assert '"daily_channel_metrics.sessions": {"from": "clicks"' in skeleton
    # An amount takes the nearest count before it. Chaining it by raw column order gave
    # "attributed_revenue from ad_spend", which means nothing.
    assert '"daily_channel_metrics.gmv": {"from": "purchasers"' in skeleton


@pytest.mark.acceptance
def test_the_skeleton_says_what_it_leaves_out_and_why():
    """The two keys a production run reached for and should not have."""
    _plan, skeleton = plan_from_ddl(SKELETON_DDL, rows=80_000, months=17)

    assert "formulas" in skeleton and "do not restate them" in skeleton
    assert "table_rows" in skeleton and "calibration" in skeleton


@pytest.mark.acceptance
def test_the_skeleton_invents_no_domain_it_could_not_read():
    """A guessed enum is worse than an empty slot: it reads as inferred and is fiction."""
    _plan, skeleton = plan_from_ddl(SKELETON_DDL, rows=80_000, months=17)

    member_level_line = next(line for line in skeleton.splitlines() if '"member_level"' in line)
    assert member_level_line.strip() == '"member_level": [],'


@pytest.mark.acceptance
def test_a_quoted_identifier_does_not_break_the_skeleton():
    """Identifiers come from the caller's DDL, where DuckDB allows quotes inside a quoted name.

    Interpolated raw, `it's_type` closed the string early and the skeleton would not parse - and
    the skeleton exists to be copied into `gen.py` and run.
    """
    import ast

    ddl = (
        'CREATE TABLE metrics ("stat_dt" DATE, "it\'s_type" VARCHAR, impressions BIGINT, '
        'clicks BIGINT, sessions BIGINT, "gm\'v" DECIMAL(18,2));'
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, "
        "paid_amount DECIMAL(18,2));"
    )

    _plan, skeleton = plan_from_ddl(ddl, rows=9000, months=6)

    ast.parse(skeleton)
    namespace = {}
    exec(compile(skeleton, "<skeleton>", "exec"), namespace)  # noqa: S102 - the engine wrote it
    assert "it's_type" in namespace["PROFILE"]["enums"]


@pytest.mark.acceptance
def test_every_metric_table_contributes_its_funnel():
    """Stopping after the first one dropped the second table's chain entirely.

    Emitting a second `"derive"` block instead would be a duplicate key that silently overwrites
    the first, so they go into one mapping.
    """
    ddl = (
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, order_time TIMESTAMP, "
        "paid_amount DECIMAL(18,2));"
        "CREATE TABLE channel_daily (stat_dt DATE, channel VARCHAR, impressions BIGINT, "
        "clicks BIGINT, sessions BIGINT, gmv DECIMAL(18,2));"
        "CREATE TABLE campaign_daily (stat_dt DATE, campaign VARCHAR, impressions BIGINT, "
        "clicks BIGINT, purchasers BIGINT, revenue DECIMAL(18,2));"
    )

    _plan, skeleton = plan_from_ddl(ddl, rows=12_000, months=6)

    assert skeleton.count('"derive": {') == 1, "one mapping, or the later block wins and the first is lost"
    namespace = {}
    exec(compile(skeleton, "<skeleton>", "exec"), namespace)  # noqa: S102 - the engine wrote it
    derive = namespace["PROFILE"]["derive"]
    assert {"channel_daily.clicks", "channel_daily.sessions"} <= set(derive)
    assert {"campaign_daily.clicks", "campaign_daily.purchasers"} <= set(derive)


# --------------------------------------------------------------------------- planning under a profile


def test_a_profile_changes_the_plan(capsys):
    """The profile used to be deliberately not an argument, and two measured runs answered "what
    will the engine do with my per_parent / dim_rows" in a 60,000-token design turn instead. The
    plan under a candidate profile is the same 0.3-second call."""
    bare, _ = plan_from_ddl(DDL, rows=20_000, months=6)
    under, _ = plan_from_ddl(DDL, rows=20_000, months=6, profile={"trend_mom": 0.0})

    assert bare != under, "the profile has to reach the engine"
    assert "profile validated" in under or "pre-check" in under, under


def test_the_tool_accepts_the_profile_as_json_or_as_the_gen_py_literal():
    from datus.tools.func_tool.database import _parse_profile_text

    assert _parse_profile_text('{"trend_mom": 0.02}') == {"trend_mom": 0.02}
    literal = '{"columns": {"t.c": {"range": (9, 320)}}, "per_parent": {"d": 0.25},}'
    assert _parse_profile_text(literal) == {"columns": {"t.c": {"range": (9, 320)}}, "per_parent": {"d": 0.25}}
    with pytest.raises((ValueError, SyntaxError)):
        _parse_profile_text("__import__('os').system('id')")
