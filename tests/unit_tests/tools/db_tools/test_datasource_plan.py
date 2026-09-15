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
    plan = plan_from_ddl(DDL, rows=50_000, months=12)

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
    plan = plan_from_ddl(DDL, rows=20_000, months=12, end_date="2026-06-30")

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
            plans[name] = plan_from_ddl(ddl_for(name), rows=5000, months=3)
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
