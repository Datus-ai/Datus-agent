# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for submit_task_result.

The validation here is the point of the tool. An orchestrator branches on
``outcome``, so an outcome that arrives without the fields that make it
actionable is worse than no call at all — it looks successful and decides
nothing.
"""

from datus.tools.func_tool.task_result_tools import PlanItem, TaskArtifact, TaskResultTool


def test_answered_records_summary_and_artifacts():
    tool = TaskResultTool()

    result = tool.submit_task_result(
        outcome="answered",
        summary="Semantic layer already had perp_notional_volume; produced last week by venue.",
        artifacts=[TaskArtifact(kind="csv", ref="s3://run/1.csv", title="Weekly volume")],
    )

    assert result.success
    assert tool.submitted["outcome"] == "answered"
    assert tool.submitted["artifacts"][0]["kind"] == "csv"


def test_needs_development_requires_both_halves():
    """gap_reasons and plan_items are two halves of one judgement — "because A
    is missing, build B" — and the caller renders them as a pair."""
    tool = TaskResultTool()

    missing_plan = tool.submit_task_result(
        outcome="needs_development",
        summary="No market-maker dimension exists.",
        gap_reasons=["no dim_market_maker"],
    )
    assert not missing_plan.success
    assert "plan_items" in missing_plan.error

    missing_gap = tool.submit_task_result(
        outcome="needs_development",
        summary="Need to build a few things.",
        plan_items=[PlanItem(kind="dimension", name="dim_market_maker")],
    )
    assert not missing_gap.success
    assert "gap_reasons" in missing_gap.error

    assert tool.submitted is None


def test_needs_development_accepted_with_both():
    tool = TaskResultTool()

    result = tool.submit_task_result(
        outcome="needs_development",
        summary="Retention by market maker needs a dimension and a backfill first.",
        gap_reasons=["no market_maker dimension", "counterparty is a raw address"],
        plan_items=[
            PlanItem(kind="dimension", name="dim_market_maker", description="42 maker addresses"),
            PlanItem(kind="metric", name="mm_taker_retention_30d"),
        ],
        estimate="1.5 person-days",
    )

    assert result.success
    assert [p["kind"] for p in tool.submitted["plan_items"]] == ["dimension", "metric"]
    assert tool.submitted["estimate"] == "1.5 person-days"


def test_blocked_requires_gap_reasons():
    """Blocked hands the request to a human, which is a real outcome — but only
    if it says why."""
    tool = TaskResultTool()

    result = tool.submit_task_result(outcome="blocked", summary="Cannot help with this.")

    assert not result.success
    assert "gap_reasons" in result.error


def test_blocked_needs_no_plan():
    tool = TaskResultTool()

    result = tool.submit_task_result(
        outcome="blocked",
        summary="The billing warehouse is not connected to this project.",
        gap_reasons=["no datasource bound for billing"],
    )

    assert result.success
    assert tool.submitted["plan_items"] == []


def test_empty_summary_rejected():
    """The summary is what the caller reads instead of the transcript; a blank
    one silently drops everything the run learned."""
    tool = TaskResultTool()

    assert not tool.submit_task_result(outcome="answered", summary="   ").success


def test_result_tells_the_model_to_stop():
    """The caller stops the run on seeing this call, so anything said afterwards
    is discarded — better to say so than to let a closing paragraph be written
    and thrown away."""
    tool = TaskResultTool()

    result = tool.submit_task_result(outcome="answered", summary="Done.")

    assert "Stop here" in result.result["note"]


def test_available_tools_exposes_one_function():
    assert len(TaskResultTool().available_tools()) == 1
