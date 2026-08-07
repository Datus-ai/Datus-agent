# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for submit_task_result.

The validation here is the point of the tool. An orchestrator branches on
``outcome``, so an outcome that arrives without the fields that make it
actionable is worse than no call at all — it looks successful and decides
nothing.
"""

import json

import pytest

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


# ── The path production actually takes ──────────────────────────────────────
#
# Calling the bound method with pydantic instances, as the tests above do,
# exercises a signature nothing invokes at runtime: the invoker built by
# ``trans_to_function_tool`` calls ``method(**args_dict)`` with the raw parsed
# JSON, so nested objects arrive as plain dicts. Every test below goes through
# ``on_invoke_tool`` so that difference cannot hide again.


async def _invoke(tool: TaskResultTool, **args) -> dict:
    return await tool.available_tools()[0].on_invoke_tool(None, json.dumps(args))


@pytest.mark.asyncio
async def test_invoker_accepts_nested_objects_as_dicts():
    tool = TaskResultTool()

    result = await _invoke(
        tool,
        outcome="needs_development",
        summary="Retention by market maker needs a dimension first.",
        gap_reasons=["no market_maker dimension"],
        plan_items=[{"kind": "dimension", "name": "dim_market_maker", "description": "42 addresses"}],
        artifacts=[{"kind": "csv", "ref": "s3://run/1.csv", "title": "Draft"}],
    )

    assert result["success"]
    assert tool.submitted["plan_items"][0]["name"] == "dim_market_maker"
    assert tool.submitted["artifacts"][0]["kind"] == "csv"


@pytest.mark.asyncio
async def test_invoker_accepts_a_nested_array_sent_as_a_json_string():
    """Some models serialise nested arrays rather than nesting them."""
    tool = TaskResultTool()

    result = await _invoke(
        tool,
        outcome="answered",
        summary="Done.",
        artifacts=json.dumps([{"kind": "report", "ref": "rpt_1"}]),
    )

    assert result["success"]
    assert tool.submitted["artifacts"][0]["ref"] == "rpt_1"


@pytest.mark.asyncio
async def test_invoker_reports_a_malformed_item_instead_of_raising():
    """A raise here aborts the whole interaction; an error string lets the model
    correct itself in-turn."""
    tool = TaskResultTool()

    result = await _invoke(
        tool,
        outcome="answered",
        summary="Done.",
        artifacts=[{"ref": "missing-kind"}],
    )

    assert not result["success"]
    assert "artifacts[0]" in result["error"]
    assert tool.submitted is None


@pytest.mark.asyncio
async def test_invoker_still_enforces_the_outcome_contract():
    tool = TaskResultTool()

    result = await _invoke(tool, outcome="needs_development", summary="Something is missing.")

    assert not result["success"]
    assert "gap_reasons" in result["error"]
