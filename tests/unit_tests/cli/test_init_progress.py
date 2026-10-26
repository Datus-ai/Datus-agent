"""Coverage is independent from todo steps, metadata observations and runtime outcome."""

import json
from types import SimpleNamespace

import pytest

from datus.cli.init_context import InitContext


@pytest.fixture
def run(tmp_path):
    context = InitContext(tmp_path, {}, "Sales and customer guidance")
    yield context
    context.close()


async def describe(context, arguments):
    return {"success": 1, "result": {"columns": [{"name": "id"}]}}


def finish(run, success=True, **kwargs):
    run.finish_execution(SimpleNamespace(success=success, error="budget exhausted" if not success else None), **kwargs)
    return run.progress()


@pytest.mark.asyncio
async def test_dependency_queries_and_cache_never_expand_selected_coverage(run):
    run.select_scope(["orders"], [], "Other domains", [])
    for name in ["orders", "customers", "customers"]:
        await run.invoke("describe_table", describe, None, json.dumps({"table_name": name}))
    assert run.progress()["selected"] == 1
    assert run.progress()["completed"] == 0
    assert run.progress()["structure_checked"] == 2
    assert run.calls == 2
    run.report_progress("Orders documented", ["orders"], True, "", "")
    await run.invoke("describe_table", describe, None, '{"table_name":"orders"}')
    assert finish(run)["status"] == "completed_scope"
    assert run.state["tables"]["orders"]["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("already_observed", [False, True])
async def test_completion_can_precede_metadata_and_namespace_qualification(run, already_observed):
    if already_observed:
        await run.invoke("describe_table", describe, None, '{"table_name":"sales.orders"}')
    run.report_progress("Existing order docs confirmed", ["orders"], True, "", "")
    await run.invoke("describe_table", describe, None, '{"table_name":"orders","schema_name":"sales"}')
    assert run.progress()["selected"] == 1
    assert run.state["tables"]["sales.orders"]["status"] == "completed"
    run.report_progress("Reviewed", ["orders"], True, "", "")
    assert run.state["selected_tables"] == ["sales.orders"]
    assert finish(run)["status"] == "completed_scope"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_mid_run_clarification_preserves_completed_work_and_scope(run, cancel):
    run.select_scope(["orders", "refunds"], ["README.md"], "", [])
    run.report_progress("Orders done", ["orders"], False, "Refund rules", "Read refund docs")

    async def answer(context, args):
        return {"success": 1, "result": [] if cancel else [{"question": "Scope?", "answer": "Only booked refunds"}]}

    await run.invoke("ask_user", answer, None, '{"questions":[{"question":"Scope?"}]}')
    assert run.progress()["selected"] == 2 and run.progress()["completed"] == 1
    assert run.state["sources"] == ["README.md"]
    if cancel:
        assert finish(run)["status"] == "partial"
        assert "cancelled" in run.state["stop_reason"]
    else:
        assert "Only booked refunds" in run.prompt_context()
        run.select_scope([], [], "Refunds deferred", ["refunds"])
        assert run.progress()["selected"] == 1 and run.progress()["completed"] == 1


@pytest.mark.parametrize(
    "kind", ["empty", "empty_plan", "metadata", "files", "partial", "complete", "error", "interrupt", "plan_mode"]
)
@pytest.mark.asyncio
async def test_outcome_requires_explicit_coverage_report_and_successful_run(run, kind):
    if kind == "empty_plan":
        run.select_scope([], [], "", [])
    elif kind == "metadata":
        run.select_scope(["orders"], [], "", [])
        await run.invoke("describe_table", describe, None, '{"table_name":"orders"}')
    elif kind == "files":
        run.report_progress("Reviewed and updated project conventions in AGENTS.md", [], True, "", "")
    elif kind not in {"empty", "empty_plan", "metadata"}:
        run.select_scope(["orders", "refunds"], [], "", [])
        run.report_progress(
            "Knowledge updated", ["orders"] if kind == "partial" else ["orders", "refunds"], True, "", ""
        )
    status = finish(run, success=kind != "error", interrupted=kind == "interrupt", plan_mode=kind == "plan_mode")
    assert status["status"] == ("completed_scope" if kind in {"files", "complete"} else "partial")
    assert status["coverage"] == "selected"


def test_large_coverage_is_queryable_without_injecting_every_table(run):
    names = [f"orders_{n:05}" for n in range(20000)]
    run.select_scope(names, [], "Other domains", [])
    run.report_progress("Reviewed first topic", names[:3], False, "More topics remain", "Read todo list")
    run.record_user_updates(["Preserve existing finance notes", "Use booked status only"])
    reminder = run.prompt_context()
    assert len(reminder) < 4000 and names[-1] not in reminder
    assert "Use booked status only" in reminder and "Read todo list" in reminder
    last = run.progress(19990, 10)
    assert len(last["selected_items"]) == 10 and not last["selected_has_more"]
    assert last["selected_items"][-1]["table"] == names[-1]
    assert run.progress()["user_context"][0]["text"] == "Preserve existing finance notes"


@pytest.mark.parametrize("tables", [[], ["orders"]])
@pytest.mark.parametrize("scope_complete", [False, True])
def test_explicit_completion_is_independent_of_handoff_wording(run, tables, scope_complete):
    remaining = "None - all selected work has been documented."
    run.select_scope(tables, ["README.md"], "Other domains", [])
    run.report_progress("Updated project guidance with source references", tables, scope_complete, remaining, "")

    progress = finish(run)

    assert progress["status"] == ("completed_scope" if scope_complete else "partial")
    assert progress["remaining"] == remaining
    assert progress["completed"] == len(tables)
