"""Ordinary todos own work progress; init only reports execution outcomes."""

import json
from types import SimpleNamespace

import pytest

from datus.cli.init_context import InitContext
from datus.tools.func_tool.plan_tools import PlanTool
from datus.utils.tool_result_store import ToolResultStore


@pytest.fixture
def run(tmp_path):
    context = InitContext(tmp_path, {}, "Sales and customer guidance")
    yield context
    context.close()


async def describe(context, arguments):
    return {"success": 1, "result": {"columns": [{"name": "id"}]}}


@pytest.mark.asyncio
async def test_metadata_and_dependencies_do_not_change_todo_progress(run):
    plan = PlanTool(None)
    plan.todo_write(json.dumps([{"title": "Order guidance", "content": "Document orders and relevant joins"}]))
    for name in ["orders", "customers", "customers"]:
        await run.invoke("describe_table", describe, None, json.dumps({"table_name": name}))
    assert plan.todo_list().result["items"][0]["status"] == "pending"
    assert run.calls == 2
    plan.todo_update(1, "completed")
    await run.invoke("describe_table", describe, None, '{"table_name":"orders"}')
    assert plan.todo_list().result["items"][0]["status"] == "completed"
    run.finish_execution(SimpleNamespace(success=True))
    assert run.state["status"] == "returned"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_clarification_preserves_work_and_retains_user_answer(run, cancel, tmp_path):
    plan = PlanTool(None)
    plan.todo_write(json.dumps([{"title": "Order guidance", "content": "Document orders"}]))
    plan.todo_update(1, "completed")
    (run.root / "AGENTS.md").write_text("Existing order guidance")

    async def answer(context, args):
        return {"success": 1, "result": [] if cancel else [{"question": "Scope?", "answer": "Only booked refunds"}]}

    await run.invoke("ask_user", answer, None, '{"questions":[{"question":"Scope?"}]}')
    assert plan.todo_list().result["completed"] == 1
    assert (run.root / "AGENTS.md").read_text() == "Existing order guidance"
    run.finish_execution(SimpleNamespace(success=True))
    if cancel:
        assert run.state["status"] == "stopped" and "cancelled" in run.state["stop_reason"]
    else:
        store = ToolResultStore(SimpleNamespace(dir=tmp_path / "archive"))
        assert "Only booked refunds" in run.prompt_context(store)
        assert run.state["status"] == "returned"


@pytest.mark.asyncio
@pytest.mark.parametrize("work", ["empty", "metadata", "files", "partial", "complete"])
async def test_normal_return_reports_execution_independently_of_business_completion(run, work):
    plan = PlanTool(None)
    if work == "metadata":
        await run.invoke("describe_table", describe, None, '{"table_name":"orders"}')
    if work == "files":
        (run.root / "AGENTS.md").write_text("Updated project conventions")
    if work in {"partial", "complete"}:
        plan.todo_write(json.dumps([{"title": "Review docs", "content": "Review project guidance"}]))
        if work == "complete":
            plan.todo_update(1, "completed")
    run.finish_execution(SimpleNamespace(success=True))
    run.close()
    # The same neutral outcome applies to file work, unfinished work and even
    # an empty reply. Business completion belongs to the model's final report.
    assert run.state["status"] == "returned" and run.state["stop_reason"] == ""
    if work == "partial":
        assert plan.todo_list().result["items"][0]["status"] == "pending"


@pytest.mark.parametrize("reason", ["budget", "error", "interrupt", "pending", "no_result"])
def test_execution_failure_overrides_finished_todos_and_preserves_files(run, reason):
    plan = PlanTool(None)
    plan.todo_write(json.dumps([{"title": "Review docs", "content": "Review project guidance"}]))
    plan.todo_update(1, "completed")
    (run.root / "AGENTS.md").write_text("Published guidance")
    result = SimpleNamespace(success=reason not in {"budget", "error"}, error=reason)
    run.finish_execution(
        None if reason == "no_result" else result,
        interrupted=reason == "interrupt",
        pending_input=reason == "pending",
    )
    run.close()
    assert run.state["status"] == "stopped" and run.state["stop_reason"]
    assert plan.todo_list().result["completed"] == 1
    assert (run.root / "AGENTS.md").read_text() == "Published guidance"


def test_large_user_clarifications_are_bounded_and_recoverable(run, tmp_path):
    store = ToolResultStore(SimpleNamespace(dir=tmp_path / "archive"))
    updates = [f"Correction {i}: " + "Preserve the documented rule. " * 30 for i in range(100)]
    run.record_user_updates(updates)
    reminder = run.prompt_context(store)
    assert len(reminder.encode()) < 20000
    context = json.loads(reminder.split("<init_context>\n", 1)[1].split("\n</init_context>", 1)[0])
    page = context["user_clarifications"]
    restored = []
    while True:
        restored.extend(page["result"]["items"])
        if not page["result"]["has_more"]:
            break
        extra = page["result"]["extra"]
        page = store.read(extra["result_id"], extra["section"], extra["next_cursor"])
    assert [item["text"] for item in restored] == list(reversed(updates))
