"""Parallel SDK tool results must reach presentation before slow siblings finish."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agents import Agent, RunConfig, RunContextWrapper, RunHooks, RunItemStreamEvent, ToolCallItem, function_tool
from agents.run_internal.agent_bindings import bind_public_agent
from agents.run_internal.run_steps import ToolRunFunction
from agents.run_internal.tool_execution import execute_function_tool_calls
from openai.types.responses import ResponseFunctionToolCall

from datus.api.services.action_sse_converter import _build_tool_result_content
from datus.models.tool_result_stream import ToolResultStream
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

ERROR = "error_code=500006, Unknown column 'missing_join_key' in 'on clause'"


def call_event(agent, name):
    return RunItemStreamEvent(
        name="tool_called",
        item=ToolCallItem(
            agent=agent,
            raw_item=ResponseFunctionToolCall(type="function_call", name=name, call_id=name, arguments="{}"),
        ),
    )


@pytest.mark.asyncio
async def test_real_sdk_fast_failure_streams_before_sibling_and_is_not_duplicated():
    release = asyncio.Event()
    hooks = ToolResultStream(RunHooks())

    @function_tool
    async def fast_failure() -> dict:
        return {"success": 0, "error": ERROR, "result": None}

    @function_tool
    async def slow_query() -> str:
        await release.wait()
        return "rows"

    agent = Agent(name="test", tools=[fast_failure, slow_query], hooks=hooks)
    run = SimpleNamespace(is_complete=False, cancel=Mock())

    async def sdk_events():
        for tool in agent.tools:
            yield call_event(agent, tool.name)
        results, _, _ = await execute_function_tool_calls(
            bindings=bind_public_agent(agent),
            tool_runs=[ToolRunFunction(call_event(agent, tool.name).item.raw_item, tool) for tool in agent.tools],
            hooks=RunHooks(),
            context_wrapper=RunContextWrapper(None),
            config=RunConfig(tracing_disabled=True),
        )
        run.is_complete = True
        for result in results:
            yield RunItemStreamEvent(name="tool_output", item=result.run_item)

    events = hooks.stream_events(sdk_events(), run)
    try:
        first = await anext(events)
        second = await anext(events)
        output = await asyncio.wait_for(anext(events), timeout=1)
        assert first.item.type == second.item.type == "tool_call_item"
        assert not release.is_set()
        assert output.item.output == {"success": 0, "error": ERROR, "result": None}
        action = ActionHistory(
            action_id="complete_fast_failure",
            role=ActionRole.TOOL,
            action_type="fast_failure",
            input={"function_name": "fast_failure"},
            output={"raw_output": output.item.output},
            status=ActionStatus.FAILED,
        )
        payload = _build_tool_result_content(action)[0].payload
        assert payload["result"]["error"] == ERROR
        release.set()
        remaining = [event async for event in events]
        assert [event.item.raw_item["call_id"] for event in remaining] == ["slow_query"]
    finally:
        release.set()
        await events.aclose()


@pytest.mark.asyncio
async def test_result_before_call_is_ordered_and_close_cancels_sdk_producer():
    hooks = ToolResultStream()
    agent = Agent(name="test")
    hooks._completed(SimpleNamespace(tool_call_id="fast"), agent, {"success": 0, "error": ERROR})
    closed = asyncio.Event()
    run = SimpleNamespace(is_complete=False, cancel=Mock())

    async def sdk_events():
        try:
            yield call_event(agent, "fast")
            await asyncio.Event().wait()
        finally:
            closed.set()

    events = hooks.stream_events(sdk_events(), run)
    assert (await anext(events)).item.type == "tool_call_item"
    assert (await anext(events)).item.output["error"] == ERROR
    await events.aclose()
    assert closed.is_set()
    run.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_sdk_exception_does_not_discard_an_already_completed_result():
    hooks = ToolResultStream()
    agent = Agent(name="test")
    run = SimpleNamespace(is_complete=True)

    async def sdk_events():
        yield call_event(agent, "fast")
        hooks._completed(SimpleNamespace(tool_call_id="fast"), agent, {"success": 0, "error": ERROR})
        raise ValueError("sibling failed")

    events = hooks.stream_events(sdk_events(), run)
    await anext(events)
    assert (await anext(events)).item.output["error"] == ERROR
    with pytest.raises(ValueError, match="sibling failed"):
        await anext(events)


@pytest.mark.asyncio
async def test_hook_delegation_and_failed_postprocessing_are_reported():
    names = ("on_start", "on_end", "on_handoff", "on_llm_start", "on_llm_end", "on_tool_start")
    delegate = SimpleNamespace(**{name: AsyncMock() for name in names})
    delegate.on_tool_end = AsyncMock(side_effect=ValueError("validation failed"))
    hooks = ToolResultStream(delegate)
    for name in names:
        await getattr(hooks, name)("context", "agent")
        getattr(delegate, name).assert_awaited_once_with("context", "agent")
    context = SimpleNamespace(tool_call_id="fast")
    with pytest.raises(ValueError, match="validation failed"):
        await hooks.on_tool_end(context, Agent(name="test"), None, "data")
    _, event = hooks.queue.get_nowait()
    assert event.item.output == {"success": 0, "error": "validation failed", "result": None}


@pytest.mark.asyncio
async def test_sdk_formatted_exception_retains_error_state(monkeypatch):
    monkeypatch.setattr(
        "datus.models.tool_result_stream.get_current_span",
        lambda: SimpleNamespace(error={"message": "database unavailable"}),
    )
    hooks = ToolResultStream()
    await hooks.on_tool_end(SimpleNamespace(tool_call_id="fast"), Agent(name="test"), None, "formatted text")
    _, event = hooks.queue.get_nowait()
    assert event.item.output == {"success": 0, "error": "database unavailable", "result": None}


@pytest.mark.asyncio
async def test_unmatched_sdk_output_and_non_tool_events_are_preserved():
    hooks = ToolResultStream()
    agent = Agent(name="test")
    hooks._completed(SimpleNamespace(), agent, "ignored")
    hooks._completed(SimpleNamespace(tool_call_id="unmatched"), agent, "orphan")
    marker = SimpleNamespace(type="agent_updated_stream_event")

    async def sdk_events():
        yield marker

    events = [event async for event in hooks.stream_events(sdk_events(), SimpleNamespace(is_complete=True))]
    assert events[0] is marker
    assert events[1].item.output == "orphan"
