# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Output-limit handling through the real SDK, LiteLLM transport, and session."""

import asyncio
import json
from pathlib import Path

import httpx
import litellm
import pytest
import pytest_asyncio
from agents import Agent, RunConfig, Runner, function_tool
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from openai import AsyncOpenAI

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.models.output_truncation import (
    ModelOutputTruncatedError,
    check_responses_output,
    checked_chat_completion_stream,
)
from datus.models.responses_model import DatusResponsesModel
from datus.models.sqlite_session import DatusSQLiteSession
from datus.schemas.action_history import ActionHistoryManager, ActionStatus

pytestmark = pytest.mark.component
ENDPOINT = "https://models.example.test/v1"
PARTIAL = "Confirmed issue 1.\nUnfinished issue"
REASONING = "Private reasoning must never become the answer."


@pytest_asyncio.fixture
async def service(monkeypatch):
    requests = []
    responses = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        spec = responses.pop(0)
        if callable(spec):
            spec = spec(body)
        message = {"role": "assistant", **spec.get("message", {})}
        finish = spec.get("finish", "stop")
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        base = {"id": f"response_{len(requests)}", "created": 1, "model": "glm-5.3-flashx"}
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    **base,
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage,
                },
            )

        chunks = []
        for key in ("reasoning_content", "content", "tool_calls"):
            if value := message.get(key):
                if key == "tool_calls":
                    value = [{"index": index, **call} for index, call in enumerate(value)]
                chunks.append(
                    {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {key: value}, "finish_reason": None}],
                    }
                )
        finish_chunk = {
            **base,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        }
        if spec.get("usage_on_finish"):
            finish_chunk["usage"] = usage
        chunks.append(finish_chunk)
        if not spec.get("usage_on_finish") and not spec.get("omit_usage"):
            chunks.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", type(litellm.in_memory_llm_clients_cache)())
        model = OpenAIModel(
            ModelConfig(type="openai", model="glm-5.3-flashx", api_key="test-key", base_url=ENDPOINT, retry_interval=0)
        )
        monkeypatch.setattr(model, "_build_run_config", lambda **kwargs: RunConfig(tracing_disabled=True))
        yield model, requests, responses
    await GLOBAL_LOGGING_WORKER.stop()
    # SDK stream generators schedule their finalizers on the event loop.
    # Let those run before pytest closes this test's loop.
    await asyncio.sleep(0)


def tool_call(arguments='{"value":"unfinished'):
    return {"id": "write_1", "type": "function", "function": {"name": "write_probe", "arguments": arguments}}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", ["text", "reasoning", "tool"])
async def test_sdk_rejects_length_before_tool_execution(service, streaming, kind):
    model, requests, responses = service
    invoked = []

    @function_tool
    def write_probe(value: str) -> str:
        """Record a write that must only execute with a complete model response."""
        invoked.append(value)
        return "written"

    message = {"reasoning_content": REASONING, "content": PARTIAL if kind == "text" else None}
    if kind == "tool":
        message["tool_calls"] = [tool_call()]
    responses.append({"message": message, "finish": "length"})
    agent = model._build_agent("Review the skill.", str, False, {}, [write_probe])
    with pytest.raises(ModelOutputTruncatedError) as caught:
        if streaming:
            result = Runner.run_streamed(agent, "Review.", run_config=RunConfig(tracing_disabled=True))
            async for _ in result.stream_events():
                pass
        else:
            await Runner.run(agent, "Review.", run_config=RunConfig(tracing_disabled=True))
    assert caught.value.partial_response == (PARTIAL if kind == "text" else "")
    assert caught.value.usage.total_tokens == 120
    assert REASONING not in str(caught.value)
    assert invoked == []
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_stream_retains_completed_tools_partial_answer_usage_and_resume(service, tmp_path):
    model, requests, responses = service
    invoked = []

    @function_tool
    def write_probe(value: str) -> str:
        """Record a completed write."""
        invoked.append(value)
        return "written"

    responses.extend(
        [
            {"message": {"tool_calls": [tool_call('{"value":"complete"}')]}, "finish": "tool_calls"},
            {"message": {"content": PARTIAL, "reasoning_content": REASONING}, "finish": "length"},
            {"message": {"content": "Review complete."}},
        ]
    )
    session = DatusSQLiteSession(create_tables=True, session_id="review", db_path=str(tmp_path / "review.db"))
    manager = ActionHistoryManager()
    try:
        with pytest.raises(ModelOutputTruncatedError) as caught:
            async for _ in model.generate_with_tools_stream(
                "Review.", tools=[write_probe], session=session, action_history_manager=manager
            ):
                pass
        assert caught.value.partial_response == PARTIAL
        assert caught.value.usage.total_tokens == 240
        assert invoked == ["complete"]
        assert len(requests) == 2  # max_retry must not restart an output-limited run.
        assert not any(action.action_type == "retry_notification" for action in manager.get_actions())
        history = await session.get_items()
        calls = [item for item in history if item.get("type") == "function_call"]
        outputs = [item for item in history if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in calls] == [item["call_id"] for item in outputs] == ["write_1"]
        assert history[-1]["content"] == PARTIAL
        assert (await session.get_turn_usage(1))["total_tokens"] == 240
        actions = [
            action
            async for action in model.generate_with_tools_stream(
                "Continue the review.",
                tools=[write_probe],
                session=session,
                action_history_manager=ActionHistoryManager(),
            )
        ]
        assert any(action.output.get("raw_output") == "Review complete." for action in actions if action.output)
        assert all(action.status != ActionStatus.FAILED for action in actions)
        assert any(message.get("content") == PARTIAL for message in requests[-1]["messages"])
        assert invoked == ["complete"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_concurrent_streams_on_one_model_keep_termination_state_local(service):
    model, requests, responses = service

    def reply(body):
        is_long = body["messages"][-1]["content"] == "long"
        return {"message": {"content": PARTIAL if is_long else "done"}, "finish": "length" if is_long else "stop"}

    responses.extend([reply, reply])
    agent = model._build_agent("Review.", str, False, {}, [])

    async def run(prompt):
        result = Runner.run_streamed(agent, prompt, run_config=RunConfig(tracing_disabled=True))
        async for _ in result.stream_events():
            pass
        return result.final_output

    long, short = await asyncio.gather(run("long"), run("short"), return_exceptions=True)
    assert isinstance(long, ModelOutputTruncatedError)
    assert long.partial_response == PARTIAL
    assert short == "done"
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_empty_content_without_length_is_not_classified_as_truncation():
    async def chunks():
        yield {"choices": [{"index": 0, "delta": {"reasoning_content": REASONING}, "finish_reason": None}]}
        yield {"choices": [], "usage": {"completion_tokens": 128000}}

    assert len([chunk async for chunk in checked_chat_completion_stream(chunks())]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("tail_error", [False, True])
async def test_length_chunk_content_and_missing_usage_survive_tail_failure(tail_error):
    closed = []

    async def chunks():
        try:
            yield {"choices": [{"index": 0, "delta": {"content": PARTIAL}, "finish_reason": "length"}]}
            if tail_error:
                raise httpx.RemoteProtocolError("Usage-only tail disconnected")
        finally:
            closed.append(True)

    with pytest.raises(ModelOutputTruncatedError) as caught:
        async for _ in checked_chat_completion_stream(chunks()):
            pass
    assert caught.value.partial_response == PARTIAL
    assert caught.value.usage.total_tokens == 0  # Missing provider usage is not estimated.
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_normal_stop_with_reasoning_and_no_answer_is_not_truncation(service, streaming):
    model, requests, responses = service
    responses.append({"message": {"content": None, "reasoning_content": REASONING}})
    agent = model._build_agent("Review.", str, False, {}, [])
    if streaming:
        result = Runner.run_streamed(agent, "Review.", run_config=RunConfig(tracing_disabled=True))
        async for _ in result.stream_events():
            pass
    else:
        result = await Runner.run(agent, "Review.", run_config=RunConfig(tracing_disabled=True))
    assert result.final_output == ""
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("omit_usage", [False, True])
async def test_glm_finish_chunk_retains_reported_usage_not_litellm_estimates(service, omit_usage):
    model, requests, responses = service
    responses.append(
        {
            "message": {"content": None, "reasoning_content": REASONING},
            "finish": "length",
            "usage_on_finish": not omit_usage,
            "omit_usage": omit_usage,
        }
    )
    with pytest.raises(ModelOutputTruncatedError) as caught:
        async for _ in model.generate_with_tools_stream("Review."):
            pass
    assert caught.value.partial_response == ""
    assert caught.value.usage.total_tokens == (0 if omit_usage else 120)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
async def test_native_responses_status_through_sdk(streaming, truncated, tmp_path):
    invoked = []

    @function_tool
    def write_probe(value: str) -> str:
        """Record a completed write."""
        invoked.append(value)
        return "written"

    def respond(request):
        assert request.url.path == "/v1/responses"
        output = [
            {
                "id": "message_1",
                "type": "message",
                "role": "assistant",
                "status": "incomplete" if truncated else "completed",
                "content": [{"type": "output_text", "text": PARTIAL, "annotations": []}],
            }
        ]
        if truncated:
            output.append(
                {
                    "id": "function_1",
                    "type": "function_call",
                    "call_id": "write_1",
                    "name": "write_probe",
                    "arguments": '{"value":"complete but response truncated"}',
                }
            )
        response = {
            "id": "response_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-test",
            "status": "incomplete" if truncated else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        }
        if not streaming:
            return httpx.Response(200, json=response)
        event_type = "response.incomplete" if truncated else "response.completed"
        events = [
            {"type": "response.output_item.done", "sequence_number": index, "output_index": index, "item": item}
            for index, item in enumerate(output)
        ]
        events.append({"type": event_type, "sequence_number": len(events), "response": response})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = AsyncOpenAI(api_key="test-key", base_url=ENDPOINT, http_client=http_client)
        agent = Agent(
            name="review",
            model=DatusResponsesModel(model="gpt-test", openai_client=client),
            tools=[write_probe],
        )
        session = DatusSQLiteSession(create_tables=True, session_id="native", db_path=str(tmp_path / "native.db"))

        async def run():
            if streaming:
                result = Runner.run_streamed(
                    agent, "Review.", session=session, run_config=RunConfig(tracing_disabled=True)
                )
                async for _ in result.stream_events():
                    pass
                return result
            return await Runner.run(agent, "Review.", session=session, run_config=RunConfig(tracing_disabled=True))

        try:
            if truncated:
                with pytest.raises(ModelOutputTruncatedError) as caught:
                    await run()
                assert caught.value.partial_response == PARTIAL
                assert caught.value.usage.total_tokens == 120
                assert all(item.get("type") != "function_call" for item in await session.get_items())
            else:
                assert (await run()).final_output == PARTIAL
            assert invoked == []
        finally:
            session.close()


def test_responses_other_incomplete_reasons_are_not_output_exhaustion():
    check_responses_output({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "reasoning", "tool"])
async def test_node_and_parent_receive_failed_partial_result(service, real_agent_config, monkeypatch, kind):
    from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
    from datus.models.base import LLMBaseModel
    from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput
    from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

    model, requests, responses = service
    message = {"content": PARTIAL if kind == "text" else None, "reasoning_content": REASONING}
    if kind == "tool":
        call = tool_call()
        call["function"] = {
            "name": "write_file",
            "arguments": '{"path":".datus/skills/unwanted/SKILL.md","content":"unfinished',
        }
        message["tool_calls"] = [call]
    responses.append({"message": message, "finish": "length"})
    monkeypatch.setattr(LLMBaseModel, "create_model", lambda *args, **kwargs: model)
    node = SkillCreatorAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SkillCreatorNodeInput(user_message="Review this skill.")
    actions = [action async for action in node.execute_stream(ActionHistoryManager())]

    assert node.result.success is False
    assert node.result.response == (PARTIAL if kind == "text" else "")
    assert node.result.tokens_used == 120
    assert "truncated" in node.result.error
    assert actions[-1].status == ActionStatus.FAILED
    assert len(requests) == 1
    assert not (Path(node._resolve_workspace_root()) / ".datus/skills/unwanted/SKILL.md").exists()
    converted = SubAgentTaskTool(agent_config=real_agent_config)._convert_to_func_result(
        node.result.model_dump(), session_id=node.session_id
    )
    assert converted.success == 0
    assert converted.result["response"] == node.result.response
    assert converted.result["tokens_used"] == 120
    assert converted.result["session_id"] == node.session_id
