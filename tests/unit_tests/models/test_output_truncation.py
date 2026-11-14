# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Output-limit handling through the real SDK, LiteLLM transport, and session."""

import asyncio
import json
from unittest.mock import patch

import httpx
import litellm
import pytest
import pytest_asyncio
from agents import RunConfig, Runner, function_tool
from agents.models.interface import ModelTracing
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from openai import AsyncOpenAI

from datus.configuration.agent_config import ModelConfig
from datus.models.observed_model import ObservedResponsesModel
from datus.models.openai_model import OpenAIModel
from datus.models.output_truncation import (
    ModelOutputTruncatedError,
    check_responses_output,
    checked_chat_completion_stream,
)
from datus.models.responses_model import DatusResponsesModel
from datus.tools.func_tool.base import trans_to_function_tool
from datus.tools.registry.contracts import tool_contract

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
@pytest.mark.parametrize("content", [PARTIAL, None])
async def test_legacy_length_returns_received_text_without_automatic_continuation(service, streaming, content):
    model, requests, responses = service
    responses.append({"message": {"content": content, "reasoning_content": REASONING}, "finish": "length"})
    agent = model._build_agent("Review.", str, False, {}, [])
    if streaming:
        result = Runner.run_streamed(agent, "Review.", run_config=RunConfig(tracing_disabled=True))
        async for _ in result.stream_events():
            pass
    else:
        result = await Runner.run(agent, "Review.", run_config=RunConfig(tracing_disabled=True))
    assert result.final_output == (content or "")
    assert result.context_wrapper.usage.total_tokens == 120
    assert len(requests) == 1


def test_legacy_direct_generation_returns_truncated_text():
    model = OpenAIModel(ModelConfig(type="openai", model="yumcode-pro", api_key="test", base_url=ENDPOINT))
    response = litellm.ModelResponse(
        model="yumcode-pro",
        choices=[{"index": 0, "message": {"role": "assistant", "content": PARTIAL}, "finish_reason": "length"}],
    )
    with patch("litellm.completion", return_value=response) as send:
        assert model.generate("Review.") == PARTIAL
    assert send.call_count == 1
    assert "max_tokens" not in send.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("incomplete", [False, True])
async def test_legacy_tool_loop_corrects_arguments_without_repeating_completed_calls(service, streaming, incomplete):
    model, requests, responses = service
    invoked = []

    class WriteProbe:
        @tool_contract(access="write", direct_call=True, category="filesystem_tools")
        def write_probe(self, value: str) -> str:
            """Record a complete write after argument validation."""
            invoked.append(value)
            return "written"

    previous = tool_call('{"value":"previous"}')
    previous["id"] = "previous_write"
    cutoff = tool_call() if incomplete else tool_call('{"value":"complete"}')
    responses.extend(
        [
            {"message": {"tool_calls": [previous]}, "finish": "tool_calls"},
            {"message": {"tool_calls": [cutoff]}, "finish": "length"},
        ]
    )
    if incomplete:
        corrected = tool_call('{"value":"complete"}')
        corrected["id"] = "corrected_write"
        responses.append({"message": {"tool_calls": [corrected]}, "finish": "tool_calls"})
    responses.append({"message": {"content": "Done."}})
    tools = [trans_to_function_tool(WriteProbe().write_probe)]
    if streaming:
        actions = [action async for action in model.generate_with_tools_stream("Build.", tools=tools)]
        assert any(action.output.get("raw_output") == "Done." for action in actions if action.output)
        assert not any(action.action_type == "retry_notification" for action in actions)
    else:
        assert (await model.generate_with_tools("Build.", tools=tools))["content"] == "Done."
    assert invoked == ["previous", "complete"]
    assert len(requests) == (4 if incomplete else 3)
    feedback = next(message for message in requests[2]["messages"] if message.get("tool_call_id") == "write_1")
    expected_feedback = "not executed" if incomplete else "written"
    assert expected_feedback in feedback["content"]


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
    assert long == PARTIAL
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
@pytest.mark.parametrize(
    "streaming,truncated,expected_kind,expected_output",
    [
        (False, False, "response", PARTIAL),
        (False, True, "response", PARTIAL),
        (True, False, "events", PARTIAL),
        (True, True, "ModelBehaviorError", "max_output_tokens"),
    ],
)
async def test_native_responses_status_matches_sdk(streaming, truncated, expected_kind, expected_output):
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
        from agents import ModelSettings

        async def outcome(transport):
            kwargs = {
                "system_instructions": "Review.",
                "input": "Review.",
                "model_settings": ModelSettings(),
                "tools": [write_probe],
                "output_schema": None,
                "handoffs": [],
                "tracing": ModelTracing.DISABLED,
            }
            try:
                if streaming:
                    events = [event.model_dump() async for event in transport.stream_response(**kwargs)]
                    return "events", events
                response = await transport.get_response(**kwargs)
                return "response", [item.model_dump() for item in response.output]
            except Exception as error:
                return type(error).__name__, str(error)

        expected = await outcome(ObservedResponsesModel(model="gpt-test", openai_client=client))
        actual = await outcome(DatusResponsesModel(model="gpt-test", openai_client=client))
        assert actual[0] == expected_kind
        assert json.dumps(expected_output)[1:-1] in json.dumps(actual[1])
        assert actual == expected
        assert invoked == []


def test_responses_other_incomplete_reasons_are_not_output_exhaustion():
    assert check_responses_output({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}) is None
