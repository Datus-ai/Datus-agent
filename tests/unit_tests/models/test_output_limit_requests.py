# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Output limits reach direct, Chat Completions and Responses API requests."""

import json
from unittest.mock import patch

import httpx
import litellm
import pytest
from agents import RunConfig
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.schemas.action_history import ActionHistoryManager

pytestmark = pytest.mark.component


@pytest.mark.asyncio
@pytest.mark.parametrize("output_limit", [None, 512, 4000], ids=["default", "override", "summary"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_wire_request_sends_output_limit(monkeypatch, output_limit, streaming):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        response = {
            "id": "response_1",
            "created": 1,
            "model": "glm-5.3-flashx",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if not streaming:
            return httpx.Response(
                200,
                json={
                    **response,
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Reviewed."}, "finish_reason": "stop"}
                    ],
                },
            )
        chunk = {
            **response,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Reviewed."}, "finish_reason": "stop"}],
        }
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        )

    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="glm-5.3-flashx",
            api_key="test",
            base_url="https://models.example.test/v1",
            max_tokens=1024,
        )
    )
    monkeypatch.setattr(model, "_build_run_config", lambda **kwargs: RunConfig(tracing_disabled=True))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", type(litellm.in_memory_llm_clients_cache)())
        try:
            if streaming:
                actions = [
                    action
                    async for action in model.generate_with_tools_stream(
                        "Review.", action_history_manager=ActionHistoryManager(), max_tokens=output_limit
                    )
                ]
                assert any(a.output and a.output.get("raw_output") == "Reviewed." for a in actions)
            elif output_limit == 4000:
                result = await model.summarize_items(
                    [{"role": "user", "content": "Earlier project findings."}],
                    instruction="Summarize the project context.",
                    prompt="Keep the findings needed to continue.",
                    max_tokens=4000,
                )
                assert result["content"] == "Reviewed."
            else:
                assert (await model.generate_with_tools("Review.", max_tokens=output_limit))["content"] == "Reviewed."
        finally:
            await GLOBAL_LOGGING_WORKER.stop()
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == (1024 if output_limit is None else output_limit)
    assert model.request_capacity().default_output == 1024


def test_direct_generation_sends_explicit_output_limit():
    model = OpenAIModel(
        ModelConfig(
            type="openai",
            model="private-deployment",
            api_key="test",
            base_url="https://models.example.test/v1",
            max_tokens=1024,
        )
    )
    response = litellm.ModelResponse(
        model="private-deployment",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Reviewed."}}],
    )
    with patch("litellm.completion", return_value=response) as send:
        assert model.generate("Review.", max_tokens=512) == "Reviewed."
    assert send.call_args.kwargs["max_tokens"] == 512
    assert model.request_capacity().default_output == 1024


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("output_limit", [None, 512, 4000], ids=["default", "override", "summary"])
async def test_native_responses_sends_output_limit_with_real_sdk(monkeypatch, streaming, output_limit):
    from openai import AsyncOpenAI

    from datus.models.responses_model import DatusResponsesModel

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        response = {
            "id": "response_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-test",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "message_1",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Reviewed.", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        if not streaming:
            return httpx.Response(200, json=response)
        event = {"type": "response.completed", "sequence_number": 1, "response": response}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(event)}\n\n")

    model = OpenAIModel(ModelConfig(type="openai", model="gpt-test", api_key="test", max_tokens=1024))
    monkeypatch.setattr(model, "_build_run_config", lambda **kwargs: RunConfig(tracing_disabled=True))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        transport = DatusResponsesModel("gpt-test", AsyncOpenAI(api_key="test", http_client=client))
        monkeypatch.setattr(model.litellm_adapter, "get_agents_sdk_model", lambda: transport)
        if streaming:
            actions = [action async for action in model.generate_with_tools_stream("Review.", max_tokens=output_limit)]
            assert any(a.output and a.output.get("raw_output") == "Reviewed." for a in actions)
        elif output_limit == 4000:
            result = await model.summarize_items(
                [{"role": "user", "content": "Earlier project findings."}],
                instruction="Summarize the project context.",
                prompt="Keep the findings needed to continue.",
                max_tokens=4000,
            )
            assert result["content"] == "Reviewed."
        else:
            assert (await model.generate_with_tools("Review.", max_tokens=output_limit))["content"] == "Reviewed."
    assert len(requests) == 1
    assert requests[0]["max_output_tokens"] == (1024 if output_limit is None else output_limit)
