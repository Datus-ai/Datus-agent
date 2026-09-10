# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json
from types import SimpleNamespace
from unittest.mock import Mock

import anthropic
import httpx
import pytest
from openai import OpenAI
from opentelemetry import trace

from datus.models.claude_model import ClaudeModel
from datus.models.codex_model import CodexModel


@pytest.fixture
def native_exporter(exported_calls, monkeypatch):
    exporter, manager, provider = exported_calls
    manager._adapters = [object()]
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: provider.get_tracer("native-test"))
    return exporter


def responses_sse(text):
    response = {
        "id": "response-body-is-not-request-id",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "test-model",
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "output": [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }
    return (
        f"data: {json.dumps({'type': 'response.completed', 'sequence_number': 0, 'response': response})}\n\n".encode()
    )


@pytest.mark.parametrize("json_output", [False, True])
def test_codex_direct_requests_and_auth_retry_keep_each_raw_id(native_exporter, json_output):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                401,
                headers={"x-request-id": "raw-auth-failure"},
                json={"error": {"message": "expired", "type": "authentication_error"}},
            )
        return httpx.Response(
            200,
            headers={"x-request-id": "raw-success", "content-type": "text/event-stream"},
            content=responses_sse('{"answer":1}' if json_output else "done"),
        )

    client = OpenAI(
        api_key="test",
        base_url="https://chatgpt.com/backend-api/codex",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    model = CodexModel.__new__(CodexModel)
    model.model_name = "test-model"
    model._get_client = lambda: client
    model._refresh_client_token = lambda: None
    model.oauth_manager = SimpleNamespace(refresh_tokens=Mock())
    try:
        result = model.generate_with_json_output("test") if json_output else model.generate("test")
    finally:
        client.close()
    assert result == ({"answer": 1} if json_output else "done")
    spans = native_exporter.get_finished_spans()
    assert len(spans) == 2
    first, second = [span.attributes for span in spans]
    assert first["datus.llm.remote_correlation_id"] == "raw-auth-failure"
    assert first["datus.llm.status"] == "error"
    assert second["datus.llm.remote_correlation_id"] == "raw-success"
    assert second["datus.llm.retry_of"] == first["datus.llm.model_call_id"]
    assert all(span.attributes["datus.llm.tools_count"] == 0 for span in spans)
    model.oauth_manager.refresh_tokens.assert_called_once()


def _claude(client, async_client=None):
    model = ClaudeModel.__new__(ClaudeModel)
    model.model_name = "claude-test"
    model.use_native_api = True
    model._is_oauth_token = False
    model.anthropic_client = client
    model.async_anthropic_client = async_client
    model.max_tokens = lambda: 100
    return model


def anthropic_message():
    return {
        "id": "message-body-id",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


@pytest.mark.asyncio
async def test_native_claude_generation_and_compact_summary_export_ids(native_exporter):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"request-id": f"anthropic-raw-{len(requests)}"}, json=anthropic_message())

    client = anthropic.Anthropic(
        api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    model = _claude(client)
    try:
        assert model.generate("hello") == "done"
        summary = await model.summarize_items(
            [{"role": "user", "content": "hello"}], instruction="summarize", prompt="summary", item_format="anthropic"
        )
        assert summary["content"] == "done"
    finally:
        client.close()
    spans = native_exporter.get_finished_spans()
    assert [s.attributes["datus.llm.remote_correlation_id"] for s in spans] == [
        "anthropic-raw-1",
        "anthropic-raw-2",
    ]
    assert [s.attributes["datus.llm.phase"] for s in spans] == ["task", "compact_summary"]
    assert all(s.attributes["datus.llm.tools_count"] == 0 for s in spans)


@pytest.mark.asyncio
async def test_native_claude_stream_parse_error_preserves_headers_and_tools(native_exporter):
    tools = [
        {"name": "lookup", "description": "Lookup a test item", "input_schema": {"type": "object", "properties": {}}}
    ]
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"request-id": "raw-stream-error", "content-type": "text/event-stream"},
            content=b"event: message_start\ndata: {invalid-json\n\n",
        )

    client = anthropic.AsyncAnthropic(
        api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )
    model = _claude(None, client)
    try:
        with pytest.raises(json.JSONDecodeError):
            async with model._anthropic_messages_stream(
                model=model.model_name, messages=[{"role": "user", "content": "test"}], max_tokens=10, tools=tools
            ) as stream:
                async for _ in stream:
                    pass
    finally:
        await client.close()
    (span,) = native_exporter.get_finished_spans()
    assert span.attributes["datus.llm.remote_correlation_id"] == "raw-stream-error"
    assert span.attributes["datus.llm.status"] == "error"
    actual = requests[0]["tools"][0]
    assert json.loads(span.attributes["llm.tools.0.tool.json_schema"]) == {
        **actual,
        "parameters": actual["input_schema"],
    }


@pytest.mark.asyncio
async def test_anthropic_observation_does_not_drain_unconsumed_stream(native_exporter):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    stream = SimpleNamespace(
        request_id="raw-stream", current_message_snapshot=SimpleNamespace(usage=None), get_final_message=AsyncMock()
    )

    @asynccontextmanager
    async def stream_manager():
        yield stream

    model = _claude(None, SimpleNamespace(base_url="https://api.anthropic.com"))
    async with model._observe_anthropic_stream(stream_manager(), {"messages": [], "tools": []}):
        # Leaving a context must release the response without reading more SSE.
        pass
    stream.get_final_message.assert_not_awaited()
