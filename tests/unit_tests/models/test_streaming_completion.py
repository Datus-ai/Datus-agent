# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Exercise chat's real SDK/LiteLLM conversion with an in-memory HTTP provider."""

import asyncio
import json
import os

import httpx
import litellm
import pytest
import pytest_asyncio
from agents import AgentHooks, ModelSettings, RunConfig
from openai import APIError

from datus.configuration.agent_config import ModelConfig
from datus.models import openai_compatible
from datus.models.base import LLMBaseModel
from datus.models.output_truncation import ModelOutputTruncatedError

pytestmark = pytest.mark.ci


def sse(events):
    """Encode provider events as an in-memory server-sent event response."""
    return "".join(
        (f"event: {event['type']}\n" if "type" in event else "") + f"data: {json.dumps(event)}\n\n" for event in events
    )


def anthropic_events(text, reason):
    """Build an Anthropic stream with input and output usage in separate events."""
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [],
                "usage": {"input_tokens": 11, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens" if reason == "length" else "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ]


@pytest_asyncio.fixture
async def provider_http(monkeypatch):
    """Replace network I/O while exercising the real SDK and LiteLLM serialization."""
    requests, clients, tls = [], [], []
    behavior = {"reason": "stop", "status": 200}
    monkeypatch.setattr(openai_compatible, "_load_model_specs", lambda: {})
    from datus.cli import provider_model_catalog

    monkeypatch.setattr(provider_model_catalog, "resolve_model_details", lambda: {})

    def handle(request):
        """Return a provider-shaped stream or the requested HTTP failure."""
        body = json.loads(request.content)
        requests.append((request, body))
        behavior.setdefault("closed_before_request", []).append(
            [clients[int(previous.headers["x-test-client"])].is_closed for previous, _ in requests[:-1]]
        )
        if behavior["status"] != 200:
            return httpx.Response(
                behavior["status"], json={"error": {"type": "invalid_request_error", "message": "rejected"}}
            )
        if body.get("stream") is False:
            if behavior.get("fail_once") and len(requests) == 1:
                return httpx.Response(500, json={"error": {"message": "temporary"}})
            message = {"role": "assistant", "content": '{"summary":"ok"}'}
            if body.get("tools") and len(requests) % 2:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_test", "type": "function", "function": {"name": "ping", "arguments": "{}"}}
                    ],
                }
            return httpx.Response(
                200,
                json={
                    "id": "chat_test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                },
            )
        if request.url.host == "api.openai.com":
            events = [
                {
                    "type": "response.completed",
                    "sequence_number": 0,
                    "response": {
                        "id": "resp_test",
                        "object": "response",
                        "created_at": 1,
                        "status": "completed",
                        "error": None,
                        "incomplete_details": None,
                        "instructions": None,
                        "model": "gpt-4.1",
                        "output": [
                            {
                                "id": "msg_test",
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": '{"summary":"ok"}', "annotations": []}],
                            }
                        ],
                        "parallel_tool_calls": False,
                        "tools": [],
                        "tool_choice": "auto",
                        "temperature": 1,
                        "top_p": 1,
                        "usage": {
                            "input_tokens": 11,
                            "output_tokens": 7,
                            "total_tokens": 18,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens_details": {"reasoning_tokens": 0},
                        },
                    },
                }
            ]
            if behavior["reason"] == "length":
                events[0]["type"] = "response.incomplete"
                events[0]["response"].update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        elif request.url.path.endswith("/messages"):
            events = anthropic_events('{"summary":"ok"}', behavior["reason"])
        elif request.url.host == "generativelanguage.googleapis.com":
            events = [
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": '{"summary":"ok"}'}]},
                            "finishReason": "MAX_TOKENS" if behavior["reason"] == "length" else "STOP",
                            "index": 0,
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
                    "modelVersion": "gemini-2.5-flash",
                }
            ]
        else:
            events = [
                {
                    "id": "chat_test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": '{"summary":"ok"}'},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chat_test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": behavior["reason"]}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                },
            ]
            if behavior.get("reasoning"):
                events.insert(
                    0,
                    {
                        "id": "chat_test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "test",
                        "choices": [{"index": 0, "delta": {"reasoning_content": "PRIVATE_REASONING"}}],
                    },
                )
        if behavior.get("omit_usage"):
            for event in events:
                event.pop("usage", None)
                if "response" in event:
                    event["response"].pop("usage", None)

        class ProviderStream(httpx.AsyncByteStream):
            """Yield real provider frames, optionally stalling after the first frame."""

            async def __aiter__(self):
                for index, event in enumerate(events):
                    yield sse([event]).encode()
                    if behavior.get("transport_error") and index == 0:
                        raise httpx.RemoteProtocolError("PRIVATE_PROVIDER_ERROR", request=request)
                    if behavior.get("stall") and index == 0:
                        behavior["stream_ready"].set()
                        await asyncio.Event().wait()

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ProviderStream())

    original_init = httpx.AsyncClient.__init__

    def init_client(client, *args, **kwargs):
        tls.append(kwargs.get("verify"))
        kwargs["transport"] = httpx.MockTransport(handle)
        original_init(client, *args, **kwargs)
        client.headers["x-test-client"] = str(len(clients))
        clients.append(client)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init_client)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", type(litellm.in_memory_llm_clients_cache)())
    yield requests, clients, tls, behavior
    for client in clients:
        await client.aclose()


class Recorder(AgentHooks):
    def __init__(self):
        self.usage = None
        self.responses = []

    async def on_llm_end(self, context, agent, response):
        self.usage = agent.model.provider_reported_usage
        self.responses.append(response)


async def consume(model, recorder, timeout=5, private=False):
    async for _ in model.generate_with_tools_stream(
        "facts",
        instruction="Return JSON.",
        tools=[],
        max_turns=1,
        max_tokens=4000,
        hooks=recorder,
        run_config=RunConfig(
            tracing_disabled=True,
            trace_include_sensitive_data=not private,
            model_settings=ModelSettings(extra_args={"timeout": timeout}),
        ),
    ):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,name,url",
    [
        ("deepseek", "deepseek-v4-flash", "https://gateway.example/v1"),
        ("deepseek", "deepseek-v4-pro", "https://api.deepseek.com"),
        ("claude", "kimi-for-coding", "https://api.kimi.com/coding"),
        ("claude", "claude-opus-4-7", "https://api.anthropic.com"),
        ("openai", "gpt-4.1", "https://api.openai.com/v1"),
        ("openai", "test", "https://custom.example/v1"),
        ("kimi", "kimi-k2.6", "https://api.moonshot.ai/v1"),
        ("qwen", "qwen3-coder-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("minimax", "MiniMax-M2.7", "https://api.minimaxi.com/v1"),
        ("glm", "glm-5", "https://open.bigmodel.cn/api/paas/v4"),
        ("openrouter", "anthropic/claude-opus-4-7", "https://openrouter.ai/api/v1"),
        ("gemini", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta"),
    ],
)
async def test_existing_chat_stream_delivers_text_and_actual_usage(provider_http, kind, name, url):
    requests, _, _, _ = provider_http
    model = LLMBaseModel.from_config(
        ModelConfig(type=kind, model=name, api_key="dummy", base_url=url, max_retry=1, context_length=1000000)
    )
    recorder = Recorder()
    await consume(model, recorder)
    assert len(requests) == 1
    assert recorder.usage.input_tokens == 11
    assert recorder.usage.output_tokens == 7
    assert len(recorder.responses) == 1
    assert recorder.responses[0].output[-1].content[0].text == '{"summary":"ok"}'
    body = requests[0][1]
    assert body.get("stream") is (None if kind == "gemini" else True)
    assert not body.get("tools")
    assert "client" not in body and "ssl_verify" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "high"])
async def test_shared_deepseek_stream_preserves_explicit_reasoning_policy(provider_http, effort):
    requests, _, _, _ = provider_http
    model = LLMBaseModel.from_config(
        ModelConfig(
            type="deepseek",
            model="deepseek-v4-flash",
            api_key="dummy",
            base_url="https://gateway.example/v1",
            reasoning_effort=effort,
            max_retry=1,
        )
    )
    await consume(model, Recorder())
    body = requests[0][1]
    assert body["thinking"] == {"type": "disabled" if effort == "none" else "enabled"}
    assert (body.get("reasoning_effort") == "none") is (effort == "none")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,name", [("deepseek", "deepseek-v4-flash"), ("openai", "gpt-4.1")])
async def test_missing_usage_is_not_replaced_by_sdk_estimates(provider_http, kind, name):
    _, _, _, behavior = provider_http
    behavior["omit_usage"] = True
    model = LLMBaseModel.from_config(ModelConfig(type=kind, model=name, api_key="dummy", max_retry=1))
    recorder = Recorder()
    await consume(model, recorder)
    assert recorder.responses
    assert recorder.usage is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,name", [("deepseek", "deepseek-v4-flash"), ("openai", "gpt-4.1")])
async def test_regular_chat_keeps_existing_tls_client_path(provider_http, monkeypatch, kind, name):
    from unittest.mock import patch

    from datus.models import litellm_model

    monkeypatch.setenv("SSL_VERIFY", "True")
    model = LLMBaseModel.from_config(ModelConfig(type=kind, model=name, api_key="dummy"))
    assert model.litellm_adapter.ssl_verify is None
    with patch.object(litellm_model, "_TLSHTTPHandler", side_effect=AssertionError("unexpected request-local client")):
        recorder = Recorder()
        await consume(model, recorder)
    assert (recorder.usage.input_tokens, recorder.usage.output_tokens) == (11, 7)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "retry", "tool_rounds"])
async def test_nonstream_closes_owned_client_after_complete_runner(provider_http, outcome):
    from agents import function_tool

    from datus.utils.exceptions import DatusException

    requests, clients, _, behavior = provider_http
    if outcome == "failure":
        behavior["status"] = 400
    if outcome == "retry":
        behavior["fail_once"] = True

    @function_tool
    def ping() -> str:
        """Return a test tool result."""
        assert not clients[int(requests[-1][0].headers["x-test-client"])].is_closed
        return "pong"

    model = LLMBaseModel.from_config(
        ModelConfig(type="deepseek", model="deepseek-v4-flash", api_key="dummy", max_retry=1, ssl_verify=True),
        configure_ssl_env=False,
    )
    for _ in range(3 if outcome == "success" else 1):
        call = model.generate_with_tools("hello", tools=[ping] if outcome == "tool_rounds" else [])
        if outcome == "failure":
            with pytest.raises(DatusException):
                await call
        else:
            result = await call
            assert result["content"] == '{"summary":"ok"}'
        assert all(clients[int(request.headers["x-test-client"])].is_closed for request, _ in requests)
    assert len(requests) == {"success": 3, "failure": 1, "retry": 2, "tool_rounds": 2}[outcome]
    assert (
        len({request.headers["x-test-client"] for request, _ in requests})
        == {
            "success": 3,
            "failure": 1,
            "retry": 2,
            "tool_rounds": 1,
        }[outcome]
    )
    assert (
        behavior["closed_before_request"]
        == {
            "success": [[], [True], [True, True]],
            "failure": [[]],
            "retry": [[], [True]],
            "tool_rounds": [[], [False]],
        }[outcome]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 500])
@pytest.mark.parametrize("kind,name", [("deepseek", "deepseek-v4-flash"), ("openai", "gpt-4.1")])
async def test_failed_shared_stream_does_not_repeat_a_single_attempt(provider_http, status, kind, name):
    requests, _, _, behavior = provider_http
    behavior["status"] = status
    model = LLMBaseModel.from_config(ModelConfig(type=kind, model=name, api_key="dummy", max_retry=1))
    with pytest.raises(Exception) as error:
        await consume(model, Recorder())
    assert error.value.status_code == status
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,name,url",
    [
        ("deepseek", "deepseek-v4-flash", "https://gateway.example/v1"),
        ("claude", "kimi-for-coding", "https://api.kimi.com/coding"),
        ("openai", "gpt-4.1", "https://api.openai.com/v1"),
    ],
)
@pytest.mark.parametrize("verify", [True, False])
async def test_shared_stream_respects_request_local_tls(provider_http, kind, name, url, verify):
    requests, clients, tls, _ = provider_http
    before = dict(os.environ)
    model = LLMBaseModel.from_config(
        ModelConfig(type=kind, model=name, api_key="dummy", base_url=url, max_retry=1, ssl_verify=verify),
        configure_ssl_env=False,
    )
    await consume(model, Recorder())
    assert len(requests) == 1
    assert dict(os.environ) == before
    used = clients[int(requests[0][0].headers["x-test-client"])]
    assert used.is_closed
    import ssl

    assert any(
        value is verify or (isinstance(value, ssl.SSLContext) and (value.verify_mode != ssl.CERT_NONE) is verify)
        for value in tls
        if value is not None
    )


@pytest.mark.asyncio
async def test_shared_stream_rejects_truncated_text(provider_http):
    requests, _, _, behavior = provider_http
    behavior["reason"] = "length"
    model = LLMBaseModel.from_config(
        ModelConfig(type="deepseek", model="deepseek-v4-flash", api_key="dummy", max_retry=1)
    )
    with pytest.raises(ModelOutputTruncatedError) as error:
        await consume(model, Recorder())
    assert error.value.usage_reported
    assert error.value.usage.input_tokens == 11
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_private_stream_does_not_log_raw_connection_errors(provider_http, monkeypatch):
    from unittest.mock import MagicMock

    requests, _, _, behavior = provider_http
    behavior["transport_error"] = True
    log = MagicMock()
    monkeypatch.setattr(openai_compatible, "logger", log)
    model = LLMBaseModel.from_config(
        ModelConfig(type="deepseek", model="deepseek-v4-flash", api_key="dummy", max_retry=1)
    )
    with pytest.raises((httpx.RemoteProtocolError, APIError)):
        await consume(model, Recorder(), private=True)
    assert len(requests) == 1
    assert "PRIVATE_PROVIDER_ERROR" not in str(log.mock_calls)
