# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Exercise chat's real SDK/LiteLLM conversion with an in-memory HTTP provider."""

import json
import os

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from datus.configuration.agent_config import ModelConfig
from datus.models import openai_compatible
from datus.models.base import LLMBaseModel
from datus.models.output_truncation import ModelOutputTruncatedError

pytestmark = pytest.mark.ci


def sse(events):
    return "".join(
        (f"event: {event['type']}\n" if "type" in event else "") + f"data: {json.dumps(event)}\n\n" for event in events
    )


def anthropic_events(text, reason):
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


@pytest.fixture
def provider_http(monkeypatch):
    requests, clients, tls = [], [], []
    behavior = {"reason": "stop", "status": 200}
    monkeypatch.setattr(openai_compatible, "_load_model_specs", lambda: {})

    def handle(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if behavior["status"] != 200:
            return httpx.Response(
                behavior["status"], json={"error": {"type": "invalid_request_error", "message": "rejected"}}
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
        if behavior.get("omit_usage"):
            for event in events:
                event.pop("usage", None)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse(events))

    def create_client(self, **kwargs):
        tls.append(kwargs["ssl_verify"])
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle), headers={"x-test-transport": str(len(clients))}
        )
        clients.append(client)
        return client

    monkeypatch.setattr(AsyncHTTPHandler, "create_client", create_client)
    return requests, clients, tls, behavior


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,name,base_url,route",
    [
        ("claude", "claude-opus-4-7", "https://api.anthropic.com", "/v1/messages"),
        ("openai", "gpt-4.1", "https://api.openai.com/v1", "/v1/responses"),
        ("claude", "kimi-k2.6", "https://coding.example/anthropic", "/anthropic/v1/messages"),
        ("openai", "test", "https://custom.example/v1", "/v1/chat/completions"),
        ("deepseek", "deepseek-chat", "https://api.deepseek.com", "/chat/completions"),
        ("kimi", "kimi-k2.6", "https://api.moonshot.ai/v1", "/v1/chat/completions"),
        (
            "qwen",
            "qwen3-coder-plus",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "/compatible-mode/v1/chat/completions",
        ),
        ("minimax", "MiniMax-M2.7", "https://api.minimaxi.com/v1", "/v1/chat/completions"),
        ("glm", "glm-5", "https://open.bigmodel.cn/api/paas/v4", "/api/paas/v4/chat/completions"),
        ("openrouter", "anthropic/claude-opus-4-7", "https://openrouter.ai/api/v1", "/api/v1/chat/completions"),
        (
            "gemini",
            "gemini-2.5-flash",
            "https://generativelanguage.googleapis.com/v1beta",
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        ),
    ],
)
async def test_single_text_call_uses_chat_routing_without_transport_fields_in_json(
    provider_http, kind, name, base_url, route
):
    requests, clients, tls, _ = provider_http
    before = dict(os.environ)
    model = LLMBaseModel.from_config(
        ModelConfig(
            type=kind, model=name, api_key="dummy", base_url=base_url, context_length=1000000, max_tokens=128000
        ),
        configure_ssl_env=False,
    )
    result = await model.complete_text("facts", instruction="Return JSON.", max_tokens=4000, timeout=5)
    assert result["content"] == '{"summary":"ok"}'
    assert result["usage"].input_tokens == 11
    assert result["usage"].output_tokens == 7
    assert len(requests) == 1
    request, body = requests[0]
    assert request.url.path == route
    assert "ssl_verify" not in body
    assert not body.get("tools") and not body.get("fallbacks")
    assert clients[0].is_closed
    assert requests[0][0].headers["x-test-transport"] == "0"
    for client in clients[1:]:
        await client.aclose()
    assert tls[0] is True
    assert dict(os.environ) == before
    if kind == "claude":
        assert "temperature" not in body and "top_p" not in body
    if kind != "gemini":
        assert body["stream"] is True
        assert body["max_output_tokens" if request.url.host == "api.openai.com" else "max_tokens"] == 4000


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,status", [("length", 200), ("stop", 400), ("stop", 500)])
async def test_failure_keeps_usage_and_never_retries(provider_http, reason, status):
    requests, clients, _, behavior = provider_http
    behavior.update(reason=reason, status=status)
    model = LLMBaseModel.from_config(
        ModelConfig(type="claude", model="claude-opus-4-7", api_key="dummy", base_url="https://api.anthropic.com")
    )
    with pytest.raises(ModelOutputTruncatedError if reason == "length" else Exception) as error:
        await model.complete_text("facts", max_tokens=4000, timeout=5)
    if reason == "length":
        assert error.value.usage.input_tokens == 11
        assert error.value.usage.output_tokens == 7
    else:
        assert error.value.status_code == status
    assert len(requests) == 1
    assert clients[0].is_closed
    assert requests[0][0].headers["x-test-transport"] == "0"
    for client in clients[1:]:
        await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_sampling_and_tls_are_request_local(provider_http):
    import asyncio

    requests, clients, tls, _ = provider_http
    before = dict(os.environ)
    models = [
        LLMBaseModel.from_config(
            ModelConfig(
                type="claude",
                model="claude-opus-4-7",
                api_key="dummy",
                base_url="https://api.anthropic.com",
                temperature=0.2,
                top_p=0.9,
                ssl_verify=verify,
            ),
            configure_ssl_env=False,
        )
        for verify in (True, False)
    ]
    results = await asyncio.gather(*(model.complete_text("facts", max_tokens=4000, timeout=5) for model in models))
    assert all(result["usage"].input_tokens == 11 for result in results)
    assert tls == [True, False]
    assert len(requests) == 2
    assert all("ssl_verify" not in body and "temperature" not in body and "top_p" not in body for _, body in requests)
    assert all(client.is_closed for client in clients)
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_inline_ca_is_trusted_without_changing_environment(provider_http):
    import datetime
    import ssl

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    _, clients, tls, _ = provider_http
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Digest Test CA")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    before = dict(os.environ)
    model = LLMBaseModel.from_config(
        ModelConfig(
            type="claude",
            model="claude-opus-4-7",
            api_key="dummy",
            base_url="https://api.anthropic.com",
            ssl_verify=cert.public_bytes(serialization.Encoding.PEM).decode(),
        ),
        configure_ssl_env=False,
    )
    await model.complete_text("facts", max_tokens=4000, timeout=5)
    assert isinstance(tls[0], ssl.SSLContext)
    assert cert.public_bytes(serialization.Encoding.DER) in tls[0].get_ca_certs(binary_form=True)
    assert all(client.is_closed for client in clients)
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_missing_provider_usage_is_not_replaced_with_estimated_tokens(provider_http):
    requests, clients, _, behavior = provider_http
    behavior["omit_usage"] = True
    model = LLMBaseModel.from_config(
        ModelConfig(type="openai", model="test", api_key="dummy", base_url="https://custom.example/v1")
    )
    with pytest.raises(ValueError, match="missing completion or usage"):
        await model.complete_text("facts", max_tokens=4000, timeout=5)
    assert len(requests) == 1
    assert clients[0].is_closed
