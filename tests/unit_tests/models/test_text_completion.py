# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Exercise chat's real SDK/LiteLLM conversion with an in-memory HTTP provider."""

import asyncio
import json
import os
from unittest.mock import MagicMock

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from datus.configuration.agent_config import ModelConfig
from datus.models import openai_compatible, text_completion
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


@pytest.fixture
def provider_http(monkeypatch):
    """Replace network I/O while exercising the real SDK and LiteLLM serialization."""
    requests, clients, tls = [], [], []
    behavior = {"reason": "stop", "status": 200}
    monkeypatch.setattr(openai_compatible, "_load_model_specs", lambda: {})

    def handle(request):
        """Return a provider-shaped stream or the requested HTTP failure."""
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

        class ProviderStream(httpx.AsyncByteStream):
            """Yield real provider frames, optionally stalling after the first frame."""

            async def __aiter__(self):
                for index, event in enumerate(events):
                    yield sse([event]).encode()
                    if behavior.get("stall") and index == 0:
                        behavior["stream_ready"].set()
                        await asyncio.Event().wait()

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ProviderStream())

    def create_client(self, **kwargs):
        """Record each client's TLS setting and route its requests to the fake provider."""
        tls.append(kwargs["ssl_verify"])
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle), headers={"x-test-transport": str(len(clients))}
        )
        clients.append(client)
        return client

    monkeypatch.setattr(AsyncHTTPHandler, "create_client", create_client)
    return requests, clients, tls, behavior


@pytest.mark.asyncio
async def test_explicit_none_disables_deepseek_thinking_on_the_wire(provider_http):
    """Check LiteLLM's provider translation rather than only inspecting our settings."""
    requests, clients, _, _ = provider_http
    model = LLMBaseModel.from_config(
        ModelConfig(type="deepseek", model="deepseek-v4-pro", api_key="dummy", reasoning_effort="none")
    )
    await model.complete_text("facts", max_tokens=32768, timeout=5)
    body = requests[0][1]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 32768
    assert len(requests) == 1
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,status,outcome", [("stop", 200, "succeeded"), ("length", 200, "truncated"), ("stop", 400, "failed")]
)
async def test_completion_logs_only_metadata_and_provider_finish_reason(
    provider_http, monkeypatch, reason, status, outcome
):
    """Keep diagnostic metadata on success and failure without disclosing content."""
    _, clients, _, behavior = provider_http
    behavior.update(reason=reason, status=status, reasoning=True)
    log = MagicMock()
    monkeypatch.setattr(text_completion, "logger", log)
    model = LLMBaseModel.from_config(ModelConfig(type="deepseek", model="deepseek-v4-pro", api_key="dummy"))
    if outcome == "succeeded":
        await model.complete_text("PRIVATE_PROMPT", max_tokens=4000, timeout=5)
    else:
        with pytest.raises(ModelOutputTruncatedError if outcome == "truncated" else litellm.BadRequestError):
            await model.complete_text("PRIVATE_PROMPT", max_tokens=4000, timeout=5)
    finished = [call.kwargs for call in log.info.call_args_list if call.args[0] == "text_completion.finished"]
    assert len(finished) == 1
    record = finished[0]
    assert record["status"] == outcome
    assert record["finish_reason"] == (reason if status == 200 else None)
    assert record["duration_ms"] >= 0
    assert record["usage_reported"] is (status == 200)
    if status == 200:
        assert 0 <= record["first_text_ms"] <= record["duration_ms"]
        assert record["output_chars"] == len('{"summary":"ok"}')
        assert record["output_tokens"] == 7
        assert record["reasoning_chars"] == len("PRIVATE_REASONING")
    else:
        assert record["input_tokens"] is None and record["output_tokens"] is None
    serialized = json.dumps([call.kwargs for call in log.info.call_args_list])
    assert all(
        secret not in serialized for secret in ["PRIVATE_PROMPT", "PRIVATE_REASONING", '{"summary":"ok"}', "rejected"]
    )
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel,reasoning", [(False, False), (False, True), (True, False)])
async def test_stalled_stream_reports_partial_progress_and_unknown_usage(provider_http, monkeypatch, cancel, reasoning):
    """Distinguish waiting after reasoning/text from a finished response with zero usage."""
    _, clients, _, behavior = provider_http
    behavior.update(stall=True, reasoning=reasoning, stream_ready=asyncio.Event())
    log = MagicMock()
    monkeypatch.setattr(text_completion, "logger", log)
    model = LLMBaseModel.from_config(ModelConfig(type="deepseek", model="deepseek-v4-pro", api_key="dummy"))
    task = asyncio.create_task(model.complete_text("PRIVATE_PROMPT", max_tokens=4000, timeout=5 if cancel else 0.2))
    await asyncio.wait_for(behavior["stream_ready"].wait(), timeout=5)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    records = [call.kwargs for call in log.info.call_args_list if call.args[0] == "text_completion.finished"]
    assert len(records) == 1
    record = records[0]
    assert record["status"] == ("cancelled" if cancel else "timeout")
    assert record["usage_reported"] is False and record["output_tokens"] is None
    assert record["finish_reason"] is None
    if reasoning:
        assert 0 <= record["first_reasoning_ms"] <= record["duration_ms"]
        assert record["reasoning_chars"] == len("PRIVATE_REASONING")
        assert record["first_text_ms"] is None and record["output_chars"] == 0
    else:
        assert 0 <= record["first_text_ms"] <= record["duration_ms"] and record["output_chars"] == len(
            '{"summary":"ok"}'
        )
    assert all(client.is_closed for client in clients)


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
    """Exercise every provider's route, output cap, usage and transport isolation."""
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
    assert tls[0] is None
    assert dict(os.environ) == before
    assert body.get("stream") is (None if kind == "gemini" else True)
    token_settings = body.get("generationConfig", body)
    token_key = "max_output_tokens" if kind == "gemini" or request.url.host == "api.openai.com" else "max_tokens"
    assert token_settings[token_key] == 4000


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,status", [("length", 200), ("stop", 400), ("stop", 500)])
async def test_failure_keeps_usage_and_never_retries(provider_http, reason, status):
    """Preserve truncation usage and avoid retrying truncation or provider HTTP errors."""
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
    """Keep simultaneous Anthropic requests' TLS and sampling settings independent."""
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
@pytest.mark.parametrize(
    "request_verify,environment_verify,expected_verify",
    [(None, "false", False), (None, "true", True), (True, "false", True), (False, "true", False)],
)
async def test_connection_retry_preserves_tls_configuration(
    monkeypatch, request_verify, environment_verify, expected_verify
):
    """Keep explicit TLS settings on connection retries and inherit env only when unset."""
    import ssl

    attempts = []
    monkeypatch.setenv("SSL_VERIFY", environment_verify)
    monkeypatch.setattr(openai_compatible, "_load_model_specs", lambda: {})
    before = dict(os.environ)

    def create_transport(**kwargs):
        """Observe the real resolved TLS policy while replacing only network I/O."""
        context = kwargs["ssl_context"]
        verify = context.verify_mode != ssl.CERT_NONE if context is not None else kwargs["ssl_verify"]

        def handle(request):
            """Fail the first connection so LiteLLM must recreate its HTTP client."""
            attempts.append(verify)
            if len(attempts) == 1:
                raise httpx.ConnectError("connection failed", request=request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=sse(anthropic_events('{"summary":"ok"}', "stop")),
            )

        return httpx.MockTransport(handle)

    monkeypatch.setattr(AsyncHTTPHandler, "_create_async_transport", staticmethod(create_transport))
    model = LLMBaseModel.from_config(
        ModelConfig(
            type="claude",
            model="claude-opus-4-7",
            api_key="dummy",
            base_url="https://api.anthropic.com",
            ssl_verify=request_verify,
        ),
        configure_ssl_env=False,
    )
    result = await model.complete_text("facts", max_tokens=4000, timeout=5)
    assert result["content"] == '{"summary":"ok"}'
    assert attempts == [expected_verify, expected_verify]
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_inline_ca_is_trusted_without_changing_environment(provider_http):
    """Trust a configured PEM certificate without changing process-wide TLS settings."""
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
    """Reject SDK estimates when the provider does not report actual token usage."""
    requests, clients, _, behavior = provider_http
    behavior["omit_usage"] = True
    model = LLMBaseModel.from_config(
        ModelConfig(type="openai", model="test", api_key="dummy", base_url="https://custom.example/v1")
    )
    with pytest.raises(ValueError, match="missing completion or usage"):
        await model.complete_text("facts", max_tokens=4000, timeout=5)
    assert len(requests) == 1
    assert clients[0].is_closed
