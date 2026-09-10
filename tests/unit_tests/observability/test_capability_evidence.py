# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from agents import Agent, RunConfig, Runner
from openai import AsyncOpenAI, BadRequestError
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from structlog.testing import capture_logs

from datus.models.observed_model import ObservedResponsesModel, _ObservedAsyncStream
from datus.observability.compaction import compact_started, observe_compaction
from datus.observability.model_call import ModelCall, capability_event, model_phase, observation_run
from datus.observability.tool_calls import observe_tool_hooks


def definition(name, description="Test"):
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object"}},
    }


def test_tool_changes_distinguish_summary_order_policy_and_mcp_evidence():
    def invoke(tools, choice="auto"):
        with ModelCall(model="m", model_impl="test", protocol="test") as call:
            call.request({"tools": tools, "tool_choice": choice})
        return call

    with capture_logs() as records, observation_run():
        initial_event = capability_event("mcp.degraded", reason="TimeoutError", server="initial")
        first = invoke([definition("read_file"), definition("bash")])
        reordered = invoke([definition("bash"), definition("read_file")])
        with model_phase("compact_summary"):
            summary = invoke([])
        event_id = capability_event("mcp.degraded", reason="TimeoutError", server="test")
        missing = invoke([definition("read_file", "Updated schema")])
        blocked = invoke([definition("read_file", "Updated schema")], choice="none")
    assert "previous_model_call_id" not in reordered.fields
    assert "previous_model_call_id" not in summary.fields
    assert missing.fields["previous_model_call_id"] == reordered.model_call_id
    assert missing.fields["tools_removed"] == ["bash"]
    assert missing.fields["tools_schema_changed"] == ["read_file"]
    assert missing.fields["tools_change_reason"] == "unknown"
    assert first.fields["capability_event_ids"] == [initial_event]
    assert missing.fields["capability_event_ids"] == [initial_event, event_id]
    assert blocked.fields["tool_choice_changed"] is True
    changes = [r for r in records if r["event"] == "tools.changed"]
    assert len(changes) == 2
    assert changes[0]["log_level"] == "warning"
    assert len([r for r in records if r["event"] == "tools.available"]) == 1
    assert first.model_call_id != summary.model_call_id


@pytest.mark.parametrize("limits", [SpanLimits(max_attributes=15), SpanLimits(max_attribute_length=64)])
def test_otel_limits_mark_incomplete_tool_definitions(exported_calls, limits):
    provider = TracerProvider(span_limits=limits)
    try:
        with provider.get_tracer(__name__).start_as_current_span("generation") as span:
            with ModelCall(model="m", model_impl="test", protocol="test") as call:
                call.bind_span(span)
                call.request({"tools": [definition("lookup", "long description " * 30)]})
            assert span.attributes["datus.llm.tools_capture_state"] == "truncated"
            assert span.attributes["datus.llm.tools_captured_count"] == 0
            assert span.attributes["datus.llm.tools_count"] == 1
    finally:
        provider.shutdown()


@pytest.mark.parametrize(
    "model,endpoint,header",
    [
        ("gpt-5", "https://api.openai.com/v1", "X-Request-ID"),
        ("claude-sonnet", "https://api.anthropic.com", "Request-ID"),
        ("deepseek/deepseek-chat", "https://api.deepseek.com", "X-DS-Trace-ID"),
        ("deepseek/private", "https://private.example", "Trace-ID"),
        ("moonshot/kimi-k3", "https://api.moonshot.cn/v1", "MSH-Request-ID"),
        ("openai/MiniMax-M2.7", "https://api.minimaxi.com/v1", "Trace-ID"),
    ],
)
def test_official_provider_correlation_headers_are_normalized(exported_calls, model, endpoint, header):
    with ModelCall(model=model, model_impl="test", protocol="test", endpoint=endpoint) as call:
        call.response(SimpleNamespace(headers={header: "Raw Value / 1", "authorization": "never-export"}))
    assert call.fields["remote_correlation_id"] == "Raw Value / 1"
    assert call.fields["remote_correlation_source"] == f"header:{header.lower()}"
    assert call.fields["remote_correlation_status"] == "captured"
    assert "never-export" not in json.dumps(call.fields)


def test_official_response_fields_are_captured_without_generic_completion_ids(exported_calls):
    with ModelCall(
        model="openai/glm-5", model_impl="test", protocol="test", endpoint="https://open.bigmodel.cn/api/paas/v4"
    ) as glm:
        glm.response(SimpleNamespace(request_id="glm-request", id="glm-completion"))
    with ModelCall(model="gemini/gemini-3", model_impl="test", protocol="test") as gemini:
        gemini.response(SimpleNamespace(id="gemini-response"))
    with ModelCall(model="unknown", model_impl="test", protocol="test") as unknown:
        unknown.response(SimpleNamespace(id="completion-id"))

    assert glm.fields["remote_correlation_id"] == "glm-request"
    assert glm.fields["remote_correlation_source"] == "field:request_id"
    assert gemini.fields["remote_correlation_id"] == "gemini-response"
    assert gemini.fields["remote_correlation_source"] == "field:responseId"
    assert "remote_correlation_id" not in unknown.fields


def test_litellm_preserved_stream_headers_are_read_when_public_headers_are_empty(exported_calls):
    response = SimpleNamespace(
        _response_headers={},
        logging_obj=SimpleNamespace(model_call_details={"_datus_response_headers": {"x-ds-trace-id": "deepseek"}}),
    )
    with ModelCall(model="deepseek/deepseek-chat", model_impl="litellm", protocol="chat_completions") as call:
        call.response(response, streaming=True)
    assert call.fields["remote_correlation_id"] == "deepseek"
    assert call.fields["remote_correlation_source"] == "header:x-ds-trace-id"


@pytest.mark.asyncio
async def test_gemini_response_id_is_observed_from_raw_stream_chunks(exported_calls):
    async def chunks():
        yield SimpleNamespace(id="gemini-stream-response")
        yield SimpleNamespace(id=None)

    with ModelCall(model="gemini/gemini-3", model_impl="litellm", protocol="chat_completions") as call:
        observed = _ObservedAsyncStream(chunks(), call)
        assert [chunk.id async for chunk in observed] == ["gemini-stream-response", None]

    assert call.fields["remote_correlation_id"] == "gemini-stream-response"
    assert call.fields["remote_correlation_source"] == "field:responseId"


@pytest.mark.asyncio
async def test_http_error_id_is_added_before_sdk_generation_ends(exported_calls):
    exporter, _, _ = exported_calls
    client = AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    400,
                    headers={"x-request-id": "raw-error-id"},
                    json={"error": {"message": "bad request", "type": "invalid_request_error"}},
                )
            )
        ),
    )
    try:
        with pytest.raises(BadRequestError):
            await Runner.run(
                Agent(name="test", model=ObservedResponsesModel("test-model", client)),
                "test",
                run_config=RunConfig(tracing_disabled=False),
            )
    finally:
        await client.close()
    spans = [s for s in exporter.get_finished_spans() if s.attributes.get("openinference.span.kind") == "LLM"]
    assert len(spans) == 1
    assert spans[0].attributes["datus.llm.remote_correlation_id"] == "raw-error-id"
    assert spans[0].attributes["datus.llm.status"] == "error"


@pytest.mark.asyncio
async def test_compaction_summary_and_following_call_share_compact_identity(exported_calls, monkeypatch):
    from opentelemetry import trace

    exporter, manager, provider = exported_calls
    manager._adapters = [object()]
    monkeypatch.setattr("datus.observability.compaction.get_observability_manager", lambda: manager)
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: provider.get_tracer("compact-test"))
    calls = []

    class Node:
        session_id = "test-session"
        tools = []

        @observe_compaction
        async def compact(self):
            compact_started(mode="major", items_before=12)
            with model_phase("compact_summary"), ModelCall(model="m", model_impl="test", protocol="test") as call:
                call.request({"tools": []})
                calls.append(call)
            return {"mode": "major", "success": True, "history_jsonl": "/tmp/history.jsonl", "summary_token": 12}

    with capture_logs() as records, observation_run():
        await Node().compact()
        with ModelCall(model="m", model_impl="test", protocol="test") as following:
            following.request({"tools": []})
    finished = next(r for r in records if r["event"] == "compact.finished")
    assert finished["compact_id"] == calls[0].fields["compact_id"] == following.fields["compact_id"]
    assert finished["recovery_tool_available"] is False
    assert any(r["event"] == "compact.recovery_tool_unavailable" for r in records)
    assert finished["status"] == "success"
    (span,) = exporter.get_finished_spans()
    assert span.name == "compact"
    assert span.attributes["datus.compact.compact_id"] == finished["compact_id"]
    assert span.attributes["datus.compact.items_before"] == 12
    assert span.attributes["datus.compact.recovery_tool_available"] is False


@pytest.mark.asyncio
async def test_tool_hooks_preserve_permissions_failure_status_and_request_identity():
    delegate = SimpleNamespace(on_tool_start=AsyncMock(), on_tool_end=AsyncMock())
    hooks = observe_tool_hooks(delegate)
    assert observe_tool_hooks(hooks) is hooks
    tool = SimpleNamespace(name="lookup")
    context = SimpleNamespace(tool_call_id="tool-1")
    with capture_logs() as records, observation_run() as state:
        with ModelCall(model="m", model_impl="test", protocol="test") as call:
            call.response(SimpleNamespace(_request_id="raw-1"))
            call.record_tool_calls([{"type": "function_call", "call_id": name} for name in ("tool-1", "tool-2")])
        await hooks.on_tool_start(context, None, tool)
        await hooks.on_tool_end(context, None, tool, '{"success": false, "error": "failed"}')
        assert set(state.tool_calls) == {"tool-2"}
        context.tool_call_id = "tool-2"
        delegate.on_tool_start.side_effect = PermissionDeniedException()
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(context, None, tool)
        assert state.tool_calls == {}
    finished = [r for r in records if r["event"] == "tool.finished"]
    assert [r["status"] for r in finished] == ["unsuccessful_result", "permission_denied"]
    assert all(r["remote_correlation_id"] == "raw-1" and r["model_call_id"] == call.model_call_id for r in finished)
    delegate.on_tool_end.assert_awaited_once()


class PermissionDeniedException(Exception):
    pass


def test_malformed_optional_usage_cannot_fail_a_successful_call(exported_calls):
    class BrokenUsage:
        def model_dump(self):
            raise ValueError("broken metadata")

    with ModelCall(model="m", model_impl="test", protocol="test") as call:
        call.request({"tools": []})
        call.usage(BrokenUsage())
        call.record_tool_calls([BrokenUsage()])
    assert call.fields["status"] == "success"


def test_connection_failure_does_not_claim_response_headers_arrived():
    with pytest.raises(ConnectionError):
        with ModelCall(model="m", model_impl="test", protocol="test") as call:
            call.request({"tools": []})
            raise ConnectionError("connection refused")
    assert "response_received_ms" not in call.fields
    assert "remote_correlation_id" not in call.fields
    assert call.fields["status"] == "error"


def test_host_run_id_is_shared_by_model_and_capability_logs(tmp_path, isolated_logging):
    from pathlib import Path

    from datus.utils.loggings import configure_logging
    from datus.utils.trace_context import TraceContext, trace_context

    manager = configure_logging(level="INFO", log_dir=tmp_path, console_output=False)
    with trace_context(TraceContext(name="test", metadata={"run_id": "host-run"}), replace=True):
        capability_event("mcp.degraded", reason="timeout")
        with ModelCall(model="m", model_impl="test", protocol="test") as call:
            call.request({"tools": []})
    manager.file_handler.flush()
    records = Path(manager.file_handler.baseFilename).read_text().splitlines()
    records = [line for line in records if "logging.configured" not in line]
    assert any("mcp.degraded" in line for line in records)
    assert any("llm.finished" in line for line in records)
    assert all("run_id=host-run" in line.split() for line in records)


def test_sdk_dataclass_usage_reaches_logs_and_trace(exported_calls):
    from agents.usage import Usage

    exporter, _, provider = exported_calls
    with capture_logs() as records, provider.get_tracer(__name__).start_as_current_span("generation") as span:
        with ModelCall(model="m", model_impl="test", protocol="test") as call:
            call.bind_span(span)
            call.usage(Usage(input_tokens=12, output_tokens=3, total_tokens=15))
    finished = next(record for record in records if record["event"] == "llm.finished")
    (generation,) = exporter.get_finished_spans()
    for field, expected in {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}.items():
        assert finished[field] == expected
        assert generation.attributes[f"datus.llm.{field}"] == expected
