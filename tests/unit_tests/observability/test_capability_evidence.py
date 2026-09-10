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

from datus.models.observed_model import ObservedResponsesModel
from datus.observability.compaction import compact_started, observe_compaction
from datus.observability.config import ObservabilityConfig, TracingConfig
from datus.observability.model_call import ModelCall, capability_event, model_phase, observation_run
from datus.observability.tool_calls import observe_tool_hooks
from datus.utils.exceptions import DatusException, ErrorCode


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


def test_explicit_raw_header_mapping_works_with_tracing_disabled(exported_calls):
    _, manager, _ = exported_calls
    config = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": False,
                "remote_id_headers": {
                    "private.example": {
                        "request_id_header": "x-provider-request",
                        "trace_id_header": "x-provider-trace",
                        "gateway_request_id_header": "x-gateway-request",
                        "issuer": "provider",
                    }
                },
            }
        }
    )
    assert manager.configure(config) is False
    with ModelCall(model="m", model_impl="test", protocol="test", endpoint="https://private.example/v1") as call:
        call.response(
            SimpleNamespace(
                headers={
                    "x-provider-request": "Raw Value / 1",
                    "x-provider-trace": "Trace-2",
                    "x-gateway-request": "Gateway-3",
                    "authorization": "never-export",
                },
                id="wrong-body-id",
            )
        )
    assert call.fields["provider_request_id"] == "Raw Value / 1"
    assert call.fields["provider_trace_id"] == "Trace-2"
    assert call.fields["gateway_request_id"] == "Gateway-3"
    assert "never-export" not in json.dumps(call.fields)
    with ModelCall(model="m", model_impl="test", protocol="test", endpoint="https://private.example") as missing:
        missing.response(SimpleNamespace(headers={}, _request_id="wrong-header"))
    assert "request_id" not in missing.fields
    assert missing.fields["request_id_status"] == "absent"
    with pytest.raises(DatusException) as error:
        TracingConfig.from_dict({"remote_id_headers": {"private.example": {"request_id_header": "authorization"}}})
    assert error.value.code == ErrorCode.COMMON_FIELD_INVALID


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
    assert spans[0].attributes["datus.llm.request_id"] == "raw-error-id"
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
    assert all(r["request_id"] == "raw-1" and r["model_call_id"] == call.model_call_id for r in finished)
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
    assert "response_headers_ms" not in call.fields
    assert "request_id" not in call.fields
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


@pytest.mark.parametrize(
    "raw,field",
    [
        ([], "remote_id_headers"),
        ({"host": []}, "remote_id_headers.host"),
        ({"host": {"unsupported": "x-id"}}, "remote_id_headers.host"),
        ({"host": {"issuer": "invalid"}}, "remote_id_headers.host.issuer"),
        ({"host": {"request_id_header": ""}}, "remote_id_headers.host.request_id_header"),
    ],
)
def test_invalid_remote_id_headers_report_field_errors(raw, field):
    with pytest.raises(DatusException) as error:
        TracingConfig.from_dict({"remote_id_headers": raw})
    assert error.value.code == ErrorCode.COMMON_FIELD_INVALID
    assert field in str(error.value)


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
