# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from opentelemetry import baggage, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import datus.observability.openai_agents as openai_agents_module
from datus.observability.adapters.langfuse import _LangfuseBaggageSpanProcessor
from datus.observability.adapters.otlp import _BaggageAttributeSpanProcessor
from datus.observability.config import TracingConfig
from datus.observability.manager import ObservabilityManager
from datus.observability.openai_agents import DatusOpenInferenceTracingProcessor


@dataclass
class FakeTrace:
    name: str = "agent/chat"
    trace_id: str = "trace_test"


@dataclass
class FakeSpan:
    span_data: object
    span_id: str
    trace_id: str = "trace_test"
    parent_id: str | None = None
    started_at: str = ""
    ended_at: str = ""
    error: dict | None = None

    def set_error(self, error: dict) -> None:
        self.error = error


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.mark.parametrize("span_type", ["response", "generation", "function"])
def test_image_bytes_are_removed_from_exported_spans_without_changing_history(span_type):
    from agents.tracing.span_data import FunctionSpanData, GenerationSpanData, ResponseSpanData
    from openinference.instrumentation import OITracer, TraceConfig

    image = {"type": "input_image", "image_url": "data:image/png;base64,c2VjcmV0"}
    history = [{"role": "user", "content": [{"type": "input_text", "text": "chart.png"}, image]}]
    if span_type == "response":
        data = ResponseSpanData(input=history)
    elif span_type == "generation":
        data = GenerationSpanData(input=history, model="test")
    else:
        data = FunctionSpanData(name="read_image", input='{"path":"chart.png"}', output=str(history))
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    processor = DatusOpenInferenceTracingProcessor(OITracer(provider.get_tracer(__name__), config=TraceConfig()))
    trace = FakeTrace()
    span = FakeSpan(span_data=data, span_id="image_span", started_at=_now_iso(), ended_at=_now_iso())
    processor.on_trace_start(trace)
    processor.on_span_start(span)
    processor.on_span_end(span)
    processor.on_trace_end(trace)
    exported = str([dict(item.attributes) for item in exporter.get_finished_spans()])
    assert "c2VjcmV0" not in exported
    assert "chart.png" in exported
    assert history[0]["content"][1]["image_url"].endswith("c2VjcmV0")
    processor.shutdown()
    provider.shutdown()


def test_openai_agents_processor_merges_first_agent_span_into_trace_root():
    from agents.tracing.span_data import AgentSpanData, FunctionSpanData
    from openinference.instrumentation import OITracer, TraceConfig

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(_BaggageAttributeSpanProcessor())
    provider.add_span_processor(_LangfuseBaggageSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer(__name__), config=TraceConfig())
    parent_context = context.get_current()
    parent_context = baggage.set_baggage("datus.trace.name", "agent/chat", context=parent_context)
    parent_context = baggage.set_baggage("session.id", "session-1", context=parent_context)
    token = context.attach(parent_context)
    processor = DatusOpenInferenceTracingProcessor(tracer)
    trace = FakeTrace()
    root_agent_span = FakeSpan(
        span_data=AgentSpanData(name="chat", tools=["describe_table"], output_type="str"),
        span_id="span_agent",
        started_at=_now_iso(),
    )
    tool_span = FakeSpan(
        span_data=FunctionSpanData(name="describe_table", input="schools", output='{"columns": 49}'),
        span_id="span_tool",
        parent_id=root_agent_span.span_id,
        started_at=_now_iso(),
    )

    try:
        processor.on_trace_start(trace)
        processor.on_span_start(root_agent_span)
        processor.on_span_start(tool_span)
        tool_span.ended_at = _now_iso()
        processor.on_span_end(tool_span)
        root_agent_span.ended_at = _now_iso()
        processor.on_span_end(root_agent_span)
        processor.on_trace_end(trace)
    finally:
        context.detach(token)
        processor.shutdown()

    spans = exporter.get_finished_spans()
    provider.shutdown()
    span_by_name = {span.name: span for span in spans}

    assert sorted(span_by_name) == ["agent/chat", "describe_table"]
    root = span_by_name["agent/chat"]
    child = span_by_name["describe_table"]
    assert root.parent is None
    assert root.attributes["openinference.span.kind"] == "AGENT"
    assert root.attributes["langfuse.trace.name"] == "agent/chat"
    assert root.attributes["langfuse.session.id"] == "session-1"
    assert child.parent == root.context


def test_openai_agents_processor_fails_lazily_when_dependency_missing(monkeypatch):
    import datus.observability.openai_agents as module

    monkeypatch.setattr(module, "_oi_processor", None)

    with pytest.raises(RuntimeError, match="openinference"):
        module.DatusOpenInferenceTracingProcessor(object())


def test_openai_agents_processor_marks_returned_tool_failures(monkeypatch):
    from agents.tracing.span_data import AgentSpanData, FunctionSpanData
    from openinference.instrumentation import OITracer, TraceConfig

    observability = ObservabilityManager()
    observability._tracing_config = TracingConfig.from_dict(
        {
            "enabled": True,
            "redact": {"patterns": [r"secret-\d+"]},
        }
    )
    monkeypatch.setattr(openai_agents_module, "get_observability_manager", lambda: observability)

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    processor = DatusOpenInferenceTracingProcessor(OITracer(provider.get_tracer(__name__), config=TraceConfig()))
    trace = FakeTrace()
    root_agent_span = FakeSpan(
        span_data=AgentSpanData(name="chat", tools=["query_metrics"], output_type="str"),
        span_id="span_agent",
        started_at=_now_iso(),
    )
    replay_error = (
        "Malformed JSON arguments for tool 'query_metrics' were repaired for replay. "
        "The tool was not executed. Retry the tool with one complete JSON object."
    )
    outputs = {
        "func_failure": {"success": 0, "error": "query failed", "result": None},
        "error_only": json.dumps({"success": "unknown", "error": f"secret-123\n{'details ' * 100}"}),
        "replay_failure": {"success": 0, "error": replay_error, "result": None},
        "native_failure": {
            "success": False,
            "raw_output": json.dumps({"success": 0, "error": "native query failed"}),
            "summary": "Failed",
        },
        "success": {"success": 1, "error": None, "result": {"rows": 1}},
    }
    tool_spans = [
        FakeSpan(
            span_data=FunctionSpanData(name=name, input="{}", output=output),
            span_id=f"span_{name}",
            parent_id=root_agent_span.span_id,
            started_at=_now_iso(),
        )
        for name, output in outputs.items()
    ]

    try:
        processor.on_trace_start(trace)
        processor.on_span_start(root_agent_span)
        for tool_span in tool_spans:
            processor.on_span_start(tool_span)
            tool_span.ended_at = _now_iso()
            processor.on_span_end(tool_span)
        root_agent_span.ended_at = _now_iso()
        processor.on_span_end(root_agent_span)
        processor.on_trace_end(trace)
    finally:
        processor.shutdown()

    spans = exporter.get_finished_spans()
    provider.shutdown()
    span_by_name = {span.name: span for span in spans}
    sdk_span_by_name = {span.span_data.name: span for span in tool_spans}

    for name in ("func_failure", "error_only", "replay_failure", "native_failure"):
        assert span_by_name[name].status.status_code is StatusCode.ERROR
        expected_output = json.loads(outputs[name]) if isinstance(outputs[name], str) else outputs[name]
        assert json.loads(span_by_name[name].attributes["output.value"]) == expected_output

    assert sdk_span_by_name["func_failure"].error["message"] == "query failed"
    redacted_message = sdk_span_by_name["error_only"].error["message"]
    assert "secret-123" not in redacted_message
    assert "[REDACTED]" in redacted_message
    assert "\n" not in redacted_message
    assert len(redacted_message) <= 500
    assert "not executed" in sdk_span_by_name["replay_failure"].error["message"]
    assert sdk_span_by_name["native_failure"].error["message"] == "native query failed"

    assert span_by_name["success"].status.status_code is StatusCode.OK
    assert sdk_span_by_name["success"].error is None
    assert json.loads(span_by_name["success"].attributes["output.value"]) == outputs["success"]


@pytest.fixture
def streamed_generation_output():
    return [
        {
            "object": "response",
            "id": "__fake_id__",
            "tools": [],
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [{"type": "summary_text", "text": "Check sources."}],
                    "encrypted_content": "opaque-reasoning",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Checking ", "annotations": []},
                        {"type": "output_text", "text": "two sources.", "annotations": []},
                    ],
                },
                {
                    "type": "function_call",
                    "id": "__fake_id__",
                    "call_id": "call-list",
                    "name": "list_models",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "id": "__fake_id__",
                    "call_id": "call-glob",
                    "name": "glob",
                    "arguments": '{"pattern":"*.yml"}',
                },
            ],
        }
    ]


def test_streamed_response_exports_messages_and_openinference_tool_calls(streamed_generation_output):
    """Validate the provider-neutral payload before it reaches any OTLP exporter."""
    from agents.tracing.span_data import GenerationSpanData
    from openinference.instrumentation import OITracer, TraceConfig

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    processor = DatusOpenInferenceTracingProcessor(OITracer(provider.get_tracer(__name__), config=TraceConfig()))
    original_output = streamed_generation_output
    data = GenerationSpanData(
        input=[{"role": "user", "content": "Inspect available sources."}],
        output=original_output,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    span = FakeSpan(span_data=data, span_id="span_generation", started_at=_now_iso())
    try:
        processor.on_trace_start(FakeTrace())
        processor.on_span_start(span)
        span.ended_at = _now_iso()
        processor.on_span_end(span)
        processor.on_trace_end(FakeTrace())
        assert data.output is original_output
        assert original_output[0]["output"][1]["content"][0]["type"] == "output_text"
        attrs = next(s.attributes for s in exporter.get_finished_spans() if s.name == "generation")
        assert attrs["llm.token_count.total"] == 15
        messages = json.loads(attrs["output.value"])["messages"]
        assert len(messages) == 1
        message = messages[0]
        assert message["role"] == "assistant"
        assert [part["text"] for part in message["content"] if part["type"] == "text"] == ["Checking ", "two sources."]
        assert message["content"][0]["encrypted_content"] == "opaque-reasoning"
        calls = message["tool_calls"]
        assert [call["id"] for call in calls] == ["call-list", "call-glob"]
        assert [call["function"]["name"] for call in calls] == ["list_models", "glob"]
        assert [json.loads(call["function"]["arguments"]) for call in calls] == [{}, {"pattern": "*.yml"}]
        for index, call in enumerate(calls):
            prefix = f"llm.output_messages.0.message.tool_calls.{index}.tool_call"
            assert attrs[f"{prefix}.id"] == call["id"]
            assert attrs[f"{prefix}.function.name"] == call["function"]["name"]
            assert attrs[f"{prefix}.function.arguments"] == call["function"]["arguments"]
        prefix = "llm.output_messages.0.message.contents.0.message_content"
        assert attrs[f"{prefix}.type"] == "reasoning"
        assert attrs[f"{prefix}.text"] == "Check sources."
        assert attrs[f"{prefix}.id"] == "reasoning-1"
        assert attrs[f"{prefix}.encrypted_content"] == "opaque-reasoning"
        assert json.loads(attrs["input.value"]) == {
            "messages": [{"role": "user", "content": "Inspect available sources."}]
        }
        assert not any(key.startswith("gen_ai.") for key in attrs)
    finally:
        processor.shutdown()
        provider.shutdown()


def test_streamed_response_masks_normalized_tool_calls_and_reasoning(streamed_generation_output):
    from agents.tracing.span_data import GenerationSpanData
    from openinference.instrumentation import OITracer, TraceConfig

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    processor = DatusOpenInferenceTracingProcessor(
        OITracer(provider.get_tracer(__name__), config=TraceConfig(hide_outputs=True))
    )
    data = GenerationSpanData(output=streamed_generation_output, model="test-model")
    span = FakeSpan(span_data=data, span_id="span_generation", started_at=_now_iso())
    try:
        processor.on_trace_start(FakeTrace())
        processor.on_span_start(span)
        span.ended_at = _now_iso()
        processor.on_span_end(span)
        processor.on_trace_end(FakeTrace())
        assert data.output is streamed_generation_output
        attrs = next(s.attributes for s in exporter.get_finished_spans() if s.name == "generation")
        assert not any(key.startswith("llm.output_messages.") for key in attrs)
        assert "gen_ai.output.messages" not in attrs
        exported = json.dumps(dict(attrs))
        assert "call-glob" not in exported
        assert "opaque-reasoning" not in exported
        assert "Check sources." not in exported
    finally:
        processor.shutdown()
        provider.shutdown()


@pytest.mark.parametrize(
    "output",
    [
        [{"role": "assistant", "content": "Already a message."}],
        [{"object": "response", "output": [{"type": "future_provider_item", "value": "preserve"}]}],
        [{"object": "response", "output": [{"type": "message", "content": "malformed"}]}],
        None,
    ],
)
def test_generation_normalization_leaves_other_formats_to_upstream(output):
    assert openai_agents_module._generation_output_messages(output) is None
