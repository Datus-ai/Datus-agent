# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from opentelemetry import baggage, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from datus.observability.adapters.langfuse import _LangfuseBaggageSpanProcessor
from datus.observability.adapters.otlp import _BaggageAttributeSpanProcessor
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    assert child.parent is not None
    assert child.parent.span_id == root.context.span_id


def test_openai_agents_processor_fails_lazily_when_dependency_missing(monkeypatch):
    import datus.observability.openai_agents as module

    monkeypatch.setattr(module, "_oi_processor", None)

    with pytest.raises(RuntimeError, match="openinference"):
        module.DatusOpenInferenceTracingProcessor(object())
