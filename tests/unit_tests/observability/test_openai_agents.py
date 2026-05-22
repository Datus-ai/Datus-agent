# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from opentelemetry import baggage, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from datus.observability.adapters.langfuse import _LangfuseBaggageSpanProcessor
from datus.observability.adapters.otlp import _BaggageAttributeSpanProcessor
from datus.observability.openai_agents import DatusOpenInferenceTracingProcessor


def test_openai_agents_processor_merges_first_agent_span_into_trace_root():
    from agents import set_trace_processors
    from agents.tracing import agent_span, function_span, trace
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

    try:
        set_trace_processors([DatusOpenInferenceTracingProcessor(tracer)])
        with trace("agent/chat", group_id="session-1"):
            with agent_span("chat"):
                with function_span("describe_table"):
                    pass
    finally:
        context.detach(token)
        set_trace_processors([])
        provider.shutdown()

    spans = exporter.get_finished_spans()
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
