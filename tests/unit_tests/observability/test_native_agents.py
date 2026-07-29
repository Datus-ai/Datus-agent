# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
from types import SimpleNamespace

import pytest
from agents.tracing.provider import DefaultTraceProvider
from agents.tracing.setup import get_trace_provider, set_trace_provider
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from datus.observability.adapters.langfuse import _LangfuseBaggageSpanProcessor
from datus.observability.adapters.otlp import _BaggageAttributeSpanProcessor
from datus.observability.config import TracingConfig
from datus.observability.manager import ObservabilityManager
from datus.observability.native_agents import (
    capture_native_trace_content,
    finish_native_span,
    start_native_generation_span,
    start_native_tool_span,
    trace_native_agent_stream,
)
from datus.observability.openai_agents import DatusOpenInferenceTracingProcessor
from datus.utils.trace_context import TraceContext, trace_context


@pytest.mark.asyncio
async def test_native_agent_stream_exports_agent_generation_tool_tree(monkeypatch):
    exporter = InMemorySpanExporter()
    otel_provider = TracerProvider()
    otel_provider.add_span_processor(_BaggageAttributeSpanProcessor())
    otel_provider.add_span_processor(_LangfuseBaggageSpanProcessor())
    otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
    oi_tracer = OITracer(otel_provider.get_tracer(__name__), config=TraceConfig())
    processor = DatusOpenInferenceTracingProcessor(oi_tracer)

    agents_provider = DefaultTraceProvider()
    agents_provider.set_processors([processor])
    previous_agents_provider = get_trace_provider()
    set_trace_provider(agents_provider)

    observability = ObservabilityManager()
    observability._adapters = [object()]
    observability._tracing_config = TracingConfig(enabled=True)
    monkeypatch.setattr(
        "datus.observability.native_agents.get_observability_manager",
        lambda: observability,
    )

    @trace_native_agent_stream
    async def native_loop(self, prompt, func_tools=None, output_type=dict, **kwargs):
        generation = start_native_generation_span(
            input_messages=capture_native_trace_content(
                "prompts",
                [{"role": "user", "content": prompt, "api_key": "must-not-leak"}],
            ),
            model="custom/sonnet",
            model_config={"provider": "anthropic", "system": "anthropic", "max_tokens": 1024},
        )
        generation.span_data.usage = {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 10,
        }
        finish_native_span(
            generation,
            output=capture_native_trace_content(
                "responses",
                [{"role": "assistant", "content": "calling sync_semantic"}],
            ),
        )

        tool = start_native_tool_span(
            name="sync_semantic",
            tool_call_id="toolu_123",
            input_value={"semantic_model": "orders", "token": "must-not-leak"},
        )
        finish_native_span(
            tool,
            output=capture_native_trace_content("tool_results", {"success": True}),
        )
        failed_tool = start_native_tool_span(
            name="publish_semantic_model",
            tool_call_id="toolu_456",
            input_value={"semantic_model": "orders"},
        )
        finish_native_span(
            failed_tool,
            output=capture_native_trace_content("tool_results", {"success": False, "error": "sync failed"}),
            error="sync failed",
        )
        yield "done"

    ctx = TraceContext(
        name="agent/gen_metrics",
        session_id="gen_metrics_session_test",
        user_id="user-1",
    )
    try:
        with trace_context(ctx):
            items = [
                item
                async for item in native_loop(
                    object(),
                    "build metrics",
                    func_tools=[SimpleNamespace(name="sync_semantic")],
                    agent_name="gen_metrics",
                )
            ]
        assert items == ["done"]
        assert agents_provider.get_current_trace() is None
        assert agents_provider.get_current_span() is None
    finally:
        set_trace_provider(previous_agents_provider)
        agents_provider.shutdown()

    spans = exporter.get_finished_spans()
    otel_provider.shutdown()
    span_by_name = {span.name: span for span in spans}

    assert sorted(span_by_name) == [
        "agent/gen_metrics",
        "generation",
        "publish_semantic_model",
        "sync_semantic",
    ]
    root = span_by_name["agent/gen_metrics"]
    generation = span_by_name["generation"]
    tool = span_by_name["sync_semantic"]

    assert root.attributes["openinference.span.kind"] == "AGENT"
    assert root.attributes["langfuse.session.id"] == "gen_metrics_session_test"
    assert generation.parent.span_id == root.context.span_id
    assert generation.attributes["openinference.span.kind"] == "LLM"
    assert generation.attributes["llm.model_name"] == "custom/sonnet"
    assert generation.attributes["llm.provider"] == "anthropic"
    assert generation.attributes["llm.system"] == "anthropic"
    assert generation.attributes["llm.token_count.prompt"] == 120
    assert generation.attributes["llm.token_count.completion"] == 30
    assert generation.attributes["llm.token_count.total"] == 150
    assert generation.attributes["llm.token_count.prompt_details.cache_read"] == 100
    assert generation.attributes["llm.token_count.prompt_details.cache_write"] == 10
    assert "[REDACTED]" in generation.attributes["input.value"]
    assert "must-not-leak" not in generation.attributes["input.value"]

    assert tool.parent.span_id == root.context.span_id
    assert tool.attributes["openinference.span.kind"] == "TOOL"
    assert tool.attributes["tool.id"] == "toolu_123"
    assert "[REDACTED]" in tool.attributes["input.value"]
    assert "must-not-leak" not in tool.attributes["input.value"]
    assert tool.attributes["output.value"] == '{"success": true}'
    failed_tool = span_by_name["publish_semantic_model"]
    assert failed_tool.parent.span_id == root.context.span_id
    assert failed_tool.status.status_code.name == "ERROR"
    assert failed_tool.attributes["output.value"] == '{"success": false, "error": "sync failed"}'


def test_native_trace_content_honors_per_field_capture(monkeypatch):
    observability = ObservabilityManager()
    observability._adapters = [object()]
    observability._tracing_config = TracingConfig.from_dict(
        {
            "enabled": True,
            "capture_content": True,
            "capture": {"prompts": False, "tool_args": False},
        }
    )
    monkeypatch.setattr(
        "datus.observability.native_agents.get_observability_manager",
        lambda: observability,
    )

    assert capture_native_trace_content("prompts", "hidden") is None
    assert capture_native_trace_content("tool_args", {"query": "hidden"}) is None
    assert capture_native_trace_content("responses", "visible") == "visible"
    assert capture_native_trace_content(
        "responses",
        {"tool_calls": [{"function": {"arguments": '{"token": "hidden", "model": "orders"}'}}]},
    ) == {"tool_calls": [{"function": {"arguments": '{"token": "[REDACTED]", "model": "orders"}'}}]}


@pytest.mark.asyncio
async def test_native_agent_stream_resets_context_when_consumer_closes(monkeypatch):
    observability = ObservabilityManager()
    observability._adapters = [object()]
    observability._tracing_config = TracingConfig(enabled=True)
    monkeypatch.setattr(
        "datus.observability.native_agents.get_observability_manager",
        lambda: observability,
    )

    agents_provider = DefaultTraceProvider()
    previous_agents_provider = get_trace_provider()
    set_trace_provider(agents_provider)

    @trace_native_agent_stream
    async def native_loop(self, **kwargs):
        yield "first"
        yield "second"

    stream = native_loop(object(), agent_name="chat")
    try:
        assert await anext(stream) == "first"
        # The traced loop runs in a dedicated producer task, so SDK/OTel context
        # tokens never leak into the consumer and can be closed from another task.
        assert agents_provider.get_current_trace() is None
        assert agents_provider.get_current_span() is None
        await asyncio.create_task(stream.aclose())
        assert agents_provider.get_current_trace() is None
        assert agents_provider.get_current_span() is None
    finally:
        await stream.aclose()
        set_trace_provider(previous_agents_provider)
        agents_provider.shutdown()
