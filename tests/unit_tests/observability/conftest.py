# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import pytest
from agents import set_trace_processors, set_tracing_disabled
from openinference.instrumentation import TraceConfig
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from datus.observability.config import TracingConfig
from datus.observability.manager import ObservabilityManager
from datus.observability.openai_agents import instrument_openai_agents


@pytest.fixture
def exported_calls(monkeypatch):
    import datus.observability.model_call as module

    manager = ObservabilityManager()
    manager._tracing_config = TracingConfig.from_dict({"enabled": True})
    monkeypatch.setattr(module, "get_observability_manager", lambda: manager)
    provider = TracerProvider(span_limits=SpanLimits(max_attributes=4096))
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrument_openai_agents(tracer_provider=provider, config=TraceConfig())
    set_tracing_disabled(False)
    yield exporter, manager, provider
    set_trace_processors([])
    provider.shutdown()
