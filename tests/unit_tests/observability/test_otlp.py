# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from types import SimpleNamespace

from opentelemetry import baggage, context
from opentelemetry.sdk.trace import TracerProvider

from datus.observability.adapters.otlp import _BaggageAttributeSpanProcessor, _build_openinference_trace_config
from datus.observability.config import ObservabilityConfig


class DummyTraceConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummySpan:
    def __init__(self, trace_id=0x123):
        self.attributes = {}
        self.trace_id = trace_id

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def get_span_context(self):
        return SimpleNamespace(trace_id=self.trace_id)


def test_openinference_trace_config_keeps_content_visible_by_default():
    cfg = ObservabilityConfig.from_dict({"tracing": {"enabled": True}})

    trace_config = _build_openinference_trace_config(DummyTraceConfig, cfg.tracing)

    assert trace_config.kwargs["hide_inputs"] is False
    assert trace_config.kwargs["hide_outputs"] is False
    assert trace_config.kwargs["hide_prompts"] is False
    assert trace_config.kwargs["hide_choices"] is False


def test_baggage_span_processor_copies_only_provider_neutral_attributes():
    parent_context = context.get_current()
    parent_context = baggage.set_baggage("session.id", "session-1", context=parent_context)
    parent_context = baggage.set_baggage("user.id", "user-1", context=parent_context)
    parent_context = baggage.set_baggage("datus.trace.name", "chat/session", context=parent_context)
    parent_context = baggage.set_baggage("datus.run_id", "run-1", context=parent_context)
    parent_context = baggage.set_baggage("langfuse.session.id", "session-1", context=parent_context)
    span = DummySpan()

    _BaggageAttributeSpanProcessor().on_start(span, parent_context)

    assert span.attributes == {
        "session.id": "session-1",
        "user.id": "user-1",
        "datus.trace.name": "chat/session",
        "datus.run_id": "run-1",
    }


def test_baggage_span_processor_reuses_trace_attributes_when_child_context_loses_baggage():
    processor = _BaggageAttributeSpanProcessor()
    parent_context = context.get_current()
    parent_context = baggage.set_baggage("session.id", "session-1", context=parent_context)
    parent_context = baggage.set_baggage("datus.trace.name", "chat/session", context=parent_context)
    root = DummySpan(trace_id=0xABC)
    child = DummySpan(trace_id=0xABC)

    processor.on_start(root, parent_context)
    processor.on_start(child, context.get_current())

    assert child.attributes["session.id"] == "session-1"
    assert child.attributes["datus.trace.name"] == "chat/session"


def test_baggage_span_processor_works_with_real_span_lifecycle():
    provider = TracerProvider()
    provider.add_span_processor(_BaggageAttributeSpanProcessor())
    tracer = provider.get_tracer(__name__)
    parent_context = context.get_current()
    parent_context = baggage.set_baggage("session.id", "session-1", context=parent_context)
    token = context.attach(parent_context)

    try:
        with tracer.start_as_current_span("chat") as span:
            pass
    finally:
        context.detach(token)
        provider.shutdown()

    assert span.attributes["session.id"] == "session-1"


def test_openinference_trace_config_hides_content_when_capture_disabled():
    cfg = ObservabilityConfig.from_dict({"tracing": {"enabled": True, "capture_content": False}})

    trace_config = _build_openinference_trace_config(DummyTraceConfig, cfg.tracing)

    assert trace_config.kwargs["hide_inputs"] is True
    assert trace_config.kwargs["hide_outputs"] is True
    assert trace_config.kwargs["hide_input_messages"] is True
    assert trace_config.kwargs["hide_output_messages"] is True


def test_openinference_trace_config_honors_prompt_response_overrides():
    cfg = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": True,
                "capture_content": False,
                "capture": {
                    "prompts": True,
                    "responses": True,
                },
            }
        }
    )

    trace_config = _build_openinference_trace_config(DummyTraceConfig, cfg.tracing)

    assert trace_config.kwargs["hide_inputs"] is False
    assert trace_config.kwargs["hide_outputs"] is False
    assert trace_config.kwargs["hide_input_messages"] is False
    assert trace_config.kwargs["hide_output_messages"] is False
    assert trace_config.kwargs["hide_prompts"] is False
    assert trace_config.kwargs["hide_choices"] is False
