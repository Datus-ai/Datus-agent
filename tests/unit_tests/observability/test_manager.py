# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import contextvars
from contextlib import contextmanager

import pytest
from opentelemetry import baggage

from datus.observability.config import ObservabilityConfig
from datus.observability.manager import (
    ObservabilityManager,
    _current_trace_run_id,
    _format_span_id,
    _format_trace_id,
    _set_span_attributes,
    _string_attr,
    _trace_baggage_attributes,
)
from datus.observability.reference import TraceReference
from datus.utils.trace_context import TraceContext, trace_context


class FakeAdapter:
    name = "fake"
    capabilities = {"traces"}

    def __init__(self):
        self.events = []
        self.flushed = False
        self.shutdown_called = False

    def setup(self, adapter_config, tracing_config):
        self.adapter_config = adapter_config
        self.tracing_config = tracing_config

    def record_event(self, event):
        self.events.append(event)

    def flush(self):
        self.flushed = True

    def shutdown(self):
        self.shutdown_called = True


class ExplodingAdapter(FakeAdapter):
    name = "exploding"

    def record_event(self, event):
        raise RuntimeError("record failed")


class ExplodingSetupAdapter(FakeAdapter):
    def setup(self, adapter_config, tracing_config):
        raise RuntimeError("setup failed")


class NoNameExplodingAdapter:
    def record_event(self, event):
        raise RuntimeError("record failed")

    def flush(self):
        raise RuntimeError("flush failed")

    def shutdown(self):
        raise RuntimeError("shutdown failed")


def test_manager_initializes_registered_adapters(monkeypatch):
    from datus.observability import manager as manager_module

    monkeypatch.setattr(manager_module.adapter_registry, "get", lambda adapter_type: FakeAdapter)
    manager = ObservabilityManager()
    config = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": True,
                "adapters": [{"type": "fake"}],
            }
        }
    )

    assert manager.configure(config) is True
    assert len(manager.adapters) == 1


def test_manager_can_configure_after_initial_disabled_call(monkeypatch):
    from datus.observability import manager as manager_module

    manager = ObservabilityManager()
    assert manager.configure(None) is False
    assert manager.initialized is False

    monkeypatch.setattr(manager_module.adapter_registry, "get", lambda adapter_type: FakeAdapter)
    config = ObservabilityConfig.from_dict({"tracing": {"enabled": True, "adapters": [{"type": "fake"}]}})

    assert manager.configure(config) is True
    assert manager.initialized is True
    assert len(manager.adapters) == 1


def test_manager_can_retry_after_all_adapters_fail(monkeypatch):
    from datus.observability import manager as manager_module

    config = ObservabilityConfig.from_dict({"tracing": {"enabled": True, "adapters": [{"type": "fake"}]}})
    manager = ObservabilityManager()
    monkeypatch.setattr(manager_module.adapter_registry, "get", lambda adapter_type: ExplodingSetupAdapter)

    assert manager.configure(config) is False
    assert manager.initialized is False

    monkeypatch.setattr(manager_module.adapter_registry, "get", lambda adapter_type: FakeAdapter)
    assert manager.configure(config) is True
    assert len(manager.adapters) == 1


def test_manager_skips_disabled_and_unknown_adapters(monkeypatch):
    from datus.observability import manager as manager_module

    looked_up = []
    monkeypatch.setattr(manager_module.adapter_registry, "get", lambda adapter_type: looked_up.append(adapter_type))
    manager = ObservabilityManager()
    config = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": True,
                "adapters": [
                    {"type": "disabled", "enabled": False},
                    {"type": "missing"},
                ],
            }
        }
    )

    assert manager.configure(config) is False
    assert manager.initialized is False
    assert looked_up == ["missing"]


def test_manager_initialized_short_circuit_returns_adapter_state():
    manager = ObservabilityManager()
    manager._initialized = True

    assert manager.configure(None) is False

    manager._adapters = [FakeAdapter()]
    assert manager.configure(None) is True


def test_manager_content_controls_and_redaction(monkeypatch):
    from datus.observability import manager as manager_module

    monkeypatch.setattr(manager_module.adapter_registry, "get", lambda adapter_type: FakeAdapter)
    manager = ObservabilityManager()
    config = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": True,
                "capture_content": False,
                "capture": {"prompts": True},
                "adapters": [{"type": "fake"}],
            }
        }
    )

    assert manager.content_enabled("prompts") is False
    value = {"api_key": "secret"}
    assert manager.redact(value) is value

    assert manager.configure(config) is True
    assert manager.content_enabled("prompts") is True
    assert manager.content_enabled("responses") is False
    assert manager.redact(value) == {"api_key": "[REDACTED]"}


def test_manager_isolates_adapter_record_failures(monkeypatch):
    from datus.observability import manager as manager_module

    monkeypatch.setattr(manager_module.adapter_registry, "get", lambda adapter_type: ExplodingAdapter)
    manager = ObservabilityManager()
    config = ObservabilityConfig.from_dict(
        {
            "tracing": {
                "enabled": True,
                "adapters": [{"type": "fake"}],
            }
        }
    )

    assert manager.configure(config) is True
    manager.record_event({"kind": "llm"})


def test_manager_exception_handlers_tolerate_adapters_without_name():
    manager = ObservabilityManager()
    manager._adapters = [NoNameExplodingAdapter()]

    manager.record_event({"kind": "llm"})
    manager.flush()
    manager.shutdown()

    assert manager.adapters == []


def test_span_propagates_body_exceptions():
    manager = ObservabilityManager()
    manager._adapters = [FakeAdapter()]

    with pytest.raises(RuntimeError, match="boom"):
        with manager.span("llm.generate"):
            raise RuntimeError("boom")


def test_span_without_adapters_is_noop():
    manager = ObservabilityManager()

    with manager.span("chat") as span:
        assert span is None

    with manager.trace_baggage("chat"):
        assert baggage.get_baggage("datus.trace.name") is None


def test_span_records_context_local_trace_reference(monkeypatch):
    from opentelemetry import trace as otel_trace

    class FakeSpanContext:
        is_valid = True
        trace_id = 0x123
        span_id = 0x456

    class FakeSpan:
        def __init__(self):
            self.attributes = {}

        def get_span_context(self):
            return FakeSpanContext()

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class FakeTracer:
        def __init__(self, span):
            self.span = span

        @contextmanager
        def start_as_current_span(self, name):
            yield self.span

    span = FakeSpan()
    monkeypatch.setattr(otel_trace, "get_tracer", lambda name: FakeTracer(span))
    manager = ObservabilityManager()
    manager._adapters = [FakeAdapter()]

    with manager.span(
        "chat",
        {"datus.trace.name": "agent/chat", "datus.session_id": "session-1"},
        run_id="run-1",
    ):
        ref = manager.get_trace_reference()
        assert ref == TraceReference(
            trace_id="00000000000000000000000000000123",
            span_id="0000000000000456",
            run_id="run-1",
            provider="fake",
        )
        assert span.attributes["datus.trace.name"] == "agent/chat"
        assert span.attributes["session.id"] == "session-1"
        assert span.attributes["datus.trace_id"] == ref.trace_id

    assert manager.get_trace_reference() == ref


def test_trace_reference_metadata_shape():
    ref = TraceReference(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        run_id="run1",
        provider="otlp",
    )

    assert ref.to_metadata() == {
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "trace_span_id": "00f067aa0ba902b7",
        "trace_run_id": "run1",
        "trace_provider": "otlp",
    }


def test_last_trace_reference_is_context_local(monkeypatch):
    ref = TraceReference(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        run_id="run1",
        provider="otlp",
    )
    manager = ObservabilityManager()

    monkeypatch.setattr(manager, "_current_otel_trace_reference", lambda: ref)
    assert manager.get_trace_reference() == ref

    monkeypatch.setattr(manager, "_current_otel_trace_reference", lambda: None)
    assert manager.get_trace_reference() == ref
    assert contextvars.Context().run(manager.get_trace_reference) is None


def test_trace_baggage_attributes_are_provider_neutral():
    attrs = _trace_baggage_attributes(
        "chat",
        {
            "datus.trace.name": "agent/chat",
            "datus.session_id": "session-1",
            "datus.user_id": "user-1",
            "datus.run_id": "run-1",
        },
    )

    assert attrs == {
        "datus.trace.name": "agent/chat",
        "session.id": "session-1",
        "user.id": "user-1",
        "datus.run_id": "run-1",
    }
    assert all(not key.startswith("langfuse.") for key in attrs)


def test_trace_baggage_attributes_accept_fallback_metadata_keys():
    attrs = _trace_baggage_attributes(
        "  ",
        {
            "session.id": " session-2 ",
            "user.id": "user-2",
            "datus.metadata.benchmark_run_id": "bench-1",
        },
    )

    assert attrs == {
        "session.id": "session-2",
        "user.id": "user-2",
        "datus.run_id": "bench-1",
    }
    assert _string_attr("  ") is None


def test_trace_baggage_attaches_provider_neutral_context():
    manager = ObservabilityManager()
    manager._adapters = [FakeAdapter()]

    with manager.trace_baggage(
        "chat",
        {
            "datus.trace.name": "agent/chat",
            "datus.session_id": "session-1",
            "datus.user_id": "user-1",
        },
    ):
        assert baggage.get_baggage("datus.trace.name") == "agent/chat"
        assert baggage.get_baggage("session.id") == "session-1"
        assert baggage.get_baggage("user.id") == "user-1"

    assert baggage.get_baggage("session.id") is None


def test_trace_reference_helpers_handle_invalid_spans_and_context():
    class InvalidSpan:
        def get_span_context(self):
            return type("SpanContext", (), {"is_valid": False})()

    class BrokenSpan:
        def get_span_context(self):
            raise RuntimeError("bad span")

    manager = ObservabilityManager()

    assert manager._trace_reference_from_span(InvalidSpan()) is None
    assert manager._trace_reference_from_span(BrokenSpan()) is None
    assert _format_trace_id(0xA) == "0000000000000000000000000000000a"
    assert _format_span_id(0xB) == "000000000000000b"

    assert _current_trace_run_id() is None
    with trace_context(TraceContext(name="chat", session_id="session-1", metadata={"run_id": "run-1"}), replace=True):
        assert _current_trace_run_id() == "run-1"
    with trace_context(
        TraceContext(name="chat", session_id="session-1", metadata={"benchmark_run_id": "bench-1"}), replace=True
    ):
        assert _current_trace_run_id() == "bench-1"
    with trace_context(TraceContext(name="chat", session_id="session-1"), replace=True):
        assert _current_trace_run_id() == "session-1"


def test_set_span_attributes_stringifies_and_ignores_failures():
    class FakeSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            if key == "bad":
                raise RuntimeError("bad attribute")
            self.attributes[key] = value

    span = FakeSpan()

    _set_span_attributes(
        span,
        {
            "none": None,
            "text": "value",
            "flag": True,
            "count": 3,
            "payload": {"a": 1},
            "bad": "ignored",
        },
    )

    assert span.attributes == {
        "text": "value",
        "flag": True,
        "count": 3,
        "payload": "{'a': 1}",
    }


def test_module_helpers_register_flush_and_delegate(monkeypatch):
    from datus.observability import manager as manager_module

    ref = TraceReference(trace_id="trace", span_id="span", run_id="run", provider="fake")

    class FakeManager:
        def __init__(self):
            self.shutdown_called = False
            self.flush_called = False

        def configure(self, config):
            return True

        def shutdown(self):
            self.shutdown_called = True

        def get_trace_reference(self):
            return ref

        def flush(self):
            self.flush_called = True
            raise RuntimeError("flush failed")

    fake_manager = FakeManager()
    registered = []
    monkeypatch.setattr(manager_module, "_manager", fake_manager)
    monkeypatch.setattr(manager_module, "_atexit_flush_registered", False)
    monkeypatch.setattr(manager_module.atexit, "register", lambda callback: registered.append(callback))

    assert manager_module.configure_observability(ObservabilityConfig()) is True
    assert manager_module.configure_observability(ObservabilityConfig()) is True
    assert registered == [manager_module._flush_observability_at_exit]
    assert manager_module.get_trace_reference() == ref

    manager_module.shutdown_observability()
    manager_module._flush_observability_at_exit()

    assert fake_manager.shutdown_called is True
    assert fake_manager.flush_called is True
