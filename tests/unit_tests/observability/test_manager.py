# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pytest
from opentelemetry import baggage

from datus.observability.config import ObservabilityConfig
from datus.observability.manager import ObservabilityManager, _trace_baggage_attributes
from datus.observability.reference import TraceReference


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


def test_span_propagates_body_exceptions():
    manager = ObservabilityManager()
    manager._adapters = [FakeAdapter()]

    with pytest.raises(RuntimeError, match="boom"):
        with manager.span("llm.generate"):
            raise RuntimeError("boom")


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
