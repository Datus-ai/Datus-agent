# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import base64
from types import SimpleNamespace

import pytest
from opentelemetry import baggage, context
from opentelemetry.sdk.trace import TracerProvider

from datus.observability.adapters.langfuse import (
    LangfuseAdapter,
    _langfuse_attributes_from_baggage,
    _LangfuseBaggageSpanProcessor,
)
from datus.observability.adapters.platforms import BraintrustAdapter, DatadogAdapter, LangSmithAdapter
from datus.observability.config import ObservabilityAdapterConfig
from datus.observability.registry import ObservabilityAdapterRegistry
from datus.utils.exceptions import DatusException


class DummySpan:
    def __init__(self, trace_id=0x123, name="span", attributes=None):
        self.name = name
        self.attributes = dict(attributes or {})
        self.trace_id = trace_id

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def get_span_context(self):
        return SimpleNamespace(trace_id=self.trace_id)


def test_registry_includes_builtin_platform_adapters():
    ObservabilityAdapterRegistry._initialized = False
    ObservabilityAdapterRegistry._adapters = {}

    assert ObservabilityAdapterRegistry.get("langfuse") is LangfuseAdapter
    assert ObservabilityAdapterRegistry.get("langsmith") is LangSmithAdapter
    assert ObservabilityAdapterRegistry.get("braintrust") is BraintrustAdapter
    assert ObservabilityAdapterRegistry.get("datadog") is DatadogAdapter


def test_langfuse_adapter_builds_otlp_endpoint_and_basic_auth(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.test")
    config = ObservabilityAdapterConfig.from_dict({"type": "langfuse"})

    resolved = LangfuseAdapter().resolve_adapter_config(config)

    auth = base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
    assert resolved.endpoint == "https://cloud.langfuse.test/api/public/otel/v1/traces"
    assert resolved.headers["Authorization"] == f"Basic {auth}"
    assert resolved.headers["x-langfuse-ingestion-version"] == "4"


def test_langfuse_adapter_maps_generic_baggage_to_langfuse_attributes():
    attrs = _langfuse_attributes_from_baggage(
        {
            "datus.trace.name": "agent/chat",
            "session.id": "session-1",
            "user.id": "user-1",
            "datus.run_id": "run-1",
        }
    )

    assert attrs == {
        "langfuse.trace.name": "agent/chat",
        "langfuse.session.id": "session-1",
        "langfuse.trace.metadata.session_id": "session-1",
        "langfuse.trace.metadata.datus_session_id": "session-1",
        "langfuse.user.id": "user-1",
        "langfuse.trace.metadata.run_id": "run-1",
    }


def test_langfuse_baggage_processor_reuses_trace_attributes_when_child_context_loses_baggage():
    processor = _LangfuseBaggageSpanProcessor()
    parent_context = context.get_current()
    parent_context = baggage.set_baggage("datus.trace.name", "agent/chat", context=parent_context)
    parent_context = baggage.set_baggage("session.id", "session-1", context=parent_context)
    root = DummySpan(trace_id=0xABC)
    child = DummySpan(trace_id=0xABC)

    processor.on_start(root, parent_context)
    processor.on_start(child, context.get_current())

    assert child.attributes["langfuse.trace.name"] == "agent/chat"
    assert child.attributes["langfuse.session.id"] == "session-1"


def test_langfuse_baggage_processor_shutdown_clears_cached_trace_attributes():
    processor = _LangfuseBaggageSpanProcessor()
    parent_context = context.get_current()
    parent_context = baggage.set_baggage("datus.trace.name", "agent/chat", context=parent_context)
    parent_context = baggage.set_baggage("session.id", "session-1", context=parent_context)
    root = DummySpan(trace_id=0xABC)
    child = DummySpan(trace_id=0xABC)

    processor.on_start(root, parent_context)
    processor.shutdown()
    processor.on_start(child, context.get_current())

    assert "langfuse.session.id" not in child.attributes


def test_langfuse_baggage_processor_works_with_real_span_lifecycle():
    provider = TracerProvider()
    provider.add_span_processor(_LangfuseBaggageSpanProcessor())
    tracer = provider.get_tracer(__name__)
    parent_context = context.get_current()
    parent_context = baggage.set_baggage("datus.trace.name", "agent/chat", context=parent_context)
    parent_context = baggage.set_baggage("session.id", "session-1", context=parent_context)
    token = context.attach(parent_context)

    try:
        with tracer.start_as_current_span("chat") as span:
            pass
    finally:
        context.detach(token)
        provider.shutdown()

    assert span.attributes["langfuse.trace.name"] == "agent/chat"
    assert span.attributes["langfuse.session.id"] == "session-1"


def test_langfuse_adapter_requires_key_pair(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    config = ObservabilityAdapterConfig.from_dict({"type": "langfuse"})

    with pytest.raises(DatusException, match="LANGFUSE_PUBLIC_KEY"):
        LangfuseAdapter().resolve_adapter_config(config)


def test_langsmith_adapter_builds_otlp_headers(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "datus-prod")
    config = ObservabilityAdapterConfig.from_dict({"type": "langsmith"})

    resolved = LangSmithAdapter().resolve_adapter_config(config)

    assert resolved.endpoint == "https://api.smith.langchain.com/otel/v1/traces"
    assert resolved.headers == {"x-api-key": "ls-key", "Langsmith-Project": "datus-prod"}


def test_langsmith_adapter_accepts_langchain_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-key")
    config = ObservabilityAdapterConfig.from_dict({"type": "langsmith", "base_url": "https://eu.api.smith.test"})

    resolved = LangSmithAdapter().resolve_adapter_config(config)

    assert resolved.endpoint == "https://eu.api.smith.test/otel/v1/traces"
    assert resolved.headers["x-api-key"] == "lc-key"


def test_langsmith_adapter_requires_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    config = ObservabilityAdapterConfig.from_dict({"type": "langsmith"})

    with pytest.raises(DatusException, match="LANGSMITH_API_KEY"):
        LangSmithAdapter().resolve_adapter_config(config)


def test_braintrust_adapter_builds_parent_header(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-key")
    monkeypatch.setenv("BRAINTRUST_PROJECT_ID", "proj-123")
    config = ObservabilityAdapterConfig.from_dict({"type": "braintrust"})

    resolved = BraintrustAdapter().resolve_adapter_config(config)

    assert resolved.endpoint == "https://api.braintrust.dev/otel/v1/traces"
    assert resolved.headers == {"Authorization": "Bearer bt-key", "x-bt-parent": "project_id:proj-123"}


def test_braintrust_adapter_requires_parent(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-key")
    monkeypatch.delenv("BRAINTRUST_PARENT", raising=False)
    monkeypatch.delenv("BRAINTRUST_PROJECT_ID", raising=False)
    monkeypatch.delenv("BRAINTRUST_PROJECT_NAME", raising=False)
    config = ObservabilityAdapterConfig.from_dict({"type": "braintrust"})

    with pytest.raises(DatusException, match="parent"):
        BraintrustAdapter().resolve_adapter_config(config)


def test_datadog_adapter_defaults_to_local_agent(monkeypatch):
    monkeypatch.delenv("DD_API_KEY", raising=False)
    config = ObservabilityAdapterConfig.from_dict({"type": "datadog"})

    resolved = DatadogAdapter().resolve_adapter_config(config)

    assert resolved.endpoint == "http://localhost:4318/v1/traces"
    assert resolved.headers == {}


def test_langfuse_maps_join_keys_onto_the_trace():
    """The whole chain, end to end: trace context -> span attrs -> baggage -> Langfuse.

    Driven by the real producers rather than a hand-built baggage dict, because
    the failure this guards against is precisely the three layers disagreeing
    about a key's spelling -- and a hand-built input would encode the mistake on
    both sides and pass.
    """
    import json as _json

    from datus.observability.manager import _trace_baggage_attributes
    from datus.utils.trace_context import build_chat_trace_context, build_trace_span_attributes

    ctx = build_chat_trace_context(
        session_id="chat_session_1",
        node_name="olist_order_analyst",
        max_turns=30,
        release="0.4.0",
        identity={"workspace_id": "ws_abc", "tenant": "acme"},
    )
    span_attributes = build_trace_span_attributes(operation="chat", ctx=ctx)
    baggage_attrs = _trace_baggage_attributes("chat", span_attributes)

    attrs = _langfuse_attributes_from_baggage(baggage_attrs)

    assert attrs["langfuse.trace.name"] == "agent/olist_order_analyst"
    assert attrs["langfuse.session.id"] == "chat_session_1"
    assert attrs["langfuse.trace.metadata.node_name"] == "olist_order_analyst"
    assert attrs["langfuse.trace.metadata.max_turns"] == "30"
    assert attrs["langfuse.trace.metadata.release"] == "0.4.0"
    # release also gets Langfuse's first-class field so runs group by deploy
    assert attrs["langfuse.release"] == "0.4.0"
    # host identity keeps the host's own key names -- the adapter strips only
    # the namespace and never interprets what is left
    assert attrs["langfuse.trace.metadata.workspace_id"] == "ws_abc"
    assert attrs["langfuse.trace.metadata.tenant"] == "acme"
    assert set(_json.loads(attrs["langfuse.trace.tags"])) == {
        "chat",
        "agent:olist_order_analyst",
        "workspace_id:ws_abc",
        "tenant:acme",
    }


def test_langfuse_omits_join_keys_that_were_never_set():
    """Absent keys must not become empty strings on the trace."""
    attrs = _langfuse_attributes_from_baggage({"datus.trace.name": "agent/chat"})

    assert attrs == {"langfuse.trace.name": "agent/chat"}
