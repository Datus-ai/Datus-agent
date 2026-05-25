# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Built-in OTLP adapter for baseline trace export."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from datus.observability.config import ObservabilityAdapterConfig, TracingConfig
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OtlpAdapter:
    name = "otlp"
    capabilities = {"traces", "otlp"}
    _lock = threading.RLock()
    _tracer_provider = None
    _instrumentor = None
    _shutdown = False

    def __init__(self) -> None:
        self._processor = None

    def setup(self, adapter_config: ObservabilityAdapterConfig, tracing_config: TracingConfig) -> None:
        resolved_config = self.resolve_adapter_config(adapter_config)
        self._setup_otlp_exporter(resolved_config, tracing_config)
        logger.info("%s tracing enabled for endpoint %s", self.name.upper(), resolved_config.endpoint)

    def resolve_adapter_config(self, adapter_config: ObservabilityAdapterConfig) -> ObservabilityAdapterConfig:
        if not adapter_config.endpoint:
            raise DatusException(ErrorCode.COMMON_FIELD_REQUIRED, message="OTLP adapter requires an endpoint")
        return adapter_config

    def _setup_otlp_exporter(self, adapter_config: ObservabilityAdapterConfig, tracing_config: TracingConfig) -> None:
        try:
            from openinference.instrumentation import TraceConfig
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            from datus.observability.openai_agents import instrument_openai_agents
        except ImportError as exc:
            raise RuntimeError(
                "OTLP tracing requires opentelemetry-exporter-otlp and "
                "openinference-instrumentation-openai-agents to be installed"
            ) from exc

        exporter_kwargs: dict[str, Any] = {
            "endpoint": adapter_config.endpoint,
        }
        if adapter_config.headers:
            exporter_kwargs["headers"] = adapter_config.headers
        if adapter_config.timeout is not None:
            exporter_kwargs["timeout"] = adapter_config.timeout

        resource_attrs = {SERVICE_NAME: tracing_config.service_name}
        if tracing_config.environment:
            resource_attrs["deployment.environment"] = tracing_config.environment
        resource_attrs.update(
            {str(key): str(value) for key, value in adapter_config.options.get("resource_attributes", {}).items()}
            if isinstance(adapter_config.options.get("resource_attributes"), dict)
            else {}
        )

        exporter = OTLPSpanExporter(**exporter_kwargs)
        processor = BatchSpanProcessor(exporter)
        baggage_processors = self.build_baggage_span_processors()

        with OtlpAdapter._lock:
            if self._processor is not None and OtlpAdapter._tracer_provider is not None and not OtlpAdapter._shutdown:
                return
            if OtlpAdapter._tracer_provider is None or OtlpAdapter._shutdown:
                tracer_provider = TracerProvider(resource=Resource.create(resource_attrs))
                tracer_provider.add_span_processor(_BaggageAttributeSpanProcessor())
                trace.set_tracer_provider(tracer_provider)

                # Replace the Agents SDK default exporter with Datus OTLP tracing.
                instrumentor = instrument_openai_agents(
                    tracer_provider=tracer_provider,
                    config=_build_openinference_trace_config(TraceConfig, tracing_config),
                )

                OtlpAdapter._tracer_provider = tracer_provider
                OtlpAdapter._instrumentor = instrumentor
                OtlpAdapter._shutdown = False

            for baggage_processor in baggage_processors:
                OtlpAdapter._tracer_provider.add_span_processor(baggage_processor)
            OtlpAdapter._tracer_provider.add_span_processor(processor)
            self._processor = processor

    def build_baggage_span_processors(self) -> list[Any]:
        return []

    def record_event(self, event: Any) -> None:
        return None

    def flush(self) -> None:
        tracer_provider = OtlpAdapter._tracer_provider
        if tracer_provider is not None:
            force_flush = getattr(tracer_provider, "force_flush", None)
            if callable(force_flush):
                force_flush()

    def shutdown(self) -> None:
        with OtlpAdapter._lock:
            if OtlpAdapter._shutdown:
                return
            try:
                if OtlpAdapter._instrumentor is not None:
                    uninstrument = getattr(OtlpAdapter._instrumentor, "uninstrument", None)
                    if callable(uninstrument):
                        uninstrument()
            finally:
                if OtlpAdapter._tracer_provider is not None:
                    shutdown = getattr(OtlpAdapter._tracer_provider, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                OtlpAdapter._tracer_provider = None
                OtlpAdapter._instrumentor = None
                OtlpAdapter._shutdown = True


try:
    from opentelemetry.sdk.trace import SpanProcessor
except Exception:  # pragma: no cover - import availability is already checked during adapter setup.
    SpanProcessor = object  # type: ignore[assignment,misc]


class _BaggageAttributeSpanProcessor(SpanProcessor):
    """Copy safe trace-level baggage keys onto every span in the context."""

    _MAX_TRACE_CACHE_SIZE = 2048

    def __init__(self) -> None:
        self._trace_attributes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._trace_lock = threading.RLock()

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        try:
            from opentelemetry import baggage

            baggage_attrs = baggage.get_all(parent_context)
        except Exception:
            baggage_attrs = {}

        trace_attrs = _provider_neutral_attributes_from_baggage(baggage_attrs)
        trace_id = _span_trace_id(span)
        with self._trace_lock:
            if trace_attrs and trace_id:
                self._remember_trace_attributes(trace_id, trace_attrs)
            elif trace_id:
                trace_attrs = self._trace_attributes.get(trace_id, {})
            trace_attrs = dict(trace_attrs)

        for key, value in trace_attrs.items():
            _set_span_attribute(span, str(key), value)

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        with self._trace_lock:
            self._trace_attributes.clear()
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _remember_trace_attributes(self, trace_id: str, attributes: dict[str, Any]) -> None:
        with self._trace_lock:
            self._trace_attributes[trace_id] = attributes
            self._trace_attributes.move_to_end(trace_id)
            while len(self._trace_attributes) > self._MAX_TRACE_CACHE_SIZE:
                self._trace_attributes.popitem(last=False)


def _is_trace_baggage_key(key: str) -> bool:
    return key in {"session.id", "user.id", "datus.trace.name", "datus.run_id"}


def _provider_neutral_attributes_from_baggage(baggage_attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value for key, value in baggage_attrs.items() if _is_trace_baggage_key(str(key)) and value is not None
    }


def _span_trace_id(span: Any) -> str | None:
    try:
        span_context = span.get_span_context()
        trace_id = getattr(span_context, "trace_id", 0)
    except Exception:
        return None
    if not trace_id:
        return None
    return f"{trace_id:032x}"


def _set_span_attribute(span: Any, key: str, value: Any) -> None:
    try:
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))
    except Exception:
        return


def _build_openinference_trace_config(trace_config_cls: type, tracing_config: TracingConfig) -> Any:
    capture = tracing_config.capture
    hide_inputs = not (capture.prompts or capture.tool_args or capture.sql or capture.artifacts)
    hide_outputs = not (capture.responses or capture.reasoning or capture.tool_results or capture.artifacts)
    return trace_config_cls(
        hide_inputs=hide_inputs,
        hide_outputs=hide_outputs,
        hide_input_messages=not capture.prompts,
        hide_output_messages=not capture.responses,
        hide_input_text=not capture.prompts,
        hide_output_text=not capture.responses,
        hide_prompts=not capture.prompts,
        hide_choices=not capture.responses,
    )
