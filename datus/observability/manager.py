# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime manager for Datus external observability."""

from __future__ import annotations

import atexit
import contextvars
import logging
import threading
from contextlib import contextmanager
from typing import Any

from datus.observability.adapters.base import ObservabilityAdapter
from datus.observability.config import ObservabilityConfig, TracingConfig
from datus.observability.privacy import redact_value
from datus.observability.reference import TraceReference
from datus.observability.registry import adapter_registry
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_CURRENT_TRACE_REFERENCE: contextvars.ContextVar[TraceReference | None] = contextvars.ContextVar(
    "datus_observability_trace_reference",
    default=None,
)
_LAST_TRACE_REFERENCE: contextvars.ContextVar[TraceReference | None] = contextvars.ContextVar(
    "datus_observability_last_trace_reference",
    default=None,
)


class ObservabilityManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._initialized = False
        self._adapters: list[ObservabilityAdapter] = []
        self._tracing_config: TracingConfig | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def adapters(self) -> list[ObservabilityAdapter]:
        return list(self._adapters)

    def configure(self, config: ObservabilityConfig | None) -> bool:
        with self._lock:
            _suppress_noisy_otel_warnings()
            if self._initialized:
                return bool(self._adapters)

            tracing = config.tracing if config is not None else None
            if tracing is None or not tracing.enabled:
                return False
            self._tracing_config = tracing

            for adapter_config in tracing.adapters:
                if not adapter_config.enabled:
                    continue
                adapter_cls = adapter_registry.get(adapter_config.type)
                if adapter_cls is None:
                    logger.warning("No observability adapter registered for type '%s'", adapter_config.type)
                    continue
                try:
                    adapter = adapter_cls()
                    adapter.setup(adapter_config, tracing)
                    self._adapters.append(adapter)
                except Exception as exc:
                    logger.warning("Failed to initialize observability adapter %s: %s", adapter_config.type, exc)

            if self._adapters:
                self._initialized = True
                return True
            self._tracing_config = None
            return False

    def content_enabled(self, field_name: str) -> bool:
        if self._tracing_config is None:
            return False
        if not self._tracing_config.capture_content:
            return bool(getattr(self._tracing_config.capture, field_name, False))
        return bool(getattr(self._tracing_config.capture, field_name, True))

    def redact(self, value: Any) -> Any:
        if self._tracing_config is None:
            return value
        return redact_value(value, self._tracing_config.redact)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None, *, run_id: str | None = None):
        if not self.adapters:
            _LAST_TRACE_REFERENCE.set(None)
            yield None
            return
        attributes = attributes or {}
        _LAST_TRACE_REFERENCE.set(None)
        baggage_token = _attach_trace_baggage(name, attributes)
        try:
            from opentelemetry import context, trace

            tracer = trace.get_tracer("datus")
        except Exception as exc:
            logger.debug("Failed to create observability span %s: %s", name, exc)
            if baggage_token is not None:
                try:
                    from opentelemetry import context

                    context.detach(baggage_token)
                except Exception:
                    pass
            yield None
            return

        try:
            with tracer.start_as_current_span(name) as span:
                _set_span_attributes(span, attributes)
                _set_span_attributes(span, _trace_baggage_attributes(name, attributes))
                ref = self._trace_reference_from_span(span, run_id=run_id)
                token = None
                if ref is not None:
                    _LAST_TRACE_REFERENCE.set(ref)
                    _set_span_attributes(
                        span,
                        {
                            "datus.trace_id": ref.trace_id,
                            "datus.span_id": ref.span_id,
                            "datus.run_id": ref.run_id,
                        },
                    )
                    token = _CURRENT_TRACE_REFERENCE.set(ref)
                try:
                    yield span
                finally:
                    if token is not None:
                        _CURRENT_TRACE_REFERENCE.reset(token)
        finally:
            if baggage_token is not None:
                try:
                    context.detach(baggage_token)
                except Exception:
                    pass

    @contextmanager
    def trace_baggage(self, name: str, attributes: dict[str, Any] | None = None):
        """Attach trace-level OTel baggage without creating an extra span."""
        if not self.adapters:
            yield
            return
        token = _attach_trace_baggage(name, attributes or {})
        try:
            yield
        finally:
            if token is not None:
                try:
                    from opentelemetry import context

                    context.detach(token)
                except Exception:
                    pass

    def record_event(self, event: Any) -> None:
        for adapter in self.adapters:
            try:
                adapter.record_event(event)
            except Exception as exc:
                logger.debug("Observability adapter %s failed to record event: %s", getattr(adapter, "name", ""), exc)

    def flush(self) -> None:
        for adapter in self.adapters:
            try:
                adapter.flush()
            except Exception as exc:
                logger.debug("Observability adapter %s failed to flush: %s", getattr(adapter, "name", ""), exc)

    def shutdown(self) -> None:
        with self._lock:
            for adapter in self._adapters:
                try:
                    adapter.shutdown()
                except Exception as exc:
                    logger.debug("Observability adapter %s failed to shutdown: %s", getattr(adapter, "name", ""), exc)
            self._adapters = []
            self._tracing_config = None
            self._initialized = False
            _LAST_TRACE_REFERENCE.set(None)

    def get_trace_reference(self) -> TraceReference | None:
        ref = self._current_otel_trace_reference()
        if ref is not None:
            _LAST_TRACE_REFERENCE.set(ref)
            return ref

        ref = _CURRENT_TRACE_REFERENCE.get()
        if ref is not None:
            return ref
        return _LAST_TRACE_REFERENCE.get()

    def _current_otel_trace_reference(self) -> TraceReference | None:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
        except Exception:
            return None
        return self._trace_reference_from_span(span)

    def _trace_reference_from_span(self, span: Any, *, run_id: str | None = None) -> TraceReference | None:
        try:
            context = span.get_span_context()
        except Exception:
            return None
        if not getattr(context, "is_valid", False):
            return None

        trace_id = _format_trace_id(getattr(context, "trace_id", 0))
        span_id = _format_span_id(getattr(context, "span_id", 0))
        return TraceReference(
            trace_id=trace_id,
            span_id=span_id,
            run_id=run_id or _current_trace_run_id(),
            provider=self._provider_name(),
        )

    def _provider_name(self) -> str | None:
        names = [getattr(adapter, "name", "") for adapter in self.adapters]
        names = [name for name in names if name]
        return ",".join(names) if names else None


_manager = ObservabilityManager()
_atexit_flush_registered = False


def get_observability_manager() -> ObservabilityManager:
    return _manager


def configure_observability(config: ObservabilityConfig | None = None) -> bool:
    configured = _manager.configure(config)
    if configured:
        _register_atexit_flush()
    return configured


def shutdown_observability() -> None:
    _manager.shutdown()


def get_trace_reference() -> TraceReference | None:
    return _manager.get_trace_reference()


def _suppress_noisy_otel_warnings() -> None:
    logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)


def _register_atexit_flush() -> None:
    global _atexit_flush_registered
    if _atexit_flush_registered:
        return
    atexit.register(_flush_observability_at_exit)
    _atexit_flush_registered = True


def _flush_observability_at_exit() -> None:
    try:
        _manager.flush()
    except Exception:
        pass


def _set_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))
        except Exception:
            continue


def _attach_trace_baggage(span_name: str, attributes: dict[str, Any]) -> Any | None:
    try:
        from opentelemetry import baggage, context

        baggage_attrs = _trace_baggage_attributes(span_name, attributes)
        if not baggage_attrs:
            return None
        baggage_context = context.get_current()
        for key, value in baggage_attrs.items():
            baggage_context = baggage.set_baggage(key, value, context=baggage_context)
        return context.attach(baggage_context)
    except Exception:
        return None


def _trace_baggage_attributes(span_name: str, attributes: dict[str, Any]) -> dict[str, str]:
    """Return trace-level attributes that should propagate to every span."""
    baggage_attrs: dict[str, str] = {}

    trace_name = _string_attr(attributes.get("datus.trace.name") or span_name)
    if trace_name:
        baggage_attrs["datus.trace.name"] = trace_name

    session_id = _string_attr(attributes.get("datus.session_id") or attributes.get("session.id"))
    if session_id:
        baggage_attrs["session.id"] = session_id

    user_id = _string_attr(attributes.get("datus.user_id") or attributes.get("user.id"))
    if user_id:
        baggage_attrs["user.id"] = user_id

    run_id = _string_attr(
        attributes.get("datus.run_id")
        or attributes.get("datus.metadata.run_id")
        or attributes.get("datus.metadata.benchmark_run_id")
    )
    if run_id:
        baggage_attrs["datus.run_id"] = run_id

    return baggage_attrs


def _string_attr(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _format_trace_id(value: int) -> str:
    return f"{value:032x}"


def _format_span_id(value: int) -> str:
    return f"{value:016x}"


def _current_trace_run_id() -> str | None:
    try:
        from datus.utils.trace_context import get_trace_context

        ctx = get_trace_context()
    except Exception:
        return None
    if ctx is None:
        return None
    metadata = getattr(ctx, "metadata", {}) or {}
    run_id = metadata.get("run_id") or metadata.get("benchmark_run_id")
    if run_id:
        return str(run_id)
    session_id = getattr(ctx, "session_id", None)
    return str(session_id) if session_id else None
