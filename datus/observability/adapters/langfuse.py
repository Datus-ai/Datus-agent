# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Langfuse OTLP observability adapter."""

from __future__ import annotations

import base64
import os
import threading
from collections import OrderedDict
from dataclasses import replace
from typing import Any

from datus.observability.adapters.otlp import OtlpAdapter
from datus.observability.config import ObservabilityAdapterConfig
from datus.utils.exceptions import DatusException, ErrorCode

try:
    from opentelemetry.sdk.trace import SpanProcessor
except Exception:  # pragma: no cover - import availability is already checked during adapter setup.
    SpanProcessor = object  # type: ignore[assignment,misc]


class LangfuseAdapter(OtlpAdapter):
    name = "langfuse"
    capabilities = {"traces", "otlp", "llm_observability", "session_attributes", "trace_metadata"}

    def resolve_adapter_config(self, adapter_config: ObservabilityAdapterConfig) -> ObservabilityAdapterConfig:
        endpoint = adapter_config.endpoint
        if not endpoint:
            host = _option_or_env(
                adapter_config,
                "host",
                "LANGFUSE_OTEL_HOST",
                "LANGFUSE_HOST",
                "LANGFUSE_BASE_URL",
                default="https://us.cloud.langfuse.com",
            )
            endpoint = _join_url(host, "/api/public/otel/v1/traces")

        auth_string = _option_or_env(adapter_config, "auth_string", "LANGFUSE_AUTH_STRING")
        if not auth_string:
            public_key = _option_or_env(adapter_config, "public_key", "LANGFUSE_PUBLIC_KEY")
            secret_key = _option_or_env(adapter_config, "secret_key", "LANGFUSE_SECRET_KEY")
            if not public_key or not secret_key:
                raise DatusException(
                    ErrorCode.COMMON_FIELD_REQUIRED,
                    message="Langfuse adapter requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY",
                )
            auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_string}",
            "x-langfuse-ingestion-version": "4",
        }
        headers.update(adapter_config.headers)
        return replace(adapter_config, endpoint=endpoint, headers=headers)

    def build_baggage_span_processors(self) -> list[Any]:
        return [_LangfuseBaggageSpanProcessor()]


class _LangfuseBaggageSpanProcessor(SpanProcessor):
    """Map generic Datus/OTel baggage attributes to Langfuse trace attributes."""

    _MAX_TRACE_CACHE_SIZE = 2048

    def __init__(self) -> None:
        self._trace_attributes: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._trace_lock = threading.RLock()

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        try:
            from opentelemetry import baggage

            baggage_attrs = baggage.get_all(parent_context)
        except Exception:
            baggage_attrs = {}

        langfuse_attrs = _langfuse_attributes_from_baggage(baggage_attrs)
        trace_id = _span_trace_id(span)
        with self._trace_lock:
            if langfuse_attrs and trace_id:
                self._remember_trace_attributes(trace_id, langfuse_attrs)
            elif trace_id:
                langfuse_attrs = self._trace_attributes.get(trace_id, {})
            langfuse_attrs = dict(langfuse_attrs)

        for key, value in langfuse_attrs.items():
            _set_span_attribute(span, key, value)

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        with self._trace_lock:
            self._trace_attributes.clear()
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _remember_trace_attributes(self, trace_id: str, attributes: dict[str, str]) -> None:
        with self._trace_lock:
            self._trace_attributes[trace_id] = attributes
            self._trace_attributes.move_to_end(trace_id)
            while len(self._trace_attributes) > self._MAX_TRACE_CACHE_SIZE:
                self._trace_attributes.popitem(last=False)


def _langfuse_attributes_from_baggage(baggage_attrs: dict[str, Any]) -> dict[str, str]:
    attributes: dict[str, str] = {}

    trace_name = _clean_string(baggage_attrs.get("datus.trace.name"))
    if trace_name:
        attributes["langfuse.trace.name"] = trace_name

    session_id = _clean_string(baggage_attrs.get("session.id"))
    if session_id:
        attributes["langfuse.session.id"] = session_id
        attributes["langfuse.trace.metadata.session_id"] = session_id
        attributes["langfuse.trace.metadata.datus_session_id"] = session_id

    user_id = _clean_string(baggage_attrs.get("user.id"))
    if user_id:
        attributes["langfuse.user.id"] = user_id

    run_id = _clean_string(baggage_attrs.get("datus.run_id"))
    if run_id:
        attributes["langfuse.trace.metadata.run_id"] = run_id

    return attributes


def _option_or_env(
    adapter_config: ObservabilityAdapterConfig,
    option_key: str,
    *env_names: str,
    default: str | None = None,
) -> str | None:
    value = _clean_string(adapter_config.options.get(option_key))
    if value is not None:
        return value
    for env_name in env_names:
        value = _clean_string(os.environ.get(env_name))
        if value is not None:
            return value
    return default


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith("<MISSING:"):
        return None
    return text


def _join_url(base_url: str | None, path: str) -> str:
    if not base_url:
        raise DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message="Langfuse adapter requires a base URL or endpoint",
        )
    return base_url.rstrip("/") + "/" + path.lstrip("/")


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
