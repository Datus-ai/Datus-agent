# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Configuration models for external observability integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from datus.utils.exceptions import DatusException, ErrorCode

CONTENT_CAPTURE_FIELDS = (
    "prompts",
    "responses",
    "reasoning",
    "tool_args",
    "tool_results",
    "sql",
    "artifacts",
)

DEFAULT_TRACING_ADAPTER_TYPE = "langfuse"


@dataclass
class CaptureConfig:
    prompts: bool = True
    responses: bool = True
    reasoning: bool = True
    tool_args: bool = True
    tool_results: bool = True
    sql: bool = True
    artifacts: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None, *, capture_content: bool = True) -> "CaptureConfig":
        values = {field_name: capture_content for field_name in CONTENT_CAPTURE_FIELDS}
        if isinstance(raw, Mapping):
            for field_name in CONTENT_CAPTURE_FIELDS:
                if field_name in raw:
                    values[field_name] = _coerce_bool(raw.get(field_name), values[field_name])
        return cls(**values)


@dataclass
class RedactConfig:
    enabled: bool = True
    fields: list[str] = field(default_factory=lambda: ["api_key", "password", "token", "secret"])
    patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "RedactConfig":
        if not isinstance(raw, Mapping):
            return cls()
        fields = raw.get("fields")
        patterns = raw.get("patterns")
        return cls(
            enabled=_coerce_bool(raw.get("enabled"), True),
            fields=[str(item) for item in fields] if isinstance(fields, list) else cls().fields,
            patterns=[str(item) for item in patterns] if isinstance(patterns, list) else [],
        )


@dataclass
class ObservabilityAdapterConfig:
    type: str
    enabled: bool = True
    endpoint: str | None = None
    protocol: str = "http/protobuf"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    features: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservabilityAdapterConfig":
        adapter_type = str(raw.get("type", "")).strip().lower()
        known = {"type", "enabled", "endpoint", "protocol", "headers", "timeout", "features"}
        timeout = raw.get("timeout")
        return cls(
            type=adapter_type,
            enabled=_coerce_bool(raw.get("enabled"), True),
            endpoint=_clean_string(raw.get("endpoint")),
            protocol=_clean_string(raw.get("protocol")) or "http/protobuf",
            headers=_parse_headers(raw.get("headers")),
            timeout=_parse_optional_float(timeout, "timeout"),
            features=dict(raw.get("features") or {}) if isinstance(raw.get("features"), Mapping) else {},
            options={str(key): value for key, value in raw.items() if key not in known},
        )


@dataclass
class TracingConfig:
    enabled: bool = False
    service_name: str = "datus-agent"
    environment: str | None = None
    capture_content: bool = True
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    redact: RedactConfig = field(default_factory=RedactConfig)
    adapters: list[ObservabilityAdapterConfig] = field(default_factory=list)
    explicit: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TracingConfig":
        if not isinstance(raw, Mapping):
            return cls(explicit=False)
        enabled = _coerce_bool(raw.get("enabled"), False)
        capture_content = _coerce_bool(raw.get("capture_content"), True)
        adapters = []
        raw_adapters = raw.get("adapters")
        if isinstance(raw_adapters, list):
            adapters = [
                ObservabilityAdapterConfig.from_dict(item)
                for item in raw_adapters
                if isinstance(item, Mapping) and item.get("type")
            ]
        elif enabled and raw_adapters is None:
            adapters = [ObservabilityAdapterConfig.from_dict({"type": DEFAULT_TRACING_ADAPTER_TYPE})]
        return cls(
            enabled=enabled,
            service_name=_clean_string(raw.get("service_name")) or "datus-agent",
            environment=_clean_string(raw.get("environment")),
            capture_content=capture_content,
            capture=CaptureConfig.from_dict(raw.get("capture"), capture_content=capture_content),
            redact=RedactConfig.from_dict(raw.get("redact")),
            adapters=adapters,
            explicit=True,
        )


@dataclass
class ObservabilityConfig:
    tracing: TracingConfig = field(default_factory=TracingConfig)
    explicit: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ObservabilityConfig":
        if not isinstance(raw, Mapping):
            return cls(explicit=False)
        tracing = TracingConfig.from_dict(raw.get("tracing"))
        return cls(tracing=tracing, explicit=bool(raw))


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith("<MISSING:"):
        return None
    return text


def _parse_optional_float(value: Any, field_name: str) -> float | None:
    text = _clean_string(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError) as exc:
        raise DatusException(
            ErrorCode.COMMON_FIELD_INVALID,
            message_args={"field_name": field_name, "except_values": "number", "your_value": value},
        ) from exc


def _parse_headers(value: Any) -> dict[str, str]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return {str(key): str(val) for key, val in value.items() if val is not None}
    headers: dict[str, str] = {}
    for part in str(value).split(","):
        if not part.strip() or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        if key:
            headers[key] = val.strip()
    return headers
