# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Content redaction helpers for trace attributes."""

from __future__ import annotations

import re
from typing import Any, Mapping

from datus.observability.config import RedactConfig


def redact_value(value: Any, config: RedactConfig) -> Any:
    if not config.enabled:
        return value
    return _redact(value, config)


def _redact(value: Any, config: RedactConfig) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, val in value.items():
            if _is_sensitive_field(str(key), config.fields):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = _redact(val, config)
        return redacted
    if isinstance(value, list):
        return [_redact(item, config) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, config) for item in value)
    if isinstance(value, str):
        text = value
        for pattern in config.patterns:
            try:
                text = re.sub(pattern, "[REDACTED]", text)
            except re.error:
                continue
        return text
    return value


def _is_sensitive_field(key: str, fields: list[str]) -> bool:
    key_parts = set(_field_parts(key))
    key_part_list = _field_parts(key)
    key_normalized = _normalize_field(key)
    for field in fields:
        field_normalized = _normalize_field(field)
        if not field_normalized:
            continue
        if key_normalized == field_normalized:
            return True
        field_parts = _field_parts(field)
        if len(field_parts) > 1 and _contains_part_sequence(key_part_list, field_parts):
            return True
        if field_normalized in key_parts:
            return True
    return False


def _field_parts(value: str) -> list[str]:
    acronym_spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    camel_spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", acronym_spaced)
    return [part for part in re.split(r"[^A-Za-z0-9]+", camel_spaced.lower()) if part]


def _normalize_field(value: str) -> str:
    return "_".join(_field_parts(value))


def _contains_part_sequence(parts: list[str], target: list[str]) -> bool:
    if not target or len(target) > len(parts):
        return False
    size = len(target)
    return any(parts[index : index + size] == target for index in range(len(parts) - size + 1))
