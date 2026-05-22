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
    normalized = key.lower()
    return any(field.lower() in normalized for field in fields)
