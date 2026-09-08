# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Standardized token usage model for LLM consumption reporting."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, model_validator


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_hit_rate: float = 0.0
    context_usage_ratio: float = 0.0
    # Session-level context information (optionally populated)
    context_length: int = 0
    session_total_tokens: int = 0  # Current context window usage (last model call's input_tokens)
    context_usage_valid: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_context_usage(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        values = dict(values)
        used = max(0, int(values.get("session_total_tokens", values.get("last_call_input_tokens", 0)) or 0))
        length = max(0, int(values.get("context_length", 0) or 0))
        valid = values.get("context_usage_valid", used > 0)
        values["context_usage_valid"] = valid
        values["session_total_tokens"] = used if valid else 0
        values["context_usage_ratio"] = round(used / length, 3) if valid and length else 0.0
        return values

    @classmethod
    def from_usage_dict(cls, d: Dict[str, Any], **overrides) -> "TokenUsage":
        """Construct a TokenUsage from a usage dict (e.g. from _extract_usage_info).

        Extra keys in *d* are silently ignored thanks to ``extra="ignore"``.
        *overrides* take precedence over values in *d*.
        """
        merged = {**d, **overrides}
        return cls(**merged)
