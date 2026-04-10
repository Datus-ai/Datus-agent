# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Standardized token usage model for LLM consumption reporting."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict


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

    @classmethod
    def from_usage_dict(cls, d: Dict[str, Any], **overrides) -> "TokenUsage":
        """Construct a TokenUsage from a usage dict (e.g. from _extract_usage_info).

        Extra keys in *d* are silently ignored thanks to ``extra="ignore"``.
        *overrides* take precedence over values in *d*.
        """
        merged = {**d, **overrides}
        return cls(**merged)
