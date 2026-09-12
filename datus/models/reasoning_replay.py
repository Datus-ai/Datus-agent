# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Reasoning-content replay policy for thinking-mode Chat Completions providers.

DeepSeek and Kimi/Moonshot thinking modes return ``reasoning_content`` on
assistant messages and expect it to be echoed back on later requests of the
same conversation. The openai-agents SDK persists each turn's reasoning as a
``reasoning`` item and replays it onto the assistant message it belongs to,
but by default only when the target model is DeepSeek.
:func:`should_replay_reasoning_content` is passed to ``LitellmModel`` as the
SDK's ``should_replay_reasoning_content`` hook so Kimi/Moonshot get the same
per-turn replay.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import litellm

_KIMI_MARKERS = ("kimi", "moonshot", "k2.5", "k2-")
REASONING_ENDPOINT_KEY = "datus_reasoning_endpoint"


def reasoning_endpoint_identity(base_url: Optional[str]) -> Optional[str]:
    """Return a stable, non-secret identity for a model endpoint."""
    if not base_url:
        return None
    normalized = str(base_url).rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_kimi_model(model_name: Optional[str]) -> bool:
    """Return True for Kimi/Moonshot model names (kimi-*, moonshot-*, k2.5, k2-*)."""
    if not model_name:
        return False
    name = model_name.lower()
    return any(marker in name for marker in _KIMI_MARKERS)


def is_deepseek_model(model_name: Optional[str]) -> bool:
    """Return True for DeepSeek model names (deepseek-chat, deepseek-reasoner, deepseek-v4, ...)."""
    if not model_name:
        return False
    return "deepseek" in model_name.lower()


def reasoning_provider_family(model_name: Optional[str]) -> Optional[str]:
    """Return ``"deepseek"`` or ``"kimi"`` for providers that echo reasoning_content, else None."""
    if is_deepseek_model(model_name):
        return "deepseek"
    if is_kimi_model(model_name):
        return "kimi"
    return None


def is_reasoning_echo_provider(model_name: Optional[str]) -> bool:
    """Return True for a LiteLLM-known reasoning model that echoes reasoning content."""
    family = reasoning_provider_family(model_name)
    if family is None:
        return False
    name = model_name or ""
    if "/" not in name:
        name = f"{'deepseek' if family == 'deepseek' else 'moonshot'}/{name}"
    try:
        return bool(litellm.supports_reasoning(model=name))
    except Exception:
        return False


def should_replay_reasoning_content(context: Any) -> bool:
    """SDK hook: replay a stored reasoning item onto its assistant message.

    ``context`` is the SDK's ``ReasoningContentReplayContext`` (``model``,
    ``base_url``, ``reasoning.origin_model``, ``reasoning.provider_data``). It is
    duck-typed here so this module does not import the agents SDK at import
    time.

    Apply a strict same-route provenance check to LiteLLM-known DeepSeek and
    Kimi/Moonshot reasoning models. Both the provider-qualified model and a
    non-secret endpoint identity must match. Missing endpoint provenance is
    rejected because the previous request may have used another compatible
    endpoint with the same model name.
    """
    model = getattr(context, "model", None)
    if not is_reasoning_echo_provider(model):
        return False

    reasoning = getattr(context, "reasoning", None)
    origin_model = getattr(reasoning, "origin_model", None)
    if not is_reasoning_echo_provider(origin_model):
        return False
    provider_data = getattr(reasoning, "provider_data", None)
    if not isinstance(provider_data, dict):
        return False
    origin_endpoint = provider_data.get(REASONING_ENDPOINT_KEY)
    current_endpoint = reasoning_endpoint_identity(getattr(context, "base_url", None))
    return origin_model == model and origin_endpoint is not None and origin_endpoint == current_endpoint


__all__ = [
    "REASONING_ENDPOINT_KEY",
    "is_deepseek_model",
    "is_kimi_model",
    "is_reasoning_echo_provider",
    "reasoning_endpoint_identity",
    "reasoning_provider_family",
    "should_replay_reasoning_content",
]
