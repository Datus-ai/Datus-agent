# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Reasoning-content replay policy for thinking-mode Chat Completions providers.

Thinking-mode providers, including DeepSeek, Kimi, and GLM, return
``reasoning_content`` on assistant messages. The openai-agents SDK stores this
field verbatim in a ``reasoning`` item's summary. Datus records the source
endpoint on these items and replays them only to the same model and endpoint.
This also supports private deployment aliases absent from LiteLLM's catalogue.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import litellm

_KIMI_MARKERS = ("kimi", "moonshot", "k2.5", "k2-")
_KIMI_REASONING_MODEL_IDS = frozenset({"kimi-for-coding"})
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
    """Return True for a verified reasoning model that echoes reasoning content."""
    family = reasoning_provider_family(model_name)
    if family is None:
        return False
    name = model_name or ""
    model_id = name.rsplit("/", 1)[-1].lower()
    # Kimi Code rolls the backing model forward behind this stable model ID.
    # It currently serves K2.8 Preview and requires reasoning_content on
    # thinking-mode tool-call replay, but LiteLLM does not yet advertise that
    # capability for either the bare or Anthropic-qualified model name.
    if family == "kimi" and model_id in _KIMI_REASONING_MODEL_IDS:
        return True
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

    Both the provider-qualified model and a non-secret endpoint identity must
    match. For unknown models, the LiteLLM transport's summary-only reasoning
    items identify a returned Chat Completions ``reasoning_content`` field.
    Signed thinking blocks and encrypted reasoning keep their provider-specific
    handling instead. Missing endpoint provenance is rejected because another
    compatible endpoint may use the same model name.
    """
    model = getattr(context, "model", None)
    if not isinstance(model, str) or not model:
        return False

    reasoning = getattr(context, "reasoning", None)
    origin_model = getattr(reasoning, "origin_model", None)
    if origin_model != model:
        return False
    provider_data = getattr(reasoning, "provider_data", None)
    if not isinstance(provider_data, dict):
        return False
    origin_endpoint = provider_data.get(REASONING_ENDPOINT_KEY)
    current_endpoint = reasoning_endpoint_identity(getattr(context, "base_url", None))
    if origin_endpoint is None or origin_endpoint != current_endpoint:
        return False
    if is_reasoning_echo_provider(model):
        return True

    # Only Datus' LiteLLM transport stamps this endpoint provenance. In that
    # path the SDK puts raw reasoning_content in summary, while signed thinking
    # and other reasoning fields use content/encrypted_content. Do not require
    # a catalogue entry or an explicit reasoning_effort for an observed field.
    item = getattr(reasoning, "item", None)
    if not isinstance(item, dict) or item.get("content") or item.get("encrypted_content"):
        return False
    return any(
        isinstance(part, dict) and isinstance(part.get("text"), str) and bool(part["text"])
        for part in item.get("summary") or []
    )


__all__ = [
    "REASONING_ENDPOINT_KEY",
    "is_deepseek_model",
    "is_kimi_model",
    "is_reasoning_echo_provider",
    "reasoning_endpoint_identity",
    "reasoning_provider_family",
    "should_replay_reasoning_content",
]
