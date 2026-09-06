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

Turns whose response carried no reasoning are left with an empty
``reasoning_content`` placeholder by
:func:`ensure_reasoning_content_placeholders`, matching how LiteLLM, OpenCode
and Hermes handle the same providers. Reasoning is never copied from one turn
to another: a message either carries the reasoning it was produced with or an
empty string.
"""

from __future__ import annotations

from typing import Any, Optional

_KIMI_MARKERS = ("kimi", "moonshot", "k2.5", "k2-")


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
    """Return True when the provider expects ``reasoning_content`` to be echoed back."""
    return reasoning_provider_family(model_name) is not None


def should_replay_reasoning_content(context: Any) -> bool:
    """SDK hook: replay a stored reasoning item onto its assistant message.

    ``context`` is the SDK's ``ReasoningContentReplayContext`` (``model``,
    ``base_url``, ``reasoning.origin_model``, ``reasoning.provider_data``). It is
    duck-typed here so this module does not import the agents SDK at import
    time.

    Replay only when the request targets a provider that echoes reasoning and
    the item came from the same provider family. Items recorded before the SDK
    tracked provider data (empty ``provider_data``) are trusted, mirroring the
    SDK's own DeepSeek default.
    """
    target = reasoning_provider_family(getattr(context, "model", None))
    if target is None:
        return False

    reasoning = getattr(context, "reasoning", None)
    origin_model = getattr(reasoning, "origin_model", None)
    if origin_model:
        return reasoning_provider_family(origin_model) == target

    provider_data = getattr(reasoning, "provider_data", None) or {}
    return not dict(provider_data)


def ensure_reasoning_content_placeholders(messages: Any, model: Optional[str]) -> Any:
    """Fill ``reasoning_content`` gaps with an empty string for thinking-mode requests.

    Applies only when ``model`` is a DeepSeek or Kimi/Moonshot model and the
    conversation is already in thinking mode, i.e. at least one assistant
    message carries ``reasoning_content``. Every other assistant message then
    gets ``reasoning_content = ""``. Assistant messages with ``tool_calls`` and
    ``content=None`` are normalised to ``content=""`` because Moonshot rejects
    ``null`` content on tool-call messages.

    Messages are mutated in place and returned; non-list payloads are returned
    untouched.
    """
    if not is_reasoning_echo_provider(model) or not isinstance(messages, list):
        return messages

    thinking_active = any(
        isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("reasoning_content") is not None
        for msg in messages
    )

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        has_tool_calls = bool(msg.get("tool_calls"))
        if has_tool_calls and msg.get("content") is None:
            msg["content"] = ""
        if thinking_active and msg.get("reasoning_content") is None:
            msg["reasoning_content"] = ""

    return messages


__all__ = [
    "ensure_reasoning_content_placeholders",
    "is_deepseek_model",
    "is_kimi_model",
    "is_reasoning_echo_provider",
    "reasoning_provider_family",
    "should_replay_reasoning_content",
]
