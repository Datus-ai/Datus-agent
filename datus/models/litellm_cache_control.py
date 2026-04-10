# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LiteLLM model subclass that injects Anthropic prompt caching markers.

LiteLLM forwards OpenAI-style ``cache_control`` annotations inside message
content blocks and tool definitions to the Anthropic Messages API when the
underlying provider is ``anthropic/*``. This enables prompt caching on the
standard LiteLLM path (used by ``ClaudeModel`` when no OAuth token is
configured), which otherwise pays full input cost on every turn.
"""

from __future__ import annotations

import contextvars
import copy
from typing import Any, List

import litellm
from agents.extensions.models.litellm_model import LitellmModel

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_EPHEMERAL = {"type": "ephemeral"}

# Per-task flag: when True the wrapper applies cache_control markers.
# ContextVar is inherently per-async-task, so concurrent requests never interfere.
_apply_cache: contextvars.ContextVar[bool] = contextvars.ContextVar("_apply_cache", default=False)


def _tag_last_block(content: Any) -> Any:
    """Normalize ``content`` into a list of blocks and tag the last block."""
    if isinstance(content, str):
        blocks: List[dict] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = [dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content]
    else:
        return content

    if not blocks:
        return content

    blocks[-1] = {**blocks[-1], "cache_control": _EPHEMERAL}
    return blocks


def apply_cache_control(messages: list | None, tools: list | None) -> tuple[list | None, list | None]:
    """Return deep-copied ``(messages, tools)`` with ephemeral cache markers.

    Adds up to 3 cache breakpoints:
    1. System prompt (first message with ``role=="system"``).
    2. Last content block of the last ``user``/``tool`` message.
    3. Last tool definition.
    """
    new_messages = copy.deepcopy(messages) if messages else messages
    new_tools = copy.deepcopy(tools) if tools else tools

    if new_messages:
        # 1. System prompt
        if new_messages and new_messages[0].get("role") == "system":
            tagged = _tag_last_block(new_messages[0].get("content"))
            if tagged is not None:
                new_messages[0]["content"] = tagged

        # 2. Last user/tool message's last content block
        for i in range(len(new_messages) - 1, -1, -1):
            role = new_messages[i].get("role")
            if role in ("user", "tool"):
                tagged = _tag_last_block(new_messages[i].get("content"))
                if tagged is not None:
                    new_messages[i]["content"] = tagged
                break

    # 3. Last tool definition
    if new_tools:
        last = new_tools[-1]
        if isinstance(last, dict):
            new_tools[-1] = {**last, "cache_control": _EPHEMERAL}

    return new_messages, new_tools


def _install_cache_control_wrapper() -> None:
    """Install a one-time wrapper around ``litellm.acompletion``.

    The wrapper checks the per-task ``_apply_cache`` ContextVar and, when set,
    injects cache_control markers into messages and tools.  Because ContextVar
    is per-async-task, concurrent requests never interfere with each other.
    """
    if getattr(litellm, "_datus_cache_control_installed", False):
        return

    original_acompletion = litellm.acompletion

    async def _cache_aware_acompletion(*args: Any, **ll_kwargs: Any) -> Any:
        if _apply_cache.get():
            try:
                messages = ll_kwargs.get("messages")
                tools = ll_kwargs.get("tools")
                new_messages, new_tools = apply_cache_control(messages, tools)
                ll_kwargs["messages"] = new_messages
                if tools is not None:
                    ll_kwargs["tools"] = new_tools
            except Exception as exc:  # defensive: never break the request
                logger.warning(f"Failed to apply Anthropic cache_control markers: {exc}")
        return await original_acompletion(*args, **ll_kwargs)

    litellm.acompletion = _cache_aware_acompletion
    litellm._datus_cache_control_installed = True  # type: ignore[attr-defined]


# Install the wrapper once at import time.
_install_cache_control_wrapper()


class CacheControlLitellmModel(LitellmModel):
    """``LitellmModel`` subclass that injects Anthropic prompt-caching markers.

    Sets the per-task ``_apply_cache`` ContextVar so that the globally-installed
    wrapper applies ephemeral ``cache_control`` annotations on the system prompt,
    the last user/tool message, and the last tool definition — but only when the
    configured model routes to the Anthropic provider (``anthropic/*``).
    """

    def _is_anthropic(self) -> bool:
        return isinstance(self.model, str) and self.model.startswith("anthropic/")

    async def _fetch_response(self, *args, **kwargs):  # type: ignore[override]
        if not self._is_anthropic():
            return await super()._fetch_response(*args, **kwargs)

        token = _apply_cache.set(True)
        try:
            return await super()._fetch_response(*args, **kwargs)
        finally:
            _apply_cache.reset(token)
