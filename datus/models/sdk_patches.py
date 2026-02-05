# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SDK Patches for openai-agents SDK.

This module provides monkey patches to extend SDK functionality for
providers not yet officially supported.

Current patches:
- Kimi/Moonshot reasoning_content support in Converter.items_to_messages()
- Kimi/Moonshot reasoning_content preservation in litellm.acompletion()

Reference: https://github.com/openai/openai-agents-python/pull/2328
The SDK already supports DeepSeek reasoning_content. This patch extends
the same support to Kimi/Moonshot models.
"""

import copy
from collections.abc import Iterable
from typing import Any

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# NOTE: Do NOT import agents SDK at module level!
# Import it inside functions to avoid circular dependencies and ensure patches are applied first.


def _is_kimi_model(model_name: str) -> bool:
    """Check if a model name is a Kimi/Moonshot model (kimi, moonshot, k2.5, k2-*, etc.)."""
    name = model_name.lower()
    return "kimi" in name or "moonshot" in name or "k2.5" in name or "k2-" in name


def _normalize_provider_data(item: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize provider_data model name to use 'deepseek' prefix if it's a
    Kimi/Moonshot model. This allows the SDK's existing DeepSeek logic to
    handle reasoning_content correctly.
    """
    if not isinstance(item, dict):
        return item

    provider_data = item.get("provider_data")
    if not provider_data or not isinstance(provider_data, dict):
        return item

    item_model = provider_data.get("model")
    if not item_model:
        return item

    if _is_kimi_model(item_model):
        # Prefix with 'deepseek-' so the SDK condition "deepseek" in model.lower() matches
        item_copy = copy.deepcopy(item)
        item_copy["provider_data"]["model"] = f"deepseek-{item_model}"
        return item_copy

    return item


def _preprocess_items_for_reasoning(
    items: str | Iterable[Any],
    model: str | None,
) -> tuple[str | list[Any], str | None]:
    """
    Preprocess items and model name to enable reasoning_content support
    for Kimi/Moonshot models.

    The SDK's items_to_messages() only handles reasoning_content for DeepSeek models.
    This function normalizes Kimi/Moonshot models to use DeepSeek format so the
    existing logic can handle them.
    """
    normalized_model = model
    if model and _is_kimi_model(model):
        normalized_model = f"deepseek-{model}"
        logger.debug(f"Normalized model name for reasoning_content support: {model} -> {normalized_model}")

    if isinstance(items, str):
        return items, normalized_model

    normalized_items = [_normalize_provider_data(item) for item in items]
    return normalized_items, normalized_model


# Store the original methods (will be initialized in apply_sdk_patches)
_original_items_to_messages = None
_original_acompletion = None


def _postprocess_messages_for_reasoning(
    messages: list[dict[str, Any]],
    model: str | None,
) -> list[dict[str, Any]]:
    """
    Post-process messages to preserve reasoning_content for Kimi/Moonshot models
    during tool calling.

    Per DeepSeek/Moonshot docs, reasoning_content must be passed back during
    tool calling to allow the model to continue reasoning.
    See: https://api-docs.deepseek.com/guides/thinking_mode
    """
    if not model or not _is_kimi_model(model):
        return messages

    # Find the last non-empty reasoning_content to reuse if needed
    last_reasoning_content = None
    for msg in messages:
        if isinstance(msg, dict) and "reasoning_content" in msg:
            rc = msg.get("reasoning_content", "")
            if rc and rc.strip():
                last_reasoning_content = rc
                logger.debug(f"[SDK Patch] Found non-empty reasoning_content, length={len(rc)}")

    # Copy reasoning_content to assistant messages with tool_calls that are missing it.
    # Only copy existing reasoning_content, never inject fake placeholders.
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            current_rc = msg.get("reasoning_content", "")
            if (not current_rc or not current_rc.strip()) and last_reasoning_content:
                msg["reasoning_content"] = last_reasoning_content
                logger.debug("[SDK Patch] Copied previous reasoning_content to assistant+tool_calls message")

            # Ensure content is empty string, not None (Moonshot requirement)
            if msg.get("content") is None:
                msg["content"] = ""

    return messages


def _patched_items_to_messages(
    cls,
    items: str | Iterable[Any],
    model: str | None = None,
    preserve_thinking_blocks: bool = False,
    preserve_tool_output_all_content: bool = False,
) -> list[dict[str, Any]]:
    """
    Patched Converter.items_to_messages that extends reasoning_content
    support from DeepSeek to Kimi/Moonshot models.
    """
    normalized_items, normalized_model = _preprocess_items_for_reasoning(items, model)

    messages = _original_items_to_messages(
        cls,
        normalized_items,
        normalized_model,
        preserve_thinking_blocks,
        preserve_tool_output_all_content,
    )

    return _postprocess_messages_for_reasoning(messages, model)


def apply_sdk_patches() -> None:
    """
    Apply all SDK patches.

    This function should be called early in application initialization,
    before any SDK methods are used.
    """
    global _original_items_to_messages, _original_acompletion

    from functools import wraps

    import litellm

    # Import agents SDK here to avoid circular dependencies
    from agents.models.chatcmpl_converter import Converter

    # Patch 1: Converter.items_to_messages for Kimi/Moonshot reasoning_content
    if _original_items_to_messages is None:
        _original_items_to_messages = Converter.items_to_messages.__func__  # type: ignore

    Converter.items_to_messages = classmethod(_patched_items_to_messages)  # type: ignore
    logger.info("Applied SDK patch: Converter.items_to_messages (Kimi/Moonshot reasoning_content)")

    # Patch 2: litellm.acompletion wrapper (safety net)
    # Re-applies reasoning_content preservation right before API calls,
    # in case the SDK modifies messages after items_to_messages.
    if _original_acompletion is None:
        _original_acompletion = litellm.acompletion

        @wraps(_original_acompletion)
        async def _patched_acompletion(*args, **kwargs):
            model = kwargs.get("model", "")
            if "messages" in kwargs:
                kwargs["messages"] = _postprocess_messages_for_reasoning(kwargs["messages"], model)
            return await _original_acompletion(*args, **kwargs)

        litellm.acompletion = _patched_acompletion
        logger.info("Applied SDK patch: litellm.acompletion (Kimi/Moonshot reasoning_content)")


def remove_sdk_patches() -> None:
    """
    Remove all SDK patches and restore original behavior.

    Useful for testing or when patches are no longer needed.
    """
    global _original_items_to_messages, _original_acompletion

    import litellm
    from agents.models.chatcmpl_converter import Converter

    if _original_items_to_messages is not None:
        Converter.items_to_messages = classmethod(_original_items_to_messages)  # type: ignore
        _original_items_to_messages = None
        logger.info("Removed SDK patch: Converter.items_to_messages")

    if _original_acompletion is not None:
        litellm.acompletion = _original_acompletion
        _original_acompletion = None
        logger.info("Removed SDK patch: litellm.acompletion")
