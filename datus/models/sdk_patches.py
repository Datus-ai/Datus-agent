# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SDK Patches for openai-agents SDK.

This module provides monkey patches to extend SDK functionality for
providers not yet officially supported.

Current patches:
- Kimi/Moonshot reasoning_content support in Converter.items_to_messages()

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

# Providers that need reasoning_content support (like DeepSeek)
REASONING_CONTENT_PROVIDERS = {"deepseek", "kimi", "moonshot"}


def _is_reasoning_provider(model_name: str | None) -> bool:
    """Check if a model name belongs to a reasoning content provider."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return any(provider in model_lower for provider in REASONING_CONTENT_PROVIDERS)


def _normalize_provider_data(item: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize provider_data model name to use 'deepseek' prefix if it's a
    Kimi/Moonshot model. This allows the SDK's existing DeepSeek logic to
    handle reasoning_content correctly.

    Args:
        item: A response item dict that may contain provider_data

    Returns:
        A modified copy of the item with normalized model name
    """
    if not isinstance(item, dict):
        return item

    provider_data = item.get("provider_data")
    if not provider_data or not isinstance(provider_data, dict):
        return item

    item_model = provider_data.get("model")
    if not item_model:
        return item

    # Check if this is a Kimi/Moonshot model that needs normalization
    # Match: kimi, moonshot, k2.5, k2-thinking, k2-turbo, etc.
    item_model_lower = item_model.lower()
    is_kimi_model = (
        "kimi" in item_model_lower
        or "moonshot" in item_model_lower
        or "k2.5" in item_model_lower
        or "k2-" in item_model_lower
    )
    if is_kimi_model:
        # Create a copy and normalize the model name to trigger DeepSeek logic
        # We prefix with 'deepseek-' so the SDK condition "deepseek" in model.lower() matches
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

    Args:
        items: Original items to convert
        model: Original model name

    Returns:
        Tuple of (normalized_items, normalized_model)
    """
    # Normalize model name
    # Match: kimi, moonshot, k2.5, k2-thinking, k2-turbo, etc.
    normalized_model = model
    if model:
        model_lower = model.lower()
        is_kimi_model = (
            "kimi" in model_lower or "moonshot" in model_lower or "k2.5" in model_lower or "k2-" in model_lower
        )
        if is_kimi_model:
            normalized_model = f"deepseek-{model}"
            logger.debug(f"Normalized model name for reasoning_content support: {model} -> {normalized_model}")

    # If items is a string, no preprocessing needed
    if isinstance(items, str):
        return items, normalized_model

    # Normalize items with Kimi/Moonshot provider_data
    normalized_items = []
    for item in items:
        normalized_items.append(_normalize_provider_data(item))

    return normalized_items, normalized_model


# Store the original methods (will be initialized in apply_sdk_patches)
_original_items_to_messages = None
_original_litellm_fetch_response = None


def _postprocess_messages_for_reasoning(
    messages: list[dict[str, Any]],
    model: str | None,
) -> list[dict[str, Any]]:
    """
    Post-process messages to ensure all assistant messages with tool_calls
    have reasoning_content for Kimi/Moonshot models.

    Moonshot API requires reasoning_content in all assistant messages that have
    tool_calls, even if there's no corresponding reasoning item. This function
    adds an empty reasoning_content to such messages.

    Args:
        messages: Converted messages from items_to_messages
        model: The model name

    Returns:
        Messages with reasoning_content added where needed
    """
    if not model:
        return messages

    # Only process for Kimi/Moonshot models
    # Match: kimi, moonshot, k2.5, k2-thinking, k2-turbo, etc.
    model_lower = model.lower()
    is_kimi_model = (
        "kimi" in model_lower
        or "moonshot" in model_lower
        or "k2.5" in model_lower
        or "k2-" in model_lower  # k2-thinking, k2-turbo, etc.
    )
    if not is_kimi_model:
        return messages

    # Find the last NON-EMPTY reasoning_content (to reuse if needed)
    last_reasoning_content = None
    for msg in messages:
        if isinstance(msg, dict) and "reasoning_content" in msg:
            rc = msg.get("reasoning_content", "")
            # Only use non-empty reasoning_content
            if rc and rc.strip():
                last_reasoning_content = rc
                logger.debug(f"[SDK Patch] Found non-empty reasoning_content: {rc[:50]}...")

    # Add reasoning_content to assistant messages with tool_calls that don't have it
    # IMPORTANT: Moonshot API requires non-empty reasoning_content when thinking is enabled
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Check if reasoning_content exists and is non-empty
            current_rc = msg.get("reasoning_content", "")
            if not current_rc or not current_rc.strip():
                # Need to add/replace with a valid reasoning_content
                if last_reasoning_content:
                    msg["reasoning_content"] = last_reasoning_content
                    logger.debug("[SDK Patch] Added non-empty reasoning_content to assistant+tool_calls message")
                else:
                    # No valid reasoning_content found - use a placeholder
                    # Moonshot thinking mode requires reasoning_content, so provide a minimal valid value
                    msg["reasoning_content"] = "Analyzing the request and planning the tool calls."
                    logger.debug("[SDK Patch] Added placeholder reasoning_content (no previous found)")

            # Ensure content is empty string, not None (Moonshot requirement)
            if msg.get("content") is None:
                msg["content"] = ""
                logger.debug("[SDK Patch] Set content to empty string for assistant message with tool_calls")

    return messages


@classmethod
def _patched_items_to_messages(
    cls,
    items: str | Iterable[Any],
    model: str | None = None,
    preserve_thinking_blocks: bool = False,
    preserve_tool_output_all_content: bool = False,
) -> list[dict[str, Any]]:
    """
    Patched version of Converter.items_to_messages that supports
    Kimi/Moonshot reasoning_content in addition to DeepSeek.

    This patch:
    1. Preprocesses items and model name to use DeepSeek format
    2. Calls the original method
    3. Post-processes to ensure all assistant messages with tool_calls have reasoning_content
    """
    logger.debug(f"[SDK Patch] items_to_messages called with model={model}")

    # Preprocess for Kimi/Moonshot support
    normalized_items, normalized_model = _preprocess_items_for_reasoning(items, model)

    # Call original method with normalized inputs
    messages = _original_items_to_messages(
        cls,
        normalized_items,
        normalized_model,
        preserve_thinking_blocks,
        preserve_tool_output_all_content,
    )

    # Post-process to add missing reasoning_content for Kimi/Moonshot
    messages = _postprocess_messages_for_reasoning(messages, model)

    # Debug: log final message structure for kimi/moonshot
    if model and ("kimi" in model.lower() or "moonshot" in model.lower()):
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                has_rc = "reasoning_content" in msg
                has_tc = "tool_calls" in msg
                logger.debug(
                    f"[SDK Patch] Message[{i}]: role=assistant, "
                    f"has_reasoning_content={has_rc}, has_tool_calls={has_tc}"
                )

    return messages


async def _patched_litellm_fetch_response(self, *args, **kwargs):
    """
    Patched version of LitellmModel._fetch_response that intercepts and patches
    the converted_messages before sending to litellm.acompletion.

    This is a wrapper that calls the original method but intercepts litellm.acompletion
    calls to add reasoning_content where needed.
    """
    # Import litellm here to patch it temporarily
    from functools import wraps

    import litellm

    # Check if this is a Kimi/Moonshot model
    model = kwargs.get("model") or self.model
    logger.debug(f"[SDK Patch] _fetch_response called with model={model}")

    is_kimi = model and ("kimi" in model.lower() or "moonshot" in model.lower())

    if is_kimi:
        # Store original acompletion
        original_acompletion = litellm.acompletion

        # Create wrapper for acompletion
        @wraps(original_acompletion)
        async def patched_acompletion(*ac_args, **ac_kwargs):
            # Patch messages if present
            if "messages" in ac_kwargs:
                messages = ac_kwargs["messages"]
                logger.debug(f"[SDK Patch] Intercepting litellm.acompletion with {len(messages)} messages")

                # Log all messages before patching
                for i, msg in enumerate(messages):
                    role = msg.get("role") if isinstance(msg, dict) else "unknown"
                    has_tc = "tool_calls" in msg if isinstance(msg, dict) else False
                    has_rc = "reasoning_content" in msg if isinstance(msg, dict) else False
                    logger.debug(
                        f"[SDK Patch] BEFORE[{i}]: role={role}, tool_calls={has_tc}, reasoning_content={has_rc}"
                    )

                # Apply reasoning_content patch
                ac_kwargs["messages"] = _postprocess_messages_for_reasoning(messages, model)

                # Log all messages after patching
                for i, msg in enumerate(ac_kwargs["messages"]):
                    role = msg.get("role") if isinstance(msg, dict) else "unknown"
                    has_tc = "tool_calls" in msg if isinstance(msg, dict) else False
                    has_rc = "reasoning_content" in msg if isinstance(msg, dict) else False
                    rc_preview = msg.get("reasoning_content", "")[:30] if has_rc else "N/A"
                    logger.debug(
                        f"[SDK Patch] AFTER[{i}]: role={role}, tool_calls={has_tc}, "
                        f"reasoning_content={has_rc}, preview={rc_preview}"
                    )

            # Call original
            return await original_acompletion(*ac_args, **ac_kwargs)

        # Temporarily replace acompletion
        litellm.acompletion = patched_acompletion
        try:
            result = await _original_litellm_fetch_response(self, *args, **kwargs)
            return result
        finally:
            # Restore original
            litellm.acompletion = original_acompletion
    else:
        # Not a Kimi model, just call original
        return await _original_litellm_fetch_response(self, *args, **kwargs)


def apply_sdk_patches() -> None:
    """
    Apply all SDK patches.

    This function should be called early in application initialization,
    before any SDK methods are used.
    """
    global _original_items_to_messages, _original_litellm_fetch_response

    # Import agents SDK here to avoid circular dependencies
    from agents.extensions.models.litellm_model import LitellmModel
    from agents.models.chatcmpl_converter import Converter

    # Store original methods if not already stored
    if _original_items_to_messages is None:
        _original_items_to_messages = Converter.items_to_messages.__func__  # type: ignore

    if _original_litellm_fetch_response is None:
        _original_litellm_fetch_response = LitellmModel._fetch_response

    # Patch Converter.items_to_messages for Kimi/Moonshot support (must use classmethod wrapper)
    Converter.items_to_messages = classmethod(_patched_items_to_messages)  # type: ignore
    logger.info("Applied SDK patch: Converter.items_to_messages (Kimi/Moonshot reasoning_content)")

    # Patch LitellmModel._fetch_response to ensure messages have reasoning_content
    LitellmModel._fetch_response = _patched_litellm_fetch_response
    logger.info("Applied SDK patch: LitellmModel._fetch_response (Kimi/Moonshot reasoning_content)")


def remove_sdk_patches() -> None:
    """
    Remove all SDK patches and restore original behavior.

    Useful for testing or when patches are no longer needed.
    """
    # Import agents SDK here
    from agents.extensions.models.litellm_model import LitellmModel
    from agents.models.chatcmpl_converter import Converter

    if _original_items_to_messages is not None:
        Converter.items_to_messages = classmethod(_original_items_to_messages)  # type: ignore
        logger.info("Removed SDK patch: Converter.items_to_messages")

    if _original_litellm_fetch_response is not None:
        LitellmModel._fetch_response = _original_litellm_fetch_response
        logger.info("Removed SDK patch: LitellmModel._fetch_response")
