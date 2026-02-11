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


def _normalize_provider_data(item: Any) -> Any:
    """
    Normalize provider_data model name to use 'deepseek' prefix if it's a
    Kimi/Moonshot model. This allows the SDK's existing DeepSeek logic to
    handle reasoning_content correctly.

    Handles both plain dicts and Pydantic model objects (e.g., ResponseReasoningItem,
    ResponseFunctionToolCall) which the agents SDK uses internally.
    """
    if isinstance(item, dict):
        provider_data = item.get("provider_data")
        if not provider_data or not isinstance(provider_data, dict):
            return item
        item_model = provider_data.get("model")
        if not item_model or not _is_kimi_model(item_model):
            return item
        item_copy = copy.deepcopy(item)
        item_copy["provider_data"]["model"] = f"deepseek-{item_model}"
        return item_copy

    # Handle Pydantic/object items with provider_data attribute
    # (e.g., ResponseReasoningItem, ResponseFunctionToolCall from agents SDK)
    provider_data = getattr(item, "provider_data", None)
    if not provider_data or not isinstance(provider_data, dict):
        return item
    item_model = provider_data.get("model")
    if not item_model or not _is_kimi_model(item_model):
        return item

    # Deep copy the Pydantic object to avoid mutating the SDK's internal state
    if hasattr(item, "model_copy"):
        item_copy = item.model_copy(deep=True)
    elif hasattr(item, "copy"):
        item_copy = item.copy(deep=True)
    else:
        item_copy = copy.deepcopy(item)
    item_copy.provider_data["model"] = f"deepseek-{item_model}"
    return item_copy


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

# Cache reasoning_content from API responses, keyed by model name.
# This provides a fallback when the SDK converter fails to extract
# reasoning_content from items (e.g., when summary is empty).
_reasoning_content_cache: dict[str, str] = {}


class _ReasoningContentStreamWrapper:
    """
    Async iterator wrapper that intercepts streaming chunks to cache
    reasoning_content for Kimi/Moonshot models.

    When stream=True, litellm.acompletion returns an async iterable (not a
    ModelResponse with .choices), so reasoning_content must be captured from
    individual delta chunks as they stream through.
    """

    def __init__(self, stream: Any, model: str):
        self._stream = stream
        self._model = model
        self._reasoning_chunks: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._flush_cache()
            raise

        try:
            for choice in getattr(chunk, "choices", []):
                delta = getattr(choice, "delta", None)
                if delta:
                    rc = getattr(delta, "reasoning_content", None)
                    if rc and isinstance(rc, str):
                        self._reasoning_chunks.append(rc)
        except Exception:
            pass

        return chunk

    def _flush_cache(self) -> None:
        """Flush accumulated reasoning_content chunks into the cache."""
        if self._reasoning_chunks:
            full_rc = "".join(self._reasoning_chunks)
            if full_rc.strip():
                _reasoning_content_cache[self._model] = full_rc
                logger.debug(
                    f"[SDK Patch] Cached reasoning_content from stream, " f"model={self._model}, length={len(full_rc)}"
                )

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


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
                logger.debug(f"[SDK Patch] Found non-empty reasoning_content in messages, length={len(rc)}")

    # Fallback: use cached reasoning_content from a previous API response
    if not last_reasoning_content and model:
        cached_rc = _reasoning_content_cache.get(model)
        if cached_rc:
            last_reasoning_content = cached_rc
            logger.debug(f"[SDK Patch] Using cached reasoning_content as fallback, length={len(cached_rc)}")

    # Ensure all assistant messages with tool_calls have reasoning_content field.
    # Kimi/Moonshot requires this field to be present when thinking is enabled.
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            current_rc = msg.get("reasoning_content", "")
            if (not current_rc or not current_rc.strip()) and last_reasoning_content:
                msg["reasoning_content"] = last_reasoning_content
                logger.debug("[SDK Patch] Injected reasoning_content into assistant+tool_calls message")
            elif "reasoning_content" not in msg:
                # No reasoning_content available from messages or cache.
                # Moonshot API requires non-empty reasoning_content for thinking models.
                # This path should be rare after the streaming cache fix.
                msg["reasoning_content"] = ""
                logger.warning(
                    "[SDK Patch] No reasoning_content available for assistant+tool_calls message. "
                    "Moonshot API may reject this request. Check if streaming cache is working."
                )

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
            response = await _original_acompletion(*args, **kwargs)

            # Cache reasoning_content from the API response for future fallback.
            # This handles cases where the SDK converter fails to extract it from items.
            if model and _is_kimi_model(model):
                stream = kwargs.get("stream", False)
                if stream:
                    # Streaming: wrap the async iterator to capture reasoning_content
                    # from delta chunks as they flow through.
                    response = _ReasoningContentStreamWrapper(response, model)
                else:
                    # Non-streaming: extract from ModelResponse.choices directly.
                    try:
                        for choice in getattr(response, "choices", []):
                            msg = getattr(choice, "message", None)
                            if msg:
                                rc = getattr(msg, "reasoning_content", None)
                                if rc and isinstance(rc, str) and rc.strip():
                                    _reasoning_content_cache[model] = rc
                                    logger.debug(
                                        f"[SDK Patch] Cached reasoning_content from response, "
                                        f"model={model}, length={len(rc)}"
                                    )
                                    break
                    except Exception:
                        pass  # Never let caching break the main flow

            return response

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

    _reasoning_content_cache.clear()
