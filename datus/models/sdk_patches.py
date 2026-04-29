# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SDK Patches for openai-agents SDK.

This module provides monkey patches to extend SDK functionality for
providers whose thinking mode requires reasoning_content to be echoed
back on every assistant-with-tool_calls turn.

Current patches:
- Kimi/Moonshot reasoning_content support in Converter.items_to_messages()
- Chat-style ``text`` block normalization for session replay through Chat
  Completions providers such as DeepSeek.
- Kimi/Moonshot + DeepSeek reasoning_content preservation in
  litellm.(a)completion() via a streaming cache fallback.
- LiteLLM ``Usage`` serialization warning suppression for provider-specific
  ``server_tool_use`` dict payloads.

Reference: https://github.com/openai/openai-agents-python/pull/2328
The SDK already supports DeepSeek reasoning_content when the streamed
`summary` is populated. For DeepSeek V4 thinking mode, the SDK sometimes
misses the reasoning delta (empty summary) and the provider rejects the
next turn with:

    The `reasoning_content` in the thinking mode must be passed back to
    the API.

This patch adds the same streaming cache + injection fallback used for
Kimi/Moonshot to DeepSeek models.
"""

import copy
import warnings
from collections.abc import Iterable, Iterator, MutableMapping
from contextvars import ContextVar
from typing import Any

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# NOTE: Do NOT import agents SDK at module level!
# Import it inside functions to avoid circular dependencies and ensure patches are applied first.


def _is_kimi_model(model_name: str) -> bool:
    """Check if a model name is a Kimi/Moonshot model (kimi, moonshot, k2.5, k2-*, etc.)."""
    name = model_name.lower()
    return "kimi" in name or "moonshot" in name or "k2.5" in name or "k2-" in name


def _is_deepseek_model(model_name: str) -> bool:
    """Check if a model name is a DeepSeek model (deepseek-chat, deepseek-reasoner, deepseek-v4, ...)."""
    if not model_name:
        return False
    return "deepseek" in model_name.lower()


def _needs_reasoning_injection(model_name: str) -> bool:
    """Providers whose thinking mode requires reasoning_content to be echoed back on tool-calling turns."""
    if not model_name:
        return False
    return _is_kimi_model(model_name) or _is_deepseek_model(model_name)


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


def _normalize_text_content_blocks(item: Any) -> Any:
    """Normalize OpenAI Chat ``text`` blocks to Agents SDK input/output blocks.

    The Agents SDK Chat Completions converter accepts ``input_text`` for input
    messages and ``output_text`` for response output messages. Session replay can
    contain OpenAI Chat-style ``{"type": "text", "text": ...}`` blocks, which
    otherwise raise ``UserError("Unknown content: ...")`` before the model is
    called.
    """
    if not isinstance(item, dict):
        return item

    replacement_type = (
        "output_text" if item.get("type") == "message" and item.get("role") == "assistant" else "input_text"
    )
    changed = False
    normalized_item = item

    for key in ("content", "output"):
        value = item.get(key)
        if not isinstance(value, list):
            continue

        normalized_value = []
        value_changed = False
        for block in value:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                normalized_block = dict(block)
                normalized_block["type"] = replacement_type
                normalized_value.append(normalized_block)
                value_changed = True
            else:
                normalized_value.append(block)

        if value_changed:
            if not changed:
                normalized_item = copy.deepcopy(item)
                changed = True
            normalized_item[key] = normalized_value

    return normalized_item


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

    normalized_items = [_normalize_text_content_blocks(_normalize_provider_data(item)) for item in items]
    return normalized_items, normalized_model


# Store the original methods (will be initialized in apply_sdk_patches)
_original_items_to_messages = None
_original_acompletion = None
_original_completion = None
_original_usage_model_dump = None
_original_usage_model_dump_json = None
_original_usage_init = None
_original_showwarning = None

# Cache reasoning_content from API responses, keyed by model name within the
# current execution context. This avoids leaking one session's hidden reasoning
# into another request while preserving the fallback inside a tool-calling run.
_reasoning_content_cache_var: ContextVar[dict[str, str] | None] = ContextVar(
    "datus_reasoning_content_cache",
    default=None,
)


def _current_reasoning_content_cache() -> dict[str, str]:
    cache = _reasoning_content_cache_var.get()
    if cache is None:
        cache = {}
        _reasoning_content_cache_var.set(cache)
    return cache


class _ReasoningContentCache(MutableMapping[str, str]):
    """Context-local mapping kept for tests and local cache helpers."""

    def __getitem__(self, key: str) -> str:
        return _current_reasoning_content_cache()[key]

    def __setitem__(self, key: str, value: str) -> None:
        _current_reasoning_content_cache()[key] = value

    def __delitem__(self, key: str) -> None:
        del _current_reasoning_content_cache()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(_current_reasoning_content_cache())

    def __len__(self) -> int:
        return len(_current_reasoning_content_cache())

    def clear(self) -> None:
        _current_reasoning_content_cache().clear()


_reasoning_content_cache: MutableMapping[str, str] = _ReasoningContentCache()


_REASONING_CONTENT_FIELD_NAMES = (
    "reasoning_content",
    "reasoning",
    "reasoning_text",
    "thinking",
)
_REASONING_CONTENT_NESTED_FIELD_NAMES = (
    "model_extra",
    "additional_kwargs",
    "provider_specific_fields",
    "extra",
    "extra_fields",
    "__pydantic_extra__",
)


def _read_field(value: Any, name: str) -> Any:
    """Safely read ``name`` from dict-like and object-like values."""
    if isinstance(value, dict):
        return value.get(name)
    try:
        return getattr(value, name)
    except Exception:
        return None


def _coerce_reasoning_text(value: Any) -> str | None:
    """Return a non-empty reasoning string from known reasoning-shaped values."""
    if isinstance(value, str):
        return value if value.strip() else None

    if isinstance(value, dict):
        for key in _REASONING_CONTENT_FIELD_NAMES + ("text", "content"):
            text = _coerce_reasoning_text(value.get(key))
            if text:
                return text
        return None

    if isinstance(value, list):
        parts = [_coerce_reasoning_text(item) for item in value]
        text = "".join(part for part in parts if part)
        return text if text.strip() else None

    return None


def _extract_reasoning_content(value: Any) -> str | None:
    """Extract reasoning text from LiteLLM/OpenAI-style dicts or objects.

    LiteLLM and provider adapters do not expose DeepSeek reasoning deltas in
    one uniform shape. Some versions use ``delta.reasoning_content`` while
    others put the same value inside dict deltas, ``model_extra`` or
    ``provider_specific_fields``. Only inspect known reasoning fields so normal
    assistant ``content`` is never mistaken for hidden reasoning.
    """
    seen: set[int] = set()
    stack = [value]

    while stack:
        current = stack.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        for field_name in _REASONING_CONTENT_FIELD_NAMES:
            text = _coerce_reasoning_text(_read_field(current, field_name))
            if text:
                return text

        for nested_field_name in _REASONING_CONTENT_NESTED_FIELD_NAMES:
            nested = _read_field(current, nested_field_name)
            if nested is not None and nested is not current:
                stack.append(nested)

    return None


def _reasoning_cache_keys(model: str | None) -> list[str]:
    """Return equivalent cache keys for prefixed/unprefixed LiteLLM model names."""
    if not model:
        return []

    raw_model = str(model).strip()
    if not raw_model:
        return []

    keys: list[str] = []

    def add(key: str | None) -> None:
        if not key:
            return
        stripped = key.strip()
        if stripped and stripped not in keys:
            keys.append(stripped)
        lowered = stripped.lower()
        if lowered and lowered not in keys:
            keys.append(lowered)

    add(raw_model)

    if "/" in raw_model:
        _, suffix = raw_model.split("/", 1)
        add(suffix)
    elif _is_deepseek_model(raw_model):
        add(f"deepseek/{raw_model}")
    elif _is_kimi_model(raw_model):
        add(f"moonshot/{raw_model}")

    return keys


def _cache_reasoning_content(model: str | None, reasoning_content: str) -> None:
    """Cache reasoning_content under all equivalent model aliases."""
    if not reasoning_content or not reasoning_content.strip():
        return
    for key in _reasoning_cache_keys(model):
        _reasoning_content_cache[key] = reasoning_content


def _get_cached_reasoning_content(model: str | None) -> str | None:
    """Fetch cached reasoning_content across prefixed/unprefixed aliases."""
    for key in _reasoning_cache_keys(model):
        cached = _reasoning_content_cache.get(key)
        if cached and cached.strip():
            return cached
    return None


def _content_text(content: Any) -> str | None:
    """Extract visible text from a chat-completion content value."""
    if isinstance(content, str):
        text = content.strip()
        return text or None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = _read_field(item, "text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        joined = "\n".join(parts).strip()
        return joined or None

    return None


def _sanitize_deepseek_history_without_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop historical tool protocol messages that DeepSeek thinking cannot replay.

    DeepSeek thinking mode requires every assistant message with ``tool_calls``
    to carry the exact reasoning_content that DeepSeek produced. When a user
    switches from another provider, old session history can contain tool-call
    turns without any DeepSeek reasoning source. The current turn still has a
    trailing user message, so only sanitize messages before the last user
    message and leave in-flight DeepSeek tool calls untouched.
    """
    last_user_index = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_index = idx

    if last_user_index <= 0:
        return messages

    sanitized: list[dict[str, Any]] = []
    dropped_count = 0
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or idx >= last_user_index:
            sanitized.append(msg)
            continue

        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls") and not _extract_reasoning_content(msg):
            text = _content_text(msg.get("content"))
            if text:
                clean_msg = {key: value for key, value in msg.items() if key not in ("tool_calls", "reasoning_content")}
                clean_msg["content"] = text
                sanitized.append(clean_msg)
            dropped_count += 1
            continue

        if role == "tool":
            dropped_count += 1
            continue

        sanitized.append(msg)

    if dropped_count:
        logger.info(
            "[SDK Patch] Dropped %s historical tool-protocol message(s) before DeepSeek thinking call "
            "because no replayable reasoning_content was available.",
            dropped_count,
        )
    return sanitized


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
            choices = _read_field(chunk, "choices") or []
            for choice in choices:
                delta = _read_field(choice, "delta")
                if delta:
                    rc = _extract_reasoning_content(delta)
                    if rc:
                        self._reasoning_chunks.append(rc)
        except Exception:
            pass

        return chunk

    def _flush_cache(self) -> None:
        """Flush accumulated reasoning_content chunks into the cache."""
        if self._reasoning_chunks:
            full_rc = "".join(self._reasoning_chunks)
            if full_rc.strip():
                _cache_reasoning_content(self._model, full_rc)
                logger.debug(
                    f"[SDK Patch] Cached reasoning_content from stream, model={self._model}, length={len(full_rc)}"
                )

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _postprocess_messages_for_reasoning(
    messages: list[dict[str, Any]],
    model: str | None,
) -> list[dict[str, Any]]:
    """
    Post-process messages to preserve reasoning_content for thinking-mode
    providers (Kimi/Moonshot, DeepSeek) during tool calling.

    Per DeepSeek/Moonshot docs, reasoning_content must be passed back during
    tool calling to allow the model to continue reasoning.
    See: https://api-docs.deepseek.com/guides/thinking_mode
    """
    if not model or not _needs_reasoning_injection(model):
        return messages

    is_kimi = _is_kimi_model(model)
    is_deepseek = _is_deepseek_model(model)

    # Find the last non-empty reasoning_content to reuse if needed
    last_reasoning_content = None
    for msg in messages:
        if isinstance(msg, dict):
            rc = _extract_reasoning_content(msg)
            if rc:
                last_reasoning_content = rc
                logger.debug(f"[SDK Patch] Found non-empty reasoning_content in messages, length={len(rc)}")

    # Fallback: use cached reasoning_content from a previous API response
    if not last_reasoning_content and model:
        cached_rc = _get_cached_reasoning_content(model)
        if cached_rc:
            last_reasoning_content = cached_rc
            logger.debug(f"[SDK Patch] Using cached reasoning_content as fallback, length={len(cached_rc)}")

    if is_deepseek and not last_reasoning_content:
        messages = _sanitize_deepseek_history_without_reasoning(messages)

    # Ensure assistant messages preserve reasoning_content on tool-calling turns.
    # DeepSeek also requires the final assistant message immediately after a tool
    # result to carry reasoning_content. Keep ordinary assistant history untouched.
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        has_tool_calls = bool(msg.get("tool_calls"))
        previous_message = messages[idx - 1] if idx > 0 else None
        follows_tool_result = (
            is_deepseek and isinstance(previous_message, dict) and previous_message.get("role") == "tool"
        )
        should_patch_message = has_tool_calls or follows_tool_result
        if should_patch_message:
            current_rc = _coerce_reasoning_text(msg.get("reasoning_content"))
            if current_rc:
                msg["reasoning_content"] = current_rc
            elif last_reasoning_content:
                msg["reasoning_content"] = last_reasoning_content
                logger.debug("[SDK Patch] Injected reasoning_content into assistant message")
            elif has_tool_calls and "reasoning_content" not in msg and is_kimi:
                # Moonshot historically tolerates an empty reasoning_content field when
                # thinking is off; DeepSeek rejects both missing and empty, so we must
                # NOT inject an empty placeholder for DeepSeek — leave the message as-is
                # and let the provider surface a clean error if thinking was actually on.
                msg["reasoning_content"] = ""
                logger.warning(
                    "[SDK Patch] No reasoning_content available for assistant+tool_calls message. "
                    "Moonshot API may reject this request. Check if streaming cache is working."
                )

            # Ensure content is empty string, not None (Moonshot requirement;
            # DeepSeek also accepts content="" for tool_calls-only messages).
            if has_tool_calls and msg.get("content") is None:
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


def _redirect_pydantic_serializer_warnings_to_log() -> None:
    """Redirect Pydantic serializer warnings from stderr/CLI to the logger.

    Pydantic emits ``UserWarning("Pydantic serializer warnings: ...")`` whenever a
    field's runtime value does not match the declared type. ``Usage.model_dump``
    is patched above to silence this for direct calls, but when a parent model
    (e.g. LiteLLM's ``ModelResponse``) serializes ``usage`` as a nested field,
    the warning fires from the parent's serializer and leaks into the CLI.

    Install a ``warnings.showwarning`` shim that diverts only those Pydantic
    serializer messages to ``logger.debug`` while leaving every other warning
    untouched.
    """
    global _original_showwarning

    if _original_showwarning is not None:
        return

    _original_showwarning = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        if "Pydantic serializer warnings" in str(message):
            logger.debug(
                "Pydantic serializer warning redirected from CLI: %s (%s:%s)",
                message,
                filename,
                lineno,
            )
            return
        _original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _showwarning


def _patch_litellm_usage_serialization() -> None:
    """Fix LiteLLM Usage.server_tool_use type mismatch and suppress residual warnings.

    LiteLLM's Usage.__init__ coerces completion_tokens_details and
    prompt_tokens_details from dict to their model types, but omits the same
    coercion for server_tool_use.  Providers such as Anthropic return
    server_tool_use as a plain dict (e.g. {"web_search_requests": 0}), which is
    stored directly on the instance, causing Pydantic's Rust core serializer to
    warn about a type mismatch every time the parent ModelResponse is serialized.

    Primary fix: patch Usage.__init__ to coerce server_tool_use dict →
    ServerToolUse at construction time, eliminating the mismatch entirely.

    Safety-net: also patch model_dump / model_dump_json with warnings=False in
    case any code path bypasses __init__ (e.g. model_construct).
    """
    global _original_usage_model_dump, _original_usage_model_dump_json, _original_usage_init

    from functools import wraps

    from litellm.types.utils import ServerToolUse, Usage

    if _original_usage_init is None:
        _original_usage_init = Usage.__init__

        @wraps(_original_usage_init)
        def _patched_usage_init(self, *args, server_tool_use=None, **kwargs):
            if isinstance(server_tool_use, dict):
                try:
                    server_tool_use = ServerToolUse(**server_tool_use)
                except Exception:
                    pass
            _original_usage_init(self, *args, server_tool_use=server_tool_use, **kwargs)

        Usage.__init__ = _patched_usage_init

    if _original_usage_model_dump is None:
        _original_usage_model_dump = Usage.model_dump

        @wraps(_original_usage_model_dump)
        def _patched_usage_model_dump(self, *args, **kwargs):
            kwargs.setdefault("warnings", False)
            return _original_usage_model_dump(self, *args, **kwargs)

        Usage.model_dump = _patched_usage_model_dump

    if _original_usage_model_dump_json is None:
        _original_usage_model_dump_json = Usage.model_dump_json

        @wraps(_original_usage_model_dump_json)
        def _patched_usage_model_dump_json(self, *args, **kwargs):
            kwargs.setdefault("warnings", False)
            return _original_usage_model_dump_json(self, *args, **kwargs)

        Usage.model_dump_json = _patched_usage_model_dump_json


def apply_sdk_patches() -> None:
    """
    Apply all SDK patches.

    This function should be called early in application initialization,
    before any SDK methods are used.
    """
    global _original_items_to_messages, _original_acompletion, _original_completion

    from functools import wraps

    import litellm

    # Import agents SDK here to avoid circular dependencies
    from agents.models.chatcmpl_converter import Converter

    _patch_litellm_usage_serialization()
    _redirect_pydantic_serializer_warnings_to_log()

    # Patch 1: Converter.items_to_messages for Kimi/Moonshot reasoning_content
    if _original_items_to_messages is None:
        _original_items_to_messages = Converter.items_to_messages.__func__  # type: ignore

    Converter.items_to_messages = classmethod(_patched_items_to_messages)  # type: ignore
    logger.info("Applied SDK patch: Converter.items_to_messages (content-block normalization + reasoning_content)")

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
            if model and _needs_reasoning_injection(model):
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
                                rc = _extract_reasoning_content(msg)
                                if rc:
                                    _cache_reasoning_content(model, rc)
                                    logger.debug(
                                        f"[SDK Patch] Cached reasoning_content from response, "
                                        f"model={model}, length={len(rc)}"
                                    )
                                    break
                    except Exception as e:
                        logger.debug(f"[SDK Patch] Failed to cache reasoning_content from async response: {e}")

            return response

        litellm.acompletion = _patched_acompletion
        logger.info("Applied SDK patch: litellm.acompletion (Kimi/Moonshot + DeepSeek reasoning_content)")

    # Patch 3: litellm.completion wrapper (sync version)
    # The generate() method uses litellm.completion (sync), which was not patched.
    # Without this, kimi-k2.5 returns empty content because reasoning_content is not exposed.
    if _original_completion is None:
        _original_completion = litellm.completion

        @wraps(_original_completion)
        def _patched_completion(*args, **kwargs):
            model = kwargs.get("model", "")
            if "messages" in kwargs:
                kwargs["messages"] = _postprocess_messages_for_reasoning(kwargs["messages"], model)
            response = _original_completion(*args, **kwargs)

            # Cache reasoning_content and inject it into message.content if empty.
            # The empty-content injection is Kimi-specific: Moonshot non-thinking
            # responses may arrive with empty content + reasoning_content. DeepSeek's
            # sync path returns real content, so we only cache (no content rewrite).
            if model and _needs_reasoning_injection(model):
                is_kimi = _is_kimi_model(model)
                try:
                    for choice in getattr(response, "choices", []):
                        msg = getattr(choice, "message", None)
                        if msg:
                            rc = _extract_reasoning_content(msg)
                            if rc:
                                _cache_reasoning_content(model, rc)
                                logger.debug(
                                    f"[SDK Patch] Cached reasoning_content from sync response, "
                                    f"model={model}, length={len(rc)}"
                                )
                                # If main content is empty, inject reasoning_content (Kimi only)
                                if is_kimi:
                                    content = getattr(msg, "content", None)
                                    if not content or not content.strip():
                                        msg.content = rc
                                        logger.debug(
                                            "[SDK Patch] Injected reasoning_content into empty sync response content"
                                        )
                                break
                except Exception as e:
                    logger.debug(f"[SDK Patch] Failed to cache reasoning_content from sync response: {e}")

            return response

        litellm.completion = _patched_completion
        logger.info("Applied SDK patch: litellm.completion (Kimi/Moonshot + DeepSeek reasoning_content sync)")


def remove_sdk_patches() -> None:
    """
    Remove all SDK patches and restore original behavior.

    Useful for testing or when patches are no longer needed.
    """
    global _original_items_to_messages, _original_acompletion, _original_completion
    global _original_usage_model_dump, _original_usage_model_dump_json, _original_usage_init
    global _original_showwarning

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

    if _original_completion is not None:
        litellm.completion = _original_completion
        _original_completion = None
        logger.info("Removed SDK patch: litellm.completion")

    try:
        from litellm.types.utils import Usage

        if _original_usage_init is not None:
            Usage.__init__ = _original_usage_init
            _original_usage_init = None
            logger.info("Removed SDK patch: LiteLLM Usage.__init__")
        if _original_usage_model_dump is not None:
            Usage.model_dump = _original_usage_model_dump
            _original_usage_model_dump = None
            logger.info("Removed SDK patch: LiteLLM Usage.model_dump")
        if _original_usage_model_dump_json is not None:
            Usage.model_dump_json = _original_usage_model_dump_json
            _original_usage_model_dump_json = None
            logger.info("Removed SDK patch: LiteLLM Usage.model_dump_json")
    except Exception as e:
        logger.debug(f"Failed to remove LiteLLM Usage serialization patch: {e}")

    if _original_showwarning is not None:
        warnings.showwarning = _original_showwarning
        _original_showwarning = None
        logger.info("Removed SDK patch: warnings.showwarning (Pydantic serializer warnings)")

    _reasoning_content_cache.clear()
