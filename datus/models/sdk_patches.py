# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SDK Patches for openai-agents SDK and LiteLLM.

Current patches:
- Chat-style ``text`` block normalization in ``Converter.items_to_messages()``
  for session replay through Chat Completions providers such as DeepSeek.
- ``reasoning_content`` placeholders on thinking-mode requests in
  ``litellm.(a)completion()`` (see :mod:`datus.models.reasoning_replay`), plus
  Kimi/Moonshot empty-content recovery on the sync path.
- LiteLLM ``Usage`` serialization warning suppression for provider-specific
  ``server_tool_use`` dict payloads.
- Pydantic serializer warnings redirected from the CLI to the logger.

Per-turn ``reasoning_content`` replay itself is handled by the SDK: DeepSeek by
default, Kimi/Moonshot through the ``should_replay_reasoning_content`` hook
wired in :meth:`datus.models.litellm_adapter.LiteLLMAdapter.get_agents_sdk_model`.
"""

import copy
import warnings
from collections.abc import Iterable
from typing import Any

from datus.models.reasoning_replay import ensure_reasoning_content_placeholders, is_kimi_model
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# NOTE: Do NOT import agents SDK at module level!
# Import it inside functions to avoid circular dependencies and ensure patches are applied first.


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


def _normalize_items(items: str | Iterable[Any]) -> str | list[Any]:
    """Apply :func:`_normalize_text_content_blocks` to every item of a session replay."""
    if isinstance(items, str):
        return items
    return [_normalize_text_content_blocks(item) for item in items]


# Store the original methods (will be initialized in apply_sdk_patches)
_original_items_to_messages = None
_original_acompletion = None
_original_completion = None
_original_usage_model_dump = None
_original_usage_model_dump_json = None
_original_usage_init = None
_original_showwarning = None


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


def _patched_items_to_messages(
    cls,
    items: str | Iterable[Any],
    model: str | None = None,
    preserve_thinking_blocks: bool = False,
    preserve_tool_output_all_content: bool = False,
    base_url: str | None = None,
    should_replay_reasoning_content: Any = None,
) -> list[dict[str, Any]]:
    """Patched ``Converter.items_to_messages`` that normalizes Chat-style text blocks first."""
    return _original_items_to_messages(
        cls,
        _normalize_items(items),
        model,
        preserve_thinking_blocks,
        preserve_tool_output_all_content,
        base_url,
        should_replay_reasoning_content,
    )


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


def _recover_empty_kimi_content(response: Any) -> None:
    """Moonshot non-thinking responses may arrive with empty content and only reasoning_content.

    Surface the reasoning as the visible content so ``generate()`` callers do not
    receive an empty string.
    """
    try:
        for choice in getattr(response, "choices", []):
            msg = getattr(choice, "message", None)
            if not msg:
                continue
            content = getattr(msg, "content", None)
            if content and content.strip():
                return
            reasoning_content = _extract_reasoning_content(msg)
            if reasoning_content:
                msg.content = reasoning_content
                logger.debug("[SDK Patch] Injected reasoning_content into empty sync response content")
            return
    except Exception as e:
        logger.debug(f"[SDK Patch] Failed to recover empty Kimi content: {e}")


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

    # Patch 1: Converter.items_to_messages content-block normalization
    if _original_items_to_messages is None:
        _original_items_to_messages = Converter.items_to_messages.__func__  # type: ignore

    Converter.items_to_messages = classmethod(_patched_items_to_messages)  # type: ignore
    logger.info("Applied SDK patch: Converter.items_to_messages (content-block normalization)")

    # Patch 2: litellm.acompletion reasoning_content placeholders
    if _original_acompletion is None:
        _original_acompletion = litellm.acompletion

        @wraps(_original_acompletion)
        async def _patched_acompletion(*args, **kwargs):
            model = kwargs.get("model", "")
            if "messages" in kwargs:
                kwargs["messages"] = ensure_reasoning_content_placeholders(kwargs["messages"], model)
            response = await _original_acompletion(*args, **kwargs)
            # Streaming returns an async iterator; only complete responses carry a message to recover.
            if is_kimi_model(model) and not kwargs.get("stream"):
                _recover_empty_kimi_content(response)
            return response

        litellm.acompletion = _patched_acompletion
        logger.info("Applied SDK patch: litellm.acompletion (reasoning_content placeholders, Kimi content recovery)")

    # Patch 3: litellm.completion reasoning_content placeholders + Kimi empty-content recovery
    if _original_completion is None:
        _original_completion = litellm.completion

        @wraps(_original_completion)
        def _patched_completion(*args, **kwargs):
            model = kwargs.get("model", "")
            if "messages" in kwargs:
                kwargs["messages"] = ensure_reasoning_content_placeholders(kwargs["messages"], model)
            response = _original_completion(*args, **kwargs)
            if is_kimi_model(model):
                _recover_empty_kimi_content(response)
            return response

        litellm.completion = _patched_completion
        logger.info("Applied SDK patch: litellm.completion (reasoning_content placeholders, Kimi content recovery)")


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
