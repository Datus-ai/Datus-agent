# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Datus compatibility transport for LiteLLM Chat Completions models."""

from collections.abc import AsyncIterator
from typing import Any

import litellm

from datus.models.observed_model import ObservedLitellmModel
from datus.models.reasoning_replay import REASONING_ENDPOINT_KEY, is_kimi_model, reasoning_endpoint_identity
from datus.utils.image_content import is_tool_image_user_message, tool_images_as_user_input


def register_image_model_capabilities(model: str) -> None:
    """Correct LiteLLM metadata for provider models verified to accept images."""
    if model == "deepseek/deepseek-v4-flash":
        litellm.register_model(
            {
                model: {
                    "litellm_provider": "deepseek",
                    "supports_vision": True,
                }
            }
        )


def _stamp_reasoning_endpoint(item: Any, endpoint_identity: str | None) -> None:
    """Attach endpoint provenance to an SDK reasoning item."""
    if endpoint_identity is None:
        return
    item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
    if item_type != "reasoning":
        return
    if isinstance(item, dict):
        provider_data = dict(item.get("provider_data") or {})
        provider_data[REASONING_ENDPOINT_KEY] = endpoint_identity
        item["provider_data"] = provider_data
        return
    provider_data = dict(getattr(item, "provider_data", None) or {})
    provider_data[REASONING_ENDPOINT_KEY] = endpoint_identity
    item.provider_data = provider_data


def _stamp_reasoning_event(event: Any, endpoint_identity: str | None) -> None:
    """Attach endpoint provenance to reasoning items in a streaming event."""
    _stamp_reasoning_endpoint(getattr(event, "item", None), endpoint_identity)
    response = getattr(event, "response", None)
    for item in getattr(response, "output", None) or []:
        _stamp_reasoning_endpoint(item, endpoint_identity)


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _is_empty_assistant_message(item: Any) -> bool:
    if _item_value(item, "type") != "message" or _item_value(item, "role") != "assistant":
        return False

    content = _item_value(item, "content")
    if not content:
        return True
    if not isinstance(content, (list, tuple)):
        return False

    for part in content:
        part_type = _item_value(part, "type")
        if part_type == "output_text":
            if _item_value(part, "text"):
                return False
        elif part_type == "refusal":
            if _item_value(part, "refusal"):
                return False
        else:
            return False
    return True


def normalize_kimi_tool_replay_items(items: Any) -> Any:
    """Keep Kimi's assistant content and parallel calls in one replay turn.

    openai-agents 0.18.1 preserves Chat Completions stream output ordering. Kimi
    can emit its ``content`` delta after one tool-call delta and before another.
    This becomes a Responses ``message`` between ``function_call`` items.
    Replaying that history through the SDK converter creates two assistant
    messages, so Moonshot rejects the first call because the next message is not
    its tool result.

    Move assistant messages observed while calls are pending immediately before
    that call group. The converter can then attach every call to the same
    assistant message. Empty messages inside the call group are redundant and
    are dropped. Messages already outside a pending call group are untouched.
    """
    if not isinstance(items, (list, tuple)):
        return items

    pending_call_ids: set[str] = set()
    pending_call_index: int | None = None
    normalized: list[Any] = []
    changed = False
    for item in items:
        item_type = _item_value(item, "type")
        if item_type == "function_call":
            if not pending_call_ids:
                pending_call_index = len(normalized)
            call_id = _item_value(item, "call_id")
            if isinstance(call_id, str) and call_id:
                pending_call_ids.add(call_id)
        elif item_type == "function_call_output":
            call_id = _item_value(item, "call_id")
            if isinstance(call_id, str):
                pending_call_ids.discard(call_id)
            if not pending_call_ids:
                pending_call_index = None
        elif pending_call_ids and item_type == "message" and _item_value(item, "role") == "assistant":
            changed = True
            if not _is_empty_assistant_message(item):
                assert pending_call_index is not None
                normalized.insert(pending_call_index, item)
                pending_call_index += 1
            continue
        normalized.append(item)

    if not changed:
        return items
    return tuple(normalized) if isinstance(items, tuple) else normalized


class DatusLitellmModel(ObservedLitellmModel):
    """LiteLLM model with Datus image, reasoning, and provider compatibility."""

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        """Return a response whose reasoning items record their source endpoint."""
        response = await super().get_response(*args, **kwargs)
        endpoint_identity = reasoning_endpoint_identity(self.base_url)
        for item in response.output:
            _stamp_reasoning_endpoint(item, endpoint_identity)
        return response

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream events after recording endpoint provenance on reasoning items."""
        endpoint_identity = reasoning_endpoint_identity(self.base_url)
        async for event in super().stream_response(*args, **kwargs):
            _stamp_reasoning_event(event, endpoint_identity)
            yield event

    async def _fetch_response(self, system_instructions: Any, input: Any, *args: Any, **kwargs: Any) -> Any:
        if is_kimi_model(self.model):
            input = normalize_kimi_tool_replay_items(input)
        return await super()._fetch_response(system_instructions, tool_images_as_user_input(input), *args, **kwargs)

    def _convert_gemini_extra_content_to_provider_specific_fields(self, messages: list[Any]) -> list[Any]:
        # An image returned by a tool is transported as user content, but does
        # not start a new turn. The SDK otherwise skips the preceding tool
        # call's thought signature when it searches for the last user message.
        conversation = [message for message in messages if not is_tool_image_user_message(message)]
        super()._convert_gemini_extra_content_to_provider_specific_fields(conversation)
        return messages
