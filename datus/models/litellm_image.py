# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Image and reasoning transport for OpenAI-compatible Chat Completions APIs."""

from collections.abc import AsyncIterator
from typing import Any

import litellm

from datus.models.observed_model import ObservedLitellmModel
from datus.models.reasoning_replay import REASONING_ENDPOINT_KEY, reasoning_endpoint_identity
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


class ImageToolLitellmModel(ObservedLitellmModel):
    """LiteLLM model that transports tool images and reasoning provenance."""

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
        return await super()._fetch_response(system_instructions, tool_images_as_user_input(input), *args, **kwargs)

    def _convert_gemini_extra_content_to_provider_specific_fields(self, messages: list[Any]) -> list[Any]:
        # An image returned by a tool is transported as user content, but does
        # not start a new turn. The SDK otherwise skips the preceding tool
        # call's thought signature when it searches for the last user message.
        conversation = [message for message in messages if not is_tool_image_user_message(message)]
        super()._convert_gemini_extra_content_to_provider_specific_fields(conversation)
        return messages
