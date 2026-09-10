# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Observe existing Agents SDK model boundaries without replacing its loops."""

from __future__ import annotations

from contextlib import aclosing
from typing import Any

from agents.extensions.models.litellm_model import LitellmModel
from agents.models.chatcmpl_converter import Converter
from agents.models.openai_responses import OpenAIResponsesModel
from agents.util._json import _to_dump_compatible

from datus.observability.model_call import ModelCall, current_model_call


class _ObservedAsyncStream:
    """Observe provider metadata on each raw LiteLLM chunk without changing it."""

    def __init__(self, stream: Any, call: ModelCall):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._call = call

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._iterator.__anext__()
        self._call.response(chunk)
        return chunk

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


class _ObservedModel:
    _datus_protocol = "chat_completions"
    _datus_impl = "litellm"

    def _new_call(self) -> ModelCall:
        endpoint = getattr(self, "base_url", None)
        if endpoint is None:
            endpoint = getattr(getattr(self, "_client", None), "base_url", None)
        call = ModelCall(
            model=str(self.model), model_impl=self._datus_impl, protocol=self._datus_protocol, endpoint=endpoint
        )
        from agents.tracing import get_current_span

        parent = get_current_span()
        if parent is not None and parent.span_data.type == "agent":
            call.bind_agent(parent)
        return call

    async def get_response(self, *args, **kwargs):
        with self._new_call() as call:
            result = await super().get_response(*args, **kwargs)
            call.usage(result.usage)
            call.record_tool_calls(result.output)
            return result

    async def stream_response(self, *args, **kwargs):
        with self._new_call() as call:
            async with aclosing(super().stream_response(*args, **kwargs)) as stream:
                async for event in stream:
                    call.stream_event()
                    response = getattr(event, "response", None)
                    if response is not None:
                        call.usage(getattr(response, "usage", None))
                        call.record_tool_calls(getattr(response, "output", None))
                    yield event


class ObservedLitellmModel(_ObservedModel, LitellmModel):
    async def _fetch_response(
        self, system_instructions, input, model_settings, tools, output_schema, handoffs, span, tracing, *args, **kwargs
    ):
        call = current_model_call()
        if call:
            call.bind_sdk_span(span)
            # Reuse SDK converters for the resolved tools (including handoffs).
            # LiteLLM may still transform them into a provider-specific payload.
            definitions = _to_dump_compatible(
                [Converter.tool_to_openai(tool) for tool in tools]
                + [Converter.convert_handoff_tool(handoff) for handoff in handoffs]
            )
            definitions = self._tools_for_observation(definitions)
            parallel = (
                True
                if model_settings.parallel_tool_calls and tools
                else False
                if model_settings.parallel_tool_calls is False
                else None
            )
            call.request(
                {
                    "tools": definitions,
                    "tool_choice": Converter.convert_tool_choice(model_settings.tool_choice),
                    "parallel_tool_calls": parallel,
                }
            )
        result = await super()._fetch_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, span, tracing, *args, **kwargs
        )
        if call:
            if isinstance(result, tuple):
                response, stream = result
                call.response(stream, streaming=True)
                result = response, _ObservedAsyncStream(stream, call)
            else:
                call.response(result)
        return result

    def _tools_for_observation(self, definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return definitions


class ObservedResponsesModel(_ObservedModel, OpenAIResponsesModel):
    _datus_protocol = "responses"
    _datus_impl = "openai_responses"

    def _build_response_create_kwargs(self, *args, **kwargs):
        params = super()._build_response_create_kwargs(*args, **kwargs)
        if call := current_model_call():
            # extra_body is merged by the OpenAI SDK after these keyword args.
            effective = dict(params)
            if isinstance(params.get("extra_body"), dict):
                effective.update(params["extra_body"])
            call.request(effective)
        return params

    async def _fetch_response(self, *args, **kwargs):
        result = await super()._fetch_response(*args, **kwargs)
        if call := current_model_call():
            # SDK 0.13.4's stream wrapper exposes correlation headers before
            # any SSE events or terminal response is consumed.
            call.response(result, streaming=hasattr(result, "__aiter__"))
        return result
