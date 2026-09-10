# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Narrow compatibility patches for provider correlation metadata."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_original_make_async_call_stream_helper = None
_original_openai_stream_chunk_parser = None
_CORRELATION_RESPONSE_HEADERS = {
    "msh-request-id",
    "request-id",
    "trace-id",
    "x-ds-trace-id",
    "x-request-id",
}


def _remember_stream_response_headers(result: Any, logging_obj: Any) -> None:
    """Keep only known correlation headers on LiteLLM's stream logging object."""
    if not isinstance(result, tuple) or len(result) != 2 or logging_obj is None:
        return
    headers = result[1]
    try:
        captured = {
            str(key).lower(): value
            for key, value in headers.items()
            if str(key).lower() in _CORRELATION_RESPONSE_HEADERS and isinstance(value, str) and value
        }
    except (AttributeError, TypeError):
        return
    if not captured:
        return
    details = getattr(logging_obj, "model_call_details", None)
    if isinstance(details, dict):
        details["_datus_response_headers"] = captured


def _preserve_stream_request_id(chunk: Any, response: Any) -> Any:
    """Copy GLM's documented streaming ``request_id`` through LiteLLM."""
    if isinstance(chunk, dict):
        request_id = chunk.get("request_id")
        if isinstance(request_id, str) and request_id and response is not None:
            response.request_id = request_id
    return response


def apply_sdk_patches() -> None:
    """Preserve streaming correlation fields dropped by LiteLLM 1.100.1."""
    global _original_make_async_call_stream_helper, _original_openai_stream_chunk_parser

    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler

    if _original_make_async_call_stream_helper is None:
        _original_make_async_call_stream_helper = BaseLLMHTTPHandler.make_async_call_stream_helper

        @wraps(_original_make_async_call_stream_helper)
        async def _patched_make_async_call_stream_helper(self, *args, **kwargs):
            result = await _original_make_async_call_stream_helper(self, *args, **kwargs)
            logging_obj = kwargs.get("logging_obj")
            if logging_obj is None:
                try:
                    bound = inspect.signature(_original_make_async_call_stream_helper).bind(self, *args, **kwargs)
                    logging_obj = bound.arguments.get("logging_obj")
                except (TypeError, ValueError):
                    pass
            _remember_stream_response_headers(result, logging_obj)
            return result

        BaseLLMHTTPHandler.make_async_call_stream_helper = _patched_make_async_call_stream_helper
        logger.debug("Applied SDK patch: LiteLLM streaming correlation headers")

    if _original_openai_stream_chunk_parser is None:
        _original_openai_stream_chunk_parser = OpenAIChatCompletionStreamingHandler.chunk_parser

        @wraps(_original_openai_stream_chunk_parser)
        def _patched_openai_stream_chunk_parser(self, chunk):
            return _preserve_stream_request_id(chunk, _original_openai_stream_chunk_parser(self, chunk))

        OpenAIChatCompletionStreamingHandler.chunk_parser = _patched_openai_stream_chunk_parser
        logger.debug("Applied SDK patch: GLM streaming request_id")


def remove_sdk_patches() -> None:
    """Restore the two patched LiteLLM methods."""
    global _original_make_async_call_stream_helper, _original_openai_stream_chunk_parser

    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler

    if _original_make_async_call_stream_helper is not None:
        BaseLLMHTTPHandler.make_async_call_stream_helper = _original_make_async_call_stream_helper
        _original_make_async_call_stream_helper = None
        logger.debug("Removed SDK patch: LiteLLM streaming correlation headers")

    if _original_openai_stream_chunk_parser is not None:
        OpenAIChatCompletionStreamingHandler.chunk_parser = _original_openai_stream_chunk_parser
        _original_openai_stream_chunk_parser = None
        logger.debug("Removed SDK patch: GLM streaming request_id")
