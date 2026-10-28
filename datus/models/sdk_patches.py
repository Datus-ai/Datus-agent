# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Narrow compatibility patch for provider correlation metadata."""

from __future__ import annotations

from functools import wraps
from typing import Any

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_original_openai_stream_chunk_parser = None


def _preserve_stream_request_id(chunk: Any, response: Any) -> Any:
    """Copy GLM's documented streaming ``request_id`` through LiteLLM."""
    if isinstance(chunk, dict):
        request_id = chunk.get("request_id")
        if isinstance(request_id, str) and request_id and response is not None:
            response.request_id = request_id
    return response


def apply_sdk_patches() -> None:
    """Preserve GLM's streaming request ID dropped by LiteLLM 1.100.1."""
    global _original_openai_stream_chunk_parser

    from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler

    if _original_openai_stream_chunk_parser is None:
        _original_openai_stream_chunk_parser = OpenAIChatCompletionStreamingHandler.chunk_parser

        @wraps(_original_openai_stream_chunk_parser)
        def _patched_openai_stream_chunk_parser(self, chunk):
            return _preserve_stream_request_id(chunk, _original_openai_stream_chunk_parser(self, chunk))

        OpenAIChatCompletionStreamingHandler.chunk_parser = _patched_openai_stream_chunk_parser
        logger.debug("Applied SDK patch: GLM streaming request_id")


def remove_sdk_patches() -> None:
    """Restore the patched LiteLLM method."""
    global _original_openai_stream_chunk_parser

    from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler

    if _original_openai_stream_chunk_parser is not None:
        OpenAIChatCompletionStreamingHandler.chunk_parser = _original_openai_stream_chunk_parser
        _original_openai_stream_chunk_parser = None
        logger.debug("Removed SDK patch: GLM streaming request_id")
