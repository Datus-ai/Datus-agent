# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Preserve explicit output-limit failures before SDK response conversion."""

from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from inspect import isawaitable
from typing import Any

from agents.exceptions import AgentsException
from agents.usage import Usage


class ModelOutputTruncatedError(AgentsException):
    """A model exhausted its output budget; partial text is not a completed answer."""

    def __init__(self, partial_response: str = "", usage: Usage | None = None, reason: str = "length"):
        super().__init__(f"Model output was truncated at the output token limit ({reason}).")
        self.partial_response = partial_response
        self.usage = usage if usage is not None else Usage(requests=1)


def _value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def chat_completion_usage(raw: Any) -> Usage:
    """Normalize received provider usage without estimating missing token counts."""
    return Usage(
        requests=1,
        input_tokens=_value(raw, "prompt_tokens", 0) or 0,
        output_tokens=_value(raw, "completion_tokens", 0) or 0,
        total_tokens=_value(raw, "total_tokens", 0) or 0,
        input_tokens_details=_value(raw, "prompt_tokens_details"),
        output_tokens_details=_value(raw, "completion_tokens_details"),
    )


def check_chat_completion(response: Any) -> None:
    """Reject a truncated non-streaming response before tools or reasoning conversion."""
    for choice in _value(response, "choices", []) or []:
        if _value(choice, "index", 0) == 0 and _value(choice, "finish_reason") == "length":
            content = _value(_value(choice, "message"), "content")
            raise ModelOutputTruncatedError(
                content if isinstance(content, str) else "", chat_completion_usage(_value(response, "usage"))
            )


def _received_chat_stream_usage(stream: Any, fallback: Any) -> Any:
    # LiteLLM 1.100.1 strips usage from provider chunks and may synthesize an
    # estimate that omits reasoning tokens. Its retained chunks still contain
    # the reported usage, including usage embedded in the finish chunk (GLM).
    chunks = getattr(stream, "chunks", None)
    if isinstance(chunks, list):
        received = [_value(chunk, "usage") for chunk in chunks if _value(chunk, "usage")]
        if not received:
            return None
        # Anthropic reports input usage at message_start and output usage at
        # message_delta. The final chunk alone has no input count. These are
        # cumulative counters for one request, so merge rather than sum them.
        merged = {
            key: max((_value(usage, key, 0) or 0) for usage in received)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        merged["total_tokens"] = max(merged["total_tokens"], merged["prompt_tokens"] + merged["completion_tokens"])
        for key in ("prompt_tokens_details", "completion_tokens_details"):
            merged[key] = next((_value(usage, key) for usage in reversed(received) if _value(usage, key)), None)
        return merged
    return fallback


async def checked_chat_completion_stream(
    stream: Any, *, on_usage: Callable[[Usage | None], None] | None = None
) -> AsyncIterator[Any]:
    """Stop conversion on length while collecting trailing usage, with per-call state.

    The SDK otherwise emits response.completed even for finish_reason=length.
    Raising before EOF also prevents its buffered, incomplete function calls
    from being flushed and executed. Only assistant content is retained, never
    reasoning deltas or function arguments.
    """
    text: list[str] = []
    usage = None
    truncated = False
    try:
        async with aclosing(stream):
            async for chunk in stream:
                if received_usage := _value(chunk, "usage"):
                    usage = received_usage
                for choice in _value(chunk, "choices", []) or []:
                    if _value(choice, "index", 0) != 0:
                        continue
                    content = _value(_value(choice, "delta"), "content")
                    if isinstance(content, str):
                        text.append(content)
                    truncated = truncated or _value(choice, "finish_reason") == "length"
                if not truncated:
                    yield chunk
    except Exception as exc:
        if truncated:
            # A failed usage-only tail cannot turn known output exhaustion
            # into a retryable transport error and repeat earlier tool calls.
            raise ModelOutputTruncatedError(
                "".join(text), chat_completion_usage(_received_chat_stream_usage(stream, usage))
            ) from exc
        raise
    received_usage = _received_chat_stream_usage(stream, usage)
    if on_usage is not None:
        on_usage(chat_completion_usage(received_usage) if received_usage is not None else None)
    if truncated:
        raise ModelOutputTruncatedError(
            "".join(text), chat_completion_usage(_received_chat_stream_usage(stream, usage))
        )


def check_responses_output(response: Any) -> None:
    """Handle the Responses API's explicit output-limit status with the same semantics."""
    details = _value(response, "incomplete_details")
    if _value(response, "status") != "incomplete" or _value(details, "reason") != "max_output_tokens":
        return
    text = []
    for item in _value(response, "output", []) or []:
        if _value(item, "type") != "message" or _value(item, "role") != "assistant":
            continue
        for part in _value(item, "content", []) or []:
            if _value(part, "type") == "output_text" and isinstance(_value(part, "text"), str):
                text.append(_value(part, "text"))
    raw_usage = _value(response, "usage")
    usage = Usage(
        requests=1,
        input_tokens=_value(raw_usage, "input_tokens", 0) or 0,
        output_tokens=_value(raw_usage, "output_tokens", 0) or 0,
        total_tokens=_value(raw_usage, "total_tokens", 0) or 0,
        input_tokens_details=_value(raw_usage, "input_tokens_details"),
        output_tokens_details=_value(raw_usage, "output_tokens_details"),
    )
    raise ModelOutputTruncatedError("".join(text), usage, reason="max_output_tokens")


async def checked_responses_stream(stream: Any) -> AsyncIterator[Any]:
    """Check terminal Responses metadata before the SDK processes its output items."""
    try:
        async for event in stream:
            response = _value(event, "response")
            if response is not None:
                check_responses_output(response)
            yield event
    finally:
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close:
            result = close()
            if isawaitable(result):
                await result
