# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Single-call text generation through the same SDK model as streaming chat."""

import asyncio
import time
from contextlib import aclosing
from typing import Any

import httpx
from agents.models.interface import ModelTracing
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from openai import AsyncOpenAI

from datus.models.observed_model import ObservedResponsesModel
from datus.models.output_truncation import ModelOutputTruncatedError
from datus.observability.model_call import current_model_call
from datus.utils.loggings import get_logger
from datus.utils.ssl_utils import resolve_ssl_verify_for_httpx

logger = get_logger(__name__)


def _usage_value(value: Any, key: str) -> Any:
    """Read usage metadata from SDK objects and provider dictionaries."""
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


class _RequestLocalHTTPHandler(AsyncHTTPHandler):
    """Retain request-local TLS settings when LiteLLM recreates a connection."""

    def create_client(self, **kwargs: Any) -> httpx.AsyncClient:
        """Use the original verification setting for initial and retry clients."""
        kwargs["ssl_verify"] = self.ssl_verify
        return super().create_client(**kwargs)


async def complete_text(
    model: Any, prompt: str, *, instruction: str, max_tokens: int | None, timeout: float
) -> dict[str, Any]:
    """Generate tool-less text with chat's model routing and provider-reported usage."""
    # Reuse chat's provider routing, sampling/reasoning rules, headers and capacity
    # checks. No Runner, sessions, tools, hooks, tracing or application retries.
    agent = model._build_agent(instruction, str, True, {}, [], max_tokens=max_tokens)
    sdk_model = agent.model
    verify = model.model_config.ssl_verify
    verify = resolve_ssl_verify_for_httpx(verify) if verify is not None else None
    transport = _RequestLocalHTTPHandler(timeout=timeout, ssl_verify=verify)
    started = time.monotonic()
    first_text_ms = first_reasoning_ms = None
    output_chars = reasoning_chars = 0
    usage = response = None
    status = "failed"
    error_type = model_call_id = None
    truncation_reason = None
    try:
        settings = agent.model_settings
        if isinstance(sdk_model, ObservedResponsesModel):
            # This SDK model is fresh, not the cached chat model/client.
            await sdk_model._client.close()
            sdk_model._client = AsyncOpenAI(
                api_key=model.api_key,
                base_url=model.base_url,
                default_headers=model.default_headers,
                http_client=transport.client,
                max_retries=0,
                timeout=timeout,
            )
        else:
            # LiteLLM's native Anthropic/Gemini paths consume HTTP handlers;
            # OpenAI-compatible paths consume an OpenAI SDK client. TLS settings
            # belong on these clients, never in the provider's JSON request body.
            client = transport
            if model.litellm_adapter.provider not in {"claude", "anthropic", "gemini", "deepseek", "openrouter"}:
                client = AsyncOpenAI(
                    api_key=model.api_key,
                    base_url=model.litellm_adapter.base_url,
                    http_client=transport.client,
                    max_retries=0,
                    timeout=timeout,
                )
            sdk_model._datus_request_options = {
                "client": client,
                "timeout": timeout,
                "num_retries": 0,
                "max_retries": 0,
                "no-log": True,
                "success_callback": [],
                "failure_callback": [],
            }
            sdk_model._datus_capture_usage = True
            sdk_model._datus_reported_usage = None
            sdk_model._datus_finish_reason = None
        # Streaming also supports providers which reject large non-streaming
        # output caps. Consume it locally; callers receive one complete result.
        async with (
            asyncio.timeout(timeout),
            aclosing(
                sdk_model.stream_response(
                    system_instructions=agent.instructions,
                    input=prompt,
                    model_settings=settings,
                    tools=[],
                    output_schema=None,
                    handoffs=[],
                    tracing=ModelTracing.DISABLED,
                )
            ) as stream,
        ):
            async for event in stream:
                if call := current_model_call():
                    model_call_id = call.model_call_id
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    elapsed = round((time.monotonic() - started) * 1000, 2)
                    if event.type == "response.output_text.delta":
                        output_chars += len(delta)
                        if first_text_ms is None:
                            first_text_ms = elapsed
                            logger.info(
                                "text_completion.first_text",
                                model=model.model_name,
                                model_call_id=model_call_id,
                                first_text_ms=first_text_ms,
                            )
                    elif event.type in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
                        reasoning_chars += len(delta)
                        if first_reasoning_ms is None:
                            first_reasoning_ms = elapsed
                if event.type == "response.completed":
                    response = event.response
                    output_chars = len(response.output_text)
        usage = (
            (response.usage if response is not None else None)
            if isinstance(sdk_model, ObservedResponsesModel)
            else getattr(sdk_model, "_datus_reported_usage", None)
        )
        if response is None or usage is None:
            raise ValueError("missing completion or usage")
        status = "succeeded"
        return {"content": response.output_text, "usage": usage, "model": model.model_name}
    except ModelOutputTruncatedError as error:
        status, error_type = "truncated", type(error).__name__
        usage = error.usage if error.usage_reported else None
        truncation_reason = "max_output_tokens" if isinstance(sdk_model, ObservedResponsesModel) else "length"
        output_chars = max(output_chars, len(error.partial_response))
        raise
    except TimeoutError as error:
        status, error_type = "timeout", type(error).__name__
        raise
    except asyncio.CancelledError as error:
        status, error_type = "cancelled", type(error).__name__
        raise
    except Exception as error:
        error_type = type(error).__name__
        raise
    finally:
        try:
            await transport.close()
        finally:
            # Only timings, counters and fixed classifications: never prompt,
            # response text, reasoning text, credentials or exception messages.
            reason = truncation_reason or getattr(sdk_model, "_datus_finish_reason", None)
            if isinstance(sdk_model, ObservedResponsesModel) and response is not None:
                reason = response.status
            if reason not in {
                None,
                "stop",
                "length",
                "max_output_tokens",
                "tool_calls",
                "function_call",
                "content_filter",
                "completed",
            }:
                reason = "other"
            logger.info(
                "text_completion.finished",
                model=model.model_name,
                model_call_id=model_call_id,
                status=status,
                error_type=error_type,
                finish_reason=reason,
                duration_ms=round((time.monotonic() - started) * 1000, 2),
                first_text_ms=first_text_ms,
                first_reasoning_ms=first_reasoning_ms,
                output_chars=output_chars,
                reasoning_chars=reasoning_chars,
                output_token_limit=agent.model_settings.max_tokens,
                timeout_seconds=timeout,
                usage_reported=usage is not None,
                input_tokens=_usage_value(usage, "input_tokens"),
                output_tokens=_usage_value(usage, "output_tokens"),
                reasoning_tokens=_usage_value(_usage_value(usage, "output_tokens_details"), "reasoning_tokens"),
            )
