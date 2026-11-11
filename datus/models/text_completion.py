# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Single-call text generation through the same SDK model as streaming chat."""

import asyncio
from contextlib import aclosing
from typing import Any

from agents.models.interface import ModelTracing
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from openai import AsyncOpenAI

from datus.models.observed_model import ObservedResponsesModel
from datus.utils.ssl_utils import resolve_ssl_verify_for_httpx


async def complete_text(
    model: Any, prompt: str, *, instruction: str, max_tokens: int | None, timeout: float
) -> dict[str, Any]:
    # Reuse chat's provider routing, sampling/reasoning rules, headers and capacity
    # checks. No Runner, sessions, tools, hooks, tracing or application retries.
    agent = model._build_agent(instruction, str, True, {}, [], max_tokens=max_tokens)
    sdk_model = agent.model
    verify = model.model_config.ssl_verify
    verify = resolve_ssl_verify_for_httpx(verify) if verify is not None else True
    transport = AsyncHTTPHandler(timeout=timeout, ssl_verify=verify)
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
        response = None
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
                if event.type == "response.completed":
                    response = event.response
        usage = (
            (response.usage if response is not None else None)
            if isinstance(sdk_model, ObservedResponsesModel)
            else getattr(sdk_model, "_datus_reported_usage", None)
        )
        if response is None or usage is None:
            raise ValueError("missing completion or usage")
        return {"content": response.output_text, "usage": usage, "model": model.model_name}
    finally:
        await transport.close()
