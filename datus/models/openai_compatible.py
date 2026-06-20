# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OpenAI-compatible base model for models that use OpenAI-compatible APIs."""

import asyncio
import hashlib
import json
import os
import threading
import time
from contextlib import nullcontext
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
import yaml
from agents import Agent, ModelSettings, Runner, Tool
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from agents.extensions.memory import AdvancedSQLiteSession
from agents.mcp import MCPServerStdio
from agents.run import CallModelData, ModelInputData, RunConfig
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from openai.types.shared.reasoning import Reasoning
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.litellm_adapter import LiteLLMAdapter, is_known_non_thinking_model, is_official_openai_endpoint
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.observability.manager import get_observability_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.tool_summary import TOOL_SUMMARY_REGISTRY, looks_like_failure
from datus.utils.constants import LLMProvider
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import to_str
from datus.utils.loggings import get_logger
from datus.utils.resource_utils import read_data_file_text
from datus.utils.text_utils import LitellmPlaceholderStreamFilter, strip_litellm_placeholder
from datus.utils.trace_context import build_agents_run_config_kwargs, build_trace_span_attributes

logger = get_logger(__name__)

# LiteLLM configuration
# Enable dropping unsupported parameters for providers that don't support them
# This allows us to set reasoning=Reasoning(effort=...) for preserve_thinking_blocks
# while LiteLLM automatically drops reasoning_effort for providers like Moonshot
litellm.drop_params = True
# Enable modify_params to handle Anthropic tool calling requirements
# When tool_choice is set but tools is empty, LiteLLM will add a dummy tool
litellm.modify_params = True
litellm.set_verbose = False
# Suppress "Provider List: ..." debug prints to stdout when LiteLLM encounters
# model names not in its built-in list (e.g. Coding Plan models like kimi-for-coding)
litellm.suppress_debug_info = True

# Module-level cache for model specs loaded from conf/providers.yml
_MODEL_SPECS_CACHE: Optional[Dict[str, Dict[str, int]]] = None
_MODEL_SPECS_LOCK = threading.Lock()


def _load_model_specs() -> Dict[str, Dict[str, int]]:
    """Load model specifications from conf/providers.yml (cached after first call, thread-safe).

    Specs from ``providers.yml`` are authoritative. The OpenRouter cache at
    ``~/.datus/cache/openrouter_models.json`` provides a fallback ``context_length``
    for slugs the YAML doesn't enumerate — this keeps the status bar and
    auto-compaction heuristic aware of fresh OpenRouter models without
    requiring a YAML edit for every new release.
    """
    global _MODEL_SPECS_CACHE
    if _MODEL_SPECS_CACHE is not None:
        return _MODEL_SPECS_CACHE
    with _MODEL_SPECS_LOCK:
        if _MODEL_SPECS_CACHE is None:
            specs: Dict[str, Dict[str, int]] = {}
            try:
                text = read_data_file_text("conf/providers.yml")
                catalog = yaml.safe_load(text)
                raw_specs = catalog.get("model_specs") or {}
                if isinstance(raw_specs, dict):
                    specs = {
                        k: dict(v) for k, v in raw_specs.items() if isinstance(k, str) and k and isinstance(v, dict)
                    }
            except Exception as e:
                logger.warning(f"Failed to load model_specs from providers.yml, using empty specs: {e}")

            # Merge OpenRouter cache: YAML wins on shared keys.
            try:
                from datus.cli.provider_model_catalog import load_cached_model_details

                cache_details = load_cached_model_details() or {}
                for entries in cache_details.values():
                    for entry in entries:
                        slug = entry.get("id") if isinstance(entry, dict) else None
                        if not isinstance(slug, str) or not slug:
                            continue
                        ctx_len = entry.get("context_length")
                        if not isinstance(ctx_len, int):
                            continue
                        specs.setdefault(slug, {}).setdefault("context_length", ctx_len)
            except Exception as e:
                logger.debug(f"Failed to merge OpenRouter cache into model_specs: {e}")

            _MODEL_SPECS_CACHE = specs
    return _MODEL_SPECS_CACHE


def _set_observability_span_attribute(span: Any, key: str, value: Any) -> None:
    if span is None or value is None:
        return
    try:
        if not isinstance(value, (str, bool, int, float)):
            value = to_str(value)
        span.set_attribute(key, value)
    except Exception:
        return


def _agents_trace_baggage(agent_name: Optional[str]):
    attrs = build_trace_span_attributes(operation=agent_name or "agent", run_type="llm")
    trace_name = attrs.get("datus.trace.name")
    if not trace_name:
        return nullcontext()
    return get_observability_manager().trace_baggage(str(trace_name), attrs)


async def _stream_events_with_trace_baggage(result, agent_name: Optional[str]):
    with _agents_trace_baggage(agent_name):
        async for event in result.stream_events():
            yield event


def classify_openai_compatible_error(error: Exception) -> tuple[ErrorCode, bool]:
    """Classify OpenAI-compatible API errors and return error code and whether it's retryable."""
    error_msg = str(error).lower()

    # TLS certificate failures must be detected first: the litellm SSL error
    # string contains "internal" (from InternalServerError) and would otherwise
    # be misclassified as a retryable MODEL_API_ERROR. Not retryable — retrying a
    # cert mismatch never helps. The dedicated code carries the ssl_verify hint.
    from datus.utils.ssl_utils import is_ssl_cert_verification_error

    if is_ssl_cert_verification_error(error):
        return ErrorCode.MODEL_SSL_CERT_ERROR, False

    if isinstance(error, APIError):
        # Handle specific HTTP status codes and error types
        if any(indicator in error_msg for indicator in ["401", "unauthorized", "authentication"]):
            return ErrorCode.MODEL_AUTHENTICATION_ERROR, False
        elif any(indicator in error_msg for indicator in ["403", "forbidden", "permission"]):
            return ErrorCode.MODEL_PERMISSION_ERROR, False
        elif any(indicator in error_msg for indicator in ["404", "not found"]):
            return ErrorCode.MODEL_NOT_FOUND, False
        elif any(indicator in error_msg for indicator in ["413", "too large", "request size"]):
            return ErrorCode.MODEL_REQUEST_TOO_LARGE, False
        elif any(indicator in error_msg for indicator in ["429", "rate limit", "quota", "billing"]):
            if any(indicator in error_msg for indicator in ["quota", "billing"]):
                return ErrorCode.MODEL_QUOTA_EXCEEDED, False
            else:
                return ErrorCode.MODEL_RATE_LIMIT, True
        elif any(indicator in error_msg for indicator in ["500", "internal", "server error"]):
            return ErrorCode.MODEL_API_ERROR, True
        elif any(indicator in error_msg for indicator in ["502", "503", "overloaded"]):
            return ErrorCode.MODEL_OVERLOADED, True
        elif any(indicator in error_msg for indicator in ["400", "bad request", "invalid"]):
            return ErrorCode.MODEL_INVALID_RESPONSE, False

    if isinstance(error, RateLimitError):
        return ErrorCode.MODEL_RATE_LIMIT, True

    if isinstance(error, APITimeoutError):
        return ErrorCode.MODEL_TIMEOUT_ERROR, True

    if isinstance(error, APIConnectionError):
        return ErrorCode.MODEL_CONNECTION_ERROR, True

    # Default to general request failure
    return ErrorCode.MODEL_REQUEST_FAILED, False


# ----------------------------------------------------------------------
# Tool result summary helpers
# ----------------------------------------------------------------------
#
# ``short_desc`` (= ``ActionHistory.output["summary"]``) is rendered in the
# CLI and surfaced as ``shortDesc`` in the streaming SSE payload. The
# per-tool formatter logic now lives in ``datus.schemas.tool_summary`` so
# CLI compact rendering and SSE share a single source of truth.


def _detect_tool_failure(output_content: Any) -> bool:
    """Return True when the tool's output payload signals failure.

    Tools built on :class:`FuncToolResult` report errors via ``success=0`` /
    non-empty ``error`` instead of raising. The Agents SDK therefore emits a
    ``tool_call_output_item`` for them and we must inspect the payload to
    render ✗ (and mark the action FAILED) correctly.
    """
    data: Optional[dict] = None
    if isinstance(output_content, dict):
        data = output_content
    elif isinstance(output_content, str):
        try:
            parsed = json.loads(output_content)
        except (TypeError, ValueError):
            return False
        if isinstance(parsed, dict):
            data = parsed
    if data is None:
        return False
    return looks_like_failure(data)


class OpenAICompatibleModel(LLMBaseModel):
    """
    Base class for models that use OpenAI-compatible APIs.

    Provides common functionality for:
    - Session management for multi-turn conversations
    - OpenAI client setup and configuration
    - Unified tool execution (replacing generate_with_mcp)
    - Streaming support
    - Error handling and retry logic
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

        self.model_config = model_config
        self.model_name = model_config.model
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        self.default_headers = dict(self.model_config.default_headers) if self.model_config.default_headers else None

        # Resolve SSL/TLS verification for this endpoint (e.g. a private gateway CA).
        # When configured, it takes priority over the SSL_VERIFY / SSL_CERT_FILE env
        # vars (house convention: config first, env fallback). We materialize it into
        # SSL_VERIFY so litellm honors it across both the agents-SDK LitellmModel path
        # (whose __init__ accepts no ssl argument) and direct litellm.completion calls.
        # The native Anthropic client (ClaudeModel) reads self.ssl_verify directly.
        #
        # Why a process-level env var and not a per-request kwarg: litellm does NOT
        # thread a per-call ``ssl_verify`` into the Anthropic provider's HTTP client
        # (verified — the kwarg is silently ignored), and a scoped set/restore around
        # each call would break the agents-SDK streaming path (Runner.run_streamed does
        # its network I/O after returning). A process global is the only mechanism that
        # works across all paths. Blast radius is limited: ``SSL_VERIFY`` is a
        # litellm-specific variable that standard libraries (requests/httpx/curl) ignore.
        from datus.utils.ssl_utils import normalize_ssl_verify, ssl_verify_to_env

        ssl_verify_cfg = getattr(self.model_config, "ssl_verify", None)
        # Only act on real bool/str config values; a non-(bool|str) (e.g. unset, or a
        # test double) leaves behavior untouched.
        if isinstance(ssl_verify_cfg, (bool, str)):
            self.ssl_verify = normalize_ssl_verify(ssl_verify_cfg)
            os.environ["SSL_VERIFY"] = ssl_verify_to_env(self.ssl_verify)
        else:
            self.ssl_verify = None

        # Initialize LiteLLM adapter for unified LLM calls
        self.litellm_adapter = LiteLLMAdapter(
            provider=model_config.type,
            model=model_config.model,
            api_key=self.api_key,
            base_url=self.base_url,
            enable_thinking=model_config.enable_thinking,
            reasoning_effort=model_config.reasoning_effort,
            default_headers=self.default_headers,
        )

        # Context for tracing ToDo: replace it with Context object
        self.current_node = None

        # Cache for model info
        self._model_info = None

    def _get_api_key(self) -> str:
        """Get API key from config or environment. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_api_key")

    def _get_base_url(self) -> Optional[str]:
        """Get base URL from config. Override in subclasses if needed."""
        return self.model_config.base_url

    def _is_official_openai_api(self) -> bool:
        """Return True only for official OpenAI API endpoints."""
        return is_official_openai_endpoint(self.model_config.type, self.base_url)

    def _default_prompt_cache_retention(self) -> Optional[str]:
        """Choose a safe default prompt cache retention policy for OpenAI."""
        if not self._is_official_openai_api():
            return None
        if self.model_name.startswith("gpt-5"):
            return "24h"
        return "in_memory"

    def _model_supports_reasoning(self) -> bool:
        """Decide whether ``/effort`` should inject ``Reasoning(effort=…)``.

        Authority chain (permissive by design — new models default to
        supported, Datus only bails out when it has *positive* evidence the
        model cannot reason):

        1. Datus-maintained deny-list (:func:`is_known_non_thinking_model`)
           kicks in first. ``deepseek-chat``, ``moonshot-v1-*`` and the bare
           ``kimi-k2`` family are the only DeepSeek/Kimi models we treat as
           non-reasoning; everything else in those providers defaults to
           supported so LiteLLM's catalog lag on DeepSeek V4 / Kimi K2.x
           doesn't mute the effort hint.
        2. LiteLLM's built-in catalog returns ``True`` → accepted.
        3. LiteLLM raises (unknown provider, self-hosted proxy, …) → accepted
           under the permissive default.
        4. LiteLLM returns ``False`` → accepted. New models routinely ship
           before LiteLLM adds them; ``drop_params=True`` still protects the
           request at the transport layer. ``_build_agent`` emits a ``skip``
           warning only when branch 1 explicitly blocks.
        """
        provider = self.litellm_adapter.provider
        if is_known_non_thinking_model(provider, self.model_name):
            return False
        try:
            if litellm.supports_reasoning(model=self.model_name, custom_llm_provider=provider):
                return True
        except Exception as e:
            logger.debug(f"litellm.supports_reasoning raised for {self.model_name}: {e}")
        return True

    def _native_thinking_extra_body(self) -> Optional[Dict[str, Any]]:
        """Extra ``extra_body`` payload that activates DeepSeek/Moonshot thinking.

        LiteLLM handles these two providers differently, and both need Datus
        intervention to land the required fields in the HTTP body:

        - **Moonshot**: its ``get_supported_openai_params`` does not list
          ``thinking`` or ``reasoning_effort``, so under
          ``litellm.drop_params=True`` both fields are stripped during param
          mapping. Pushing ``thinking`` through ``extra_body`` bypasses the
          filter because the OpenAI client merges ``extra_body`` verbatim into
          the request body. ``reasoning_effort`` is intentionally NOT added
          for Moonshot — Kimi K2.x does not accept it and would reject the
          request.
        - **DeepSeek**: its transformation *does* recognise both fields but
          pops ``reasoning_effort`` during mapping (it only keeps ``thinking``
          internally). Per
          https://api-docs.deepseek.com/zh-cn/guides/thinking_mode, DeepSeek
          requires ``reasoning_effort`` alongside ``thinking.type=enabled`` to
          control depth. Adding ``reasoning_effort`` to ``extra_body`` re-injects
          it into the final request body after the transformation has run.

        Returns ``None`` for non-DeepSeek/Kimi providers or explicitly
        non-thinking models (see :func:`is_known_non_thinking_model`).
        """
        provider = self.litellm_adapter.provider
        if provider not in ("deepseek", "kimi"):
            return None
        if is_known_non_thinking_model(provider, self.model_name):
            return None
        body: Dict[str, Any] = {"thinking": {"type": "enabled"}}
        if provider == "deepseek":
            effort = self.litellm_adapter.reasoning_effort_level
            if effort:
                body["reasoning_effort"] = effort
        return body

    def _default_prompt_cache_key(self, agent_name: str) -> Optional[str]:
        """Build a stable prompt cache key for requests with shared prefixes."""
        if not self._is_official_openai_api():
            return None

        node_name = ""
        datasource = ""
        if getattr(self, "current_node", None):
            try:
                node_name = self.current_node.get_node_name()
            except Exception:
                node_name = getattr(self.current_node, "node_type", "") or ""
            agent_config = getattr(self.current_node, "agent_config", None)
            if agent_config:
                datasource = getattr(agent_config, "current_datasource", "") or ""

        raw_key = "|".join(
            [
                "openai-prompt-cache",
                self.model_name,
                agent_name,
                node_name,
                datasource,
            ]
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _setup_custom_json_encoder():
        """Setup custom JSON encoder for special types (AnyUrl, date, datetime).

        Note: For snowflake mcp server compatibility, can be removed after using native db tools.
        """

        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

    def _with_retry(
        self, operation_func, operation_name: str = "operation", max_retries: int = 3, base_delay: float = 1.0
    ):
        """
        Generic retry wrapper for synchronous operations.

        Args:
            operation_func: Function to execute (should raise API exceptions on failure)
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff

        Returns:
            Result from operation_func
        """
        for attempt in range(max_retries + 1):
            try:
                return operation_func()
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"API error in {operation_name} (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_code.code} - {error_code.desc}. Retrying in {delay:.1f}s... upstream={e!s}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Max retries reached or non-retryable error. Surface the
                    # upstream message verbatim — the wrapper code/desc is
                    # often too generic (300002 = "Invalid request format,
                    # content, or model response") and operators end up
                    # grepping litellm tracebacks to figure out what really
                    # broke. Keep ``str(e)`` on the same line so a single
                    # CloudWatch entry has everything.
                    logger.error(
                        f"API error in {operation_name} after {attempt + 1} attempts: "
                        f"{error_code.code} - {error_code.desc}. upstream={e!s}"
                    )
                    raise DatusException(error_code)
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {str(e)}")
                raise

    async def _with_retry_async(
        self, operation_func, operation_name: str = "operation", max_retries: int = 3, base_delay: float = 1.0
    ):
        """
        Generic retry wrapper for asynchronous operations.

        Args:
            operation_func: Async function to execute (should raise API exceptions on failure)
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff

        Returns:
            Result from operation_func
        """
        for attempt in range(max_retries + 1):
            try:
                return await operation_func()
            except ModelBehaviorError as e:
                # LLM hallucinated a non-existent tool or produced malformed output.
                # Retry since the LLM may behave correctly on the next attempt.
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Model behavior error in {operation_name} (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{str(e)}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Model behavior error in {operation_name} after {attempt + 1} attempts: {str(e)}")
                    raise
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_openai_compatible_error(e)

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"API error in {operation_name} (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{error_code.code} - {error_code.desc}. Retrying in {delay:.1f}s... upstream={e!s}"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Max retries reached or non-retryable error — same
                    # reasoning as the sync ``_with_retry`` path: append
                    # the upstream message so a single log line carries
                    # both the wrapper classification and the raw cause.
                    logger.error(
                        f"API error in {operation_name} after {attempt + 1} attempts: "
                        f"{error_code.code} - {error_code.desc}. upstream={e!s}"
                    )
                    raise DatusException(error_code)
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {str(e)}")
                raise

    def generate(self, prompt: Any, enable_thinking: bool | None = None, **kwargs) -> str:
        """
        Generate a response from the model with error handling and retry logic.

        Uses LiteLLM for unified provider support and consistent tracing.

        Args:
            prompt: The input prompt (string or list of messages)
            enable_thinking: Enable thinking mode for hybrid models (default: uses model_config)
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        # Fall back to model_config.enable_thinking if not explicitly provided
        if enable_thinking is None:
            enable_thinking = self.model_config.enable_thinking

        def _generate_operation():
            # Use LiteLLM model name for unified provider support
            params = {
                "model": self.litellm_adapter.litellm_model_name,
                "api_key": self.api_key,
            }

            # Add base_url if specified
            if self.base_url:
                params["api_base"] = self.base_url

            # Add custom headers for Coding Plan endpoints
            if self.default_headers:
                params["extra_headers"] = self.default_headers

            # Anthropic rejects requests carrying BOTH ``temperature`` and
            # ``top_p`` with HTTP 400 / ``invalid_request_error`` (the
            # claude-sonnet-4.x family enforces this strictly). The gate is
            # the **routed** provider (``litellm_adapter.provider``), not the
            # model class — operators may configure ``type=openai`` but a
            # Claude model name, in which case ``LiteLLMAdapter`` auto-routes
            # to Anthropic; the suppression must still apply or the request
            # 400s. ``temperature`` wins (mirrors the historical
            # ``ClaudeModel.generate`` contract); ``top_p`` is dropped
            # unconditionally for Claude regardless of kwargs / model_config
            # / non-reasoning default.
            routed_provider = getattr(self.litellm_adapter, "provider", "")
            suppress_top_p = routed_provider == LLMProvider.CLAUDE

            # Add temperature: priority is kwargs > model_config > default (0.7).
            # An explicit ``None`` in kwargs is the caller's "drop this param"
            # signal — kept for symmetry with the top_p path even though
            # Anthropic accepts ``temperature`` alone.
            if "temperature" in kwargs:
                if kwargs["temperature"] is not None:
                    params["temperature"] = kwargs["temperature"]
            elif self.model_config.temperature is not None:
                # Use temperature from model config (e.g., kimi-k2.5 requires temperature=1)
                params["temperature"] = self.model_config.temperature
            elif not hasattr(self, "_uses_completion_tokens_parameter") or not self._uses_completion_tokens_parameter():
                # Add default temperature only for non-reasoning models
                params["temperature"] = 0.7

            # Add top_p: same priority + same explicit-None contract as
            # temperature above, EXCEPT for Anthropic where we never send it.
            if suppress_top_p:
                pass
            elif "top_p" in kwargs:
                if kwargs["top_p"] is not None:
                    params["top_p"] = kwargs["top_p"]
            elif self.model_config.top_p is not None:
                # Use top_p from model config (e.g., kimi-k2.5 requires top_p=0.95)
                params["top_p"] = self.model_config.top_p
            elif not hasattr(self, "_uses_completion_tokens_parameter") or not self._uses_completion_tokens_parameter():
                # Add default top_p only for non-reasoning models
                params["top_p"] = 1.0

            # Resolve max_tokens / max_completion_tokens.
            # Priority: kwargs > model_specs.max_tokens. If neither is set, omit
            # the parameter so the provider's own default applies.
            explicit_max_tokens = kwargs.get("max_tokens")
            explicit_max_completion = kwargs.get("max_completion_tokens")
            if explicit_max_tokens is not None:
                params["max_tokens"] = explicit_max_tokens
            if explicit_max_completion is not None:
                params["max_completion_tokens"] = explicit_max_completion
            if explicit_max_tokens is None and explicit_max_completion is None:
                spec_max_tokens = self.max_tokens()
                if isinstance(spec_max_tokens, int) and spec_max_tokens > 0:
                    uses_completion = (
                        hasattr(self, "_uses_completion_tokens_parameter") and self._uses_completion_tokens_parameter()
                    )
                    if uses_completion:
                        params["max_completion_tokens"] = spec_max_tokens
                    else:
                        params["max_tokens"] = spec_max_tokens

            # Filter out handled parameters from remaining kwargs
            excluded_params = ["temperature", "top_p", "max_tokens", "max_completion_tokens"]
            params.update({k: v for k, v in kwargs.items() if k not in excluded_params})

            # Convert prompt to messages format
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            observability = get_observability_manager()
            span_attributes = {
                "datus.operation": "llm.generate",
                "gen_ai.request.model": self.model_name,
                "gen_ai.request.provider": str(routed_provider) if routed_provider else None,
                "gen_ai.system": str(routed_provider) if routed_provider else None,
                "datus.model.litellm_name": params["model"],
            }
            if observability.content_enabled("prompts"):
                span_attributes["datus.llm.prompt"] = to_str(observability.redact(messages))

            with observability.span("llm.generate", span_attributes) as span:
                # Use LiteLLM for unified provider support
                response = litellm.completion(messages=messages, **params)
                message = response.choices[0].message
                content = strip_litellm_placeholder(message.content)

                # Handle reasoning content for reasoning models (DeepSeek R1, OpenAI O-series)
                reasoning_content = None
                if enable_thinking:
                    if hasattr(message, "reasoning_content") and message.reasoning_content:
                        reasoning_content = message.reasoning_content
                        # If main content is empty but reasoning_content exists, use reasoning_content
                        if not content or content.strip() == "":
                            content = reasoning_content
                        logger.debug(f"Found reasoning_content: {reasoning_content[:100]}...")

                final_content = content or ""

                if hasattr(self, "_save_llm_trace"):
                    self._save_llm_trace(messages, final_content, reasoning_content)

                # Extract usage information for LangSmith tracking
                usage_info = {}
                if hasattr(response, "usage") and response.usage:
                    usage_info = {
                        "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "output_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }
                    logger.debug(f"Token usage: {usage_info}")

                finish_reason = response.choices[0].finish_reason if response.choices else None
                response_model = response.model if hasattr(response, "model") else self.model_name
                _set_observability_span_attribute(span, "gen_ai.response.finish_reason", finish_reason)
                _set_observability_span_attribute(span, "gen_ai.response.model", response_model)
                _set_observability_span_attribute(span, "gen_ai.usage.input_tokens", usage_info.get("input_tokens"))
                _set_observability_span_attribute(span, "gen_ai.usage.output_tokens", usage_info.get("output_tokens"))
                _set_observability_span_attribute(span, "gen_ai.usage.total_tokens", usage_info.get("total_tokens"))
                if observability.content_enabled("responses"):
                    _set_observability_span_attribute(
                        span, "datus.llm.response", to_str(observability.redact(final_content))
                    )
                if reasoning_content and observability.content_enabled("reasoning"):
                    _set_observability_span_attribute(
                        span, "datus.llm.reasoning", to_str(observability.redact(reasoning_content))
                    )

                # Return structured data for LangSmith to capture
                return {
                    "content": final_content or "",
                    "usage": usage_info,
                    "model": self.model_name,
                    "response_metadata": {
                        "finish_reason": finish_reason,
                        "model": response_model,
                    },
                }

        result = self._with_retry(_generate_operation, "text generation")

        # Return just the content for backward compatibility, but LangSmith will capture the full result
        if isinstance(result, dict):
            return result.get("content", "")
        return result

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a JSON response with error handling.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Parsed JSON dictionary
        """
        # Set JSON mode
        json_kwargs = kwargs.copy()
        json_kwargs["response_format"] = {"type": "json_object"}

        # Pass through enable_thinking if provided
        enable_thinking_param = json_kwargs.pop("enable_thinking", None)
        response_text = self.generate(prompt, enable_thinking=enable_thinking_param, **json_kwargs)

        try:
            parsed_json = json.loads(response_text)
            # For LangSmith tracing, we want to capture metadata but return the actual JSON for backward compatibility
            return parsed_json
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                    return parsed_json
                except json.JSONDecodeError:
                    pass

            return {"error": "Failed to parse JSON response", "raw_response": response_text}

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        strict_json_schema: bool = True,
        max_turns: int = 10,
        session: Optional[AdvancedSQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        **kwargs,
    ) -> Dict:
        """
        Generate response with unified tool support (replaces generate_with_mcp).

        Args:
            prompt: Input prompt
            mcp_servers: Optional MCP servers to use
            tools: Optional regular tools to use
            instruction: System instruction
            output_type: Expected output type (use Pydantic models for structured output)
            strict_json_schema: Enable strict JSON schema mode for structured output (default: True)
            max_turns: Maximum conversation turns
            session: Optional session for context
            action_history_manager: Action history manager for tracking
            **kwargs: Additional parameters

        Returns:
            Dict with content and sql_contexts
        """
        # Use the internal method that returns a Dict
        result = await self._generate_with_tools_internal(
            prompt,
            mcp_servers,
            tools,
            instruction,
            output_type,
            strict_json_schema,
            max_turns,
            session,
            hooks,
            **kwargs,
        )

        # Enhance result with tracing metadata
        enhanced_result = {
            **result,
            "model": self.model_name,
            "max_turns": max_turns,
            "tool_count": len(tools) if tools else 0,
            "mcp_server_count": len(mcp_servers) if mcp_servers else 0,
            "instruction_length": len(instruction),
            "prompt_length": len(prompt),
        }

        return enhanced_result

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        tools: Optional[List[Any]] = None,
        instruction: str = "",
        output_type: type = str,
        strict_json_schema: bool = True,
        max_turns: int = 10,
        session: Optional[AdvancedSQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        interrupt_controller=None,
        pending_input_queue=None,
        interaction_broker=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Generate response with streaming and tool support (replaces generate_with_mcp_stream).

        Args:
            prompt: Input prompt
            mcp_servers: Optional MCP servers
            tools: Optional regular tools
            instruction: System instruction
            output_type: Expected output type (use Pydantic models for structured output)
            strict_json_schema: Enable strict JSON schema mode for structured output (default: True)
            max_turns: Maximum turns
            session: Optional session
            action_history_manager: Action history manager
            **kwargs: Additional parameters

        Yields:
            ActionHistory objects for streaming updates
        """
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        async for action in self._generate_with_tools_stream_internal(
            prompt,
            mcp_servers,
            tools,
            instruction,
            output_type,
            strict_json_schema,
            max_turns,
            session,
            action_history_manager,
            hooks,
            interrupt_controller=interrupt_controller,
            pending_input_queue=pending_input_queue,
            interaction_broker=interaction_broker,
            **kwargs,
        ):
            yield action

    def _build_run_config(
        self,
        pending_input_queue=None,
        session: Optional[AdvancedSQLiteSession] = None,
        interrupt_controller=None,
        interaction_broker=None,
        agent_name: Optional[str] = None,
    ) -> Optional[RunConfig]:
        """Build a :class:`RunConfig` carrying our mid-run input filter.

        Returns ``None`` when neither tracing context nor ``pending_input_queue``
        is wired in — the SDK then runs with its default config and behavior is
        unchanged.

        The filter is invoked by the SDK before every LLM turn within a
        ``Runner.run_streamed`` invocation (``agents/run.py`` ~1483). It
        drains the queue (every queued item, FIFO) and appends each as a
        structured Responses-API user message, also persisting them onto
        the session so future runs see them.

        When ``interaction_broker`` is provided, each drained item is also
        emitted as a USER ``ActionHistory`` via
        :meth:`InteractionBroker.emit_user_insert`. That makes the
        injection visible to the TUI's streaming display and to the API
        SSE consumer at the exact moment it's being flushed to the model.

        All exceptions in the filter body are swallowed: the SDK re-raises
        uncaught filter errors and aborts the entire run, which we must
        never let a queue or sqlite hiccup do.
        """
        trace_kwargs = build_agents_run_config_kwargs(agent_name=agent_name)
        if pending_input_queue is None and not trace_kwargs:
            return None

        async def _filter(data: CallModelData) -> ModelInputData:
            try:
                if interrupt_controller is not None and getattr(interrupt_controller, "is_interrupted", False):
                    return data.model_data
                pending = pending_input_queue.drain()
                if not pending:
                    return data.model_data
                new_items = list(data.model_data.input)
                for text in pending:
                    user_item = {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                    new_items.append(user_item)
                    if session is not None:
                        try:
                            await session.add_items([user_item])
                        except Exception:  # noqa: BLE001 — never abort the run on a session write
                            logger.exception("Failed to persist injected user item; in-memory only.")
                    if interaction_broker is not None:
                        try:
                            interaction_broker.emit_user_insert(text)
                        except Exception:  # noqa: BLE001 — broker emission is best-effort
                            logger.exception("Failed to emit user_insert ActionHistory; model still sees the text.")
                return ModelInputData(input=new_items, instructions=data.model_data.instructions)
            except Exception:  # noqa: BLE001 — filter exceptions abort the SDK run
                logger.exception("call_model_input_filter raised; returning original input unchanged.")
                return data.model_data

        if pending_input_queue is None:
            return RunConfig(**trace_kwargs)

        return RunConfig(call_model_input_filter=_filter, **trace_kwargs)

    def _build_agent(
        self,
        instruction: str,
        output_type: type,
        strict_json_schema: bool,
        connected_servers: dict,
        tools: Optional[List[Tool]],
        hooks=None,
        agent_name: str = "default_agent",
    ) -> Agent:
        """Build Agent with consistent configuration for both streaming and non-streaming paths."""
        actual_output_type = output_type
        enable_structured_output = False
        if output_type is not str:
            from agents import AgentOutputSchema

            actual_output_type = AgentOutputSchema(output_type, strict_json_schema=strict_json_schema)
            enable_structured_output = True
            logger.debug(
                f"Wrapped output_type with AgentOutputSchema: type={output_type.__name__}, strict={strict_json_schema}"
            )

        litellm_model = self.litellm_adapter.get_agents_sdk_model()

        # DeepSeek requires "json" keyword in prompt for JSON mode
        final_instruction = instruction
        if enable_structured_output and self.litellm_adapter.provider == "deepseek":
            if "json" not in instruction.lower():
                final_instruction = f"{instruction}\n\nIMPORTANT: Return output in valid JSON format."
                logger.debug("Added JSON keyword to instructions for DeepSeek")

        agent_kwargs = {
            "name": agent_name,
            "instructions": final_instruction,
            "output_type": actual_output_type,
            "model": litellm_model,
        }

        # Build ModelSettings with provider-specific configurations
        model_settings_kwargs = {"include_usage": True}

        if self.model_config.temperature is not None:
            model_settings_kwargs["temperature"] = self.model_config.temperature

        if self.model_config.top_p is not None:
            model_settings_kwargs["top_p"] = self.model_config.top_p

        # Apply model_specs.max_tokens as a fallback so streaming/tool calls
        # don't silently rely on each provider's server-side default. If the
        # spec is unknown, omit the field and let the provider decide.
        spec_max_tokens = self.max_tokens()
        if isinstance(spec_max_tokens, int) and spec_max_tokens > 0:
            model_settings_kwargs["max_tokens"] = spec_max_tokens

        if self.default_headers:
            model_settings_kwargs["extra_headers"] = self.default_headers

        effort = self.litellm_adapter.reasoning_effort_level
        if effort:
            if self._model_supports_reasoning():
                model_settings_kwargs["reasoning"] = Reasoning(effort=effort)
                logger.debug(f"Enabled reasoning (effort={effort}) for model: {self.model_name}")
                native_thinking = self._native_thinking_extra_body()
                if native_thinking:
                    # Must go through ``extra_args["extra_body"]`` (not
                    # ``ModelSettings.extra_body``). The agents SDK *flattens*
                    # ``extra_body`` into the acompletion kwargs, which means
                    # Moonshot's transformation — whose ``get_supported_openai_params``
                    # list does not include ``thinking`` — would silently drop the
                    # flag under ``drop_params=True``. Nesting it one level deeper
                    # keeps ``extra_body={...}`` as a reserved acompletion kwarg,
                    # which the OpenAI client then merges verbatim into the HTTP
                    # POST body, bypassing the per-provider supported-params filter.
                    existing_extra_args = dict(model_settings_kwargs.get("extra_args") or {})
                    nested_extra_body = dict(existing_extra_args.get("extra_body") or {})
                    nested_extra_body.update(native_thinking)
                    existing_extra_args["extra_body"] = nested_extra_body
                    model_settings_kwargs["extra_args"] = existing_extra_args
                    logger.debug(f"Added native thinking payload for {self.model_name}: {native_thinking}")
            else:
                logger.warning(
                    "Skipping reasoning (effort=%s) for %s: Datus recognises this model "
                    "as not supporting thinking. Use `/effort off` or switch to a "
                    "thinking-capable model (gpt-5*, o-series, claude-4*, gemini-2.5*+, "
                    "deepseek-v4*, deepseek-reasoner, kimi-k2.5+) to silence this warning.",
                    effort,
                    self.model_name,
                )

        prompt_cache_retention = self._default_prompt_cache_retention()
        if prompt_cache_retention:
            model_settings_kwargs["prompt_cache_retention"] = prompt_cache_retention
            prompt_cache_key = self._default_prompt_cache_key(agent_name)
            if prompt_cache_key:
                existing_extra_args = model_settings_kwargs.get("extra_args", {})
                existing_extra_args["prompt_cache_key"] = prompt_cache_key
                model_settings_kwargs["extra_args"] = existing_extra_args

        agent_kwargs["model_settings"] = ModelSettings(**model_settings_kwargs)

        if connected_servers:
            agent_kwargs["mcp_servers"] = list(connected_servers.values())

        if tools:
            agent_kwargs["tools"] = tools

        if hooks:
            agent_kwargs["hooks"] = hooks

        return Agent(**agent_kwargs)

    async def _generate_with_tools_internal(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]],
        tools: Optional[List[Tool]],
        instruction: str,
        output_type: type,
        strict_json_schema: bool,
        max_turns: int,
        session: Optional[AdvancedSQLiteSession] = None,
        hooks=None,
        **kwargs,
    ) -> Dict:
        """Internal method for tool execution with error handling."""

        # Custom JSON encoder for special types
        # (for snowflake mcp server, we can remove it after using native db tools)
        self._setup_custom_json_encoder()

        async def _tools_operation():
            # Use multiple_mcp_servers context manager with empty dict if no MCP servers
            async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
                agent_name = kwargs.get("agent_name", "default_agent")
                agent = self._build_agent(
                    instruction=instruction,
                    output_type=output_type,
                    strict_json_schema=strict_json_schema,
                    connected_servers=connected_servers,
                    tools=tools,
                    hooks=hooks,
                    agent_name=agent_name,
                )
                run_config = self._build_run_config(agent_name=agent_name)

                # OpenAI Agents SDK tracing is configured by observability adapters.
                compact_callback = kwargs.get("compact_callback")
                overflow_retried = False
                try:
                    with _agents_trace_baggage(agent_name):
                        result = await Runner.run(
                            agent,
                            input=prompt,
                            max_turns=max_turns,
                            session=session,
                            run_config=run_config,
                        )
                except MaxTurnsExceeded as e:
                    # Overflow fallback: when a compact_callback is wired in,
                    # ask the caller to run a major compact (so the session
                    # is short enough to retry) and try the run exactly once
                    # more. The single-retry guard prevents a degenerate loop
                    # if the compact itself fails to shrink the session
                    # below max_turns.
                    if compact_callback is not None and not overflow_retried:
                        overflow_retried = True
                        logger.warning("MaxTurnsExceeded — invoking compact_callback and retrying once.")
                        try:
                            await compact_callback(mode="major", reason="overflow")
                        except Exception as compact_exc:  # noqa: BLE001
                            logger.error("Overflow compact_callback failed: %s", compact_exc)
                            raise DatusException(
                                ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns}
                            ) from compact_exc
                        try:
                            result = await Runner.run(
                                agent,
                                input=prompt,
                                max_turns=max_turns,
                                session=session,
                                run_config=run_config,
                            )
                        except MaxTurnsExceeded as retry_e:
                            logger.error(f"Max turns exceeded after overflow compact: {retry_e}")
                            raise DatusException(
                                ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns}
                            ) from retry_e
                    else:
                        logger.error(f"Max turns exceeded: {str(e)}")
                        raise DatusException(
                            ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns}
                        ) from e

                # Save LLM trace if method exists (for models that support it like DeepSeekModel)
                if hasattr(self, "_save_llm_trace"):
                    # For tools calls, we need to extract messages from the result
                    messages = [{"role": "user", "content": prompt}]
                    if instruction:
                        messages.insert(0, {"role": "system", "content": instruction})

                    # Get complete conversation history including tool calls
                    conversation_history = None
                    if hasattr(result, "to_input_list"):
                        try:
                            conversation_history = result.to_input_list()
                        except Exception as e:
                            logger.debug(f"Failed to get conversation history: {e}")

                    self._save_llm_trace(messages, result.final_output, conversation_history)

                # Store per-turn usage in turn_usage table
                if session and hasattr(session, "store_run_usage"):
                    try:
                        await session.store_run_usage(result)
                    except Exception as e:
                        logger.warning(f"Failed to store run usage: {e}")

                # Extract usage information from the correct location: result.context_wrapper.usage
                usage_info = {}
                if hasattr(result, "context_wrapper") and hasattr(result.context_wrapper, "usage"):
                    usage_info = self._extract_usage_info(result.context_wrapper.usage)
                    logger.debug(f"Agent execution usage: {usage_info}")
                else:
                    logger.warning("No usage information found in result.context_wrapper")

                return {
                    "content": result.final_output,
                    "sql_contexts": extract_sql_contexts(result),
                    "usage": usage_info,
                    "model": self.model_name,
                    "turns_used": getattr(result, "turn_count", 0),
                    "final_output_length": len(result.final_output) if result.final_output else 0,
                }

        return await self._with_retry_async(_tools_operation, "tool execution")

    async def _generate_with_tools_stream_internal(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]],
        tools: Optional[List[Tool]],
        instruction: str,
        output_type: type,
        strict_json_schema: bool,
        max_turns: int,
        session: Optional[AdvancedSQLiteSession],
        action_history_manager: ActionHistoryManager,
        hooks=None,
        interrupt_controller=None,
        pending_input_queue=None,
        interaction_broker=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Internal method for tool streaming execution with error handling.

        Strategy: Use streaming events only for progress display, then rebuild
        the complete action history from result.to_input_list() after streaming completes.
        This avoids issues with duplicate call_ids and out-of-order events.
        """

        # Custom JSON encoder
        self._setup_custom_json_encoder()

        async def _stream_operation():
            # Use multiple_mcp_servers context manager with empty dict if no MCP servers
            async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
                agent_name = kwargs.get("agent_name", "Tools_Agent")
                agent = self._build_agent(
                    instruction=instruction,
                    output_type=output_type,
                    strict_json_schema=strict_json_schema,
                    connected_servers=connected_servers,
                    tools=tools,
                    hooks=hooks,
                    agent_name=agent_name,
                )

                run_config = self._build_run_config(
                    pending_input_queue=pending_input_queue,
                    session=session,
                    interrupt_controller=interrupt_controller,
                    interaction_broker=interaction_broker,
                    agent_name=agent_name,
                )
                try:
                    with _agents_trace_baggage(agent_name):
                        result = Runner.run_streamed(
                            agent,
                            input=prompt,
                            max_turns=max_turns,
                            session=session,
                            run_config=run_config,
                        )
                except MaxTurnsExceeded as e:
                    logger.error(f"Max turns exceeded in streaming: {str(e)}")
                    raise DatusException(
                        ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns}
                    ) from e

                # Streaming phase: yield progress actions in real-time
                # After streaming completes, generate final summary report
                import uuid

                from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

                # Phase 1: Stream events with detailed progress
                # Track tool calls and results for immediate feedback
                temp_tool_calls = {}  # {call_id: ActionHistory}
                early_assistant_yielded = False  # Flag to skip duplicate message_output_item
                tool_call_seen = False
                tool_output_seen = False
                final_assistant_yielded = False
                last_assistant_text = ""

                # Streaming thinking state: accumulate text deltas for real-time output
                thinking_stream_id: Optional[str] = None
                thinking_accumulated = ""
                placeholder_filter = LitellmPlaceholderStreamFilter()

                def mark_assistant_response(text: str, is_thinking: bool) -> None:
                    nonlocal final_assistant_yielded, last_assistant_text
                    last_assistant_text = text
                    if not is_thinking and (not tool_call_seen or tool_output_seen):
                        final_assistant_yielded = True

                while not result.is_complete:
                    if interrupt_controller and interrupt_controller.is_interrupted:
                        from datus.cli.execution_state import ExecutionInterrupted

                        raise ExecutionInterrupted("Interrupted by user")
                    async for event in _stream_events_with_trace_baggage(result, agent_name):
                        if interrupt_controller and interrupt_controller.is_interrupted:
                            from datus.cli.execution_state import ExecutionInterrupted

                            raise ExecutionInterrupted("Interrupted by user")

                        # Capture assistant text from raw response events for streaming output.
                        # We process three raw event types:
                        # 1. response.output_text.delta - stream text chunks as thinking_delta
                        # 2. response.content_part.done - emit final thinking action
                        # 3. response.output_item.done type="message" - fallback (skipped if early captured)
                        if hasattr(event, "type") and event.type == "raw_response_event":
                            raw_data = getattr(event, "data", None)
                            raw_type = getattr(raw_data, "type", None) if raw_data else None

                            # Stream text delta: yield thinking_delta for real-time display
                            if raw_type == "response.output_text.delta":
                                raw_delta = getattr(raw_data, "delta", None) or ""
                                delta_text = placeholder_filter.feed(raw_delta)
                                if delta_text:
                                    if thinking_stream_id is None:
                                        thinking_stream_id = f"thinking_stream_{uuid.uuid4().hex[:8]}"
                                    thinking_accumulated += delta_text
                                    delta_action = ActionHistory(
                                        action_id=thinking_stream_id,
                                        role=ActionRole.ASSISTANT,
                                        messages="",
                                        action_type="thinking_delta",
                                        input={},
                                        output={"delta": delta_text, "accumulated": thinking_accumulated},
                                        status=ActionStatus.PROCESSING,
                                    )
                                    # Do NOT add to action_history_manager (transient, prevents dedup)
                                    yield delta_action
                                continue

                            # Content part done: emit completed thinking action
                            if raw_type == "response.content_part.done":
                                tail = placeholder_filter.finalize()
                                if tail:
                                    thinking_accumulated += tail
                                full_text = strip_litellm_placeholder(thinking_accumulated.strip())
                                if full_text:
                                    text_content_split = full_text if len(full_text) <= 200 else f"{full_text[:200]}..."
                                    is_thinking = len(temp_tool_calls) > 0
                                    thinking_action = ActionHistory(
                                        action_id=thinking_stream_id or f"assistant_{uuid.uuid4().hex[:8]}",
                                        role=ActionRole.ASSISTANT,
                                        messages=f"Thinking: {text_content_split}",
                                        action_type="response",
                                        input={},
                                        output={"raw_output": full_text, "is_thinking": is_thinking},
                                        status=ActionStatus.SUCCESS,
                                    )
                                    action_history_manager.add_action(thinking_action)
                                    yield thinking_action
                                    mark_assistant_response(full_text, is_thinking)
                                    early_assistant_yielded = True
                                # Reset stream state for next content part
                                thinking_stream_id = None
                                thinking_accumulated = ""
                                continue

                            # Fallback: response.output_item.done type="message"
                            if raw_type == "response.output_item.done":
                                raw_item = getattr(raw_data, "item", None)
                                if raw_item and getattr(raw_item, "type", None) == "message":
                                    if not early_assistant_yielded:
                                        content_list = getattr(raw_item, "content", [])
                                        text_parts = []
                                        for content_part in content_list or []:
                                            part_text = getattr(content_part, "text", None)
                                            if part_text:
                                                text_parts.append(part_text)
                                        full_text = strip_litellm_placeholder("\n".join(text_parts).strip())
                                        if full_text:
                                            text_content_split = (
                                                full_text if len(full_text) <= 200 else f"{full_text[:200]}..."
                                            )
                                            is_thinking = len(temp_tool_calls) > 0
                                            thinking_action = ActionHistory(
                                                action_id=f"assistant_{uuid.uuid4().hex[:8]}",
                                                role=ActionRole.ASSISTANT,
                                                messages=f"Thinking: {text_content_split}",
                                                action_type="response",
                                                input={},
                                                output={"raw_output": full_text, "is_thinking": is_thinking},
                                                status=ActionStatus.SUCCESS,
                                            )
                                            action_history_manager.add_action(thinking_action)
                                            yield thinking_action
                                            mark_assistant_response(full_text, is_thinking)
                                            early_assistant_yielded = True
                            continue

                        if not hasattr(event, "type") or event.type != "run_item_stream_event":
                            continue

                        if not (hasattr(event, "item") and hasattr(event.item, "type")):
                            continue

                        item_type = event.item.type

                        # Handle tool call start
                        if item_type == "tool_call_item":
                            tool_call_seen = True
                            final_assistant_yielded = False
                            raw_item = getattr(event.item, "raw_item", None)
                            if raw_item:
                                tool_name = getattr(raw_item, "name", None)
                                if not tool_name:
                                    logger.warning(f"Tool call has no name field: {type(raw_item)}, {dir(raw_item)}")
                                    tool_name = "unknown"

                                arguments = getattr(raw_item, "arguments", "{}")
                                call_id = getattr(raw_item, "call_id", None)

                                # Generate call_id if missing
                                if not call_id:
                                    call_id = f"tool_{uuid.uuid4().hex[:8]}"
                                    logger.warning(f"Tool call missing call_id, generated: {call_id}")

                                # Try to format arguments
                                try:
                                    args_dict = json.loads(arguments) if arguments else {}
                                    args_str = to_str(args_dict)[:80]
                                except Exception:
                                    args_str = str(arguments)[:80]

                                # Store tool call info for matching with result
                                temp_tool_calls[call_id] = {
                                    "tool_name": tool_name,
                                    "arguments": arguments,
                                    "args_display": args_str,
                                    "start_time": datetime.now(),
                                }
                                start_action = ActionHistory(
                                    action_id=call_id,
                                    role=ActionRole.TOOL,
                                    messages=f"Tool call: {tool_name}('{args_str}...')",
                                    action_type=tool_name,
                                    input={"function_name": tool_name, "arguments": arguments},
                                    output={},
                                    status=ActionStatus.PROCESSING,
                                )
                                logger.debug(
                                    f"Stored tool call: {tool_name} (call_id={call_id[:20] if call_id else 'None'}...)"
                                )
                                action_history_manager.add_action(start_action)
                                yield start_action

                        # Handle tool call completion
                        elif item_type == "tool_call_output_item":
                            tool_output_seen = True
                            raw_item = getattr(event.item, "raw_item", None)
                            output_content = getattr(event.item, "output", "")

                            # Extract call_id from raw_item
                            # raw_item can be either a dict or an object
                            call_id = None
                            if raw_item:
                                if isinstance(raw_item, dict):
                                    call_id = raw_item.get("call_id")
                                else:
                                    call_id = getattr(raw_item, "call_id", None)

                            logger.debug(
                                f"🔍 Tool output call_id={call_id}, type={type(output_content)}, "
                                f"stored={list(temp_tool_calls.keys())}"
                            )

                            # Try to match with stored tool call
                            if call_id and call_id in temp_tool_calls:
                                # Found matching tool call
                                tool_info = temp_tool_calls[call_id]
                                tool_name = tool_info["tool_name"]
                                args_display = tool_info["args_display"]

                                # Format result summary (only count info)
                                # output_content might already be a dict or string
                                if isinstance(output_content, dict):
                                    result_summary = self._format_tool_result_from_dict(output_content, tool_name)
                                elif isinstance(output_content, str):
                                    result_summary = self._format_tool_result(output_content, tool_name)
                                else:
                                    # Log unexpected type and try to convert
                                    logger.warning(f"Unexpected output_content type: {type(output_content)}")
                                    result_summary = self._format_tool_result(str(output_content), tool_name)

                                # Detect failure from FuncToolResult-shaped payload so the
                                # CLI renders ✗ (and SSE reports FAILED) when the tool
                                # returned success=0 / non-empty error, even though the
                                # Agents SDK call itself did not raise.
                                tool_failed = _detect_tool_failure(output_content)

                                # Create complete action with both input and output
                                # Put result_summary as the status message to replace default "Success"
                                complete_action = ActionHistory(
                                    action_id="complete_" + call_id,
                                    role=ActionRole.TOOL,
                                    messages=f"Tool call: {tool_name}('{args_display}...')",
                                    action_type=tool_name,
                                    input={"function_name": tool_name, "arguments": tool_info["arguments"]},
                                    output={
                                        "success": not tool_failed,
                                        "raw_output": output_content,
                                        "summary": result_summary,
                                        "status_message": result_summary,
                                    },
                                    status=ActionStatus.FAILED if tool_failed else ActionStatus.SUCCESS,
                                    start_time=tool_info["start_time"],
                                )
                                complete_action.end_time = datetime.now()

                                logger.debug(f"Matched tool: {tool_name}({args_display[:30]}...) -> {result_summary}")

                                # Add to action_history_manager before yielding (consistent with thinking messages)
                                action_history_manager.add_action(complete_action)
                                yield complete_action

                                # Remove from temp storage to avoid duplicates
                                del temp_tool_calls[call_id]

                            else:
                                # No matching tool call found
                                logger.warning(
                                    f"Orphan tool result: call_id={call_id}, stored={list(temp_tool_calls.keys())[:3]}"
                                )

                                # Yield result anyway
                                orphan_action = ActionHistory(
                                    action_id=call_id or f"orphan_{uuid.uuid4().hex[:8]}",
                                    role=ActionRole.TOOL,
                                    messages="Tool call (orphan)",
                                    action_type="tool_result",
                                    input={"function_name": "unknown"},
                                    output={"success": True, "raw_output": output_content},
                                    status=ActionStatus.SUCCESS,
                                )
                                orphan_action.end_time = datetime.now()

                                # Add to action_history_manager before yielding (consistent with other actions)
                                action_history_manager.add_action(orphan_action)
                                yield orphan_action

                        # Handle thinking messages
                        elif item_type == "message_output_item":
                            # Skip if already captured from raw response event
                            if early_assistant_yielded:
                                early_assistant_yielded = False
                                continue
                            raw_item = getattr(event.item, "raw_item", None)
                            if raw_item and hasattr(raw_item, "content"):
                                content = raw_item.content
                                if isinstance(content, list) and content:
                                    text_content = content[0].text if hasattr(content[0], "text") else str(content[0])
                                else:
                                    text_content = str(content)

                                text_content = strip_litellm_placeholder(text_content)
                                if text_content and len(text_content.strip()) > 0:
                                    text_content = text_content.strip()
                                    text_content_split = (
                                        text_content if len(text_content) <= 200 else f"{text_content[:200]}..."
                                    )
                                    is_thinking = len(temp_tool_calls) > 0
                                    thinking_action = ActionHistory(
                                        action_id=f"assistant_{uuid.uuid4().hex[:8]}",
                                        role=ActionRole.ASSISTANT,
                                        messages=f"Thinking: {text_content_split}",
                                        action_type="response",
                                        input={},
                                        output={"raw_output": text_content, "is_thinking": is_thinking},
                                        status=ActionStatus.SUCCESS,
                                    )
                                    action_history_manager.add_action(thinking_action)
                                    yield thinking_action
                                    mark_assistant_response(text_content, is_thinking)

                final_output = getattr(result, "final_output", "") or ""
                if not isinstance(final_output, str):
                    final_output = str(final_output)
                final_output = strip_litellm_placeholder(final_output.strip())
                if final_output and not final_assistant_yielded and final_output != last_assistant_text:
                    final_output_preview = final_output if len(final_output) <= 200 else f"{final_output[:200]}..."
                    final_action = ActionHistory(
                        action_id=f"assistant_{uuid.uuid4().hex[:8]}",
                        role=ActionRole.ASSISTANT,
                        messages=f"Thinking: {final_output_preview}",
                        action_type="response",
                        input={},
                        output={"raw_output": final_output, "is_thinking": False},
                        status=ActionStatus.SUCCESS,
                    )
                    action_history_manager.add_action(final_action)
                    yield final_action
                    mark_assistant_response(final_output, False)

                # Save LLM trace if method exists
                if hasattr(self, "_save_llm_trace"):
                    # For tools calls, we need to extract messages from the result
                    messages = [{"role": "user", "content": prompt}]
                    if instruction:
                        messages.insert(0, {"role": "system", "content": instruction})

                    # Get complete conversation history including tool calls
                    conversation_history = None
                    if hasattr(result, "to_input_list"):
                        try:
                            conversation_history = result.to_input_list()
                        except Exception as e:
                            logger.debug(f"Failed to get conversation history: {e}")

                    final_output = result.final_output if hasattr(result, "final_output") else ""
                    self._save_llm_trace(messages, final_output, conversation_history)

                # Store per-turn usage in turn_usage table
                if session and hasattr(session, "store_run_usage"):
                    try:
                        await session.store_run_usage(result)
                    except Exception as e:
                        logger.warning(f"Failed to store run usage: {e}")

                # After streaming completes, extract usage information from the final result
                # and add it to the final assistant action
                await self._extract_and_distribute_token_usage(result, action_history_manager)

        # Execute the streaming operation with retry logic for connection errors
        max_retries = getattr(self.model_config, "max_retry", 3)
        retry_delay = getattr(self.model_config, "retry_interval", 2.0)

        # Track already processed action IDs to prevent duplicates across retries
        processed_action_ids = set()

        from datus.cli.execution_state import ExecutionInterrupted

        for attempt in range(max_retries):
            try:
                async for action in _stream_operation():
                    # Skip actions that were already yielded in previous retry attempts
                    # (thinking_delta actions share a stream ID and must not be deduped)
                    if action.action_type != "thinking_delta" and action.action_id in processed_action_ids:
                        logger.debug(f"Skipping duplicate action: {action.action_id}")
                        continue

                    # Mark this action as processed (skip transient deltas)
                    if action.action_type != "thinking_delta":
                        processed_action_ids.add(action.action_id)
                    yield action
                # If we successfully complete, break out of retry loop
                break

            except ExecutionInterrupted:
                # User-initiated interrupt: propagate immediately, do not retry
                raise

            except ModelBehaviorError as e:
                # LLM hallucinated a non-existent tool or produced malformed output.
                # Retry since the LLM may behave correctly on the next attempt.
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Model behavior error in stream (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )

                    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

                    retry_action = ActionHistory.create_action(
                        role=ActionRole.ASSISTANT,
                        action_type="retry_notification",
                        messages=f"Model behavior error, retrying ({attempt + 2}/{max_retries})...",
                        input_data={"error": str(e), "attempt": attempt + 1},
                        status=ActionStatus.PROCESSING,
                    )
                    action_history_manager.add_action(retry_action)
                    processed_action_ids.add(retry_action.action_id)
                    yield retry_action

                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.exception(f"Model behavior error after {max_retries} attempts: {type(e).__name__}: {e}")
                    raise

            except (httpx.RemoteProtocolError, APIConnectionError, APITimeoutError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Stream connection error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )

                    # Yield a retry notification action to inform user
                    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

                    retry_action = ActionHistory.create_action(
                        role=ActionRole.ASSISTANT,
                        action_type="retry_notification",
                        messages=f"Connection interrupted, retrying ({attempt + 2}/{max_retries})...",
                        input_data={"error": str(e), "attempt": attempt + 1},
                        status=ActionStatus.PROCESSING,
                    )
                    action_history_manager.add_action(retry_action)
                    processed_action_ids.add(retry_action.action_id)
                    yield retry_action

                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # All retries exhausted
                    logger.exception(f"Stream failed after {max_retries} attempts: {type(e).__name__}: {e}")
                    raise

    def _extract_usage_info(self, usage) -> dict:
        """Extract standardized usage info from SDK usage object.

        Used by both streaming and non-streaming code paths to avoid duplication.
        """
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        cached_tokens = 0
        if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
            cached_tokens = getattr(usage.input_tokens_details, "cached_tokens", 0)

        reasoning_tokens = 0
        if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0)

        cache_hit_rate = round(cached_tokens / input_tokens, 3) if input_tokens > 0 else 0

        context_usage_ratio = 0
        max_context = self.context_length()
        if max_context and total_tokens > 0:
            context_usage_ratio = round(total_tokens / max_context, 3)

        # Last model call's input_tokens = real context window usage
        last_call_input_tokens = 0
        if hasattr(usage, "request_usage_entries") and usage.request_usage_entries:
            last_call_input_tokens = getattr(usage.request_usage_entries[-1], "input_tokens", 0)

        return {
            "requests": getattr(usage, "requests", 0),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_hit_rate": cache_hit_rate,
            "context_usage_ratio": context_usage_ratio,
            "last_call_input_tokens": last_call_input_tokens,
        }

    async def _extract_and_distribute_token_usage(self, result, action_history_manager: ActionHistoryManager) -> None:
        """Extract token usage from completed streaming result and distribute to ActionHistory objects."""
        try:
            if not (hasattr(result, "context_wrapper") and hasattr(result.context_wrapper, "usage")):
                logger.warning("No usage information found in streaming result")
                return

            usage_info = self._extract_usage_info(result.context_wrapper.usage)
            logger.debug(f"Extracted streaming token usage: {usage_info}")

            self._distribute_token_usage_to_actions(action_history_manager, usage_info)

        except Exception as e:
            logger.error(f"Error extracting and distributing token usage: {e}")

    def _distribute_token_usage_to_actions(
        self, action_history_manager: ActionHistoryManager, usage_info: dict
    ) -> None:
        """
        Distribute token usage information to ActionHistory objects.
        Only adds full usage to final assistant action to avoid double-counting.

        Args:
            action_history_manager: ActionHistoryManager containing actions
            usage_info: Usage information dictionary with token counts
        """
        try:
            actions = action_history_manager.get_actions()
            if not actions:
                return

            total_tokens = usage_info.get("total_tokens", 0)
            assistant_actions = [a for a in actions if a.role == "assistant"]

            # Add full usage to the final assistant action (represents the complete conversation cost)
            if assistant_actions:
                final_assistant = assistant_actions[-1]
                self._add_usage_to_action(final_assistant, usage_info)
                logger.debug(f"Distributed {total_tokens} tokens to final assistant action")

            # Note: Tool actions don't get token counts to avoid double-counting

        except Exception as e:
            logger.error(f"Error distributing token usage: {e}")

    def _add_usage_to_action(self, action: ActionHistory, usage_info: dict) -> None:
        """Add usage information to an action's output."""
        if action.output is None:
            action.output = {}
        elif not isinstance(action.output, dict):
            action.output = {"raw_output": action.output}

        action.output["usage"] = usage_info

    def _format_tool_result_from_dict(self, data: dict, tool_name: str = "") -> str:
        """Delegate to the shared :class:`ToolSummaryRegistry`."""
        return TOOL_SUMMARY_REGISTRY.summarize_dict(data, tool_name)

    def _format_tool_result(self, content: str, tool_name: str = "") -> str:
        """Delegate to the shared :class:`ToolSummaryRegistry`."""
        return TOOL_SUMMARY_REGISTRY.summarize_content(content, tool_name)

    @property
    def model_specs(self) -> Dict[str, Dict[str, int]]:
        """Model specifications loaded from conf/providers.yml (cached)."""
        return _load_model_specs()

    def _lookup_spec(self, field: str) -> Optional[int]:
        """Look up ``field`` in ``model_specs`` with exact-then-longest-prefix matching.

        Longest-prefix is important when specs declare both a generic family key
        (e.g. ``gpt-5``) and a more specific variant (e.g. ``gpt-5.3-codex``): the
        generic key would otherwise bleed into unrelated slugs that happen to
        share a short prefix. Returns None when the field is missing —
        OpenRouter-cache-only entries do not carry ``max_tokens``, so callers
        must tolerate absence.
        """
        specs = self.model_specs
        exact = specs.get(self.model_name)
        if isinstance(exact, dict) and isinstance(exact.get(field), int):
            return exact[field]
        best_key: Optional[str] = None
        for spec_model, spec in specs.items():
            if not isinstance(spec, dict) or not isinstance(spec.get(field), int):
                continue
            if self.model_name.startswith(spec_model) and (best_key is None or len(spec_model) > len(best_key)):
                best_key = spec_model
        if best_key is None:
            return None
        return specs[best_key][field]

    def max_tokens(self) -> Optional[int]:
        """Max tokens from model specs with prefix matching, or None if unavailable."""
        return self._lookup_spec("max_tokens")

    def context_length(self) -> Optional[int]:
        """Context length from model specs with prefix matching, or None if unavailable."""
        return self._lookup_spec("context_length")

    def token_count(self, prompt: str) -> int:
        """
        Count tokens in prompt using LiteLLM's token counter.
        Supports automatic tokenizer selection for different model providers.
        """
        try:
            # Use LiteLLM's unified token counter
            return litellm.token_counter(model=self.litellm_adapter.litellm_model_name, text=str(prompt))
        except Exception as e:
            # Fallback to character approximation if token counting fails
            logger.debug(f"Token counting failed for {self.model_name}, using approximation: {e}")
            return len(str(prompt)) // 4

    def _save_llm_trace(self, prompt: Any, response_content: str, reasoning_content: Any = None):
        """Save LLM input/output trace to YAML file if tracing is enabled.

        Args:
            prompt: The input prompt (str or list of messages)
            response_content: The response content from the model
            reasoning_content: Optional reasoning content for reasoning models
        """
        if not self.model_config.save_llm_trace:
            return

        try:
            # Get workflow and node context from current execution
            if (
                not hasattr(self, "workflow")
                or not self.workflow
                or not hasattr(self, "current_node")
                or not self.current_node
            ):
                logger.debug("No workflow or node context available for trace saving")
                return

            # Create trace directory
            trajectory_dir = Path(self.workflow.global_config.trajectory_dir)
            task_id = self.workflow.task.id
            trace_dir = trajectory_dir / task_id
            trace_dir.mkdir(parents=True, exist_ok=True)

            # Parse prompt to separate system and user content
            system_prompt = ""
            user_prompt = ""

            if isinstance(prompt, list):
                # Handle message format like [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                for message in prompt:
                    if isinstance(message, dict):
                        role = message.get("role", "")
                        content = message.get("content", "")
                        if role == "system":
                            # Concatenate multiple system messages with newlines
                            if system_prompt:
                                system_prompt += "\n" + content
                            else:
                                system_prompt = content
                        elif role == "user":
                            # Concatenate multiple user messages with newlines
                            if user_prompt:
                                user_prompt += "\n" + content
                            else:
                                user_prompt = content
                        elif role == "assistant":
                            # Skip assistant messages in prompt parsing
                            continue
            elif isinstance(prompt, str):
                # Handle string prompt - put it all in user_prompt
                user_prompt = prompt
            else:
                # Handle other types by converting to string
                user_prompt = str(prompt)

            # Ensure we have valid strings
            system_prompt = system_prompt or ""
            user_prompt = user_prompt or ""
            response_content = response_content or ""

            # Create trace data with improved structure
            trace_data = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "reason_content": reasoning_content or "",
                "output_content": response_content,
            }

            # Save to YAML file named after node ID
            trace_file = trace_dir / f"{self.current_node.id}.yml"
            with open(trace_file, "w", encoding="utf-8") as f:
                yaml.dump(trace_data, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)

            logger.debug(f"LLM trace saved to {trace_file}")

        except Exception as e:
            logger.error(f"Failed to save LLM trace: {str(e)}")
            # Don't re-raise to avoid breaking the main execution flow
