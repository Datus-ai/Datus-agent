# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LiteLLM Adapter - Unified LLM calling layer.

Provides:
1. Unified completion/chat API across all providers
2. Model name mapping for different providers
3. Integration with openai-agents SDK's LitellmModel
"""

from typing import TYPE_CHECKING, Dict, Optional
from urllib.parse import urlparse

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from agents.models.interface import Model

logger = get_logger(__name__)

# Note: SDK patches are applied in datus/models/__init__.py to ensure
# they are applied before any agents SDK usage

# Datus-maintained DENY-list of DeepSeek/Moonshot models that are known NOT to
# support thinking/reasoning. Everything else in these two providers defaults
# to "supports reasoning" so that future releases (DeepSeek V5, Kimi K3, …)
# receive ``reasoning_effort`` and the native ``thinking`` switch without
# waiting for a config edit.
#
# Why a deny-list (not allow-list):
# - Every new LLM release in 2025+ ships thinking; the exceptions are legacy
#   lines that the vendor explicitly keeps non-reasoning (DeepSeek-Chat,
#   Moonshot-v1 classic, the bare ``kimi-k2`` non-thinking fork).
# - LiteLLM's built-in capability table lags behind Chinese providers, so
#   ``litellm.supports_reasoning`` returns False for many thinking models.
#   Defaulting to "supported" compensates for that drift.
#
# Sources (verified 2026-04):
# - DeepSeek: https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
# - Moonshot/Kimi: https://platform.kimi.com/docs/api/models-overview
#
# Matching rules:
# - Entries ending in ``*`` are prefix-matched (``moonshot-v1*`` covers
#   ``moonshot-v1-8k``, ``moonshot-v1-32k``, ``moonshot-v1-128k``, ...).
# - Plain entries are compared by exact equality so ``kimi-k2`` does NOT
#   accidentally match ``kimi-k2.5``, ``kimi-k2.6``, or ``kimi-k2-thinking``.
NON_THINKING_MODEL_RULES: Dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-chat",),
    "kimi": ("moonshot-v1*", "kimi-k2"),
}


def is_official_openai_endpoint(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True iff ``provider`` is ``openai`` and ``base_url`` resolves to ``api.openai.com``.

    A missing ``base_url`` implies the OpenAI SDK default
    (``https://api.openai.com/v1``); any other host is treated as a third-party
    OpenAI-compatible proxy (vLLM, OpenRouter relays, Coding Plan endpoints).
    """
    if provider != "openai":
        return False
    if not base_url:
        return True
    try:
        hostname = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    return hostname == "api.openai.com"


def is_known_non_thinking_model(provider: Optional[str], model_name: Optional[str]) -> bool:
    """Return True only when the (provider, model) is on the deny-list.

    Unknown models — including every entry outside DeepSeek and Kimi, and every
    new release not explicitly listed — return False, which lets the caller
    treat them as thinking-capable.
    """
    if not provider or not model_name:
        return False
    rules = NON_THINKING_MODEL_RULES.get(provider.lower())
    if not rules:
        return False
    name = model_name.lower()
    for rule in rules:
        if rule.endswith("*"):
            if name.startswith(rule[:-1].lower()):
                return True
        elif name == rule.lower():
            return True
    return False


class LiteLLMAdapter:
    """
    Unified LiteLLM adapter for calling various LLM providers.

    Supports:
    - OpenAI (gpt-4o, gpt-5, o3, etc.)
    - Anthropic (claude-sonnet-4, claude-opus-4, etc.)
    - DeepSeek (deepseek-chat, deepseek-reasoner, etc.)
    - Qwen/DashScope (qwen3-coder, etc.)
    - Google Gemini (gemini-2.5-pro, gemini-3-pro, etc.)
    - Moonshot/Kimi (kimi-k2.5, kimi-k2-thinking, etc.)
    """

    # Model name prefix mapping for LiteLLM
    # See: https://docs.litellm.ai/docs/providers
    MODEL_PREFIX_MAP = {
        "openai": "",  # LiteLLM supports OpenAI models natively without prefix
        "claude": "anthropic/",
        "deepseek": "deepseek/",
        "qwen": "dashscope/",
        "gemini": "gemini/",
        "kimi": "moonshot/",  # Moonshot AI - https://docs.litellm.ai/docs/providers/moonshot
        "openrouter": "openrouter/",  # OpenRouter unified gateway
        "minimax": "openai/",  # MiniMax - OpenAI-compatible API
        "glm": "openai/",  # Zhipu GLM - OpenAI-compatible API
    }

    # Provider-specific base URLs (if not using default)
    DEFAULT_BASE_URLS = {
        "openai": None,  # Use LiteLLM default
        "claude": None,  # Use LiteLLM default
        "deepseek": "https://api.deepseek.com",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "gemini": None,  # Use LiteLLM default (native Gemini API, not OpenAI-compatible)
        "kimi": "https://api.moonshot.ai/v1",  # Moonshot AI global endpoint
        "openrouter": None,  # Use LiteLLM default (https://openrouter.ai/api/v1)
        "minimax": "https://api.minimaxi.com/v1",  # MiniMax OpenAI-compatible endpoint
        "glm": "https://open.bigmodel.cn/api/paas/v4",  # Zhipu GLM OpenAI-compatible endpoint
    }

    # Model name prefixes for auto-detection
    # When provider is generic (e.g., "openai"), detect actual provider from model name
    MODEL_NAME_PREFIXES = {
        "kimi": "kimi",  # kimi-k2, kimi-k2.5, kimi-k2-thinking
        "moonshot": "kimi",  # moonshot-v1-8k, etc.
        "claude": "claude",  # claude-sonnet-4, etc.
        "gpt": "openai",  # gpt-4o, gpt-5, etc.
        "o1": "openai",  # o1, o1-mini, o1-preview
        "o3": "openai",  # o3, o3-mini
        "deepseek": "deepseek",  # deepseek-chat, deepseek-reasoner
        "qwen": "qwen",  # qwen3-coder, etc.
        "gemini": "gemini",  # gemini-2.5-pro, gemini-3-flash
        "minimax": "minimax",  # MiniMax-M2.5, MiniMax-M2.7
        "glm": "glm",  # glm-5, glm-4.7, etc.
    }

    # Known domains for each provider (used to validate auto-detection against base_url)
    PROVIDER_DOMAINS = {
        "openai": ["api.openai.com"],
        "claude": ["api.anthropic.com"],
        "deepseek": ["api.deepseek.com"],
        "qwen": ["dashscope.aliyuncs.com"],
        "gemini": ["generativelanguage.googleapis.com"],
        "kimi": ["api.moonshot.ai", "api.moonshot.cn", "api.kimi.com"],
        "minimax": ["api.minimaxi.com"],
        "glm": ["open.bigmodel.cn"],
    }

    # Protocol keywords that appear in proxy URL paths (e.g. /apps/anthropic, /coding/)
    # When a keyword is found in the base_url path, auto-detection is skipped to
    # preserve the configured provider. Supports Coding Plan endpoints where
    # Anthropic-compatible proxies use vendor-specific paths like /coding/.
    PROVIDER_PROTOCOL_KEYWORDS = {
        "claude": ["anthropic", "coding"],
        "openai": ["openai"],
    }

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        enable_thinking: bool = False,
        reasoning_effort: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the LiteLLM adapter.

        Args:
            provider: The model provider (openai, claude, deepseek, qwen, gemini, kimi)
            model: The model name (e.g., gpt-4o, claude-sonnet-4, kimi-k2.5)
            api_key: API key for the provider
            base_url: Optional custom base URL (overrides default)
            enable_thinking: Legacy bool switch; True is equivalent to reasoning_effort="medium".
            reasoning_effort: One of off|minimal|low|medium|high. Takes precedence
                over ``enable_thinking`` when set; ``None`` defers to the bool.
            default_headers: Optional custom HTTP headers (e.g., User-Agent for Coding Plan endpoints)
        """
        # Auto-detect provider from model name if provider is generic
        detected_provider = self._detect_provider_from_model(provider, model, base_url)
        self.provider = detected_provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URLS.get(self.provider)
        self._enable_thinking = enable_thinking
        self._reasoning_effort = reasoning_effort.strip().lower() if isinstance(reasoning_effort, str) else None
        self.default_headers = default_headers
        self._litellm_model_name = None

    def _detect_provider_from_model(self, provider: str, model: str, base_url: Optional[str] = None) -> str:
        """
        Auto-detect provider from model name when provider is generic.

        This allows configurations like type: "openai" with model: "kimi-k2.5"
        to be automatically detected as kimi provider for correct LiteLLM routing.

        When a custom base_url is provided and its domain doesn't match the detected
        provider's known domains, auto-detection is skipped to preserve the user's
        configured provider. This supports Coding Plan endpoints where models like
        "qwen3-coder-plus" are accessed via Anthropic-compatible proxy URLs.

        Args:
            provider: The configured provider (e.g., "openai")
            model: The model name (e.g., "kimi-k2.5")
            base_url: Optional custom base URL for domain validation

        Returns:
            The detected provider name
        """
        # Skip auto-detection for providers that must not be overridden
        # (e.g., openrouter models contain provider/ prefix that would trigger false detection)
        if provider.lower() == "openrouter":
            return provider

        model_lower = model.lower()

        # Check if model name starts with a known prefix
        for prefix, detected in self.MODEL_NAME_PREFIXES.items():
            if model_lower.startswith(prefix):
                # If base_url is set and detected provider differs from configured,
                # check if the base_url domain matches the detected provider's known domains
                if base_url and detected != provider.lower():
                    parsed = urlparse(base_url)
                    domain = parsed.hostname or ""
                    path = parsed.path.lower()
                    known_domains = self.PROVIDER_DOMAINS.get(detected, [])
                    domain_matches = any(domain == d or domain.endswith(f".{d}") for d in known_domains)

                    # Skip auto-detection if domain doesn't match detected provider
                    if not domain_matches:
                        logger.info(
                            f"Keeping provider '{provider}' — base_url domain '{domain}' "
                            f"doesn't match detected provider '{detected}'"
                        )
                        return provider

                    # Even if domain matches, check if URL path indicates a proxy
                    # for the configured provider (e.g. /apps/anthropic, /coding/)
                    protocol_keywords = self.PROVIDER_PROTOCOL_KEYWORDS.get(provider.lower(), [])
                    matched_keyword = next((kw for kw in protocol_keywords if kw in path), None)
                    if matched_keyword:
                        logger.info(
                            f"Keeping provider '{provider}' — base_url path indicates "
                            f"'{matched_keyword}' proxy on '{domain}'"
                        )
                        return provider
                if provider.lower() != detected:
                    logger.info(
                        f"Auto-detected provider '{detected}' from model name '{model}' (configured as '{provider}')"
                    )
                return detected

        # No match, use configured provider
        return provider

    @property
    def litellm_model_name(self) -> str:
        """
        Get the LiteLLM-formatted model name.

        Returns:
            Model name with appropriate provider prefix for LiteLLM
        """
        if self._litellm_model_name is None:
            self._litellm_model_name = self._get_litellm_model_name()
        return self._litellm_model_name

    def _get_litellm_model_name(self) -> str:
        """
        Build the LiteLLM model name with provider prefix.

        Examples:
            - openai/gpt-4o -> gpt-4o (no prefix for OpenAI)
            - claude/claude-sonnet-4 -> anthropic/claude-sonnet-4
            - deepseek/deepseek-chat -> deepseek/deepseek-chat
            - openai + custom base_url + Qwen3.5-397B -> openai/Qwen3.5-397B
        """
        prefix = self.MODEL_PREFIX_MAP.get(self.provider, "")

        # OpenRouter models always need the openrouter/ prefix,
        # even when model name contains / (e.g., anthropic/claude-sonnet-4)
        if self.provider == "openrouter":
            return self.model if self.model.startswith("openrouter/") else f"openrouter/{self.model}"

        # If model already has a prefix, don't add another
        if "/" in self.model:
            return self.model

        # For OpenAI provider with custom base_url (e.g. self-hosted vLLM),
        # add "openai/" prefix so LiteLLM uses OpenAI-compatible API format
        # for model names not in its built-in list (e.g. Qwen3.5-397B).
        if self.provider == "openai" and not prefix and self.base_url:
            parsed = urlparse(self.base_url)
            domain = parsed.hostname or ""
            if domain not in ("api.openai.com",):
                return f"openai/{self.model}"

        # For OpenAI native models (gpt-4o, o3, etc.), no prefix needed
        if not prefix:
            return self.model

        return f"{prefix}{self.model}"

    @property
    def reasoning_effort_level(self) -> Optional[str]:
        """Resolved reasoning effort level or ``None`` when disabled.

        Priority: explicit ``reasoning_effort`` (except ``"off"``) wins; a
        legacy ``enable_thinking=True`` falls back to ``"medium"`` to keep
        existing configs working; otherwise reasoning is off. LiteLLM maps
        the returned level into each vendor's dialect (OpenAI
        ``reasoning_effort``, Anthropic ``thinking.budget_tokens``, Gemini
        ``thinking_config.thinking_budget``, etc.).
        """
        if self._reasoning_effort == "off":
            return None
        if self._reasoning_effort:
            return self._reasoning_effort
        if self._enable_thinking:
            return "medium"
        return None

    @property
    def is_thinking_model(self) -> bool:
        """
        Check if thinking/reasoning mode is active for this model.

        True when :attr:`reasoning_effort_level` resolves to a non-None level.
        When enabled, the model returns reasoning_content in responses and
        needs special handling to preserve thinking blocks in multi-turn
        conversations.
        """
        return self.reasoning_effort_level is not None

    def _is_official_openai(self) -> bool:
        """True when this adapter targets the canonical OpenAI API host.

        The Responses API (``/v1/responses``) is the preferred endpoint for
        reasoning models (o-series, gpt-5*) because ``chat/completions``
        rejects ``reasoning_effort`` alongside function tools. LiteLLM goes
        through ``chat/completions`` by default; routing official OpenAI
        traffic through :class:`OpenAIResponsesModel` sidesteps the
        restriction without adding per-model branches.
        """
        return is_official_openai_endpoint(self.provider, self.base_url)

    def get_agents_sdk_model(self) -> "Model":
        """
        Get an openai-agents SDK compatible Model instance.

        Routing:

        - **Official OpenAI** (``provider=="openai"`` and ``base_url`` on
          ``api.openai.com``) uses :class:`OpenAIResponsesModel`, which
          speaks the Responses API. This is the only endpoint that accepts
          ``reasoning_effort`` together with function tools for o-series /
          gpt-5* reasoning models.
        - **Anthropic Claude** uses :class:`CacheControlLitellmModel` so the
          prompt caching control markers survive the LiteLLM transform.
        - **Every other OpenAI-compatible provider** (DeepSeek, Kimi, Qwen,
          Gemini, OpenRouter, GLM, MiniMax, vLLM, and self-hosted proxies)
          uses :class:`LitellmModel` as before.

        Kimi/Moonshot thinking models still go through the LiteLLM path and
        continue to rely on :mod:`datus.models.sdk_patches` for the
        ``reasoning_content`` echo-back behaviour required on tool-calling
        turns.

        Returns:
            Model instance configured for this adapter
        """
        try:
            from agents.extensions.models.litellm_model import LitellmModel
        except ImportError as err:
            raise ImportError(
                "LitellmModel not found. Please install openai-agents with litellm support: "
                "pip install 'openai-agents[litellm]'"
            ) from err

        if self._is_official_openai():
            return self._build_openai_responses_model()

        # Build model kwargs for the LiteLLM path
        model_kwargs = {
            "model": self.litellm_model_name,
        }

        # Add API key - LiteLLM uses different env var names per provider
        # We pass it directly to avoid env var conflicts
        if self.api_key:
            model_kwargs["api_key"] = self.api_key

        # Add base URL if specified
        if self.base_url:
            model_kwargs["base_url"] = self.base_url

        # Note: default_headers are NOT passed here — LitellmModel.__init__ only
        # accepts model/base_url/api_key. Headers are injected via ModelSettings.extra_headers
        # at call time in OpenAICompatibleModel.generate_with_tools/_stream.

        logger.debug(f"Creating LitellmModel with model={self.litellm_model_name}")

        if self.provider == "claude":
            from datus.models.litellm_cache_control import CacheControlLitellmModel

            return CacheControlLitellmModel(**model_kwargs)

        return LitellmModel(**model_kwargs)

    def _build_openai_responses_model(self) -> "Model":
        """Construct an :class:`OpenAIResponsesModel` bound to this adapter.

        A fresh :class:`AsyncOpenAI` client is created per adapter instance
        so the SDK is free to reuse HTTP connections for repeated calls
        without leaking state between ``/model`` switches. Custom
        ``default_headers`` flow into the client so Coding Plan-style
        User-Agent overrides still apply on the Responses path.
        """
        from agents.models.openai_responses import OpenAIResponsesModel
        from openai import AsyncOpenAI

        client_kwargs: Dict[str, object] = {"api_key": self.api_key or ""}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        async_client = AsyncOpenAI(**client_kwargs)
        logger.debug(f"Creating OpenAIResponsesModel for official OpenAI model={self.model}")
        return OpenAIResponsesModel(model=self.model, openai_client=async_client)

    def get_completion_kwargs(self) -> dict:
        """
        Get kwargs for direct litellm.completion() calls.

        Returns:
            Dict of kwargs for litellm.completion()
        """
        kwargs = {
            "model": self.litellm_model_name,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.base_url:
            kwargs["api_base"] = self.base_url

        if self.default_headers:
            kwargs["extra_headers"] = self.default_headers

        return kwargs


def create_litellm_adapter(
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    enable_thinking: bool = False,
    reasoning_effort: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> LiteLLMAdapter:
    """
    Factory function to create a LiteLLM adapter.

    Args:
        provider: The model provider (openai, claude, deepseek, qwen, gemini)
        model: The model name
        api_key: API key for the provider
        base_url: Optional custom base URL
        enable_thinking: Legacy bool switch; True is equivalent to reasoning_effort="medium".
        reasoning_effort: One of off|minimal|low|medium|high; takes precedence
            over ``enable_thinking`` when set.
        default_headers: Optional custom HTTP headers

    Returns:
        Configured LiteLLMAdapter instance
    """
    return LiteLLMAdapter(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        default_headers=default_headers,
    )
