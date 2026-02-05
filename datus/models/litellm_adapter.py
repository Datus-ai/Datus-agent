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

from typing import TYPE_CHECKING, Optional

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from agents.models.interface import Model

logger = get_logger(__name__)

# Note: SDK patches are applied in datus/models/__init__.py to ensure
# they are applied before any agents SDK usage


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
    }

    # Provider-specific base URLs (if not using default)
    DEFAULT_BASE_URLS = {
        "openai": None,  # Use LiteLLM default
        "claude": None,  # Use LiteLLM default
        "deepseek": "https://api.deepseek.com",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "gemini": None,  # Use LiteLLM default (native Gemini API, not OpenAI-compatible)
        "kimi": "https://api.moonshot.ai/v1",  # Moonshot AI global endpoint
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
    }

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        enable_thinking: bool = False,
    ):
        """
        Initialize the LiteLLM adapter.

        Args:
            provider: The model provider (openai, claude, deepseek, qwen, gemini, kimi)
            model: The model name (e.g., gpt-4o, claude-sonnet-4, kimi-k2.5)
            api_key: API key for the provider
            base_url: Optional custom base URL (overrides default)
            enable_thinking: Whether to enable thinking/reasoning mode (default: False)
        """
        # Auto-detect provider from model name if provider is generic
        detected_provider = self._detect_provider_from_model(provider, model)
        self.provider = detected_provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URLS.get(self.provider)
        self._enable_thinking = enable_thinking
        self._litellm_model_name = None

    def _detect_provider_from_model(self, provider: str, model: str) -> str:
        """
        Auto-detect provider from model name when provider is generic.

        This allows configurations like type: "openai" with model: "kimi-k2.5"
        to be automatically detected as kimi provider for correct LiteLLM routing.

        Args:
            provider: The configured provider (e.g., "openai")
            model: The model name (e.g., "kimi-k2.5")

        Returns:
            The detected provider name
        """
        model_lower = model.lower()

        # Check if model name starts with a known prefix
        for prefix, detected in self.MODEL_NAME_PREFIXES.items():
            if model_lower.startswith(prefix):
                if provider.lower() != detected:
                    logger.info(
                        f"Auto-detected provider '{detected}' from model name '{model}' "
                        f"(configured as '{provider}')"
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
        """
        prefix = self.MODEL_PREFIX_MAP.get(self.provider, "")

        # If model already has a prefix, don't add another
        if "/" in self.model:
            return self.model

        # For OpenAI, no prefix needed
        if not prefix:
            return self.model

        return f"{prefix}{self.model}"

    @property
    def is_thinking_model(self) -> bool:
        """
        Check if thinking/reasoning mode is explicitly enabled for this model.

        When enabled, the model returns reasoning_content in responses and needs
        special handling to preserve thinking blocks in multi-turn conversations.

        Disabled by default. Set enable_thinking: true in config to enable.
        """
        return bool(self._enable_thinking)

    def get_agents_sdk_model(self) -> "Model":
        """
        Get an openai-agents SDK compatible Model instance.

        Returns a LitellmModel for all providers. Kimi/Moonshot thinking models
        are supported via SDK patches that extend the reasoning_content handling.

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

        # Build model kwargs
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

        logger.debug(f"Creating LitellmModel with model={self.litellm_model_name}")

        return LitellmModel(**model_kwargs)

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

        return kwargs


def create_litellm_adapter(
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    enable_thinking: bool = False,
) -> LiteLLMAdapter:
    """
    Factory function to create a LiteLLM adapter.

    Args:
        provider: The model provider (openai, claude, deepseek, qwen, gemini)
        model: The model name
        api_key: API key for the provider
        base_url: Optional custom base URL
        enable_thinking: Whether to enable thinking/reasoning mode (default: False)

    Returns:
        Configured LiteLLMAdapter instance
    """
    return LiteLLMAdapter(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=enable_thinking,
    )
