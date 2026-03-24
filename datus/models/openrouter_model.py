# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
OpenRouter Model - Unified AI gateway supporting 300+ models.

Thin wrapper over OpenAICompatibleModel. OpenRouter provides an OpenAI-compatible
API, so all features (streaming, tool calling, JSON mode) work out of the box
via LiteLLM's openrouter/ prefix.
"""

import os
from typing import Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OpenRouterModel(OpenAICompatibleModel):
    """
    OpenRouter model implementation.

    Routes requests through OpenRouter's unified gateway to any supported
    provider (OpenAI, Anthropic, Google, DeepSeek, Meta, Mistral, etc.).

    Model names follow the OpenRouter convention: provider/model-name
    e.g., anthropic/claude-sonnet-4, openai/gpt-4o, google/gemini-2.5-pro
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

    def _get_api_key(self) -> str:
        """Get OpenRouter API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(ErrorCode.COMMON_ENV, message_args={"env_var": "OPENROUTER_API_KEY"})
        return api_key

    def _get_base_url(self) -> Optional[str]:
        """Get OpenRouter base URL from config or default."""
        return self.model_config.base_url or "https://openrouter.ai/api/v1"
