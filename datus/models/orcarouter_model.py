# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
OrcaRouter Model - Unified AI gateway supporting 150+ models.

Thin wrapper over OpenAICompatibleModel. OrcaRouter provides an OpenAI-compatible
API, so all features (streaming, tool calling, JSON mode) work out of the box.

Model names keep the OrcaRouter ``vendor/slug`` namespace (e.g.
``deepseek/deepseek-chat``, ``openai/gpt-5.5``); the LiteLLM adapter prepends
the ``openai/`` provider prefix and routes the full namespace to the base URL.
"""

import os
from typing import Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OrcaRouterModel(OpenAICompatibleModel):
    """
    OrcaRouter model implementation.

    Routes requests through OrcaRouter's unified gateway to any supported
    provider (OpenAI, Anthropic, Google, DeepSeek, Qwen, etc.).

    Model names follow the OrcaRouter convention: provider/model-name
    e.g., deepseek/deepseek-chat, openai/gpt-5.5, orcarouter/auto
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

    def _get_api_key(self) -> str:
        """Get OrcaRouter API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("ORCAROUTER_API_KEY")
        if not api_key:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(ErrorCode.COMMON_ENV, message_args={"env_var": "ORCAROUTER_API_KEY"})
        return api_key

    def _get_base_url(self) -> Optional[str]:
        """Get OrcaRouter base URL from config or default."""
        return self.model_config.base_url or "https://api.orcarouter.ai/v1"
