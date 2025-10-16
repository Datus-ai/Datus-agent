# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os

from agents import set_tracing_disabled

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

# Import typing fix for Python 3.12+ compatibility
try:
    from datus.utils.typing_fix import patch_agents_typing_issue

    patch_agents_typing_issue()
except ImportError:
    pass

logger = get_logger(__name__)
MAX_INPUT_DEEPSEEK = 52000  # 57344 - buffer of ~5000 tokens

set_tracing_disabled(True)


class DeepSeekModel(OpenAICompatibleModel):
    """
    Implementation of the BaseModel for DeepSeek's API.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)
        logger.debug(f"Using DeepSeek model: {self.model_name} base Url: {self.base_url}")

    def _get_api_key(self) -> str:
        """Get DeepSeek API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key must be provided or set as DEEPSEEK_API_KEY environment variable")
        return api_key

    def _get_base_url(self) -> str:
        """Get DeepSeek base URL from config or environment."""
        return self.model_config.base_url or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    def token_count(self, prompt: str) -> int:
        """
        Estimate the number of tokens in a text using the deepseek tokenizer.
        """
        return int(len(prompt) * 0.3 + 0.5)

    def max_tokens(self) -> int:
        return MAX_INPUT_DEEPSEEK
