# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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
