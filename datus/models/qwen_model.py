# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class QwenModel(OpenAICompatibleModel):
    def __init__(self, model_config: ModelConfig, **kwargs):
        """
        Initialize the Qwen model.

        Args:
            model_config: Model configuration object
            **kwargs: Additional parameters
        """
        super().__init__(model_config, **kwargs)

    def _get_api_key(self) -> str:
        """Get Qwen API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise ValueError("Qwen API key must be provided or set as QWEN_API_KEY environment variable")
        return api_key

    def _get_base_url(self) -> str:
        """Get Qwen base URL from config or environment."""
        return self.model_config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
