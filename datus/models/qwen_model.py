# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any

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

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """
        Generate a response from the Qwen model with thinking support.

        Args:
            prompt: The input prompt to send to the model
            enable_thinking: Enable thinking mode (default: False)
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Merge default parameters with any provided kwargs
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "top_p"]},
        }

        # Add enable_thinking to extra_body for Qwen API
        if "extra_body" not in params:
            params["extra_body"] = {}
        params["extra_body"]["enable_thinking"] = enable_thinking

        # Convert prompt to messages format
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        response = self.client.chat.completions.create(messages=messages, **params)
        return response.choices[0].message.content
