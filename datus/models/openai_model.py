# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OpenAIModel(OpenAICompatibleModel):
    """
    Implementation of the BaseModel for OpenAI's API.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        """
        Initialize the OpenAI model.

        Args:
            model_config: Model configuration object
            **kwargs: Additional parameters for the OpenAI API
        """
        super().__init__(model_config, **kwargs)

    def _get_api_key(self) -> str:
        """Get OpenAI API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        return api_key

    def _uses_completion_tokens_parameter(self) -> bool:
        """Check if this model uses max_completion_tokens instead of max_tokens.

        OpenAI's reasoning models (o-series) require max_completion_tokens parameter
        instead of the traditional max_tokens parameter.

        Returns:
            True if model uses max_completion_tokens, False if it uses max_tokens
        """
        # O-series reasoning models use max_completion_tokens
        reasoning_model_patterns = ["o1", "o1-", "o2", "o2-", "o3", "o3-", "o4", "o4-"]

        # Also check for gpt-o4 pattern (like gpt-o4-mini)
        if self.model_name.startswith("gpt-o"):
            return True

        return any(self.model_name.startswith(pattern) for pattern in reasoning_model_patterns)

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """Generate response with OpenAI-specific parameter handling.

        For reasoning models (o-series), removes unsupported parameters and
        transforms max_tokens to max_completion_tokens.

        Args:
            prompt: The input prompt to send to the model
            enable_thinking: Enable thinking mode for hybrid models (default: False)
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Handle reasoning models (o1, o3, o4-mini, etc.)
        if self._uses_completion_tokens_parameter():
            # Transform max_tokens to max_completion_tokens
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                logger.debug(f"Transformed max_tokens to max_completion_tokens for model {self.model_name}")

            # Remove unsupported parameters for reasoning models
            unsupported_params = [
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "logprobs",
                "top_logprobs",
                "logit_bias",
            ]

            removed_params = []
            for param in unsupported_params:
                if param in kwargs:
                    kwargs.pop(param)
                    removed_params.append(param)

            if removed_params:
                logger.debug(f"Removed unsupported parameters for reasoning model {self.model_name}: {removed_params}")

        if self.model_name.startswith("gpt-5"):
            kwargs["temperature"] = 1

        return super().generate(prompt, enable_thinking, **kwargs)
