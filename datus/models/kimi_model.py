# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os

from agents import set_tracing_disabled

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

set_tracing_disabled(True)


class KimiModel(OpenAICompatibleModel):
    """
    Implementation of the BaseModel for Moonshot Kimi's API.

    Kimi K2 and K2.5 models support a "thinking" mode that returns reasoning_content.
    The sdk_patches.py module handles reasoning_content preservation by monkey-patching
    the agents SDK to treat Kimi models like DeepSeek (which has native support).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)
        # Kimi models may have issues with structured outputs + tool calls
        # See: https://github.com/openai/openai-agents-python/issues/1778
        self._supports_structured_output = False
        logger.debug(f"Using Kimi model: {self.model_name} base_url: {self.base_url}")

    def _is_k25_model(self) -> bool:
        """Check if this is a K2.5 model that requires temperature=1."""
        return "k2.5" in self.model_name.lower()

    def generate(self, prompt, enable_thinking: bool = False, **kwargs) -> str:
        """Generate response with Kimi-specific parameter handling.

        For K2.5 models, temperature must be 1 and top_p must be 0.95.

        Args:
            prompt: The input prompt to send to the model
            enable_thinking: Enable thinking mode for hybrid models (default: False)
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        if self._is_k25_model():
            kwargs["temperature"] = 1
            kwargs["top_p"] = 0.95
            logger.debug(f"Set temperature=1, top_p=0.95 for K2.5 model {self.model_name}")

        return super().generate(prompt, enable_thinking, **kwargs)

    def _get_api_key(self) -> str:
        """Get Kimi API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("KIMI_API_KEY")
        if not api_key:
            raise ValueError("Kimi API key must be provided or set as KIMI_API_KEY environment variable")
        return api_key

    def _get_base_url(self) -> str:
        """Get Kimi base URL from config or environment."""
        return self.model_config.base_url or os.environ.get("KIMI_API_BASE", "https://api.moonshot.cn/v1")
