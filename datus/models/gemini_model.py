# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Gemini Model - Google Gemini model implementation.

Inherits from OpenAICompatibleModel and uses LiteLLM for unified API access.
"""

import os
from typing import Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GeminiModel(OpenAICompatibleModel):
    """Google Gemini model implementation using LiteLLM."""

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)
        logger.debug(f"Initialized Gemini model: {self.model_name}")

    def _get_api_key(self) -> str:
        """Get Gemini API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key must be provided or set as GEMINI_API_KEY environment variable")
        return api_key

    def _get_base_url(self) -> Optional[str]:
        """Get Gemini base URL. Returns None to use LiteLLM's native Gemini support."""
        return self.model_config.base_url  # Don't provide a default, let LiteLLM handle it
