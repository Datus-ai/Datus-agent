# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Gemini Model - Google Gemini model implementation.

Inherits from OpenAICompatibleModel and uses LiteLLM for unified API access.
"""

import os
from typing import Dict, Optional

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

    @property
    def model_specs(self) -> Dict[str, Dict[str, int]]:
        """Model specifications for Gemini models."""
        return {
            # Gemini 2.x series
            "gemini-2.5-pro": {"context_length": 1048576, "max_tokens": 65535},
            "gemini-2.5-flash": {"context_length": 1048576, "max_tokens": 8192},
            "gemini-2.5-flash-lite": {"context_length": 1048576, "max_tokens": 8192},
            "gemini-2.0-flash": {"context_length": 1048576, "max_tokens": 8192},
            # Gemini 3.x series (preview) - Nov 2025 specs
            "gemini-3-pro-preview": {"context_length": 1048576, "max_tokens": 65536},
            "gemini-3-flash-preview": {"context_length": 1048576, "max_tokens": 65536},
            # Gemini 1.x series
            "gemini-1.5-pro": {"context_length": 2097152, "max_tokens": 8192},
            "gemini-1.5-flash": {"context_length": 1048576, "max_tokens": 8192},
        }
