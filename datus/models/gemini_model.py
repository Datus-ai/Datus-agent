# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Dict, List, Union

import google.generativeai as genai

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GeminiModel(OpenAICompatibleModel):
    """Google Gemini model implementation"""

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)
        # Initialize Gemini-specific client
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel(self.model_name)

    def _get_api_key(self) -> str:
        """Get Gemini API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key must be provided or set as GEMINI_API_KEY environment variable")
        return api_key

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.7),
                max_output_tokens=kwargs.get("max_tokens", 10000),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 40),
            )

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=kwargs.get("safety_settings", None),
            )

            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                logger.warning("No candidates returned from Gemini model")
                return ""

        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            raise

    def _get_base_url(self) -> str:
        """Get Gemini base URL for OpenAI compatibility."""
        return "https://generativelanguage.googleapis.com/v1beta/openai"

    def token_count(self, prompt: str) -> int:
        try:
            model = genai.GenerativeModel(self.model_name)
            token_count = model.count_tokens(prompt)
            return token_count.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens with Gemini: {str(e)}")
            return len(prompt) // 4
