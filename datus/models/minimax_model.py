# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MiniMaxModel(OpenAICompatibleModel):
    """Implementation of the BaseModel for MiniMax's OpenAI-compatible API."""

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)
        logger.debug(f"Using MiniMax model: {self.model_name} base_url: {self.base_url}")

    def _get_api_key(self) -> str:
        """Get MiniMax API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise DatusException(ErrorCode.COMMON_ENV, message_args={"env_var": "MINIMAX_API_KEY"})
        return api_key

    def _get_base_url(self) -> str:
        """Get MiniMax base URL from config or environment."""
        return self.model_config.base_url or os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/v1")
