# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os

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


class KimiModel(OpenAICompatibleModel):
    """
    Implementation of the BaseModel for Moonshot Kimi's API.

    Kimi K2 and K2.5 models support a "thinking" mode that returns reasoning_content.
    The sdk_patches.py module handles reasoning_content preservation during tool calling.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)
        logger.debug(f"Using Kimi model: {self.model_name} base_url: {self.base_url}")

    def _get_api_key(self) -> str:
        """Get Kimi API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("KIMI_API_KEY")
        if not api_key:
            raise ValueError("Kimi API key must be provided or set as KIMI_API_KEY environment variable")
        return api_key

    def _get_base_url(self) -> str:
        """Get Kimi base URL from config or environment."""
        return self.model_config.base_url or os.environ.get("KIMI_API_BASE", "https://api.moonshot.cn/v1")
