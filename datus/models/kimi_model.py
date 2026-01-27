# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any, Dict

from agents import set_tracing_disabled

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

set_tracing_disabled(True)


class KimiModel(OpenAICompatibleModel):
    """
    Implementation of the BaseModel for Moonshot Kimi's API.

    Kimi K2 and K2.5 models support a "thinking" mode that returns reasoning_content.
    When using tool calls with the agents SDK, thinking mode must be disabled because
    the SDK doesn't preserve reasoning_content in assistant messages, which causes
    API errors: "thinking is enabled but reasoning_content is missing".
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

    def _build_tool_extra_body(self) -> Dict[str, Any]:
        """Build extra_body for tool calls with Kimi-specific settings.

        Disables thinking mode to avoid 'reasoning_content is missing' errors
        when using tool calls. The agents SDK doesn't preserve reasoning_content
        in assistant messages, which Kimi's API requires when thinking is enabled.

        Returns:
            Dict with extra_body parameters including thinking disabled
        """
        extra_body = super()._build_tool_extra_body()
        # Disable thinking mode for Kimi to avoid reasoning_content requirement
        # See: https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model
        extra_body["thinking"] = {"type": "disabled"}
        logger.debug("Kimi model: disabled thinking mode for tool calls")
        return extra_body

    def token_count(self, prompt: str) -> int:
        """
        Estimate the number of tokens in a text.
        Kimi uses a similar tokenization to OpenAI models.
        """
        return int(len(prompt) * 0.3 + 0.5)
