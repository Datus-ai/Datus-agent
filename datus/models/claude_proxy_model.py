# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Claude Proxy Model - Integration with claude-max-api-proxy.

Thin wrapper over OpenAICompatibleModel for the claude-max-api-proxy local proxy.
The proxy translates OpenAI-compatible requests to the real Claude CLI,
ensuring all models (including Opus) are available via subscription quota.

Install: npm install -g claude-max-api-proxy && claude-max-api
Default endpoint: http://localhost:3456/v1
"""

import os
from typing import Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ClaudeProxyModel(OpenAICompatibleModel):
    """
    Claude Proxy model via claude-max-api-proxy.

    The proxy handles authentication through the real Claude CLI,
    so no API key is required. Provides an OpenAI-compatible endpoint.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

    def _get_api_key(self) -> str:
        """API key is not required for the local proxy — return a placeholder."""
        return self.model_config.api_key or os.environ.get("CLAUDE_PROXY_API_KEY") or "not-needed"

    def _get_base_url(self) -> Optional[str]:
        """Get proxy base URL from config or default to localhost:3456."""
        return self.model_config.base_url or "http://localhost:3456/v1"
