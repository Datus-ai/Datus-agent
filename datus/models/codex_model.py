# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Codex model implementation using OAuth authentication and Responses API."""

from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from agents import SQLiteSession, Tool
from agents.mcp import MCPServerStdio

from datus.auth.oauth_config import CODEX_API_BASE_URL
from datus.auth.oauth_manager import OAuthManager
from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Known Codex model specifications
_CODEX_MODEL_SPECS: Dict[str, Dict[str, int]] = {
    "gpt-5.3-codex": {"context_length": 192000, "max_tokens": 16384},
    "gpt-5.1-codex-mini": {"context_length": 192000, "max_tokens": 16384},
    "o3-codex": {"context_length": 200000, "max_tokens": 100000},
}


class CodexModel(LLMBaseModel):
    """Access OpenAI Codex models via OAuth tokens and the Responses API.

    Unlike standard OpenAI models, Codex uses:
    - OAuth token authentication (ChatGPT subscription) instead of API keys
    - The Responses API format (POST /responses) instead of Chat Completions
    - A different base URL (chatgpt.com/backend-api/codex)
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config)
        self.model_name = model_config.model
        self.oauth_manager = OAuthManager()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI client with OAuth token."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.oauth_manager.get_access_token(),
                base_url=CODEX_API_BASE_URL,
            )
        return self._client

    def _refresh_client_token(self):
        """Refresh the OAuth token on the existing client."""
        client = self._get_client()
        client.api_key = self.oauth_manager.get_access_token()

    @staticmethod
    def _convert_prompt_to_input(prompt: Any) -> Any:
        """Convert messages-format prompt to Responses API input format.

        The Responses API accepts either a plain string or a list of
        message dicts with 'role' and 'content' keys.
        """
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return prompt
        return str(prompt)

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """Generate a response via the Codex Responses API.

        Args:
            prompt: Input prompt (string or messages list)
            enable_thinking: Not supported for Codex models (ignored)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        self._refresh_client_token()
        input_data = self._convert_prompt_to_input(prompt)

        try:
            response = self._get_client().responses.create(
                model=self.model_name,
                input=input_data,
            )
            return response.output_text
        except Exception as e:
            # On 401, try refreshing token once and retry
            if "401" in str(e) or "unauthorized" in str(e).lower():
                logger.info("Got 401, refreshing OAuth token and retrying...")
                self.oauth_manager.refresh_tokens()
                self._refresh_client_token()
                response = self._get_client().responses.create(
                    model=self.model_name,
                    input=input_data,
                )
                return response.output_text
            raise

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """Generate a JSON-structured response via the Codex Responses API.

        Args:
            prompt: Input prompt (string or messages list)
            **kwargs: May contain 'output_schema' for structured output

        Returns:
            Parsed JSON response as a dictionary
        """
        import json

        self._refresh_client_token()
        input_data = self._convert_prompt_to_input(prompt)

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": input_data,
        }

        output_schema = kwargs.get("output_schema")
        if output_schema:
            create_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "schema": output_schema,
                }
            }
        else:
            create_kwargs["text"] = {"format": {"type": "json_object"}}

        response = self._get_client().responses.create(**create_kwargs)
        return json.loads(response.output_text)

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        **kwargs,
    ) -> Dict:
        """Not implemented for Codex models in this initial version."""
        raise NotImplementedError("generate_with_tools is not yet supported for Codex models")

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        interrupt_controller=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Not implemented for Codex models in this initial version."""
        raise NotImplementedError("generate_with_tools_stream is not yet supported for Codex models")
        yield  # pragma: no cover — make this a generator

    def token_count(self, prompt: str) -> int:
        """Estimate token count using a simple heuristic."""
        return len(prompt) // 4

    def context_length(self) -> Optional[int]:
        """Return the context length for the current model."""
        specs = _CODEX_MODEL_SPECS.get(self.model_name)
        if specs:
            return specs["context_length"]
        return 192000  # default
