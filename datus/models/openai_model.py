import json
import os
from datetime import date, datetime
from typing import Any, Dict, List

from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
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

    def token_count(self, prompt: str) -> int:
        """
        Count the number of tokens in the given prompt using tiktoken.

        Args:
            prompt: The input text to count tokens for

        Returns:
            The number of tokens in the prompt
        """
        try:
            # Use OpenAI's tiktoken library for token counting
            import tiktoken

            # Get the encoding for the model
            encoding = tiktoken.encoding_for_model(self.model_name)

            # Count tokens
            tokens = encoding.encode(prompt)
            return len(tokens)
        except ImportError:
            # Fallback: rough estimation (1 token ≈ 4 characters for English text)
            return len(prompt) // 4
        except Exception:
            # Fallback: rough estimation if model encoding is not found
            return len(prompt) // 4
