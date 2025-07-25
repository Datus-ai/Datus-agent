import json
import os
import time
from datetime import date, datetime
from typing import Any, Dict, List

import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI
from pydantic import AnyUrl
from transformers import AutoTokenizer

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
MAX_INPUT_QEN = 98000  # 98304 - buffer of ~300 tokens


class QwenModel(OpenAICompatibleModel):
    def __init__(self, model_config: ModelConfig):
        """
        Initialize the Qwen model.

        Args:
            model_config: Model configuration object
        """
        super().__init__(model_config)
        # Initialize Qwen-specific tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        self._async_client = None
    
    def _get_api_key(self) -> str:
        """Get Qwen API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise ValueError("Qwen API key must be provided or set as QWEN_API_KEY environment variable")
        return api_key
    
    def _get_base_url(self) -> str:
        """Get Qwen base URL from config or environment."""
        return self.model_config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def async_client(self):
        if self._async_client is None:
            logger.debug(f"Creating async OpenAI client with base_url: {self.api_base}, model: {self.model_name}")

            async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
            try:
                self._async_client = wrap_openai(async_client)
            except Exception as e:
                logger.error(f"Error wrapping async OpenAI client: {str(e)}. Use the original client.")
                self._async_client = async_client
        return self._async_client


    def token_count(self, messages: List[Dict[str, str]]) -> int:
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        return len(self.tokenizer.encode(input_text))

    def max_tokens(self) -> int:
        return MAX_INPUT_QEN

