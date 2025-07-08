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
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
MAX_INPUT_QEN = 98000  # 98304 - buffer of ~300 tokens


class QwenModel(LLMBaseModel):
    def __init__(self, model_config: ModelConfig):
        """
        Initialize the Qwen model.

        Args:
            model_name: The specific Qwen model to use
            api_key: Qwen API key. If not provided, will look for QWEN_API_KEY env variable
            api_base: Base URL for the Qwen API. If not provided, will use the default
            **kwargs: Additional parameters for the Qwen API
        """
        super().__init__(model_config)

        # Use provided API key or get from environment
        self.api_key = model_config.api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("Qwen API key must be provided or set as QWEN_API_KEY environment variable")

        # Set API base URL
        self.api_base = model_config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = wrap_openai(OpenAI(api_key=self.api_key, base_url=self.api_base))
        self.model_name = model_config.model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        self._async_client = None

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

    @traceable
    def generate(self, prompt: Any, **kwargs) -> str:
        """
        Generate a response from the Qwen model.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Merge default parameters with any provided kwargs
        params = {
            "model": self.model_config.model,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "top_p": 1.0,
            **kwargs,
        }

        # Create messages format expected by Qwen
        if type(prompt) is list:
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        logger.debug(f"params: {params}")

        chunks = []
        is_answering = False
        is_thinking = False
        for _ in range(3):
            try:
                completion = self.client.chat.completions.create(
                    messages=messages,
                    stream=True,
                    response_format={"type": "text"},
                    **params,
                )
                break
            except Exception as e:
                logger.warning(f"Match schema failed: {str(e)}")
                if isinstance(e, openai.RateLimitError) or isinstance(e, openai.APITimeoutError):
                    time.sleep(700)
                    continue

                raise e
        logger.debug("=" * 20 + "QWEN generate start " + "=" * 20)

        for chunk in completion:
            # if chunk.choices is empty, print usage
            if not chunk.choices:
                logger.debug("\nUsage:")
                logger.debug(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # print thinking process
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if not is_thinking:
                        logger.debug("thinking:")
                        is_thinking = True
                    if is_thinking:
                        print(f"{delta.reasoning_content}", end="")
                else:
                    # start answering
                    if delta.content != "" and is_answering is False:
                        logger.debug("\n" + "=" * 20 + "complete answer" + "=" * 20 + "\n")
                        is_answering = True
                    # print answering process
                    print(delta.content, end="")
                    chunks.append(delta.content)
        if not chunks:
            raise ValueError("No answer content from LLM")

        final_answer = "".join(chunks)
        logger.debug(final_answer)
        logger.debug("=" * 20 + "QWEN generate end " + "=" * 20)
        # Extract and return the generated text
        return final_answer

    @traceable
    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a response and ensure it conforms to the provided JSON schema.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            A dictionary representing the JSON response
        """
        # Generate the response
        response_text = self.generate(prompt, **kwargs)

        # Parse the JSON response
        try:
            return llm_result2json(response_text)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            return {}

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

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate a response using multiple MCP (Machine Conversation Protocol) servers.

        Args:
            prompt: The input prompt to send to the model
            mcp_servers: Dictionary of MCP servers to use for execution
            instruction: The instruction for the agent
            output_type: The type of output expected from the agent
            max_turns: Maximum number of conversation turns
            **kwargs: Additional parameters for the agent

        Returns:
            The result from the MCP agent execution with content and sql_contexts
        """

        # Custom JSON encoder to handle special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        async_client = self.async_client()
        model_params = {"model": self.model_name}

        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

        # Define the agent instructions
        logger.debug("Starting run_agent")
        from agents.models.openai_chatcompletions import ResponseTextDeltaEvent

        try:
            # Use context manager to manage multiple MCP servers
            from datus.models.mcp_utils import multiple_mcp_servers

            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                logger.debug("MCP servers started successfully")

                agent = Agent(
                    name=kwargs.pop("agent_name", "MCP_Agent"),
                    instructions=instruction,
                    mcp_servers=list(connected_servers.values()),
                    output_type=output_type,
                    model=async_model,
                )
                logger.debug(f"Agent created with name: {agent.name}, {output_type}")

                logger.debug(f"Running agent with max_turns: {max_turns}")
                result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns)
                while not result.is_complete:
                    async for event in result.stream_events():
                        # FIXME print temp
                        if event.type == "raw_response_event":
                            if isinstance(event.data, ResponseTextDeltaEvent):
                                print(event.data.delta, end="")

                logger.debug("Agent execution completed")
                # Wrap in object so .content and .sql_contexts are accessible
                return {
                    "content": result.final_output,
                    "sql_contexts": extract_sql_contexts(result),
                }
        except Exception as e:
            logger.error(f"Error in run_agent: {str(e)}")
            raise

    def set_context(self, workflow=None, current_node=None):
        pass
