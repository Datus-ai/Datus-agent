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
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OpenAIModel(LLMBaseModel):
    """
    Implementation of the BaseModel for OpenAI's API.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        """
        Initialize the OpenAI model.

        Args:
            model_name: The specific OpenAI model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo')
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env variable
            **kwargs: Additional parameters for the OpenAI API
        """
        super().__init__(model_config, **kwargs)

        # Store model config for later use
        self.model_config = model_config

        # Use provided API key or get from environment
        self.api_key = model_config.api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_config.model
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if model_config.base_url:
            client_kwargs["base_url"] = model_config.base_url
        self.client = OpenAI(**client_kwargs)

        # Store reference to workflow and current node for trace saving
        self.workflow = None
        self.current_node = None

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the OpenAI model.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Merge default parameters with any provided kwargs
        params = {
            "model": self.model_name,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            **kwargs,
        }

        # Create messages format expected by OpenAI
        from openai.types.chat import ChatCompletionMessageParam

        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

        # Call the OpenAI API
        response = self.client.chat.completions.create(messages=messages, **params)

        # Extract and return the generated text
        return response.choices[0].message.content

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a response and ensure it conforms to the provided JSON schema.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            A dictionary representing the JSON response
        """
        # Add instructions to format the response as JSON according to the schema
        json_prompt = f"{prompt}\n\nRespond with a JSON object that conforms to the following schema:\n"

        # Set response format to JSON
        params = {**kwargs, "response_format": {"type": "json_object"}}

        # Generate the response
        response_text = self.generate(json_prompt, **params)

        # Parse the JSON response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            # Return a basic error response if all parsing attempts fail
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response_text,
            }

    def generate_with_tools(self, prompt: str, tools: List[Any], **kwargs) -> Dict:
        """Generate a response using tools.

        Args:
            prompt: The input prompt to send to the model
            tools: List of tools to use
            **kwargs: Additional generation parameters

        Returns:
            A dictionary containing the response
        """
        # TODO: Implement tool-based generation
        return {"content": "Tool-based generation not implemented yet"}

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

        # Create async OpenAI client
        logger.debug(f"Creating async OpenAI client with model: {self.model_name}")
        async_client_kwargs = {"api_key": self.api_key}
        if self.model_config.base_url:
            async_client_kwargs["base_url"] = self.model_config.base_url
        async_client = wrap_openai(AsyncOpenAI(**async_client_kwargs))

        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

        logger.debug("Starting run_agent with OpenAI")
        try:
            # Use context manager to manage multiple MCP servers
            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                logger.debug("MCP servers started successfully")

                agent = Agent(
                    name=kwargs.pop("agent_name", "MCP_Agent"),
                    instructions=instruction,
                    mcp_servers=list(connected_servers.values()),
                    output_type=output_type,
                    model=async_model,
                )
                logger.debug(f"Agent created with name: {agent.name}")

                logger.debug(f"Running agent with max_turns: {max_turns}")
                result = await Runner.run(agent, input=prompt, max_turns=max_turns)

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
        """Set workflow and node context for trace saving.

        Args:
            workflow: Current workflow instance
            current_node: Current node instance
        """
        self.workflow = workflow
        self.current_node = current_node

    def token_count(self, prompt: str) -> int:
        """
        Count the number of tokens in the given prompt.

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
