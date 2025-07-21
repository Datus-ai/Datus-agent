from typing import Any, AsyncGenerator, Dict, Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.client_factory import ClientFactory, ModelConfigHelper
from datus.models.common_mixins import JSONParsingMixin, MCPMixin, StreamingMixin
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OpenAIModel(LLMBaseModel, JSONParsingMixin, MCPMixin, StreamingMixin):
    """
    Implementation of the BaseModel for OpenAI's API.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

        self.api_key = ModelConfigHelper.get_api_key(model_config, "OPENAI_API_KEY")
        self.model_name = model_config.model

        # Initialize OpenAI client
        self.client = ClientFactory.create_openai_client(self.api_key)

        # Store reference to workflow and current node for trace saving
        self.workflow = None
        self.current_node = None

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the OpenAI model.
        """
        base_params = {
            "model": self.model_name,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
        }
        params = ModelConfigHelper.merge_params(base_params, **kwargs)
        messages = ModelConfigHelper.prepare_messages(prompt)

        response = self.client.chat.completions.create(messages=messages, **params)
        return response.choices[0].message.content

    def generate_with_json_output(self, prompt: Any, json_schema: Dict = None, **kwargs) -> Dict:
        """
        Generate a response and ensure it conforms to the provided JSON schema.
        """
        params = {**kwargs, "response_format": {"type": "json_object"}}
        response_text = self.generate(prompt, **params)
        return self.parse_json_response(response_text)

    def generate_with_tools(self, prompt: str, tools: list, **kwargs) -> Dict:
        """Generate a response using tools."""
        # TODO: Implement tool-based generation
        return {"content": "Tool-based generation not implemented yet"}

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: dict,
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate a response using multiple MCP servers."""

        def async_model_factory(**factory_kwargs):
            return ClientFactory.create_async_model(
                model_name=self.model_name,
                api_key=self.api_key,
            )

        return await self.generate_with_mcp_base(
            prompt=prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            async_model_factory=async_model_factory,
            **kwargs,
        )

    async def generate_with_mcp_stream(
        self,
        prompt: str,
        mcp_servers: dict,
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate a response using multiple MCP servers with streaming."""

        def async_model_factory(**factory_kwargs):
            return ClientFactory.create_async_model(
                model_name=self.model_name,
                api_key=self.api_key,
            )

        async for action in self.generate_with_mcp_stream_base(
            prompt=prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            action_history_manager=action_history_manager,
            async_model_factory=async_model_factory,
            **kwargs,
        ):
            yield action

    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context for trace saving."""
        self.workflow = workflow
        self.current_node = current_node

    def token_count(self, prompt: str) -> int:
        """Count the number of tokens in the given prompt."""
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model_name)
            tokens = encoding.encode(prompt)
            return len(tokens)
        except Exception:
            return len(prompt) // 4
