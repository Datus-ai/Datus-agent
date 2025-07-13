import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict

from agents.mcp import MCPServerStdio

from datus.configuration.agent_config import AgentConfig, ModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LLMBaseModel(ABC):  # Changed from BaseModel to LLMBaseModel
    """
    Abstract base class for all language model implementations.
    Provides a common interface for different LLM providers.
    """

    MODEL_TYPE_MAP: ClassVar[Dict[str, str]] = {
        "deepseek": "DeepSeekModel",
        "qwen": "QwenModel",
        "openai": "OpenAIModel",
        "claude": "ClaudeModel",
        "gemini": "GeminiModel",
    }

    def __init__(self, model_config: ModelConfig):
        """Initialize model with configuration and parameters"""
        self.model_config = model_config  # Model configuration

    @classmethod
    def create_model(cls, agent_config: AgentConfig, model_name: str = None, **kwargs) -> "LLMBaseModel":
        if not model_name or model_name == "default":
            target_config = agent_config.active_model()
        elif model_name in agent_config.models:
            target_config = agent_config.model_config(model_name)
        else:
            raise KeyError(f"Model {model_name} not found in agent_config")

        model_type = target_config.type

        if (model_class_name := cls.MODEL_TYPE_MAP.get(model_type)) is None:
            raise KeyError(f"Unsupported model type: {model_type}")

        module = __import__(f"datus.models.{model_type}_model", fromlist=[model_class_name])
        model_class = getattr(module, model_class_name)

        return model_class(model_config=target_config)

    @abstractmethod
    def generate(self, prompt: Any, **kwargs) -> str:
        """
        Generate a response from the language model.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """

    @abstractmethod
    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a response and ensure it conforms to the provided JSON schema.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            A dictionary representing the JSON response
        """

    def to_dict(self) -> Dict[str, str]:
        return {"model_name": self.model_config.model}

    @abstractmethod
    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context for potential trace saving.

        Args:
            workflow: Current workflow instance
            current_node: Current node instance

        Note:
            This is a default implementation. Subclasses can override this
            method to implement specific tracing functionality.
        """

    @abstractmethod
    def token_count(self, prompt: str) -> int:
        pass

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type[Any],
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
        return {}
