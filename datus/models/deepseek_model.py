from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import yaml
from agents import set_tracing_disabled

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.client_factory import ClientFactory, ModelConfigHelper
from datus.models.common_mixins import JSONParsingMixin, MCPMixin, StreamingMixin
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
MAX_INPUT_DEEPSEEK = 52000  # 57344 - buffer of ~5000 tokens

set_tracing_disabled(True)


class DeepSeekModel(LLMBaseModel, JSONParsingMixin, MCPMixin, StreamingMixin):
    """Implementation of the BaseModel for DeepSeek's API."""

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

        self.api_key = ModelConfigHelper.get_api_key(model_config, "DEEPSEEK_API_KEY")
        self.api_base = ModelConfigHelper.get_base_url(model_config, "https://api.deepseek.com")
        self.model_name = model_config.model

        logger.debug(f"Using DeepSeek model: {self.model_name} base Url: {self.api_base}")

        self.client = ClientFactory.create_openai_client(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        # Store reference to workflow and current node for trace saving
        self.workflow = None
        self.current_node = None

    def _save_llm_trace(self, prompt: Any, response_content: str, reasoning_content: Any = None):
        """Save LLM input/output trace to YAML file if tracing is enabled."""
        if not self.model_config.save_llm_trace:
            return

        try:
            # Get workflow and node context from current execution
            if (
                not hasattr(self, "workflow")
                or not self.workflow
                or not hasattr(self, "current_node")
                or not self.current_node
            ):
                logger.debug("No workflow or node context available for trace saving")
                return

            # Create trace directory
            trajectory_dir = Path(self.workflow.global_config.trajectory_dir)
            task_id = self.workflow.task.id
            trace_dir = trajectory_dir / task_id
            trace_dir.mkdir(parents=True, exist_ok=True)

            # Parse prompt to separate system and user content
            system_prompt = ""
            user_prompt = ""

            if isinstance(prompt, list):
                # Handle message format
                for message in prompt:
                    if message.get("role") == "system":
                        system_prompt = message.get("content", "")
                    elif message.get("role") == "user":
                        user_prompt = message.get("content", "")
            else:
                # Handle string prompt
                user_prompt = str(prompt)

            # Create trace data
            trace_data = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "reason_content": reasoning_content or "",
                "output_content": response_content,
            }

            # Save to YAML file named after node ID
            trace_file = trace_dir / f"{self.current_node.id}.yml"
            with open(trace_file, "w", encoding="utf-8") as f:
                yaml.dump(trace_data, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)

            logger.debug(f"LLM trace saved to {trace_file}")

        except Exception as e:
            logger.error(f"Failed to save LLM trace: {str(e)}")

    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context for trace saving."""
        self.workflow = workflow
        self.current_node = current_node

    def generate(self, prompt: Any, **kwargs) -> str:
        """Generate a response from the DeepSeek model."""
        base_params = {
            "model": self.model_name,
            "temperature": 0.7,
            "max_tokens": 5000,
            "top_p": 1.0,
        }
        params = ModelConfigHelper.merge_params(base_params, **kwargs)
        messages = ModelConfigHelper.prepare_messages(prompt)

        # Call the OpenAI API
        response = self.client.chat.completions.create(messages=messages, **params)

        # Get response content
        response_content = response.choices[0].message.content

        # Check for reasoning content (for deepseek-reasoner and similar models)
        reasoning_content = None
        if hasattr(response.choices[0].message, "reasoning_content"):
            reasoning_content = response.choices[0].message.reasoning_content
        elif hasattr(response, "reasoning_content"):
            reasoning_content = response.reasoning_content

        # Save trace if enabled
        self._save_llm_trace(prompt, response_content, reasoning_content)

        # Log the response
        logger.debug(f"Model response: {response_content}")

        return response_content

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """Generate a response and ensure it conforms to the provided JSON schema."""
        response_text = self.generate(prompt, response_format={"type": "json_object"}, **kwargs)
        # Note: trace is already saved in self.generate() call above
        return self.parse_json_response(response_text)

    def generate_with_tools(self, prompt: str, tools: list, **kwargs) -> Dict:
        """Generate with tools - placeholder for future implementation."""
        # TODO: Implement tool-based generation
        return {"content": "Tool-based generation not implemented yet"}

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: dict,
        instruction: str,
        output_type: type[Any],
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate a response using multiple MCP servers."""

        def async_model_factory(**factory_kwargs):
            return ClientFactory.create_async_model(
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=self.api_base,
            )

        try:
            # DeepSeek doesn't support structured output, force to str
            result = await self.generate_with_mcp_base(
                prompt=prompt,
                mcp_servers=mcp_servers,
                instruction=instruction,
                output_type=str,  # Force to str for DeepSeek
                max_turns=max_turns,
                async_model_factory=async_model_factory,
                **kwargs,
            )

            # Create reasoning content from the full interaction list
            reasoning_content = None
            # Additional tracing logic would go here if needed

            self._save_llm_trace(
                prompt=prompt,
                response_content=result["content"],
                reasoning_content=reasoning_content,
            )
            return result
        except Exception as e:
            logger.error(f"Error in run_agent: {str(e)}")
            # Save trace even on error
            self._save_llm_trace(
                prompt=f"Instruction: {instruction}\n\nUser Prompt: {prompt}",
                response_content=f"ERROR: {str(e)}",
                reasoning_content="",
            )
            raise

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
        """Generate a response using multiple MCP servers with streaming support."""

        def async_model_factory(**factory_kwargs):
            return ClientFactory.create_async_model(
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=self.api_base,
            )

        # DeepSeek doesn't support structured output, force to str
        async for action in self.generate_with_mcp_stream_base(
            prompt=prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=str,  # Force to str for DeepSeek
            max_turns=max_turns,
            action_history_manager=action_history_manager,
            async_model_factory=async_model_factory,
            **kwargs,
        ):
            yield action

    def token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in a text using DeepSeek tokenizer."""
        return int(len(prompt) * 0.3 + 0.5)

    def max_tokens(self) -> int:
        return MAX_INPUT_DEEPSEEK
