import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.mcp import MCPServerStdio
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
MAX_INPUT_DEEPSEEK = 52000  # 57344 - buffer of ~5000 tokens

set_tracing_disabled(True)


class DeepSeekModel(LLMBaseModel):
    """
    Implementation of the BaseModel for DeepSeek's API.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)

        self.api_key = model_config.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key must be provided or set as DEEPSEEK_API_KEY environment variable")

        self.api_base = model_config.base_url or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.model_name = model_config.model
        # Initialize OpenAI client and langsmith wrapper
        logger.debug(f"Using DeepSeek model: {self.model_name} base Url: {self.api_base}")
        from langsmith.wrappers import wrap_openai

        self.client = wrap_openai(OpenAI(api_key=self.api_key, base_url=self.api_base))

        # Store reference to workflow and current node for trace saving
        self.workflow = None
        self.current_node = None

    def _save_llm_trace(self, prompt: Any, response_content: str, reasoning_content: Any = None):
        """Save LLM input/output trace to YAML file if tracing is enabled.

        Args:
            prompt: The input prompt (str or list of messages)
            response_content: The response content from the model
            reasoning_content: Optional reasoning content for reasoning models
        """
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
                # Handle message format like [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                for message in prompt:
                    if message.get("role") == "system":
                        system_prompt = message.get("content", "")
                    elif message.get("role") == "user":
                        user_prompt = message.get("content", "")
            else:
                # Handle string prompt - put it all in user_prompt
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
        """Set workflow and node context for trace saving.

        Args:
            workflow: Current workflow instance
            current_node: Current node instance
        """
        self.workflow = workflow
        self.current_node = current_node

    def generate(self, prompt: Any, **kwargs) -> str:
        """Generate a response from the DeepSeek model.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Merge default parameters with any provided kwargs
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 5000),
            "top_p": 1.0,
            **kwargs,
        }

        # Create messages format expected by OpenAI
        if type(prompt) is list:
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

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
        """
        Generate a response and ensure it conforms to the provided JSON schema.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            A dictionary representing the JSON response
        """
        # Add instructions to format the response as JSON according to the schema
        # json_prompt = f"{prompt}\n\nRespond with a JSON object that conforms
        # to the following schema:\n{json.dumps(json_schema, indent=2)}"

        # Generate the response
        response_text = self.generate(prompt, response_format={"type": "json_object"}, **kwargs)

        # Parse the JSON response
        try:
            json_result = json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    json_result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    json_result = {}
            else:
                json_result = {}

        # Note: trace is already saved in self.generate() call above
        return json_result

    def generate_with_tools(self, prompt: str, tools: List[Any], **kwargs) -> Dict:
        # flow control and context cache here
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

        # Custom JSON encoder to handle special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        # Initialize reasoning content list to track the entire MCP conversation process
        reasoning_steps = []

        # Create async OpenAI client
        logger.debug(f"Creating async OpenAI client with base_url: {self.api_base}, model: {self.model_name}")
        async_client = wrap_openai(
            AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        )

        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

        # Define the agent instructions
        logger.debug("Starting run_agent")
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

                result = await Runner.run(agent, input=prompt, max_turns=max_turns)

                logger.info(f"deepseek mcp run Result: {result}")
                # Build the result
                final_result = {
                    "content": result.final_output,
                    "sql_contexts": extract_sql_contexts(result),
                }

                # Create reasoning content from the full interaction list
                reasoning_content = None
                if hasattr(result, "to_input_list"):
                    try:
                        # Pass the raw list of interactions
                        reasoning_content = result.to_input_list()
                    except Exception as e:
                        logger.error(f"Error getting reasoning content list: {e}")
                        # Fallback to a simple string representation
                        reasoning_content = str(result.to_input_list())

                logger.debug(f"Reasoning content: {reasoning_content}")
                self._save_llm_trace(
                    prompt=prompt,
                    response_content=result.final_output,
                    reasoning_content=reasoning_content,
                )
                return {
                    "content": result.final_output,
                    "sql_contexts": extract_sql_contexts(result),
                }
            return final_result
        except Exception as e:
            logger.error(f"Error in run_agent: {str(e)}")
            reasoning_steps.append("=== Error Occurred ===")
            reasoning_steps.append(f"Error: {str(e)}")

            # Save trace even on error
            full_reasoning_content = "\n".join(reasoning_steps)
            self._save_llm_trace(
                prompt=f"Instruction: {instruction}\n\nUser Prompt: {prompt}",
                response_content=f"ERROR: {str(e)}",
                reasoning_content=full_reasoning_content,
            )
            raise

    def token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in a text using the deepseek tokenizer.

        Args:
            prompt (str): The text to count the tokens of.

        Returns:
            int: The number of tokens in the text.
        """
        return int(len(prompt) * 0.3 + 0.5)

    def max_tokens(self) -> int:
        return MAX_INPUT_DEEPSEEK
