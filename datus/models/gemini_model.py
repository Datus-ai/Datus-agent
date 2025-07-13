import json
import os
from datetime import date, datetime
from typing import Any, Dict

import google.generativeai as genai
from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GeminiModel(LLMBaseModel):
    """Google Gemini model implementation"""

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

        self.api_key = model_config.api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_config.model
        if not self.api_key:
            raise ValueError("Gemini API key must be provided or set as GEMINI_API_KEY environment variable")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.workflow = None
        self.current_node = None

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.7),
                max_output_tokens=kwargs.get("max_tokens", 1000),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 40),
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=kwargs.get("safety_settings", None),
            )

            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                logger.warning("No candidates returned from Gemini model")
                return ""

        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            raise

    def generate_with_json_output(self, prompt: Any, json_schema: Dict = None, **kwargs) -> Dict:
        if json_schema:
            json_prompt = (
                f"{prompt}\n\nRespond with a JSON object that conforms to the following schema:\n"
                f"{json.dumps(json_schema, indent=2)}"
            )
        else:
            json_prompt = f"{prompt}\n\nRespond with a valid JSON object."

        response_text = self.generate(json_prompt, **kwargs)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON response: {response_text}")
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response_text,
            }

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        try:
            logger.debug(f"Creating async OpenAI client for Gemini model: {self.model_name}")

            base_url = kwargs.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai")
            async_client = wrap_openai(
                AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=base_url,
                )
            )

            model_params = {"model": self.model_name}
            async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

            logger.debug("Starting run_agent with Gemini via OpenAI compatibility")

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
                return {
                    "content": result.final_output,
                    "sql_contexts": extract_sql_contexts(result),
                }
        except Exception as e:
            logger.error(f"Error in run_agent with Gemini: {str(e)}")
            logger.warning("MCP execution failed, falling back to basic generation")
            basic_response = self.generate(f"{instruction}\n\n{prompt}", **kwargs)
            return {
                "content": basic_response,
                "sql_contexts": [],
            }

    def set_context(self, workflow=None, current_node=None):
        self.workflow = workflow
        self.current_node = current_node

    def token_count(self, prompt: str) -> int:
        try:
            model = genai.GenerativeModel(self.model_name)
            token_count = model.count_tokens(prompt)
            return token_count.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens with Gemini: {str(e)}")
            return len(prompt) // 4
