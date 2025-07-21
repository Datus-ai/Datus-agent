import json
from typing import Any, Dict

import google.generativeai as genai

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.client_factory import ClientFactory, ModelConfigHelper
from datus.models.common_mixins import JSONParsingMixin, MCPMixin
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GeminiModel(LLMBaseModel, JSONParsingMixin, MCPMixin):
    """Google Gemini model implementation"""

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config, **kwargs)

        self.api_key = ModelConfigHelper.get_api_key(model_config, "GEMINI_API_KEY")
        self.model_name = model_config.model

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.workflow = None
        self.current_node = None

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini."""
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
        """Generate JSON response."""
        if json_schema:
            json_prompt = (
                f"{prompt}\n\nRespond with a JSON object that conforms to the following schema:\n"
                f"{json.dumps(json_schema, indent=2)}"
            )
        else:
            json_prompt = f"{prompt}\n\nRespond with a valid JSON object."

        response_text = self.generate(json_prompt, **kwargs)
        return self.parse_json_response(response_text)

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: dict,
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate using MCP with fallback to basic generation."""
        self.setup_json_encoder()

        try:
            logger.debug(f"Creating async OpenAI client for Gemini model: {self.model_name}")

            base_url = kwargs.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai")

            def async_model_factory(**factory_kwargs):
                return ClientFactory.create_async_model(
                    model_name=self.model_name,
                    api_key=self.api_key,
                    base_url=base_url,
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
        except Exception as e:
            logger.error(f"Error in run_agent with Gemini: {str(e)}")
            logger.warning("MCP execution failed, falling back to basic generation")
            basic_response = self.generate(f"{instruction}\n\n{prompt}", **kwargs)
            return {
                "content": basic_response,
                "sql_contexts": [],
            }

    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context."""
        self.workflow = workflow
        self.current_node = current_node

    def token_count(self, prompt: str) -> int:
        """Count tokens using Gemini."""
        try:
            model = genai.GenerativeModel(self.model_name)
            token_count = model.count_tokens(prompt)
            return token_count.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens with Gemini: {str(e)}")
            return len(prompt) // 4
