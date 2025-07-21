import time
from typing import Any, Dict, List

import openai
from langsmith import traceable
from transformers import AutoTokenizer

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.client_factory import ClientFactory, ModelConfigHelper
from datus.models.common_mixins import JSONParsingMixin, MCPMixin
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
MAX_INPUT_QEN = 98000  # 98304 - buffer of ~300 tokens


class QwenModel(LLMBaseModel, JSONParsingMixin, MCPMixin):
    """Qwen model implementation."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.api_key = ModelConfigHelper.get_api_key(model_config, "QWEN_API_KEY")
        self.api_base = ModelConfigHelper.get_base_url(
            model_config, "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_config.model

        self.client = ClientFactory.create_openai_client(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        self._async_client = None

    def async_client(self):
        """Get or create async client."""
        if self._async_client is None:
            logger.debug(f"Creating async OpenAI client with base_url: {self.api_base}, model: {self.model_name}")
            try:
                self._async_client = ClientFactory.create_async_openai_client(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )
            except Exception as e:
                logger.error(f"Error creating async OpenAI client: {str(e)}.")
                raise
        return self._async_client

    @traceable
    def generate(self, prompt: Any, **kwargs) -> str:
        """Generate a response from the Qwen model."""
        base_params = {
            "model": self.model_config.model,
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0,
        }
        params = ModelConfigHelper.merge_params(base_params, **kwargs)
        messages = ModelConfigHelper.prepare_messages(prompt)

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
                if isinstance(e, (openai.RateLimitError, openai.APITimeoutError)):
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
        return final_answer

    @traceable
    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """Generate a response and ensure it conforms to the provided JSON schema."""
        response_text = self.generate(prompt, **kwargs)

        try:
            return llm_result2json(response_text)
        except Exception:
            return self.parse_json_response(response_text)

    def token_count(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens using tokenizer."""
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return len(self.tokenizer.encode(input_text))

    def max_tokens(self) -> int:
        return MAX_INPUT_QEN

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
        self.setup_json_encoder()

        def async_model_factory(**factory_kwargs):
            return ClientFactory.create_async_model(
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=self.api_base,
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

    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context."""
        pass
