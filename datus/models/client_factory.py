import os
from typing import Any, Dict, Optional

import httpx
from agents import OpenAIChatCompletionsModel
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI

from datus.configuration.agent_config import ModelConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ClientFactory:
    """Factory for creating standardized API clients."""

    @staticmethod
    def create_openai_client(
        api_key: str,
        base_url: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ) -> OpenAI:
        """Create OpenAI client with optional proxy support."""
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        if proxy_url:
            proxy_client = httpx.Client(
                transport=httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url)),
                timeout=60.0,
            )
            client_kwargs["http_client"] = proxy_client

        return wrap_openai(OpenAI(**client_kwargs))

    @staticmethod
    def create_async_openai_client(
        api_key: str,
        base_url: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ) -> AsyncOpenAI:
        """Create async OpenAI client with optional proxy support."""
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        if proxy_url:
            proxy_client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(proxy=httpx.Proxy(url=proxy_url)),
                timeout=60.0,
            )
            client_kwargs["http_client"] = proxy_client

        return wrap_openai(AsyncOpenAI(**client_kwargs))

    @staticmethod
    def create_async_model(
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ) -> OpenAIChatCompletionsModel:
        """Create async OpenAI chat completions model."""
        async_client = ClientFactory.create_async_openai_client(
            api_key=api_key,
            base_url=base_url,
            proxy_url=proxy_url,
        )

        return OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=async_client,
        )


class ModelConfigHelper:
    """Helper for processing model configurations."""

    @staticmethod
    def get_api_key(model_config: ModelConfig, env_var: str) -> str:
        """Get API key from config or environment."""
        api_key = model_config.api_key or os.environ.get(env_var)
        if not api_key:
            raise ValueError(f"API key must be provided or set as {env_var} environment variable")
        return api_key

    @staticmethod
    def get_base_url(model_config: ModelConfig, default_url: Optional[str] = None) -> Optional[str]:
        """Get base URL from config or default."""
        return model_config.base_url or default_url

    @staticmethod
    def get_proxy_url() -> Optional[str]:
        """Get proxy URL from environment."""
        return os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")

    @staticmethod
    def prepare_messages(prompt: Any) -> list:
        """Prepare messages in OpenAI format."""
        if isinstance(prompt, list):
            return prompt
        else:
            return [{"role": "user", "content": str(prompt)}]

    @staticmethod
    def merge_params(base_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Merge base parameters with kwargs."""
        params = base_params.copy()
        params.update(kwargs)
        return params
