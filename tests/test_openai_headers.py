from unittest.mock import patch

from openai import OpenAI

from datus.configuration.agent_config import ModelConfig, load_model_config
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.traceable_utils import create_openai_client


def test_model_config_headers():
    config_data = {
        "type": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "model": "gpt-4",
        "default_headers": {"X-Custom-Header": "test-value"},
    }
    config = load_model_config(config_data)
    assert config.default_headers == {"X-Custom-Header": "test-value"}


def test_create_openai_client_headers():
    headers = {"X-Custom-Header": "test-value"}
    client = create_openai_client(OpenAI, "test-key", "https://api.openai.com/v1", default_headers=headers)
    # client.default_headers contains standard headers too, so we check if our custom header is present
    assert client.default_headers["X-Custom-Header"] == "test-value"


class ConcreteOpenAIModel(OpenAICompatibleModel):
    def _get_api_key(self) -> str:
        return "test-key"


def test_openai_compatible_model_headers():
    config = ModelConfig(
        type="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4",
        default_headers={"X-Custom-Header": "test-value"},
    )

    with patch("datus.models.openai_compatible.create_openai_client") as mock_create_client:
        model = ConcreteOpenAIModel(config)
        mock_create_client.assert_called_with(
            OpenAI, "test-key", "https://api.openai.com/v1", default_headers={"X-Custom-Header": "test-value"}
        )
        assert model.default_headers == {"X-Custom-Header": "test-value"}
