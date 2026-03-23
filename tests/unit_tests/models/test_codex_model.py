# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for Codex model."""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import ModelConfig


@pytest.fixture
def model_config():
    return ModelConfig(
        type="codex",
        api_key="",
        model="gpt-5.3-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        auth_type="oauth",
    )


@pytest.fixture
def mock_oauth():
    with patch("datus.models.codex_model.OAuthManager") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_access_token.return_value = "test_oauth_token"
        mock_instance.refresh_tokens.return_value = {"access_token": "refreshed_token"}
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestCodexModelInit:
    def test_init(self, model_config, mock_oauth):
        from datus.models.codex_model import CodexModel

        model = CodexModel(model_config=model_config)
        assert model.model_name == "gpt-5.3-codex"
        assert model._client is None  # lazy init

    def test_model_specs(self, model_config, mock_oauth):
        from datus.models.codex_model import _CODEX_MODEL_SPECS

        assert "gpt-5.3-codex" in _CODEX_MODEL_SPECS
        assert "gpt-5.1-codex-mini" in _CODEX_MODEL_SPECS
        assert "o3-codex" in _CODEX_MODEL_SPECS


class TestCodexModelGenerate:
    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_string_prompt(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Hello from Codex!"
        mock_client.responses.create.return_value = mock_response
        model._client = mock_client

        result = model.generate("Say hello")
        assert result == "Hello from Codex!"
        mock_client.responses.create.assert_called_once_with(
            model="gpt-5.3-codex",
            input="Say hello",
        )

    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_list_prompt(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Result"
        mock_client.responses.create.return_value = mock_response
        model._client = mock_client

        messages = [{"role": "user", "content": "Hi"}]
        result = model.generate(messages)
        assert result == "Result"

    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_401_retry(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Retried result"

        # First call raises 401, second succeeds
        mock_client.responses.create.side_effect = [
            Exception("401 Unauthorized"),
            mock_response,
        ]
        model._client = mock_client

        result = model.generate("test")
        assert result == "Retried result"
        mock_oauth.refresh_tokens.assert_called_once()


class TestCodexModelJsonOutput:
    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_with_json_output(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({"sql": "SELECT 1"})
        mock_client.responses.create.return_value = mock_response
        model._client = mock_client

        result = model.generate_with_json_output("Generate SQL")
        assert result == {"sql": "SELECT 1"}

    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_with_schema(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({"answer": 42})
        mock_client.responses.create.return_value = mock_response
        model._client = mock_client

        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        result = model.generate_with_json_output("test", output_schema=schema)
        assert result["answer"] == 42

        # Verify schema was passed
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["text"]["format"]["type"] == "json_schema"


class TestCodexModelUtils:
    @patch("datus.models.codex_model.OAuthManager")
    def test_token_count(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        model = CodexModel(model_config=model_config)
        # Simple heuristic: len / 4
        assert model.token_count("Hello World!") == 3

    @patch("datus.models.codex_model.OAuthManager")
    def test_context_length_known_model(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        model = CodexModel(model_config=model_config)
        assert model.context_length() == 192000

    @patch("datus.models.codex_model.OAuthManager")
    def test_context_length_unknown_model(self, mock_oauth_cls):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(type="codex", api_key="", model="unknown-codex-model", auth_type="oauth")
        model = CodexModel(model_config=config)
        assert model.context_length() == 192000  # default

    @patch("datus.models.codex_model.OAuthManager")
    def test_convert_prompt_to_input(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        assert CodexModel._convert_prompt_to_input("hello") == "hello"
        assert CodexModel._convert_prompt_to_input([{"role": "user"}]) == [{"role": "user"}]
        assert CodexModel._convert_prompt_to_input(123) == "123"
