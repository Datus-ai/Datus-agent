# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/claude_proxy_model.py.

CI-level: zero external dependencies. All I/O mocked.
"""

from unittest.mock import MagicMock, patch

from datus.models.claude_proxy_model import ClaudeProxyModel


def _make_model_config(
    model="claude-sonnet-4",
    api_key="",
    base_url=None,
    temperature=None,
    top_p=None,
    enable_thinking=False,
):
    cfg = MagicMock()
    cfg.model = model
    cfg.type = "claude_proxy"
    cfg.api_key = api_key
    cfg.base_url = base_url
    cfg.use_native_api = False
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.enable_thinking = enable_thinking
    cfg.default_headers = {}
    cfg.max_retry = 3
    cfg.retry_interval = 0.0
    cfg.strict_json_schema = True
    cfg.auth_type = "api_key"
    return cfg


def _make_proxy_model(model_config=None):
    if model_config is None:
        model_config = _make_model_config()

    mock_litellm_adapter = MagicMock()
    mock_litellm_adapter.litellm_model_name = "claude-sonnet-4"
    mock_litellm_adapter.provider = "claude_proxy"
    mock_litellm_adapter.is_thinking_model = False
    mock_litellm_adapter.get_agents_sdk_model.return_value = MagicMock()

    with (
        patch("datus.models.openai_compatible.setup_tracing"),
        patch("datus.models.openai_compatible.LiteLLMAdapter", return_value=mock_litellm_adapter),
    ):
        model = ClaudeProxyModel(model_config)
        model.litellm_adapter = mock_litellm_adapter
        return model


class TestClaudeProxyModelInit:
    def test_model_name_set(self):
        model = _make_proxy_model()
        assert model.model_name == "claude-sonnet-4"

    def test_api_key_fallback_not_needed(self):
        cfg = _make_model_config(api_key="")
        model = _make_proxy_model(cfg)
        assert model.api_key == "not-needed"

    def test_api_key_from_config(self):
        cfg = _make_model_config(api_key="custom-key")
        model = _make_proxy_model(cfg)
        assert model.api_key == "custom-key"

    def test_api_key_from_env(self):
        cfg = _make_model_config(api_key="")
        with patch.dict("os.environ", {"CLAUDE_PROXY_API_KEY": "env-key"}):
            mock_litellm_adapter = MagicMock()
            mock_litellm_adapter.litellm_model_name = "claude-sonnet-4"
            mock_litellm_adapter.provider = "claude_proxy"
            mock_litellm_adapter.is_thinking_model = False
            mock_litellm_adapter.get_agents_sdk_model.return_value = MagicMock()

            with (
                patch("datus.models.openai_compatible.setup_tracing"),
                patch("datus.models.openai_compatible.LiteLLMAdapter", return_value=mock_litellm_adapter),
            ):
                model = ClaudeProxyModel(cfg)
            assert model.api_key == "env-key"


class TestClaudeProxyModelBaseUrl:
    def test_default_base_url(self):
        cfg = _make_model_config(base_url=None)
        model = _make_proxy_model(cfg)
        assert model._get_base_url() == "http://localhost:3456/v1"

    def test_custom_base_url(self):
        cfg = _make_model_config(base_url="http://myproxy:8080/v1")
        model = _make_proxy_model(cfg)
        assert model.base_url == "http://myproxy:8080/v1"


class TestClaudeProxyModelGenerate:
    def test_generate_calls_parent(self):
        model = _make_proxy_model()
        with patch(
            "datus.models.openai_compatible.OpenAICompatibleModel.generate", return_value="proxy response"
        ) as mock_gen:
            result = model.generate("hello")
        mock_gen.assert_called_once()
        assert result == "proxy response"
