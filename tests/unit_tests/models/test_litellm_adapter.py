# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/litellm_adapter.py

CI-level: zero external dependencies (LiteLLM / openai-agents SDK calls are mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.models.litellm_adapter import LiteLLMAdapter, create_litellm_adapter


class TestLiteLLMAdapterInit:
    def test_basic_init(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="sk-test")
        assert adapter.provider == "openai"
        assert adapter.model == "gpt-4o"
        assert adapter.api_key == "sk-test"

    def test_auto_detect_claude(self):
        adapter = LiteLLMAdapter(provider="openai", model="claude-sonnet-4", api_key="key")
        assert adapter.provider == "claude"

    def test_auto_detect_deepseek(self):
        adapter = LiteLLMAdapter(provider="openai", model="deepseek-chat", api_key="key")
        assert adapter.provider == "deepseek"

    def test_auto_detect_qwen(self):
        adapter = LiteLLMAdapter(provider="openai", model="qwen3-coder", api_key="key")
        assert adapter.provider == "qwen"

    def test_auto_detect_gemini(self):
        adapter = LiteLLMAdapter(provider="openai", model="gemini-2.5-pro", api_key="key")
        assert adapter.provider == "gemini"

    def test_auto_detect_kimi(self):
        adapter = LiteLLMAdapter(provider="openai", model="kimi-k2.5", api_key="key")
        assert adapter.provider == "kimi"

    def test_auto_detect_gpt_stays_openai(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        assert adapter.provider == "openai"

    def test_custom_base_url(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key", base_url="https://custom.api.com")
        assert adapter.base_url == "https://custom.api.com"

    def test_default_base_url_deepseek(self):
        adapter = LiteLLMAdapter(provider="deepseek", model="deepseek-chat", api_key="key")
        assert adapter.base_url == "https://api.deepseek.com"

    def test_thinking_disabled_by_default(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        assert adapter.is_thinking_model is False

    def test_thinking_enabled(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key", enable_thinking=True)
        assert adapter.is_thinking_model is True


class TestLiteLLMAdapterModelName:
    def test_openai_no_prefix(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        assert adapter.litellm_model_name == "gpt-4o"

    def test_claude_with_prefix(self):
        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="key")
        assert adapter.litellm_model_name == "anthropic/claude-sonnet-4"

    def test_deepseek_with_prefix(self):
        adapter = LiteLLMAdapter(provider="deepseek", model="deepseek-chat", api_key="key")
        assert adapter.litellm_model_name == "deepseek/deepseek-chat"

    def test_model_already_has_prefix(self):
        adapter = LiteLLMAdapter(provider="claude", model="anthropic/claude-sonnet-4", api_key="key")
        assert adapter.litellm_model_name == "anthropic/claude-sonnet-4"

    def test_litellm_model_name_cached(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        name1 = adapter.litellm_model_name
        name2 = adapter.litellm_model_name
        assert name1 is name2  # same object (cached)

    def test_unknown_provider_no_prefix(self):
        adapter = LiteLLMAdapter(provider="unknown_provider", model="my-model", api_key="key")
        assert adapter.litellm_model_name == "my-model"

    def test_gemini_prefix(self):
        adapter = LiteLLMAdapter(provider="gemini", model="gemini-2.5-pro", api_key="key")
        assert adapter.litellm_model_name == "gemini/gemini-2.5-pro"

    def test_qwen_prefix(self):
        adapter = LiteLLMAdapter(provider="qwen", model="qwen3-coder", api_key="key")
        assert adapter.litellm_model_name == "dashscope/qwen3-coder"

    def test_kimi_prefix(self):
        adapter = LiteLLMAdapter(provider="kimi", model="kimi-k2.5", api_key="key")
        assert adapter.litellm_model_name == "moonshot/kimi-k2.5"


class TestGetCompletionKwargs:
    def test_includes_model(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="sk-test")
        kwargs = adapter.get_completion_kwargs()
        assert kwargs["model"] == "gpt-4o"

    def test_includes_api_key(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="sk-test")
        kwargs = adapter.get_completion_kwargs()
        assert kwargs["api_key"] == "sk-test"

    def test_includes_base_url_as_api_base(self):
        adapter = LiteLLMAdapter(
            provider="deepseek", model="deepseek-chat", api_key="key", base_url="https://api.deepseek.com"
        )
        kwargs = adapter.get_completion_kwargs()
        assert kwargs["api_base"] == "https://api.deepseek.com"

    def test_no_base_url_key_when_absent(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key", base_url=None)
        # Override the default to ensure no base_url
        adapter.base_url = None
        kwargs = adapter.get_completion_kwargs()
        assert "api_base" not in kwargs


class TestGetAgentsSdkModel:
    def test_import_error_raised(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        with patch.dict("sys.modules", {"agents.extensions.models.litellm_model": None}):
            with patch("builtins.__import__", side_effect=ImportError("no litellm")):
                with pytest.raises(ImportError):
                    adapter.get_agents_sdk_model()

    def test_returns_litellm_model(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        mock_model = MagicMock()
        mock_litellm_model_cls = MagicMock(return_value=mock_model)
        mock_module = MagicMock()
        mock_module.LitellmModel = mock_litellm_model_cls

        with patch.dict("sys.modules", {"agents.extensions.models.litellm_model": mock_module}):
            result = adapter.get_agents_sdk_model()
        assert result is mock_model
        mock_litellm_model_cls.assert_called_once()


class TestCreateLiteLLMAdapter:
    def test_factory_function(self):
        adapter = create_litellm_adapter(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            enable_thinking=False,
        )
        assert isinstance(adapter, LiteLLMAdapter)
        assert adapter.model == "gpt-4o"

    def test_factory_with_base_url(self):
        adapter = create_litellm_adapter(
            provider="deepseek",
            model="deepseek-chat",
            api_key="key",
            base_url="https://custom.url",
        )
        assert adapter.base_url == "https://custom.url"
