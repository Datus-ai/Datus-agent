# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for OpenAIModel, QwenModel, KimiModel, GeminiModel.

CI-level: zero external deps, zero network, zero API keys.
"""

from unittest.mock import patch

import pytest

from datus.configuration.agent_config import ModelConfig


def _make_model_config(model="gpt-4o", api_key="test-key", base_url=None):
    return ModelConfig(
        type="openai",
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# OpenAIModel
# ---------------------------------------------------------------------------


class TestOpenAIModel:
    def _make(self, model_name="gpt-4o", api_key="test-key"):
        from datus.models.openai_model import OpenAIModel

        config = _make_model_config(model=model_name, api_key=api_key)
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = OpenAIModel.__new__(OpenAIModel)
            instance.model_config = config
            instance.model_name = model_name
            instance.base_url = None
        return instance

    def test_get_api_key_from_config(self):
        node = self._make(api_key="my-openai-key")
        assert node._get_api_key() == "my-openai-key"

    def test_get_api_key_from_env(self, monkeypatch):
        from datus.models.openai_model import OpenAIModel

        config = _make_model_config(api_key="")
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = OpenAIModel.__new__(OpenAIModel)
            instance.model_config = config
            instance.model_name = "gpt-4o"
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert instance._get_api_key() == "env-key"

    def test_get_api_key_raises_when_missing(self, monkeypatch):
        from datus.models.openai_model import OpenAIModel

        config = _make_model_config(api_key="")
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = OpenAIModel.__new__(OpenAIModel)
            instance.model_config = config
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            instance._get_api_key()

    def test_uses_completion_tokens_o1(self):
        node = self._make(model_name="o1-mini")
        assert node._uses_completion_tokens_parameter() is True

    def test_uses_completion_tokens_o3(self):
        node = self._make(model_name="o3-mini")
        assert node._uses_completion_tokens_parameter() is True

    def test_uses_completion_tokens_o4(self):
        node = self._make(model_name="o4-mini")
        assert node._uses_completion_tokens_parameter() is True

    def test_uses_completion_tokens_gpt_o4(self):
        node = self._make(model_name="gpt-o4-mini")
        assert node._uses_completion_tokens_parameter() is True

    def test_does_not_use_completion_tokens_gpt4(self):
        node = self._make(model_name="gpt-4o")
        assert node._uses_completion_tokens_parameter() is False

    def test_generate_transforms_max_tokens_for_reasoning(self):
        node = self._make(model_name="o1-mini")
        with patch(
            "datus.models.openai_compatible.OpenAICompatibleModel.generate", return_value="response"
        ) as mock_gen:
            node.generate("prompt", max_tokens=1000)
        call_kwargs = mock_gen.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 1000

    def test_generate_removes_unsupported_params_for_reasoning(self):
        node = self._make(model_name="o1-mini")
        with patch(
            "datus.models.openai_compatible.OpenAICompatibleModel.generate", return_value="response"
        ) as mock_gen:
            node.generate("prompt", temperature=0.7, top_p=0.9, max_tokens=500)
        call_kwargs = mock_gen.call_args[1]
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs

    def test_generate_passes_through_for_normal_model(self):
        node = self._make(model_name="gpt-4o")
        with patch(
            "datus.models.openai_compatible.OpenAICompatibleModel.generate", return_value="response"
        ) as mock_gen:
            node.generate("prompt", temperature=0.5)
        call_kwargs = mock_gen.call_args[1]
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_generate_sets_temperature_1_for_gpt5(self):
        node = self._make(model_name="gpt-5-turbo")
        with patch(
            "datus.models.openai_compatible.OpenAICompatibleModel.generate", return_value="response"
        ) as mock_gen:
            node.generate("prompt")
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("temperature") == 1


# ---------------------------------------------------------------------------
# QwenModel
# ---------------------------------------------------------------------------


class TestQwenModel:
    def _make(self, api_key="test-qwen-key", base_url=None):
        from datus.models.qwen_model import QwenModel

        config = _make_model_config(model="qwen-max", api_key=api_key, base_url=base_url)
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = QwenModel.__new__(QwenModel)
            instance.model_config = config
        return instance

    def test_get_api_key_from_config(self):
        node = self._make(api_key="my-qwen-key")
        assert node._get_api_key() == "my-qwen-key"

    def test_get_api_key_from_env(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.setenv("QWEN_API_KEY", "env-qwen-key")
        assert node._get_api_key() == "env-qwen-key"

    def test_get_api_key_raises_when_missing(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="QWEN_API_KEY"):
            node._get_api_key()

    def test_get_base_url_default(self):
        node = self._make(base_url=None)
        url = node._get_base_url()
        assert "aliyuncs.com" in url

    def test_get_base_url_custom(self):
        node = self._make(base_url="https://custom.api.example.com/v1")
        assert node._get_base_url() == "https://custom.api.example.com/v1"


# ---------------------------------------------------------------------------
# KimiModel
# ---------------------------------------------------------------------------


class TestKimiModel:
    def _make(self, api_key="test-kimi-key", base_url=None):
        from datus.models.kimi_model import KimiModel

        config = _make_model_config(model="moonshot-v1", api_key=api_key, base_url=base_url)
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = KimiModel.__new__(KimiModel)
            instance.model_config = config
            instance.model_name = "moonshot-v1"
            instance.base_url = None
        return instance

    def test_get_api_key_from_config(self):
        node = self._make(api_key="my-kimi-key")
        assert node._get_api_key() == "my-kimi-key"

    def test_get_api_key_from_env(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.setenv("KIMI_API_KEY", "env-kimi-key")
        assert node._get_api_key() == "env-kimi-key"

    def test_get_api_key_raises_when_missing(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Kimi API key"):
            node._get_api_key()

    def test_get_base_url_default(self, monkeypatch):
        node = self._make(base_url=None)
        monkeypatch.delenv("KIMI_API_BASE", raising=False)
        url = node._get_base_url()
        assert "moonshot.cn" in url

    def test_get_base_url_from_env(self, monkeypatch):
        node = self._make(base_url=None)
        monkeypatch.setenv("KIMI_API_BASE", "https://custom-kimi.example.com/v1")
        url = node._get_base_url()
        assert "custom-kimi" in url

    def test_get_base_url_custom(self):
        node = self._make(base_url="https://my-kimi.example.com/v1")
        assert node._get_base_url() == "https://my-kimi.example.com/v1"


# ---------------------------------------------------------------------------
# GeminiModel
# ---------------------------------------------------------------------------


class TestGeminiModel:
    def _make(self, api_key="test-gemini-key", base_url=None):
        from datus.models.gemini_model import GeminiModel

        config = _make_model_config(model="gemini-2.0-flash", api_key=api_key, base_url=base_url)
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = GeminiModel.__new__(GeminiModel)
            instance.model_config = config
            instance.model_name = "gemini-2.0-flash"
        return instance

    def test_get_api_key_from_config(self):
        node = self._make(api_key="my-gemini-key")
        assert node._get_api_key() == "my-gemini-key"

    def test_get_api_key_from_env(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
        assert node._get_api_key() == "env-gemini-key"

    def test_get_api_key_raises_when_missing(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Gemini API key"):
            node._get_api_key()

    def test_get_base_url_returns_config_value(self):
        node = self._make(base_url="https://my-gemini.example.com/v1")
        assert node._get_base_url() == "https://my-gemini.example.com/v1"

    def test_get_base_url_returns_none_when_not_configured(self):
        node = self._make(base_url=None)
        assert node._get_base_url() is None

    def test_model_specs_contains_gemini_models(self):
        node = self._make()
        specs = node.model_specs
        assert "gemini-2.0-flash" in specs
        assert "gemini-2.5-pro" in specs
        assert "context_length" in specs["gemini-2.0-flash"]
        assert "max_tokens" in specs["gemini-2.0-flash"]


# ---------------------------------------------------------------------------
# OpenRouterModel
# ---------------------------------------------------------------------------


class TestOpenRouterModel:
    def _make(self, api_key="test-openrouter-key", base_url=None):
        from datus.models.openrouter_model import OpenRouterModel

        config = ModelConfig(
            type="openrouter",
            api_key=api_key,
            model="anthropic/claude-sonnet-4",
            base_url=base_url,
        )
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = OpenRouterModel.__new__(OpenRouterModel)
            instance.model_config = config
            instance.model_name = "anthropic/claude-sonnet-4"
        return instance

    def test_get_api_key_from_config(self):
        node = self._make(api_key="sk-or-config-key")
        assert node._get_api_key() == "sk-or-config-key"

    def test_get_api_key_from_env(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        assert node._get_api_key() == "sk-or-env-key"

    def test_get_api_key_raises_when_missing(self, monkeypatch):
        from datus.utils.exceptions import DatusException, ErrorCode

        node = self._make(api_key="")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(DatusException) as exc_info:
            node._get_api_key()
        assert exc_info.value.code == ErrorCode.COMMON_ENV

    def test_get_base_url_returns_config_value(self):
        node = self._make(base_url="https://custom.openrouter.ai/api/v1")
        assert node._get_base_url() == "https://custom.openrouter.ai/api/v1"

    def test_get_base_url_defaults_to_openrouter(self):
        node = self._make(base_url=None)
        assert node._get_base_url() == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# MiniMaxModel
# ---------------------------------------------------------------------------


class TestMiniMaxModel:
    def _make(self, api_key="test-minimax-key", base_url=None):
        from datus.models.minimax_model import MiniMaxModel

        config = _make_model_config(model="MiniMax-M2.7", api_key=api_key, base_url=base_url)
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = MiniMaxModel.__new__(MiniMaxModel)
            instance.model_config = config
            instance.model_name = "MiniMax-M2.7"
        return instance

    def test_get_api_key_from_config(self):
        node = self._make(api_key="my-minimax-key")
        assert node._get_api_key() == "my-minimax-key"

    def test_get_api_key_from_env(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.setenv("MINIMAX_API_KEY", "env-minimax-key")
        assert node._get_api_key() == "env-minimax-key"

    def test_get_api_key_raises_when_missing(self, monkeypatch):
        from datus.utils.exceptions import DatusException, ErrorCode

        node = self._make(api_key="")
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(DatusException) as exc_info:
            node._get_api_key()
        assert exc_info.value.code == ErrorCode.COMMON_ENV

    def test_get_base_url_default(self, monkeypatch):
        node = self._make(base_url=None)
        monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
        assert "minimaxi.com" in node._get_base_url()

    def test_get_base_url_custom(self):
        node = self._make(base_url="https://my-minimax.example.com/v1")
        assert node._get_base_url() == "https://my-minimax.example.com/v1"


# ---------------------------------------------------------------------------
# GLMModel
# ---------------------------------------------------------------------------


class TestGLMModel:
    def _make(self, api_key="test-glm-key", base_url=None):
        from datus.models.glm_model import GLMModel

        config = _make_model_config(model="glm-5", api_key=api_key, base_url=base_url)
        with patch("datus.models.openai_compatible.OpenAICompatibleModel.__init__", return_value=None):
            instance = GLMModel.__new__(GLMModel)
            instance.model_config = config
            instance.model_name = "glm-5"
        return instance

    def test_get_api_key_from_config(self):
        node = self._make(api_key="my-glm-key")
        assert node._get_api_key() == "my-glm-key"

    def test_get_api_key_from_env(self, monkeypatch):
        node = self._make(api_key="")
        monkeypatch.setenv("GLM_API_KEY", "env-glm-key")
        assert node._get_api_key() == "env-glm-key"

    def test_get_api_key_raises_when_missing(self, monkeypatch):
        from datus.utils.exceptions import DatusException, ErrorCode

        node = self._make(api_key="")
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        with pytest.raises(DatusException) as exc_info:
            node._get_api_key()
        assert exc_info.value.code == ErrorCode.COMMON_ENV

    def test_get_base_url_default(self, monkeypatch):
        node = self._make(base_url=None)
        monkeypatch.delenv("GLM_API_BASE", raising=False)
        assert "bigmodel.cn" in node._get_base_url()

    def test_get_base_url_custom(self):
        node = self._make(base_url="https://my-glm.example.com/v4")
        assert node._get_base_url() == "https://my-glm.example.com/v4"
