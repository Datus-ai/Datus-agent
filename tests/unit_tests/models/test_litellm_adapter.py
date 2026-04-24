# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/litellm_adapter.py

CI-level: zero external dependencies (LiteLLM / openai-agents SDK calls are mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.models.litellm_adapter import (
    LiteLLMAdapter,
    create_litellm_adapter,
    is_known_non_thinking_model,
)


class TestLiteLLMAdapterInit:
    def test_basic_init(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="sk-test")
        assert adapter.provider == "openai"
        assert adapter.model == "gpt-4o"
        assert adapter.api_key == "sk-test"

    @pytest.mark.parametrize(
        "model,expected_provider",
        [
            ("claude-sonnet-4", "claude"),
            ("deepseek-chat", "deepseek"),
            ("qwen3-coder", "qwen"),
            ("gemini-2.5-pro", "gemini"),
            ("kimi-k2.5", "kimi"),
            ("gpt-4o", "openai"),
        ],
    )
    def test_auto_detect_provider(self, model, expected_provider):
        adapter = LiteLLMAdapter(provider="openai", model=model, api_key="key")
        assert adapter.provider == expected_provider

    def test_custom_base_url(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key", base_url="https://custom.api.com")
        assert adapter.base_url == "https://custom.api.com"

    def test_default_base_url_deepseek(self):
        adapter = LiteLLMAdapter(provider="deepseek", model="deepseek-chat", api_key="key")
        assert adapter.base_url == "https://api.deepseek.com"

    def test_thinking_disabled_by_default(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        assert adapter.is_thinking_model is False
        assert adapter.reasoning_effort_level is None

    def test_thinking_enabled(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key", enable_thinking=True)
        assert adapter.is_thinking_model is True
        # Legacy bool defaults to "medium" when no explicit effort is set.
        assert adapter.reasoning_effort_level == "medium"

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
    def test_reasoning_effort_level_explicit(self, effort):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="key",
            reasoning_effort=effort,
        )
        assert adapter.reasoning_effort_level == effort
        assert adapter.is_thinking_model is True

    def test_reasoning_effort_off_overrides_enable_thinking(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="key",
            enable_thinking=True,
            reasoning_effort="off",
        )
        assert adapter.reasoning_effort_level is None
        assert adapter.is_thinking_model is False

    def test_reasoning_effort_wins_over_enable_thinking(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="key",
            enable_thinking=True,
            reasoning_effort="high",
        )
        assert adapter.reasoning_effort_level == "high"

    def test_reasoning_effort_normalizes_case(self):
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="key",
            reasoning_effort="HIGH",
        )
        assert adapter.reasoning_effort_level == "high"


class TestLiteLLMAdapterModelName:
    @pytest.mark.parametrize(
        "provider,model,expected_litellm_name",
        [
            ("openai", "gpt-4o", "gpt-4o"),
            ("claude", "claude-sonnet-4", "anthropic/claude-sonnet-4"),
            ("deepseek", "deepseek-chat", "deepseek/deepseek-chat"),
            ("claude", "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4"),  # already has prefix
            ("unknown_provider", "my-model", "my-model"),
            ("gemini", "gemini-2.5-pro", "gemini/gemini-2.5-pro"),
            ("qwen", "qwen3-coder", "dashscope/qwen3-coder"),
            ("kimi", "kimi-k2.5", "moonshot/kimi-k2.5"),
        ],
    )
    def test_litellm_model_name(self, provider, model, expected_litellm_name):
        adapter = LiteLLMAdapter(provider=provider, model=model, api_key="key")
        assert adapter.litellm_model_name == expected_litellm_name

    def test_litellm_model_name_cached(self):
        adapter = LiteLLMAdapter(provider="openai", model="gpt-4o", api_key="key")
        name1 = adapter.litellm_model_name
        name2 = adapter.litellm_model_name
        assert name1 is name2  # same object (cached)


class TestOpenRouterModelName:
    def test_openrouter_prefixes_model_with_slash(self):
        """OpenRouter models like anthropic/claude-sonnet-4 need openrouter/ prefix."""
        adapter = LiteLLMAdapter(provider="openrouter", model="anthropic/claude-sonnet-4", api_key="key")
        assert adapter.litellm_model_name == "openrouter/anthropic/claude-sonnet-4"

    def test_openrouter_prefixes_simple_model(self):
        adapter = LiteLLMAdapter(provider="openrouter", model="mistralai/mistral-large", api_key="key")
        assert adapter.litellm_model_name == "openrouter/mistralai/mistral-large"

    def test_openrouter_default_base_url_is_none(self):
        """LiteLLM handles the default OpenRouter base URL."""
        adapter = LiteLLMAdapter(provider="openrouter", model="openai/gpt-4o", api_key="key")
        assert adapter.base_url is None

    def test_openrouter_custom_base_url(self):
        adapter = LiteLLMAdapter(
            provider="openrouter", model="openai/gpt-4o", api_key="key", base_url="https://custom.openrouter.ai/api/v1"
        )
        assert adapter.base_url == "https://custom.openrouter.ai/api/v1"

    def test_openrouter_not_autodetected(self):
        """OpenRouter models have provider/ prefix — should NOT be auto-detected as another provider."""
        adapter = LiteLLMAdapter(provider="openrouter", model="anthropic/claude-sonnet-4", api_key="key")
        assert adapter.provider == "openrouter"

    def test_openrouter_unprefixed_gpt_stays_openrouter(self):
        """Unprefixed model like gpt-4o should NOT be auto-detected away from openrouter."""
        adapter = LiteLLMAdapter(provider="openrouter", model="gpt-4o", api_key="key")
        assert adapter.provider == "openrouter"
        assert adapter.litellm_model_name == "openrouter/gpt-4o"

    def test_openrouter_unprefixed_deepseek_stays_openrouter(self):
        """Unprefixed model like deepseek-chat should NOT be auto-detected away from openrouter."""
        adapter = LiteLLMAdapter(provider="openrouter", model="deepseek-chat", api_key="key")
        assert adapter.provider == "openrouter"
        assert adapter.litellm_model_name == "openrouter/deepseek-chat"

    def test_openrouter_unprefixed_claude_stays_openrouter(self):
        """Unprefixed model like claude-sonnet-4 should NOT be auto-detected away from openrouter."""
        adapter = LiteLLMAdapter(provider="openrouter", model="claude-sonnet-4", api_key="key")
        assert adapter.provider == "openrouter"
        assert adapter.litellm_model_name == "openrouter/claude-sonnet-4"

    def test_openrouter_no_double_prefix(self):
        """Model already prefixed with openrouter/ should NOT get double-prefixed."""
        adapter = LiteLLMAdapter(provider="openrouter", model="openrouter/anthropic/claude-sonnet-4", api_key="key")
        assert adapter.litellm_model_name == "openrouter/anthropic/claude-sonnet-4"


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
        # deepseek is not routed to the Responses API, so it still uses LitellmModel.
        adapter = LiteLLMAdapter(provider="deepseek", model="deepseek-chat", api_key="key")
        mock_model = MagicMock()
        mock_litellm_model_cls = MagicMock(return_value=mock_model)
        mock_module = MagicMock()
        mock_module.LitellmModel = mock_litellm_model_cls

        with patch.dict("sys.modules", {"agents.extensions.models.litellm_model": mock_module}):
            result = adapter.get_agents_sdk_model()
        assert result is mock_model
        mock_litellm_model_cls.assert_called_once()

    def test_extra_headers_not_passed_to_litellm_model_constructor(self):
        """default_headers should NOT be passed to LitellmModel constructor.
        Headers are injected via ModelSettings.extra_headers at call time instead."""
        headers = {"User-Agent": "datus-agent (cli)"}
        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="key", default_headers=headers)
        mock_model = MagicMock()
        with patch(
            "datus.models.litellm_cache_control.CacheControlLitellmModel",
            return_value=mock_model,
        ) as mock_cls:
            adapter.get_agents_sdk_model()
        call_kwargs = mock_cls.call_args
        assert "extra_headers" not in (call_kwargs.kwargs or {})


class TestAutoDetectWithBaseUrl:
    """Tests for auto-detection bypass when base_url doesn't match detected provider."""

    def test_skip_auto_detect_when_base_url_mismatches(self):
        """type: claude + model: qwen3-coder-plus + alibaba base_url → provider stays claude."""
        adapter = LiteLLMAdapter(
            provider="claude",
            model="qwen3-coder-plus",
            api_key="test-key",
            base_url="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
        )
        assert adapter.provider == "claude"
        assert adapter.litellm_model_name == "anthropic/qwen3-coder-plus"

    def test_skip_auto_detect_kimi_on_alibaba(self):
        """type: claude + model: kimi-k2.5 + alibaba base_url → provider stays claude."""
        adapter = LiteLLMAdapter(
            provider="claude",
            model="kimi-k2.5",
            api_key="test-key",
            base_url="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
        )
        assert adapter.provider == "claude"
        assert adapter.litellm_model_name == "anthropic/kimi-k2.5"

    def test_auto_detect_works_when_url_matches(self):
        """type: openai + model: kimi-k2.5 + moonshot base_url → provider becomes kimi."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="kimi-k2.5",
            api_key="test-key",
            base_url="https://api.moonshot.cn/v1",
        )
        assert adapter.provider == "kimi"
        assert adapter.litellm_model_name == "moonshot/kimi-k2.5"

    def test_auto_detect_works_without_base_url(self):
        """type: openai + model: kimi-k2.5 + no base_url → provider becomes kimi (backward compat)."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="kimi-k2.5",
            api_key="test-key",
        )
        assert adapter.provider == "kimi"

    def test_coding_plan_claude_model_name_passthrough(self):
        """type: claude + model: claude-sonnet-4 + alibaba base_url → provider stays claude."""
        adapter = LiteLLMAdapter(
            provider="claude",
            model="claude-sonnet-4",
            api_key="test-key",
            base_url="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
        )
        assert adapter.provider == "claude"
        assert adapter.litellm_model_name == "anthropic/claude-sonnet-4"

    def test_skip_auto_detect_glm_on_alibaba(self):
        """type: claude + model: glm-5 + alibaba base_url → provider stays claude."""
        adapter = LiteLLMAdapter(
            provider="claude",
            model="glm-5",
            api_key="test-key",
            base_url="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
        )
        assert adapter.provider == "claude"
        assert adapter.litellm_model_name == "anthropic/glm-5"

    def test_skip_auto_detect_kimi_coding_endpoint(self):
        """type: claude + model: kimi-for-coding + kimi coding base_url → provider stays claude."""
        adapter = LiteLLMAdapter(
            provider="claude",
            model="kimi-for-coding",
            api_key="test-key",
            base_url="https://api.kimi.com/coding/",
        )
        assert adapter.provider == "claude"
        assert adapter.litellm_model_name == "anthropic/kimi-for-coding"

    def test_skip_auto_detect_minimax_on_alibaba(self):
        """type: claude + model: MiniMax-M2.5 + alibaba base_url → provider stays claude."""
        adapter = LiteLLMAdapter(
            provider="claude",
            model="MiniMax-M2.5",
            api_key="test-key",
            base_url="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
        )
        assert adapter.provider == "claude"
        assert adapter.litellm_model_name == "anthropic/MiniMax-M2.5"

    def test_domain_match_rejects_label_substring(self):
        """Ensure 'evil-deepseek.com' does NOT match known domain 'deepseek.com'."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="deepseek-chat",
            api_key="test-key",
            base_url="https://evil-deepseek.com/v1",
        )
        # Domain 'evil-deepseek.com' should NOT match 'deepseek.com',
        # so auto-detection should be skipped and provider stays 'openai'
        assert adapter.provider == "openai"

    def test_domain_match_accepts_exact_domain(self):
        """Ensure exact domain 'api.deepseek.com' matches correctly."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="deepseek-chat",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
        )
        # Exact domain match → auto-detection applies
        assert adapter.provider == "deepseek"

    def test_domain_match_accepts_subdomain(self):
        """Ensure 'api.deepseek.com' matches via .deepseek.com suffix."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="deepseek-chat",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
        )
        assert adapter.provider == "deepseek"

    def test_vllm_custom_endpoint_openai_prefix(self):
        """type: openai + custom vLLM base_url + Qwen3.5-397B → openai/Qwen3.5-397B (#532)."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="Qwen3.5-397B",
            api_key="key",
            base_url="http://192.168.1.100:8015/v1",
        )
        assert adapter.provider == "openai"
        assert adapter.litellm_model_name == "openai/Qwen3.5-397B"

    def test_native_openai_no_extra_prefix(self):
        """type: openai + api.openai.com → no openai/ prefix for native models."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="key",
            base_url="https://api.openai.com/v1",
        )
        assert adapter.litellm_model_name == "gpt-4o"

    def test_localhost_vllm_openai_prefix(self):
        """type: openai + localhost vLLM → openai/ prefix for unknown model."""
        adapter = LiteLLMAdapter(
            provider="openai",
            model="my-custom-model",
            api_key="key",
            base_url="http://localhost:8000/v1",
        )
        assert adapter.litellm_model_name == "openai/my-custom-model"


class TestDefaultHeaders:
    """Tests for default_headers passthrough to LiteLLM and Anthropic clients."""

    def test_default_headers_stored(self):
        headers = {"User-Agent": "my-tool/1.0"}
        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="key", default_headers=headers)
        assert adapter.default_headers == headers

    def test_default_headers_none_by_default(self):
        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="key")
        assert adapter.default_headers is None

    def test_default_headers_in_completion_kwargs(self):
        headers = {"User-Agent": "my-tool/1.0", "X-Custom": "value"}
        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="key", default_headers=headers)
        kwargs = adapter.get_completion_kwargs()
        assert kwargs["extra_headers"] == headers

    def test_no_extra_headers_when_none(self):
        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="key")
        kwargs = adapter.get_completion_kwargs()
        assert "extra_headers" not in kwargs

    def test_coding_plan_with_custom_headers(self):
        """Coding Plan endpoint with custom headers — provider stays claude, headers preserved."""
        headers = {"User-Agent": "my-coding-tool/1.0"}
        adapter = LiteLLMAdapter(
            provider="claude",
            model="qwen3-coder-plus",
            api_key="sk-sp-test",
            base_url="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
            default_headers=headers,
        )
        assert adapter.provider == "claude"
        assert adapter.default_headers == headers
        kwargs = adapter.get_completion_kwargs()
        assert kwargs["extra_headers"] == headers


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

    def test_factory_with_default_headers(self):
        headers = {"User-Agent": "test/1.0"}
        adapter = create_litellm_adapter(
            provider="claude",
            model="claude-sonnet-4",
            api_key="key",
            default_headers=headers,
        )
        assert adapter.default_headers == headers


class TestIsKnownNonThinkingModel:
    """The deny-list is narrow by design: only models explicitly known NOT to
    support thinking are listed; everything else is treated as supported so
    new DeepSeek/Kimi releases don't require a config edit."""

    @pytest.mark.parametrize(
        "provider,model",
        [
            ("deepseek", "deepseek-chat"),
            ("deepseek", "DeepSeek-Chat"),  # case-insensitive
            ("kimi", "kimi-k2"),
            ("kimi", "moonshot-v1-8k"),
            ("kimi", "moonshot-v1-32k"),
            ("kimi", "moonshot-v1-128k"),
            ("kimi", "moonshot-v1-auto"),
        ],
    )
    def test_deny_listed_models(self, provider, model):
        assert is_known_non_thinking_model(provider, model) is True

    @pytest.mark.parametrize(
        "provider,model",
        [
            # DeepSeek thinking-capable
            ("deepseek", "deepseek-reasoner"),
            ("deepseek", "deepseek-v4"),
            ("deepseek", "deepseek-v4-pro"),
            ("deepseek", "deepseek-v5-future"),  # unknown future release
            # Kimi thinking-capable (must NOT match bare kimi-k2)
            ("kimi", "kimi-k2.5"),
            ("kimi", "kimi-k2.6"),
            ("kimi", "kimi-k2-thinking"),
            ("kimi", "kimi-k3-future"),
            # Other providers are never on the deny-list
            ("openai", "gpt-4.1"),
            ("claude", "claude-haiku-4-5"),
            ("gemini", "gemini-2.5-flash"),
        ],
    )
    def test_allow_listed_by_omission(self, provider, model):
        assert is_known_non_thinking_model(provider, model) is False

    def test_none_inputs_are_safe(self):
        assert is_known_non_thinking_model(None, "anything") is False
        assert is_known_non_thinking_model("deepseek", None) is False
        assert is_known_non_thinking_model(None, None) is False


class TestGetAgentsSdkModelRouting:
    def test_claude_returns_cache_control_subclass(self):
        from agents.extensions.models.litellm_model import LitellmModel

        from datus.models.litellm_cache_control import CacheControlLitellmModel

        adapter = LiteLLMAdapter(provider="claude", model="claude-sonnet-4", api_key="sk-test")
        model = adapter.get_agents_sdk_model()
        assert isinstance(model, CacheControlLitellmModel)
        assert isinstance(model, LitellmModel)

    def test_official_openai_returns_responses_model(self):
        """Official OpenAI endpoint routes through /v1/responses so reasoning +
        function tools coexist without chat/completions limitations."""
        from agents.models.openai_responses import OpenAIResponsesModel

        adapter = LiteLLMAdapter(provider="openai", model="gpt-5.4", api_key="sk-test")
        model = adapter.get_agents_sdk_model()
        assert isinstance(model, OpenAIResponsesModel)

    def test_openai_with_api_openai_com_base_url_uses_responses(self):
        from agents.models.openai_responses import OpenAIResponsesModel

        adapter = LiteLLMAdapter(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        model = adapter.get_agents_sdk_model()
        assert isinstance(model, OpenAIResponsesModel)

    def test_openai_compatible_custom_endpoint_keeps_litellm(self):
        """vLLM / self-hosted OpenAI-compatible proxies should stay on the LiteLLM path."""
        from agents.extensions.models.litellm_model import LitellmModel
        from agents.models.openai_responses import OpenAIResponsesModel

        adapter = LiteLLMAdapter(
            provider="openai",
            model="Qwen3.5-397B",
            api_key="key",
            base_url="http://192.168.1.100:8015/v1",
        )
        model = adapter.get_agents_sdk_model()
        assert isinstance(model, LitellmModel)
        assert not isinstance(model, OpenAIResponsesModel)

    def test_deepseek_still_uses_litellm(self):
        from agents.extensions.models.litellm_model import LitellmModel
        from agents.models.openai_responses import OpenAIResponsesModel

        adapter = LiteLLMAdapter(provider="deepseek", model="deepseek-chat", api_key="sk-test")
        model = adapter.get_agents_sdk_model()
        assert isinstance(model, LitellmModel)
        assert not isinstance(model, OpenAIResponsesModel)

    def test_kimi_still_uses_litellm(self):
        """Kimi/Moonshot relies on sdk_patches.py for reasoning_content echo-back,
        which only applies on the LiteLLM path. It must not be routed to Responses."""
        from agents.extensions.models.litellm_model import LitellmModel
        from agents.models.openai_responses import OpenAIResponsesModel

        adapter = LiteLLMAdapter(provider="kimi", model="kimi-k2.5", api_key="sk-test")
        model = adapter.get_agents_sdk_model()
        assert isinstance(model, LitellmModel)
        assert not isinstance(model, OpenAIResponsesModel)
