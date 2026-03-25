# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/claude_model.py.

CI-level: zero external dependencies. Anthropic client and all I/O mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.models.claude_model import ClaudeModel, convert_tools_for_anthropic, wrap_prompt_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(
    model="claude-sonnet-4-5",
    api_key="sk-ant-test",
    base_url=None,
    use_native_api=False,
    temperature=None,
    top_p=None,
    enable_thinking=False,
    auth_type="api_key",
):
    cfg = MagicMock()
    cfg.model = model
    cfg.type = "claude"
    cfg.api_key = api_key
    cfg.base_url = base_url
    cfg.use_native_api = use_native_api
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.enable_thinking = enable_thinking
    cfg.default_headers = {}
    cfg.max_retry = 3
    cfg.retry_interval = 0.0
    cfg.strict_json_schema = True
    cfg.auth_type = auth_type
    return cfg


def _make_claude_model(model_config=None):
    """Create ClaudeModel with all external dependencies mocked."""
    if model_config is None:
        model_config = _make_model_config()

    mock_litellm_adapter = MagicMock()
    mock_litellm_adapter.litellm_model_name = "anthropic/claude-sonnet-4-5"
    mock_litellm_adapter.provider = "anthropic"
    mock_litellm_adapter.is_thinking_model = False
    mock_litellm_adapter.get_agents_sdk_model.return_value = MagicMock()

    mock_anthropic_client = MagicMock()

    with (
        patch("datus.models.openai_compatible.setup_tracing"),
        patch("datus.models.openai_compatible.LiteLLMAdapter", return_value=mock_litellm_adapter),
        patch("anthropic.Anthropic", return_value=mock_anthropic_client),
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: "sk-ant-test" if key == "ANTHROPIC_API_KEY" else default,
        ),
    ):
        model = ClaudeModel(model_config)
        model.litellm_adapter = mock_litellm_adapter
        model.anthropic_client = mock_anthropic_client
        return model


# ---------------------------------------------------------------------------
# wrap_prompt_cache
# ---------------------------------------------------------------------------


class TestWrapPromptCache:
    def test_adds_cache_control_to_last_content_block(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}]
        result = wrap_prompt_cache(messages)
        last_content = result[-1]["content"]
        assert last_content[-1].get("cache_control") == {"type": "ephemeral"}

    def test_does_not_modify_original(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "original"}]}]
        wrap_prompt_cache(messages)
        assert "cache_control" not in messages[0]["content"][0]

    def test_string_content_not_modified(self):
        messages = [{"role": "user", "content": "plain string"}]
        result = wrap_prompt_cache(messages)
        # String content should remain unchanged (not list, so no cache_control added)
        assert result[-1]["content"] == "plain string"


# ---------------------------------------------------------------------------
# convert_tools_for_anthropic
# ---------------------------------------------------------------------------


class TestConvertToolsForAnthropic:
    def _make_mcp_tool(self, name="query_db", description="run query", input_schema=None):
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputSchema = input_schema or {"type": "object", "properties": {"query": {"type": "string"}}}
        tool.annotations = None
        return tool

    def test_converts_single_tool(self):
        tool = self._make_mcp_tool()
        result = convert_tools_for_anthropic([tool])
        assert len(result) == 1
        assert result[0]["name"] == "query_db"
        assert result[0]["description"] == "run query"

    def test_adds_cache_control_to_last_tool(self):
        tools = [self._make_mcp_tool("t1"), self._make_mcp_tool("t2")]
        result = convert_tools_for_anthropic(tools)
        assert "cache_control" in result[-1]
        assert "cache_control" not in result[0]

    def test_empty_tools_returns_empty(self):
        result = convert_tools_for_anthropic([])
        assert result == []

    def test_desc_key_renamed_to_description(self):
        tool = self._make_mcp_tool(input_schema={"type": "object", "properties": {"q": {"desc": "the query"}}})
        result = convert_tools_for_anthropic([tool])
        prop = result[0]["input_schema"]["properties"]["q"]
        assert "description" in prop
        assert "desc" not in prop

    def test_annotations_added_when_present(self):
        tool = self._make_mcp_tool()
        tool.annotations = {"readOnlyHint": True}
        result = convert_tools_for_anthropic([tool])
        assert result[0]["annotations"] == {"readOnlyHint": True}


# ---------------------------------------------------------------------------
# ClaudeModel.__init__ / properties
# ---------------------------------------------------------------------------


class TestClaudeModelInit:
    def test_model_name_set(self):
        model = _make_claude_model()
        assert model.model_name == "claude-sonnet-4-5"

    def test_use_native_api_false_by_default(self):
        model = _make_claude_model()
        assert model.use_native_api is False

    def test_use_native_api_true_when_configured(self):
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)
        assert model.use_native_api is True

    def test_anthropic_client_initialized(self):
        model = _make_claude_model()
        assert model.anthropic_client is not None

    def test_model_specs_contains_expected_models(self):
        model = _make_claude_model()
        specs = model.model_specs
        assert "claude-sonnet-4-5" in specs
        assert "claude-sonnet-4" in specs
        assert "context_length" in specs["claude-sonnet-4-5"]
        assert "max_tokens" in specs["claude-sonnet-4-5"]


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_returns_config_api_key(self):
        cfg = _make_model_config(api_key="sk-ant-explicit")
        model = _make_claude_model(cfg)
        # The api_key attr should be set from config
        assert model.api_key == "sk-ant-explicit"

    def test_raises_when_no_api_key(self):
        cfg = _make_model_config(api_key=None)
        cfg.api_key = None

        from datus.utils.exceptions import DatusException

        with (
            patch("datus.models.openai_compatible.setup_tracing"),
            patch("datus.models.openai_compatible.LiteLLMAdapter"),
            patch("anthropic.Anthropic"),
            patch.dict("os.environ", {}, clear=True),
        ):
            with pytest.raises(DatusException) as exc_info:
                ClaudeModel(cfg)
            assert "300011" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _get_base_url
# ---------------------------------------------------------------------------


class TestGetBaseUrl:
    def test_returns_config_base_url(self):
        cfg = _make_model_config(base_url="https://myproxy.com")
        model = _make_claude_model(cfg)
        assert model.base_url == "https://myproxy.com"

    def test_defaults_to_anthropic_api(self):
        cfg = _make_model_config(base_url=None)
        model = _make_claude_model(cfg)
        # When base_url is None, _get_base_url falls back to anthropic.com
        assert model._get_base_url() == "https://api.anthropic.com"


# ---------------------------------------------------------------------------
# generate (litellm path vs native path)
# ---------------------------------------------------------------------------


class TestClaudeModelGenerate:
    def test_litellm_path_calls_super(self):
        model = _make_claude_model()
        with patch(
            "datus.models.openai_compatible.OpenAICompatibleModel.generate", return_value="from litellm"
        ) as mock_super:
            result = model.generate("hello")
        mock_super.assert_called_once()
        assert result == "from litellm"

    def test_litellm_path_passes_top_p_none(self):
        model = _make_claude_model()
        captured_kwargs = {}

        def capture_generate(self_inner, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok"

        with patch("datus.models.openai_compatible.OpenAICompatibleModel.generate", capture_generate):
            model.generate("hello")
        assert captured_kwargs.get("top_p") is None

    def test_native_api_path_calls_anthropic_client(self):
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        content_block = MagicMock()
        content_block.text = "native response"
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_create = MagicMock(return_value=mock_response)
        model.anthropic_client.messages.create = mock_create

        result = model.generate("hello world")
        assert result == "native response"
        mock_create.assert_called_once()

    def test_native_api_extracts_system_message(self):
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        content_block = MagicMock()
        content_block.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_create = MagicMock(return_value=mock_response)
        model.anthropic_client.messages.create = mock_create

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        model.generate(messages)
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    def test_native_api_returns_empty_when_no_content(self):
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        mock_response = MagicMock()
        mock_response.content = []
        mock_create = MagicMock(return_value=mock_response)
        model.anthropic_client.messages.create = mock_create

        result = model.generate("hello")
        assert result == ""


# ---------------------------------------------------------------------------
# generate_with_tools routing
# ---------------------------------------------------------------------------


class TestClaudeModelGenerateWithTools:
    @pytest.mark.asyncio
    async def test_native_api_with_mcp_routes_to_generate_with_mcp(self):
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)
        mock_mcp_servers = {"server1": MagicMock()}

        with patch.object(
            model, "generate_with_mcp", new_callable=AsyncMock, return_value={"content": "x", "sql_contexts": []}
        ) as mock_mcp:
            await model.generate_with_tools(
                prompt="test",
                mcp_servers=mock_mcp_servers,
                instruction="instr",
                output_type=str,
            )
        mock_mcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_litellm_path_when_not_native_api(self):
        cfg = _make_model_config(use_native_api=False)
        model = _make_claude_model(cfg)

        from datus.models.openai_compatible import OpenAICompatibleModel

        with patch.object(
            OpenAICompatibleModel,
            "generate_with_tools",
            new_callable=AsyncMock,
            return_value={"content": "litellm", "sql_contexts": []},
        ) as mock_parent:
            await model.generate_with_tools(prompt="test", instruction="instr")
        mock_parent.assert_called_once()

    @pytest.mark.asyncio
    async def test_litellm_path_when_native_with_regular_tools(self):
        """native_api=True but tools (not mcp_servers) provided -> use parent."""
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)
        regular_tools = [MagicMock()]

        from datus.models.openai_compatible import OpenAICompatibleModel

        with patch.object(
            OpenAICompatibleModel,
            "generate_with_tools",
            new_callable=AsyncMock,
            return_value={"content": "ok", "sql_contexts": []},
        ) as mock_parent:
            await model.generate_with_tools(prompt="test", tools=regular_tools, instruction="instr")
        mock_parent.assert_called_once()


# ---------------------------------------------------------------------------
# aclose / close
# ---------------------------------------------------------------------------


class TestClaudeModelClose:
    def test_close_calls_proxy_client_close(self):
        model = _make_claude_model()
        model.proxy_client = MagicMock()
        model.close()
        model.proxy_client.close.assert_called_once()

    def test_close_calls_anthropic_client_close(self):
        model = _make_claude_model()
        model.close()
        model.anthropic_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_closes_clients(self):
        model = _make_claude_model()
        model.proxy_client = MagicMock()
        await model.aclose()
        model.proxy_client.close.assert_called_once()
        model.anthropic_client.close.assert_called_once()

    def test_context_manager_calls_close(self):
        model = _make_claude_model()
        with patch.object(model, "close") as mock_close:
            with model:
                pass
        mock_close.assert_called_once()

    def test_close_handles_exception_gracefully(self):
        model = _make_claude_model()
        model.anthropic_client.close.side_effect = RuntimeError("already closed")
        # Should not raise
        model.close()


# ---------------------------------------------------------------------------
# Subscription auth
# ---------------------------------------------------------------------------


class TestClaudeModelSubscriptionAuth:
    def test_subscription_auth_calls_credential_resolver(self):
        cfg = _make_model_config(api_key="sk-ant-oat01-sub-token")
        cfg.auth_type = "subscription"

        with (
            patch("datus.models.openai_compatible.setup_tracing"),
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic"),
            patch(
                "datus.auth.claude_credential.get_claude_subscription_token",
                return_value="sk-ant-oat01-sub-token",
            ) as mock_resolver,
        ):
            mock_adapter = MagicMock()
            mock_adapter.litellm_model_name = "anthropic/claude-sonnet-4-5"
            mock_adapter.provider = "anthropic"
            mock_adapter.is_thinking_model = False
            mock_adapter.get_agents_sdk_model.return_value = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            model = ClaudeModel(cfg)
            mock_resolver.assert_called_once_with("sk-ant-oat01-sub-token")
            assert model.api_key == "sk-ant-oat01-sub-token"

    def test_subscription_auth_type_in_config(self):
        cfg = _make_model_config(api_key="")
        cfg.auth_type = "subscription"

        with (
            patch("datus.models.openai_compatible.setup_tracing"),
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic"),
            patch(
                "datus.auth.claude_credential.get_claude_subscription_token",
                return_value="sk-ant-oat01-from-env",
            ),
        ):
            mock_adapter = MagicMock()
            mock_adapter.litellm_model_name = "anthropic/claude-sonnet-4-5"
            mock_adapter.provider = "anthropic"
            mock_adapter.is_thinking_model = False
            mock_adapter.get_agents_sdk_model.return_value = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            model = ClaudeModel(cfg)
            assert model.api_key == "sk-ant-oat01-from-env"

    def test_non_subscription_auth_ignores_resolver(self):
        """Default auth_type='api_key' should not call the credential resolver."""
        cfg = _make_model_config(api_key="sk-ant-regular-key")
        cfg.auth_type = "api_key"
        model = _make_claude_model(cfg)
        assert model.api_key == "sk-ant-regular-key"


# ---------------------------------------------------------------------------
# OAuth token: Bearer auth + client headers
# ---------------------------------------------------------------------------


class TestClaudeModelOAuthHeaders:
    def test_oauth_token_forces_native_api(self):
        """When auth_type='subscription', use_native_api should be forced to True."""
        cfg = _make_model_config(api_key="sk-ant-oat01-test-token", use_native_api=False)
        cfg.auth_type = "subscription"
        model = _make_claude_model(cfg)
        assert model._is_oauth_token is True
        assert model.use_native_api is True

    def test_oauth_uses_auth_token_not_api_key(self):
        """Native client should be created with auth_token for OAuth tokens."""
        cfg = _make_model_config(api_key="sk-ant-oat01-test-token")
        cfg.auth_type = "subscription"

        with (
            patch("datus.models.openai_compatible.setup_tracing"),
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic") as mock_anthropic_cls,
        ):
            mock_adapter = MagicMock()
            mock_adapter.litellm_model_name = "anthropic/claude-sonnet-4-5"
            mock_adapter.provider = "anthropic"
            mock_adapter.is_thinking_model = False
            mock_adapter.get_agents_sdk_model.return_value = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            ClaudeModel(cfg)

            # Verify Anthropic was called with auth_token, not api_key
            call_kwargs = mock_anthropic_cls.call_args[1]
            assert call_kwargs["auth_token"] == "sk-ant-oat01-test-token"
            assert call_kwargs["api_key"] is None

    def test_oauth_injects_client_headers(self):
        """OAuth tokens should inject user-agent, x-app, and dangerous-direct-browser-access headers."""
        cfg = _make_model_config(api_key="sk-ant-oat01-test-token")
        cfg.auth_type = "subscription"

        with (
            patch("datus.models.openai_compatible.setup_tracing"),
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic") as mock_anthropic_cls,
        ):
            mock_adapter = MagicMock()
            mock_adapter.litellm_model_name = "anthropic/claude-sonnet-4-5"
            mock_adapter.provider = "anthropic"
            mock_adapter.is_thinking_model = False
            mock_adapter.get_agents_sdk_model.return_value = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            ClaudeModel(cfg)

            call_kwargs = mock_anthropic_cls.call_args[1]
            headers = call_kwargs["default_headers"]
            assert "user-agent" in headers
            assert headers["x-app"] == "cli"
            assert headers["anthropic-dangerous-direct-browser-access"] == "true"

    def test_oauth_beta_headers_correct(self):
        """OAuth beta headers should contain the expected values."""
        assert "claude-code-20250219" in ClaudeModel.OAUTH_BETA_HEADERS
        assert "oauth-2025-04-20" in ClaudeModel.OAUTH_BETA_HEADERS
        assert "interleaved-thinking-2025-05-14" in ClaudeModel.OAUTH_BETA_HEADERS
        assert "prompt-caching-scope-2026-01-05" in ClaudeModel.OAUTH_BETA_HEADERS
        # fine-grained-tool-streaming should NOT be present
        assert "fine-grained-tool-streaming-2025-05-14" not in ClaudeModel.OAUTH_BETA_HEADERS

    def test_non_oauth_uses_api_key(self):
        """Regular API key (auth_type='api_key') should use api_key, not auth_token."""
        cfg = _make_model_config(api_key="sk-ant-oat01-looks-like-oauth-but-not")
        cfg.auth_type = "api_key"

        with (
            patch("datus.models.openai_compatible.setup_tracing"),
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic") as mock_anthropic_cls,
        ):
            mock_adapter = MagicMock()
            mock_adapter.litellm_model_name = "anthropic/claude-sonnet-4-5"
            mock_adapter.provider = "anthropic"
            mock_adapter.is_thinking_model = False
            mock_adapter.get_agents_sdk_model.return_value = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            ClaudeModel(cfg)

            call_kwargs = mock_anthropic_cls.call_args[1]
            assert call_kwargs["api_key"] == "sk-ant-oat01-looks-like-oauth-but-not"
            assert "auth_token" not in call_kwargs


class TestDiagnoseOAuth401:
    """Tests for _diagnose_oauth_401 smart error handling."""

    def test_non_oauth_token_does_nothing(self):
        """Non-OAuth tokens should pass through without raising."""
        cfg = _make_model_config(api_key="sk-ant-regular-key", auth_type="api_key")
        model = _make_claude_model(cfg)
        original_error = Exception("401 Unauthorized")
        # Should return without raising
        model._diagnose_oauth_401(original_error)

    def test_expired_token_raises_expired_error(self, tmp_path):
        """When credentials file shows expired token, raise CLAUDE_SUBSCRIPTION_TOKEN_EXPIRED."""
        import json
        import time

        from datus.utils.exceptions import DatusException, ErrorCode

        # Create a credentials file with expired token
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        cred_file = claude_dir / ".credentials.json"
        expired_ms = int((time.time() - 3600) * 1000)  # 1 hour ago
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-test", "expiresAt": expired_ms}})
        )

        cfg = _make_model_config(api_key="sk-ant-oat01-test", auth_type="subscription")
        model = _make_claude_model(cfg)

        original_error = Exception("401 Unauthorized")
        with patch("datus.models.claude_model.Path.home", return_value=tmp_path):
            with pytest.raises(DatusException) as exc_info:
                model._diagnose_oauth_401(original_error)
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_EXPIRED

    def test_valid_token_raises_auth_failed(self, tmp_path):
        """When token is not expired but 401, raise CLAUDE_SUBSCRIPTION_AUTH_FAILED."""
        import json
        import time

        from datus.utils.exceptions import DatusException, ErrorCode

        # Create a credentials file with valid (non-expired) token
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        cred_file = claude_dir / ".credentials.json"
        future_ms = int((time.time() + 3600) * 1000)  # 1 hour from now
        cred_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-test", "expiresAt": future_ms}})
        )

        cfg = _make_model_config(api_key="sk-ant-oat01-test", auth_type="subscription")
        model = _make_claude_model(cfg)

        original_error = Exception("401 Unauthorized")
        with patch("datus.models.claude_model.Path.home", return_value=tmp_path):
            with pytest.raises(DatusException) as exc_info:
                model._diagnose_oauth_401(original_error)
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_AUTH_FAILED

    def test_no_credentials_file_raises_auth_failed(self, tmp_path):
        """When no credentials file exists, raise CLAUDE_SUBSCRIPTION_AUTH_FAILED."""
        from datus.utils.exceptions import DatusException, ErrorCode

        cfg = _make_model_config(api_key="sk-ant-oat01-test", auth_type="subscription")
        model = _make_claude_model(cfg)

        original_error = Exception("401 Unauthorized")
        with patch("datus.models.claude_model.Path.home", return_value=tmp_path):
            with pytest.raises(DatusException) as exc_info:
                model._diagnose_oauth_401(original_error)
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_AUTH_FAILED

    def test_malformed_credentials_file_raises_auth_failed(self, tmp_path):
        """When credentials file is malformed, fall through to auth_failed."""
        from datus.utils.exceptions import DatusException, ErrorCode

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        cred_file = claude_dir / ".credentials.json"
        cred_file.write_text("not-valid-json{{{")

        cfg = _make_model_config(api_key="sk-ant-oat01-test", auth_type="subscription")
        model = _make_claude_model(cfg)

        original_error = Exception("401 Unauthorized")
        with patch("datus.models.claude_model.Path.home", return_value=tmp_path):
            with pytest.raises(DatusException) as exc_info:
                model._diagnose_oauth_401(original_error)
            assert exc_info.value.code == ErrorCode.CLAUDE_SUBSCRIPTION_AUTH_FAILED

    def test_preserves_original_error_as_cause(self, tmp_path):
        """The original 401 error should be chained as __cause__."""
        from datus.utils.exceptions import DatusException

        cfg = _make_model_config(api_key="sk-ant-oat01-test", auth_type="subscription")
        model = _make_claude_model(cfg)

        original_error = Exception("401 from Anthropic API")
        with patch("datus.models.claude_model.Path.home", return_value=tmp_path):
            with pytest.raises(DatusException) as exc_info:
                model._diagnose_oauth_401(original_error)
            assert exc_info.value.__cause__ is original_error
