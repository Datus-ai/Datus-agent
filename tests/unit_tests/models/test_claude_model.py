# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/claude_model.py.

CI-level: zero external dependencies. Anthropic client and all I/O mocked.
"""

from types import SimpleNamespace
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
        patch("datus.models.openai_compatible.LiteLLMAdapter", return_value=mock_litellm_adapter),
        patch("anthropic.Anthropic", return_value=mock_anthropic_client),
        patch("langsmith.wrappers.wrap_anthropic", side_effect=lambda c: c),
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: "sk-ant-test" if key == "ANTHROPIC_API_KEY" else default,
        ),
    ):
        model = ClaudeModel(model_config)
        model.litellm_adapter = mock_litellm_adapter
        model.anthropic_client = mock_anthropic_client
        # Existing tests mock the sync ``anthropic_client.messages.create`` API.
        # Disabling the async streaming client routes those tests through the
        # non-streaming fallback in ``_generate_with_mcp_stream`` so they keep
        # working without rewriting every mock; the streaming code path has
        # dedicated tests in ``TestGenerateWithMcpStreamTextDeltas``.
        model.async_anthropic_client = None
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
        with (
            patch("datus.models.openai_compatible.LiteLLMAdapter"),
            patch("anthropic.Anthropic") as mock_anthropic_cls,
            patch("langsmith.wrappers.wrap_anthropic", side_effect=lambda c: c),
            patch(
                "os.environ.get",
                side_effect=lambda key, default=None: "sk-ant-test" if key == "ANTHROPIC_API_KEY" else default,
            ),
        ):
            model = ClaudeModel(_make_model_config())
        # Verify anthropic.Anthropic constructor was called (client is not merely assigned)
        mock_anthropic_cls.assert_called_once()
        assert model.anthropic_client is mock_anthropic_cls.return_value

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

    def test_claude_generate_does_not_send_top_p_to_litellm(self):
        """Regression test for the Anthropic "temperature and top_p" 400.

        Exercises the full pipeline (no mock at the parent boundary) so
        a regression in either layer surfaces. The suppression is
        owned by ``OpenAICompatibleModel._generate_operation`` and gates
        on ``litellm_adapter.provider == LLMProvider.CLAUDE``, so this
        test relies on ``_make_claude_model`` exposing the correct
        runtime provider — override the fixture's default value
        explicitly to make the contract visible.
        """
        model = _make_claude_model()
        model.litellm_adapter.provider = "claude"
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        mock_resp.choices[0].message.reasoning_content = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.model = "claude-sonnet-4-5"
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp) as mock_lit:
            model.generate("hello")
        call_kwargs = mock_lit.call_args[1]
        assert "top_p" not in call_kwargs, (
            "Anthropic rejects requests with both temperature and top_p; "
            f"the suppression contract must reach litellm.completion. Got top_p={call_kwargs.get('top_p')!r}."
        )

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
        with patch("datus.models.claude_model.logger") as mock_logger:
            # Should not raise — exception is swallowed and logged
            model.close()
        # Exception must be logged as a warning (see claude_model.py close())
        mock_logger.warning.assert_called_once()
        logged_msg = mock_logger.warning.call_args[0][0]
        assert "already closed" in logged_msg


# ---------------------------------------------------------------------------
# Subscription auth
# ---------------------------------------------------------------------------


class TestClaudeModelSubscriptionAuth:
    def test_subscription_auth_calls_credential_resolver(self):
        cfg = _make_model_config(api_key="sk-ant-oat01-sub-token")
        cfg.auth_type = "subscription"

        with (
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic"),
            patch(
                "datus.auth.claude_credential.get_claude_subscription_token",
                return_value=("sk-ant-oat01-sub-token", "config (agent.yml)"),
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
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic"),
            patch(
                "datus.auth.claude_credential.get_claude_subscription_token",
                return_value=("sk-ant-oat01-from-env", "env CLAUDE_CODE_OAUTH_TOKEN"),
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
        try:
            result = model._diagnose_oauth_401(original_error)
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"_diagnose_oauth_401 unexpectedly raised for non-OAuth token: {exc}")
        # Helper is fire-and-return for non-OAuth tokens; verify it did not
        # mutate state into an exception object or substitute the input error.
        assert result is None

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
            with patch("datus.auth.claude_credential._read_keychain_credentials", return_value=None):
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
            with patch("datus.auth.claude_credential._read_keychain_credentials", return_value=None):
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
            with patch("datus.auth.claude_credential._read_keychain_credentials", return_value=None):
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


# ---------------------------------------------------------------------------
# _generate_with_mcp_stream & generate_with_mcp
# ---------------------------------------------------------------------------


def _make_text_block(text="final answer"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name="read_query", block_id="tool_1", input_data=None):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = block_id
    block.input = input_data or {"query": "SELECT 1"}
    return block


def _make_response(content_blocks, input_tokens=100, output_tokens=50):
    response = MagicMock()
    response.content = content_blocks
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    response.usage = usage
    return response


class TestGenerateWithMcpStream:
    @pytest.mark.asyncio
    async def test_no_tool_calls_yields_final_action(self):
        """When API returns no tool_use, should yield a single ASSISTANT action."""
        from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        response = _make_response([_make_text_block("hello world")])
        model.anthropic_client.messages.create.return_value = response

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
            ):
                actions.append(action)

        assert len(actions) == 1
        assert actions[0].role == ActionRole.ASSISTANT
        assert actions[0].action_type == "final_response"
        assert actions[0].status == ActionStatus.SUCCESS
        assert "hello world" in actions[0].output["raw_output"]

    @pytest.mark.asyncio
    async def test_tool_call_yields_processing_and_success(self):
        """When API returns tool_use, should yield PROCESSING + SUCCESS + final."""
        from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_block = _make_tool_use_block(name="list_tables", block_id="call_1", input_data={"db": "main"})
        # First call: returns tool_use, second call: returns text (done)
        resp_tool = _make_response([tool_block], input_tokens=200, output_tokens=80)
        resp_final = _make_response([_make_text_block("done")], input_tokens=300, output_tokens=100)
        model.anthropic_client.messages.create.side_effect = [resp_tool, resp_final]

        # Mock func_tool
        func_tool = MagicMock()
        func_tool.name = "list_tables"
        func_tool.description = "List tables"
        func_tool.params_json_schema = {"type": "object"}
        func_tool.on_invoke_tool = AsyncMock(return_value='["table1", "table2"]')

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                func_tools=[func_tool],
                action_history_manager=ahm,
            ):
                actions.append(action)

        # Should have: PROCESSING, SUCCESS, ASSISTANT
        assert len(actions) == 3
        assert actions[0].role == ActionRole.TOOL
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[0].action_type == "list_tables"
        assert actions[1].role == ActionRole.TOOL
        assert actions[1].status == ActionStatus.SUCCESS
        assert actions[2].role == ActionRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_func_tool_receives_tool_call_id_matching_block_id(self):
        """The func tool must be invoked with a context whose ``tool_call_id``
        equals the tool_use block id (== the PROCESSING action's ``action_id``).

        Regression: claude_model previously passed a bare ``RunContextWrapper``
        (no ``tool_call_id``), so the ``task`` tool could not link sub-agent
        actions' ``parent_action_id`` to the wrapping task action and the CLI
        renderer mis-rendered the group as separate ``<node>_request`` /
        ``<node>_response`` blocks.
        """
        from agents import RunContextWrapper

        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_block = _make_tool_use_block(name="task", block_id="toolu_abc123", input_data={"type": "gen_sql"})
        resp_tool = _make_response([tool_block], input_tokens=200, output_tokens=80)
        resp_final = _make_response([_make_text_block("done")], input_tokens=300, output_tokens=100)
        model.anthropic_client.messages.create.side_effect = [resp_tool, resp_final]

        func_tool = MagicMock()
        func_tool.name = "task"
        func_tool.description = "Spawn a sub-agent"
        func_tool.params_json_schema = {"type": "object"}
        func_tool.on_invoke_tool = AsyncMock(return_value='{"success": 1}')

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                func_tools=[func_tool],
                action_history_manager=ahm,
            ):
                actions.append(action)

        func_tool.on_invoke_tool.assert_awaited_once()
        passed_ctx = func_tool.on_invoke_tool.await_args.args[0]
        # Must remain a RunContextWrapper subtype so on_invoke_tool accepts it,
        # and must expose the block id as tool_call_id for parent-action linking.
        assert isinstance(passed_ctx, RunContextWrapper)
        assert getattr(passed_ctx, "tool_call_id", None) == "toolu_abc123"
        # The PROCESSING action's id is the same block id the tool now sees,
        # so the sub-agent's parent_action_id resolves to a real anchor.
        assert actions[0].action_id == "toolu_abc123"

    @pytest.mark.asyncio
    async def test_token_usage_accumulated(self):
        """Token usage should be accumulated across turns and included in final action."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_block = _make_tool_use_block()
        resp1 = _make_response([tool_block], input_tokens=100, output_tokens=50)
        resp2 = _make_response([_make_text_block("answer")], input_tokens=200, output_tokens=80)
        model.anthropic_client.messages.create.side_effect = [resp1, resp2]

        func_tool = MagicMock()
        func_tool.name = "read_query"
        func_tool.description = ""
        func_tool.params_json_schema = {"type": "object"}
        func_tool.on_invoke_tool = AsyncMock(return_value="result")

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                func_tools=[func_tool],
                action_history_manager=ahm,
            ):
                actions.append(action)

        # Final action should have accumulated usage
        final = actions[-1]
        usage = final.output["usage"]
        assert usage["input_tokens"] == 300  # 100 + 200
        assert usage["output_tokens"] == 130  # 50 + 80
        assert usage["total_tokens"] == 430
        assert usage["requests"] == 2

    @pytest.mark.asyncio
    async def test_token_usage_folds_cache_into_input(self):
        """Anthropic reports cache_read / cache_creation separately from
        ``input_tokens``. The native loop must fold them back in so the reported
        input, total, cache-hit-rate and context-usage ratios are correct —
        otherwise heavy prompt caching collapses input to a few tokens and
        inflates cache_hit_rate beyond 1.0 (the original bug)."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_block = _make_tool_use_block()
        # Mimic Claude Code subscription: tiny fresh input, large cache read.
        resp1 = _make_response([tool_block], input_tokens=3, output_tokens=50)
        resp1.usage.cache_creation_input_tokens = 12000
        resp1.usage.cache_read_input_tokens = 0
        resp2 = _make_response([_make_text_block("answer")], input_tokens=1, output_tokens=80)
        resp2.usage.cache_creation_input_tokens = 0
        resp2.usage.cache_read_input_tokens = 12003
        model.anthropic_client.messages.create.side_effect = [resp1, resp2]

        func_tool = MagicMock()
        func_tool.name = "read_query"
        func_tool.description = ""
        func_tool.params_json_schema = {"type": "object"}
        func_tool.on_invoke_tool = AsyncMock(return_value="result")

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                func_tools=[func_tool],
                action_history_manager=ahm,
            ):
                actions.append(action)

        usage = actions[-1].output["usage"]
        # input = (3 + 0 + 12000) + (1 + 12003 + 0) = 24007
        assert usage["input_tokens"] == 24007
        assert usage["output_tokens"] == 130  # 50 + 80
        assert usage["total_tokens"] == 24137
        assert usage["cached_tokens"] == 12003  # cumulative cache_read
        # cache_hit_rate must stay a sane fraction (12003 / 24007), never > 1.
        assert 0 < usage["cache_hit_rate"] <= 1
        assert usage["cache_hit_rate"] == round(12003 / 24007, 3)
        # last call's real context window = 1 + 12003 + 0 = 12004
        assert usage["last_call_input_tokens"] == 12004

    @pytest.mark.asyncio
    async def test_tool_failure_yields_failed_action(self):
        """When a tool fails, should yield FAILED action."""
        from datus.schemas.action_history import ActionHistoryManager, ActionStatus

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_block = _make_tool_use_block(name="bad_tool", block_id="call_fail")
        resp_tool = _make_response([tool_block])
        resp_final = _make_response([_make_text_block("fallback")])
        model.anthropic_client.messages.create.side_effect = [resp_tool, resp_final]

        # No func_tools and no MCP servers → tool cannot be executed
        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
            ):
                actions.append(action)

        # PROCESSING + FAILED + ASSISTANT
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[1].status == ActionStatus.FAILED
        assert actions[1].output["summary"] == "Failed"

    @pytest.mark.asyncio
    async def test_session_persists_user_and_assistant_across_turns(self):
        """OAuth-subscription native path must persist multi-turn history through ``session``.

        Regression: prior to the fix, ``_generate_with_mcp_stream`` ignored the
        ``session`` parameter entirely, so subsequent turns started from an empty
        history and the assistant could not see the user's prior message.
        """
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        # Two independent invocations on the same session — turn 1 then turn 2.
        resp1 = _make_response([_make_text_block("answer-1")])
        resp2 = _make_response([_make_text_block("answer-2")])
        model.anthropic_client.messages.create.side_effect = [resp1, resp2]

        # Session stub mimicking AdvancedSQLiteSession's get_items / add_items contract.
        session = MagicMock()
        session_store: list = []
        session.get_items = AsyncMock(side_effect=lambda: list(session_store))
        session.add_items = AsyncMock(side_effect=lambda items: session_store.extend(items))

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            # Turn 1
            async for _ in model._generate_with_mcp_stream(
                prompt="hello",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
                session=session,
            ):
                pass

            # After turn 1: session should contain user "hello" and assistant "answer-1".
            assert session.add_items.await_count == 1
            assert any(
                isinstance(item, dict) and item.get("role") == "user" and "hello" in str(item.get("content"))
                for item in session_store
            ), f"user prompt not stored: {session_store}"
            assert any(
                isinstance(item, dict) and item.get("role") == "assistant" and "answer-1" in str(item.get("content"))
                for item in session_store
            ), f"assistant final not stored: {session_store}"

            # Turn 2
            async for _ in model._generate_with_mcp_stream(
                prompt="follow up",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
                session=session,
            ):
                pass

        # Anthropic API on turn 2 must have seen the prior turn's history.
        assert model.anthropic_client.messages.create.call_count == 2
        turn2_messages = model.anthropic_client.messages.create.call_args_list[1].kwargs["messages"]
        flattened = str(turn2_messages)
        assert "hello" in flattened, f"turn2 messages missing turn1 user: {turn2_messages}"
        assert "answer-1" in flattened, f"turn2 messages missing turn1 assistant: {turn2_messages}"
        assert "follow up" in flattened

    @pytest.mark.asyncio
    async def test_session_get_items_failure_falls_back_to_fresh_history(self):
        """A broken session must not abort the turn — log and start fresh instead."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        model.anthropic_client.messages.create.return_value = _make_response([_make_text_block("ok")])

        session = MagicMock()
        session.get_items = AsyncMock(side_effect=RuntimeError("disk error"))
        session.add_items = AsyncMock()

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            actions = []
            async for action in model._generate_with_mcp_stream(
                prompt="probe",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
                session=session,
            ):
                actions.append(action)

        # Native turn still completed despite the load failure.
        assert any(a.action_type == "final_response" for a in actions)
        # And we still tried to persist this turn for the next call.
        assert session.add_items.await_count == 1

    @pytest.mark.asyncio
    async def test_session_skip_persist_when_final_content_empty(self):
        """When ``max_turns`` is exhausted while still tool-calling, ``final_content``
        stays empty. Persisting an empty assistant text block would be rejected by
        Anthropic on replay (``text content blocks must be non-empty``), so the
        guard must skip ``add_items`` entirely.
        """
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        # Every turn keeps tool-calling; the loop exits via max_turns with no
        # text response. Use max_turns=2 so the test is fast.
        tool_block = _make_tool_use_block()
        model.anthropic_client.messages.create.return_value = _make_response([tool_block])

        func_tool = MagicMock()
        func_tool.name = "read_query"
        func_tool.description = ""
        func_tool.params_json_schema = {"type": "object"}
        func_tool.on_invoke_tool = AsyncMock(return_value="result")

        session = MagicMock()
        session.get_items = AsyncMock(return_value=[])
        session.add_items = AsyncMock()

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            actions = []
            async for action in model._generate_with_mcp_stream(
                prompt="probe",
                mcp_servers={},
                instruction="sys",
                output_type={},
                max_turns=2,
                func_tools=[func_tool],
                action_history_manager=ahm,
                session=session,
            ):
                actions.append(action)

        # final_response still yielded so the caller gets a response.
        final = next(a for a in actions if a.action_type == "final_response")
        assert final.output["raw_output"] == ""
        # Critically: we must NOT have persisted an empty assistant text block.
        session.add_items.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prompt_list_variant_is_normalized_to_string(self):
        """The ``prompt`` parameter signature accepts ``List[Dict[str, str]]`` for
        legacy callers; the native Anthropic ``text`` field requires a string, so
        list inputs must be serialised rather than handed through verbatim.
        """
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        model.anthropic_client.messages.create.return_value = _make_response([_make_text_block("ok")])

        list_prompt = [{"role": "user", "content": "structured-input"}]
        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for _ in model._generate_with_mcp_stream(
                prompt=list_prompt,
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
            ):
                pass

        # The user message sent to Anthropic must carry a single string ``text``
        # field; the list payload should have been json-serialised.
        call_messages = model.anthropic_client.messages.create.call_args.kwargs["messages"]
        user_text = call_messages[0]["content"][0]["text"]
        assert isinstance(user_text, str)
        assert "structured-input" in user_text

    @pytest.mark.asyncio
    async def test_session_add_items_failure_does_not_break_turn(self):
        """If persistence fails after a successful turn, the user still gets the response."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        model.anthropic_client.messages.create.return_value = _make_response([_make_text_block("ok")])

        session = MagicMock()
        session.get_items = AsyncMock(return_value=[])
        session.add_items = AsyncMock(side_effect=RuntimeError("disk full"))

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            actions = []
            async for action in model._generate_with_mcp_stream(
                prompt="probe",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
                session=session,
            ):
                actions.append(action)

        # The final action is still yielded even though persistence raised.
        assert any(a.action_type == "final_response" for a in actions)


class TestNativeTokenUsageStreaming:
    """The native OAuth loop bypasses the SDK Runner, so it must drive the
    per-LLM-call token-usage hook and persist durable turn usage itself.
    These tests pin that contract (the bug: status bar showed 0 usage /
    context for native Claude because neither happened)."""

    def _usage_hook(self):
        """Build a real ``TokenUsageHook`` over a fake node that records the
        emitted snapshots — exercises the genuine emit pipeline, not a stub."""
        from datus.agent.node.token_usage_hook import TokenUsageHook
        from datus.schemas.action_history import ActionHistoryManager

        emitted: list = []

        node = MagicMock()
        node.model = MagicMock()
        node.context_length = 200_000
        node._current_action_history = ActionHistoryManager()
        node.action_bus = None
        node.session_id = "chat_session_native"
        node.actions = []
        node.running_turn_usage = None

        sm = MagicMock()
        sm.upsert_running_turn_usage = MagicMock(side_effect=lambda **kw: emitted.append(kw))
        node.session_manager = sm
        node._notify_status_dirty = MagicMock()

        return TokenUsageHook(node), node, emitted

    @pytest.mark.asyncio
    async def test_native_loop_drives_per_call_usage_updates(self):
        """Each Anthropic response must push a cumulative snapshot through the
        hook so the status bar refreshes mid-turn (one update per LLM call)."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_block = _make_tool_use_block(name="read_query", block_id="c1")
        resp1 = _make_response([tool_block], input_tokens=100, output_tokens=40)
        resp2 = _make_response([_make_text_block("done")], input_tokens=250, output_tokens=90)
        model.anthropic_client.messages.create.side_effect = [resp1, resp2]

        func_tool = MagicMock()
        func_tool.name = "read_query"
        func_tool.description = ""
        func_tool.params_json_schema = {"type": "object"}
        func_tool.on_invoke_tool = AsyncMock(return_value="ok")

        hook, node, emitted = self._usage_hook()

        ahm = ActionHistoryManager()
        node._current_action_history = ahm
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for _ in model._generate_with_mcp_stream(
                prompt="q",
                mcp_servers={},
                instruction="sys",
                output_type={},
                func_tools=[func_tool],
                action_history_manager=ahm,
                hooks=hook,
            ):
                pass

        # Two Anthropic responses → two per-call usage updates persisted.
        assert len(emitted) == 2
        # Cumulative grows monotonically across calls.
        assert emitted[0]["cumulative"]["input_tokens"] == 100
        assert emitted[0]["cumulative"]["output_tokens"] == 40
        assert emitted[1]["cumulative"]["input_tokens"] == 350  # 100 + 250
        assert emitted[1]["cumulative"]["output_tokens"] == 130  # 40 + 90
        assert emitted[1]["cumulative"]["total_tokens"] == 480
        # Context length flows through so the status bar can render the ratio.
        assert emitted[1]["context_length"] == 200_000
        # The node's live snapshot reflects the final cumulative for the
        # status bar's next paint.
        assert node.running_turn_usage.total_tokens == 480

    @pytest.mark.asyncio
    async def test_native_loop_persists_durable_turn_usage(self):
        """At turn end the native loop must write the durable ``turn_usage``
        row via ``store_run_usage`` (with cached tokens), otherwise the status
        bar's cumulative total resets to 0 once the running snapshot clears."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        # Single response carrying a cache read so we can assert cached flows
        # into the persisted Usage.
        resp = _make_response([_make_text_block("answer")], input_tokens=500, output_tokens=120)
        resp.usage.cache_read_input_tokens = 200
        model.anthropic_client.messages.create.return_value = resp

        stored: list = []

        async def _store_run_usage(result):
            stored.append(result.context_wrapper.usage)

        session = MagicMock()
        session.get_items = AsyncMock(return_value=[])
        session.add_items = AsyncMock()
        session.store_run_usage = _store_run_usage

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for _ in model._generate_with_mcp_stream(
                prompt="q",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
                session=session,
            ):
                pass

        assert len(stored) == 1, "durable turn usage must be persisted exactly once"
        usage = stored[0]
        # Anthropic's ``input_tokens`` (500) excludes the cache_read (200); the
        # native loop folds the cache components back in to match OpenAI's
        # "input includes cached" semantics, so the persisted input is 700.
        assert usage.input_tokens == 700
        assert usage.output_tokens == 120
        assert usage.total_tokens == 820
        # cached_tokens must survive into the durable schema.
        assert usage.input_tokens_details.cached_tokens == 200

    @pytest.mark.asyncio
    async def test_native_loop_skips_durable_usage_when_no_final_text(self):
        """When the loop exhausts max_turns mid-tool-call (no final text), the
        session is not persisted — and neither should the usage row be, to
        avoid a turn_usage row with no matching message."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_block = _make_tool_use_block(name="read_query", block_id="loop")
        # Always returns a tool call → never produces final text → max_turns hit.
        looping = _make_response([tool_block])
        model.anthropic_client.messages.create.return_value = looping

        func_tool = MagicMock()
        func_tool.name = "read_query"
        func_tool.description = ""
        func_tool.params_json_schema = {"type": "object"}
        func_tool.on_invoke_tool = AsyncMock(return_value="row")

        stored: list = []

        async def _store_run_usage(result):
            stored.append(result)

        session = MagicMock()
        session.get_items = AsyncMock(return_value=[])
        session.add_items = AsyncMock()
        session.store_run_usage = _store_run_usage

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for _ in model._generate_with_mcp_stream(
                prompt="q",
                mcp_servers={},
                instruction="sys",
                output_type={},
                func_tools=[func_tool],
                action_history_manager=ahm,
                session=session,
                max_turns=2,
            ):
                pass

        assert stored == [], "no durable usage row when the turn produced no final text"
        session.add_items.assert_not_awaited()


class TestNativeTokenUsageHooks:
    """The native loop drives token-usage hooks manually (no SDK Runner)."""

    def test_iter_yields_bare_hook_and_composite_inner_hooks(self):
        model = _make_claude_model(_make_model_config(use_native_api=True))
        bare = MagicMock(spec=["emit_manual"])
        assert list(model._iter_token_usage_hooks(bare)) == [bare]
        assert list(model._iter_token_usage_hooks(None)) == []

        # Composite: hooks_list with a mix of usage and non-usage hooks.
        usage_hook = MagicMock(spec=["emit_manual", "on_start"])
        other_hook = MagicMock(spec=["on_start"])  # no emit_manual → skipped
        composite = SimpleNamespace(hooks_list=[usage_hook, other_hook])
        assert list(model._iter_token_usage_hooks(composite)) == [usage_hook]

    @pytest.mark.asyncio
    async def test_reset_and_emit_drive_composite_inner_hooks(self):
        model = _make_claude_model(_make_model_config(use_native_api=True))
        usage_hook = MagicMock()
        usage_hook.on_start = AsyncMock()
        usage_hook.emit_manual = AsyncMock()
        composite = SimpleNamespace(hooks_list=[usage_hook])

        await model._reset_token_usage_hook(composite)
        usage_hook.on_start.assert_awaited_once()

        await model._emit_native_token_usage(composite, {"total_tokens": 42})
        usage_hook.emit_manual.assert_awaited_once_with({"total_tokens": 42})

    @pytest.mark.asyncio
    async def test_emit_swallows_hook_failure(self):
        model = _make_claude_model(_make_model_config(use_native_api=True))
        bad = MagicMock()
        bad.emit_manual = AsyncMock(side_effect=RuntimeError("boom"))
        # Must not propagate — the native loop keeps running.
        await model._emit_native_token_usage(bad, {"total_tokens": 1})
        # The hook WAS invoked (so the raise happened inside and was swallowed),
        # proving the suppression path — not a silent skip — was exercised.
        bad.emit_manual.assert_awaited_once_with({"total_tokens": 1})


class TestStoreNativeTurnUsage:
    """Direct coverage of the durable-usage persistence helper used by the
    native Anthropic loop."""

    @pytest.mark.asyncio
    async def test_noop_when_session_is_none(self):
        model = _make_claude_model(_make_model_config(use_native_api=True))
        # No session → nothing to persist; the guard returns None without raising.
        result = await model._store_native_turn_usage(None, {"total_tokens": 100})
        assert result is None

    @pytest.mark.asyncio
    async def test_noop_when_session_lacks_store_run_usage(self):
        model = _make_claude_model(_make_model_config(use_native_api=True))
        # A session object that does not expose ``store_run_usage`` (spec omits
        # it) must trip the guard and no-op rather than AttributeError.
        session = MagicMock(spec=["add_items"])
        assert not hasattr(session, "store_run_usage")
        result = await model._store_native_turn_usage(session, {"total_tokens": 100})
        assert result is None

    @pytest.mark.asyncio
    async def test_builds_usage_and_calls_store_run_usage(self):
        model = _make_claude_model(_make_model_config(use_native_api=True))
        stored = []

        async def _store(result):
            stored.append(result.context_wrapper.usage)

        session = MagicMock()
        session.store_run_usage = _store
        await model._store_native_turn_usage(
            session,
            {
                "requests": 3,
                "input_tokens": 52499,
                "output_tokens": 1932,
                "total_tokens": 54431,
                "cached_tokens": 37747,
                "reasoning_tokens": 0,
            },
        )
        assert len(stored) == 1
        usage = stored[0]
        assert usage.input_tokens == 52499
        assert usage.output_tokens == 1932
        assert usage.total_tokens == 54431
        assert usage.input_tokens_details.cached_tokens == 37747

    @pytest.mark.asyncio
    async def test_swallows_store_run_usage_failure(self):
        model = _make_claude_model(_make_model_config(use_native_api=True))

        attempts = []

        async def _boom(result):
            attempts.append(result)
            raise RuntimeError("db down")

        session = MagicMock()
        session.store_run_usage = _boom
        # The warning path must not propagate the exception.
        await model._store_native_turn_usage(session, {"total_tokens": 100})
        # Storage WAS attempted (and the raised error swallowed), confirming the
        # except branch ran rather than the call being skipped entirely.
        assert len(attempts) == 1


class TestGenerateWithMcpWrapper:
    @pytest.mark.asyncio
    async def test_returns_dict_with_content(self):
        """generate_with_mcp wrapper should return dict with content and sql_contexts."""
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        response = _make_response([_make_text_block("the answer")])
        model.anthropic_client.messages.create.return_value = response

        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await model.generate_with_mcp(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ActionHistoryManager(),
            )

        assert isinstance(result, dict)
        assert result["content"] == "the answer"
        assert result["sql_contexts"] == []


class TestGenerateWithToolsRouting:
    @pytest.mark.asyncio
    async def test_oauth_token_routes_to_native(self):
        """When _is_oauth_token, generate_with_tools routes to generate_with_mcp."""
        cfg = _make_model_config(api_key="sk-ant-oat01-test", auth_type="subscription", use_native_api=True)
        model = _make_claude_model(cfg)

        mock_result = {"content": "ok", "sql_contexts": []}
        model.generate_with_mcp = AsyncMock(return_value=mock_result)

        result = await model.generate_with_tools(
            prompt="test",
            tools=[],
            mcp_servers={},
            instruction="sys",
        )
        assert result == mock_result
        model.generate_with_mcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_oauth_stream_routes_to_native_stream(self):
        """When _is_oauth_token, generate_with_tools_stream routes to _generate_with_mcp_stream."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        cfg = _make_model_config(api_key="sk-ant-oat01-test", auth_type="subscription", use_native_api=True)
        model = _make_claude_model(cfg)

        mock_action = ActionHistory(
            action_id="test",
            role=ActionRole.ASSISTANT,
            messages="ok",
            action_type="final_response",
            status=ActionStatus.SUCCESS,
            output={"raw_output": "ok", "sql_contexts": []},
        )

        async def mock_stream(**kwargs):
            yield mock_action

        model._generate_with_mcp_stream = mock_stream

        actions = []
        async for action in model.generate_with_tools_stream(
            prompt="test",
            tools=[],
            mcp_servers={},
            instruction="sys",
        ):
            actions.append(action)

        assert len(actions) == 1
        assert actions[0].role == ActionRole.ASSISTANT


# ---------------------------------------------------------------------------
# _count_session_tokens fallback
# ---------------------------------------------------------------------------


class TestCountSessionTokensFallback:
    @pytest.mark.asyncio
    async def test_uses_last_call_input_tokens_from_latest_action(self):
        """Primary: uses last_call_input_tokens from the most recent action with usage."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        mock_node = MagicMock()
        mock_node._session = MagicMock()
        mock_node.actions = [
            ActionHistory(
                action_id="a1",
                role=ActionRole.ASSISTANT,
                messages="ok",
                action_type="final_response",
                status=ActionStatus.SUCCESS,
                output={
                    "raw_output": "answer",
                    "usage": {"last_call_input_tokens": 500, "input_tokens": 800, "total_tokens": 1200},
                },
            ),
            ActionHistory(
                action_id="a2",
                role=ActionRole.ASSISTANT,
                messages="ok2",
                action_type="final_response",
                status=ActionStatus.SUCCESS,
                output={
                    "raw_output": "answer2",
                    "usage": {"last_call_input_tokens": 900, "input_tokens": 1500, "total_tokens": 2000},
                },
            ),
        ]

        from datus.agent.node.agentic_node import AgenticNode

        result = await AgenticNode._count_session_tokens(mock_node)
        # Should return last_call_input_tokens from the LAST action (900), not sum
        assert result == 900

    @pytest.mark.asyncio
    async def test_falls_back_to_input_tokens_when_no_last_call(self):
        """When last_call_input_tokens is 0, fall back to input_tokens."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        mock_node = MagicMock()
        mock_node._session = MagicMock()
        mock_node.actions = [
            ActionHistory(
                action_id="a1",
                role=ActionRole.ASSISTANT,
                messages="ok",
                action_type="final_response",
                status=ActionStatus.SUCCESS,
                output={"usage": {"last_call_input_tokens": 0, "input_tokens": 999, "total_tokens": 1500}},
            ),
        ]

        from datus.agent.node.agentic_node import AgenticNode

        result = await AgenticNode._count_session_tokens(mock_node)
        assert result == 999


class TestInjectOAuthHeaders:
    def test_injects_headers_when_oauth(self):
        """_inject_oauth_headers should add bearer + client headers for OAuth tokens."""
        cfg = _make_model_config(auth_type="subscription")
        model = _make_claude_model(cfg)
        kwargs: dict = {}
        model._inject_oauth_headers(kwargs)
        headers = kwargs["extra_headers"]
        assert "anthropic-beta" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["x-app"] == "cli"

    def test_no_headers_when_not_oauth(self):
        """_inject_oauth_headers should be a no-op for regular API keys."""
        cfg = _make_model_config(auth_type="api_key")
        model = _make_claude_model(cfg)
        kwargs: dict = {}
        model._inject_oauth_headers(kwargs)
        assert "extra_headers" not in kwargs


class TestNativeGenerateAuthError:
    def test_native_generate_auth_error_calls_diagnose(self):
        """Native generate() should call _diagnose_oauth_401 on AuthenticationError."""
        import anthropic as anthropic_mod

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        error = anthropic_mod.AuthenticationError(
            message="auth failed",
            response=MagicMock(status_code=401, headers={}, content=b""),
            body={"error": {"message": "auth failed"}},
        )
        model.anthropic_client.messages.create.side_effect = error
        model._diagnose_oauth_401 = MagicMock()

        with pytest.raises(anthropic_mod.AuthenticationError):
            model.generate(prompt="test", instruction="sys")

        model._diagnose_oauth_401.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_auth_error_calls_diagnose(self):
        """_generate_with_mcp_stream should call _diagnose_oauth_401 on AuthenticationError."""
        import anthropic as anthropic_mod

        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        error = anthropic_mod.AuthenticationError(
            message="auth failed",
            response=MagicMock(status_code=401, headers={}, content=b""),
            body={"error": {"message": "auth failed"}},
        )
        model.anthropic_client.messages.create.side_effect = error
        model._diagnose_oauth_401 = MagicMock()

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(anthropic_mod.AuthenticationError):
                async for _ in model._generate_with_mcp_stream(
                    prompt="test",
                    mcp_servers={},
                    instruction="sys",
                    output_type={},
                    action_history_manager=ahm,
                ):
                    pass

        model._diagnose_oauth_401.assert_called_once()


# ---------------------------------------------------------------------------
# generate_with_mcp: tool routing and duplicate tool names
# ---------------------------------------------------------------------------


class TestGenerateWithMcpToolRouting:
    """Tests for MCP tool_server_map construction and routing (Issue #2 + #6)."""

    def _make_mcp_tool(self, name):
        tool = MagicMock()
        tool.name = name
        return tool

    @pytest.mark.asyncio
    async def test_duplicate_tool_name_logs_warning(self):
        """When two MCP servers expose a tool with the same name, a warning should be logged."""
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_a = self._make_mcp_tool("shared_tool")
        tool_b = self._make_mcp_tool("shared_tool")

        server1 = AsyncMock()
        server1.list_tools = AsyncMock(return_value=[tool_a])
        server2 = AsyncMock()
        server2.list_tools = AsyncMock(return_value=[tool_b])

        connected_servers = {"server1": server1, "server2": server2}

        # Mock the context manager to yield our servers
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_mcp_ctx(servers):
            yield connected_servers

        # Mock anthropic response with no tool calls (stop immediately)
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = "done"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_response.usage = mock_usage
        model.anthropic_client.messages.create = MagicMock(return_value=mock_response)

        with (
            patch("datus.models.claude_model.multiple_mcp_servers", side_effect=mock_mcp_ctx),
            patch("datus.models.claude_model.convert_tools_for_anthropic", return_value=[]),
            patch("datus.models.claude_model.logger") as mock_logger,
        ):
            await model.generate_with_mcp(
                prompt="test",
                mcp_servers={"s1": MagicMock(), "s2": MagicMock()},
                instruction="instr",
                output_type=str,
            )

        # Verify warning was logged for the duplicate tool name
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("shared_tool" in w for w in warning_calls), (
            f"Expected warning about duplicate tool 'shared_tool', got: {warning_calls}"
        )

    @pytest.mark.asyncio
    async def test_tool_call_uses_shallow_copy_of_input(self):
        """block.input should be shallow-copied before passing to call_tool."""
        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        tool_mock = self._make_mcp_tool("my_tool")
        server = AsyncMock()
        server.list_tools = AsyncMock(return_value=[tool_mock])
        tool_content = MagicMock()
        tool_content.text = "query result"
        tool_result_obj = MagicMock()
        tool_result_obj.content = [tool_content]
        server.call_tool = AsyncMock(return_value=tool_result_obj)
        connected_servers = {"server1": server}

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_mcp_ctx(servers):
            yield connected_servers

        # First response: tool_use block, second response: text block (stop)
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "my_tool"
        tool_block.id = "call_1"
        original_input = {"query": "SELECT 1"}
        tool_block.input = original_input

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "done"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0
        response1 = MagicMock()
        response1.content = [tool_block]
        response1.usage = mock_usage
        response2 = MagicMock()
        response2.content = [text_block]
        response2.usage = mock_usage

        model.anthropic_client.messages.create = MagicMock(side_effect=[response1, response2])

        with (
            patch("datus.models.claude_model.multiple_mcp_servers", side_effect=mock_mcp_ctx),
            patch("datus.models.claude_model.convert_tools_for_anthropic", return_value=[]),
        ):
            await model.generate_with_mcp(
                prompt="test",
                mcp_servers={"s1": MagicMock()},
                instruction="instr",
                output_type=str,
            )

        # Verify call_tool was called with a copy, not the original dict
        call_args = server.call_tool.call_args
        passed_args = call_args[1]["arguments"]
        assert passed_args == {"query": "SELECT 1"}
        assert passed_args is not original_input  # must be a different object


# ---------------------------------------------------------------------------
# Streaming text deltas (native Anthropic ``messages.stream`` path)
# ---------------------------------------------------------------------------


class _FakeStreamEvent:
    """Lightweight stand-in for the SDK's stream event objects."""

    def __init__(self, type_, delta=None, content_block=None):
        self.type = type_
        if delta is not None:
            self.delta = delta
        if content_block is not None:
            self.content_block = content_block


class _FakeAsyncStreamManager:
    """Async context manager that replays a fixed sequence of stream events.

    Mirrors the shape of ``anthropic.lib.streaming._messages.AsyncMessageStreamManager``
    just enough for ``_generate_with_mcp_stream``: ``async with`` enters, the
    body iterates via ``async for event in stream``, then awaits
    ``stream.get_final_message()`` to retrieve the assembled ``Message``.
    """

    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        events = list(self._events)

        async def gen():
            for ev in events:
                yield ev

        return gen()

    async def get_final_message(self):
        return self._final_message


class TestGenerateWithMcpStreamTextDeltas:
    @pytest.mark.asyncio
    async def test_streams_text_deltas_as_thinking_delta_actions(self):
        """When async_anthropic_client is set, yield thinking_delta per text_delta event."""
        from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        # Build a fake stream: start text block, two text deltas, stop text block.
        text_block_start = MagicMock()
        text_block_start.type = "text"
        delta1 = MagicMock()
        delta1.type = "text_delta"
        delta1.text = "Hello, "
        delta2 = MagicMock()
        delta2.type = "text_delta"
        delta2.text = "world!"
        events = [
            _FakeStreamEvent("content_block_start", content_block=text_block_start),
            _FakeStreamEvent("content_block_delta", delta=delta1),
            _FakeStreamEvent("content_block_delta", delta=delta2),
            _FakeStreamEvent("content_block_stop"),
        ]
        final_msg = _make_response([_make_text_block("Hello, world!")])

        stream_manager = _FakeAsyncStreamManager(events, final_msg)

        async_client = MagicMock()
        async_client.messages.stream = MagicMock(return_value=stream_manager)
        model.async_anthropic_client = async_client

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
            ):
                actions.append(action)

        # Stream invoked once
        async_client.messages.stream.assert_called_once()

        # Exactly two thinking_delta events with incremental accumulation, both transient.
        delta_actions = [a for a in actions if a.action_type == "thinking_delta"]
        assert len(delta_actions) == 2
        assert all(a.role == ActionRole.ASSISTANT for a in delta_actions)
        assert all(a.status == ActionStatus.PROCESSING for a in delta_actions)
        assert delta_actions[0].output == {"delta": "Hello, ", "accumulated": "Hello, "}
        assert delta_actions[1].output == {"delta": "world!", "accumulated": "Hello, world!"}
        # Delta actions share one stream id (incremental display in CLI groups them).
        assert delta_actions[0].action_id == delta_actions[1].action_id

        # Paired terminal ``response`` SUCCESS emitted at content_block_stop —
        # CLI ``_print_completed_action`` consumes this to finalize the
        # markdown stream and clear the pinned live region. Without it the
        # tail paragraph stays visible AND gets reflushed via ``__exit__``,
        # producing a visible duplicate.
        responses = [a for a in actions if a.action_type == "response"]
        assert len(responses) == 1
        assert responses[0].role == ActionRole.ASSISTANT
        assert responses[0].status == ActionStatus.SUCCESS
        assert responses[0].output["raw_output"] == "Hello, world!"
        # Shares stream id with deltas so CLI dedup matches the paired turn.
        assert responses[0].action_id == delta_actions[0].action_id
        assert responses[0].output["is_thinking"] is False

        # Final assistant action carries the assembled text.
        finals = [a for a in actions if a.action_type == "final_response"]
        assert len(finals) == 1
        assert finals[0].output["raw_output"] == "Hello, world!"
        assert finals[0].status == ActionStatus.SUCCESS

        # Transient deltas are NOT persisted into the action history manager;
        # only the final_response should land there. The paired ``response``
        # action is also transient (yield-only, like delta) so it should not
        # land in the manager either.
        persisted_types = {a.action_type for a in ahm.actions}
        assert "thinking_delta" not in persisted_types
        assert "response" not in persisted_types
        assert "final_response" in persisted_types

    @pytest.mark.asyncio
    async def test_interrupt_during_stream_raises_execution_interrupted(self):
        """When ``interrupt_controller.is_interrupted`` is True during the stream
        loop, the generator must raise ``ExecutionInterrupted`` so the caller can
        unwind cleanly instead of yielding more delta events.
        """
        from datus.cli.execution_state import ExecutionInterrupted
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "ignored"
        events = [_FakeStreamEvent("content_block_delta", delta=delta)]
        final_msg = _make_response([_make_text_block("ignored")])
        stream_manager = _FakeAsyncStreamManager(events, final_msg)

        async_client = MagicMock()
        async_client.messages.stream = MagicMock(return_value=stream_manager)
        model.async_anthropic_client = async_client

        interrupt_ctrl = MagicMock()
        interrupt_ctrl.is_interrupted = True

        ahm = ActionHistoryManager()
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(ExecutionInterrupted):
                async for _ in model._generate_with_mcp_stream(
                    prompt="test",
                    mcp_servers={},
                    instruction="sys",
                    output_type={},
                    action_history_manager=ahm,
                    interrupt_controller=interrupt_ctrl,
                ):
                    pass

    @pytest.mark.asyncio
    async def test_non_text_delta_event_is_skipped(self):
        """A ``content_block_delta`` whose ``delta.type`` is not ``text_delta``
        (e.g. ``input_json_delta`` from tool argument streaming) must be ignored
        — no ``thinking_delta`` should be emitted for it.
        """
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        text_block_start = MagicMock()
        text_block_start.type = "text"
        json_delta = MagicMock()
        json_delta.type = "input_json_delta"
        json_delta.text = "{should-be-ignored}"
        real_delta = MagicMock()
        real_delta.type = "text_delta"
        real_delta.text = "kept"
        events = [
            _FakeStreamEvent("content_block_start", content_block=text_block_start),
            _FakeStreamEvent("content_block_delta", delta=json_delta),
            _FakeStreamEvent("content_block_delta", delta=real_delta),
            _FakeStreamEvent("content_block_stop"),
        ]
        final_msg = _make_response([_make_text_block("kept")])
        stream_manager = _FakeAsyncStreamManager(events, final_msg)

        async_client = MagicMock()
        async_client.messages.stream = MagicMock(return_value=stream_manager)
        model.async_anthropic_client = async_client

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
            ):
                actions.append(action)

        delta_actions = [a for a in actions if a.action_type == "thinking_delta"]
        assert len(delta_actions) == 1
        assert delta_actions[0].output["delta"] == "kept"

    @pytest.mark.asyncio
    async def test_empty_text_delta_is_skipped(self):
        """A ``text_delta`` event with an empty ``text`` field carries no payload
        and must be silently skipped so the CLI never receives a no-op delta.
        """
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        text_block_start = MagicMock()
        text_block_start.type = "text"
        empty_delta = MagicMock()
        empty_delta.type = "text_delta"
        empty_delta.text = ""
        real_delta = MagicMock()
        real_delta.type = "text_delta"
        real_delta.text = "real"
        events = [
            _FakeStreamEvent("content_block_start", content_block=text_block_start),
            _FakeStreamEvent("content_block_delta", delta=empty_delta),
            _FakeStreamEvent("content_block_delta", delta=real_delta),
            _FakeStreamEvent("content_block_stop"),
        ]
        final_msg = _make_response([_make_text_block("real")])
        stream_manager = _FakeAsyncStreamManager(events, final_msg)

        async_client = MagicMock()
        async_client.messages.stream = MagicMock(return_value=stream_manager)
        model.async_anthropic_client = async_client

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
            ):
                actions.append(action)

        delta_actions = [a for a in actions if a.action_type == "thinking_delta"]
        assert len(delta_actions) == 1
        assert delta_actions[0].output["delta"] == "real"

    @pytest.mark.asyncio
    async def test_text_delta_without_prior_block_start_gets_fresh_stream_id(self):
        """If a ``text_delta`` arrives before any ``content_block_start`` is seen,
        a stream id must be created inline so the CLI can still pair the deltas.
        """
        from datus.schemas.action_history import ActionHistoryManager

        cfg = _make_model_config(use_native_api=True)
        model = _make_claude_model(cfg)

        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "orphan"
        events = [_FakeStreamEvent("content_block_delta", delta=delta)]
        final_msg = _make_response([_make_text_block("orphan")])
        stream_manager = _FakeAsyncStreamManager(events, final_msg)

        async_client = MagicMock()
        async_client.messages.stream = MagicMock(return_value=stream_manager)
        model.async_anthropic_client = async_client

        ahm = ActionHistoryManager()
        actions = []
        with patch("datus.models.claude_model.multiple_mcp_servers") as mock_mcp:
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            async for action in model._generate_with_mcp_stream(
                prompt="test",
                mcp_servers={},
                instruction="sys",
                output_type={},
                action_history_manager=ahm,
            ):
                actions.append(action)

        delta_actions = [a for a in actions if a.action_type == "thinking_delta"]
        assert len(delta_actions) == 1
        # The stream id was minted on the delta path (no prior content_block_start
        # set one), proving the fallback at line 549 fired.
        assert delta_actions[0].action_id.startswith("thinking_stream_")


# ---------------------------------------------------------------------------
# Proxy client init via HTTP_PROXY / HTTPS_PROXY env vars
# ---------------------------------------------------------------------------


class TestProxyClientInit:
    def test_async_proxy_client_created_when_http_proxy_env_set(self):
        """When ``HTTP_PROXY`` is set, both sync and async proxy clients must be
        wired up — the async client backs the streaming code path, which would
        otherwise bypass the proxy and leak traffic.
        """
        cfg = _make_model_config(api_key="sk-ant-test")
        with (
            patch("datus.models.openai_compatible.LiteLLMAdapter") as mock_adapter_cls,
            patch("anthropic.Anthropic"),
            patch("anthropic.AsyncAnthropic"),
            patch("langsmith.wrappers.wrap_anthropic", side_effect=lambda c: c),
            patch.dict(
                "os.environ",
                {"HTTP_PROXY": "http://proxy.example.com:8080", "ANTHROPIC_API_KEY": "sk-ant-test"},
                clear=True,
            ),
        ):
            mock_adapter = MagicMock()
            mock_adapter.litellm_model_name = "anthropic/claude-sonnet-4-5"
            mock_adapter.provider = "anthropic"
            mock_adapter.is_thinking_model = False
            mock_adapter.get_agents_sdk_model.return_value = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            model = ClaudeModel(cfg)

        import httpx

        # Both sides of the client pair must be the matching httpx flavour —
        # an async-only or sync-only setup would still pass ``is not None`` but
        # would break the streaming path the test is meant to protect.
        assert isinstance(model.proxy_client, httpx.Client)
        assert isinstance(model.async_proxy_client, httpx.AsyncClient)


# ---------------------------------------------------------------------------
# _anthropic_messages_stream routing
# ---------------------------------------------------------------------------


class TestAnthropicMessagesStream:
    def test_raises_when_async_client_is_none(self):
        """Streaming without an initialised async client must raise
        ``MODEL_AUTHENTICATION_ERROR`` so callers can fall back to the
        non-streaming path instead of silently hanging.
        """
        from datus.utils.exceptions import DatusException, ErrorCode

        model = _make_claude_model()
        model.async_anthropic_client = None

        with pytest.raises(DatusException) as exc_info:
            model._anthropic_messages_stream(model="test", messages=[])
        assert exc_info.value.code == ErrorCode.MODEL_AUTHENTICATION_ERROR

    def test_routes_to_beta_stream_for_oauth_token(self):
        """OAuth subscription tokens require the OAuth beta endpoint —
        ``beta.messages.stream`` — because the standard endpoint rejects Bearer auth.
        """
        cfg = _make_model_config(api_key="sk-ant-oat01-token", auth_type="subscription")
        model = _make_claude_model(cfg)

        async_client = MagicMock()
        async_client.beta.messages.stream.return_value = "beta-stream-ctx"
        async_client.messages.stream.return_value = "standard-stream-ctx"
        model.async_anthropic_client = async_client

        result = model._anthropic_messages_stream(model="m", messages=[])

        assert result == "beta-stream-ctx"
        async_client.beta.messages.stream.assert_called_once()
        async_client.messages.stream.assert_not_called()

    def test_routes_to_standard_stream_for_api_key(self):
        """Standard ``x-api-key`` auth must route to ``messages.stream`` —
        the beta endpoint is reserved for OAuth and would reject the request.
        """
        cfg = _make_model_config(api_key="sk-ant-regular", auth_type="api_key")
        model = _make_claude_model(cfg)

        async_client = MagicMock()
        async_client.beta.messages.stream.return_value = "beta-stream-ctx"
        async_client.messages.stream.return_value = "standard-stream-ctx"
        model.async_anthropic_client = async_client

        result = model._anthropic_messages_stream(model="m", messages=[])

        assert result == "standard-stream-ctx"
        async_client.messages.stream.assert_called_once()
        async_client.beta.messages.stream.assert_not_called()


# ---------------------------------------------------------------------------
# aclose: async proxy + async anthropic client cleanup
# ---------------------------------------------------------------------------


class TestAcloseAsyncClients:
    @pytest.mark.asyncio
    async def test_aclose_closes_async_proxy_client(self):
        """A configured ``async_proxy_client`` must be awaited-closed on aclose."""
        model = _make_claude_model()
        async_proxy = MagicMock()
        async_proxy.aclose = AsyncMock()
        model.async_proxy_client = async_proxy

        await model.aclose()

        async_proxy.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_logs_warning_on_async_proxy_close_failure(self):
        """Exceptions from ``async_proxy_client.aclose()`` must be swallowed and
        logged as a warning — cleanup failures must not propagate and abort shutdown.
        """
        model = _make_claude_model()
        async_proxy = MagicMock()
        async_proxy.aclose = AsyncMock(side_effect=RuntimeError("net error"))
        model.async_proxy_client = async_proxy

        with patch("datus.models.claude_model.logger") as mock_logger:
            await model.aclose()

        warned = [str(c.args[0]) for c in mock_logger.warning.call_args_list]
        assert any("async proxy client" in w.lower() and "net error" in w for w in warned), (
            f"Expected warning naming 'async proxy client' with the underlying error, got: {warned}"
        )

    @pytest.mark.asyncio
    async def test_aclose_closes_async_anthropic_client(self):
        """A configured ``async_anthropic_client`` must be awaited-closed on aclose."""
        model = _make_claude_model()
        async_client = MagicMock()
        async_client.close = AsyncMock()
        model.async_anthropic_client = async_client

        await model.aclose()

        async_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_logs_warning_on_async_anthropic_close_failure(self):
        """Exceptions from ``async_anthropic_client.close()`` must be swallowed
        and logged as a warning — keeping shutdown best-effort.
        """
        model = _make_claude_model()
        async_client = MagicMock()
        async_client.close = AsyncMock(side_effect=RuntimeError("already closed"))
        model.async_anthropic_client = async_client

        with patch("datus.models.claude_model.logger") as mock_logger:
            await model.aclose()

        warned = [str(c.args[0]) for c in mock_logger.warning.call_args_list]
        assert any("async anthropic client" in w.lower() and "already closed" in w for w in warned), (
            f"Expected warning naming 'async anthropic client' with the underlying error, got: {warned}"
        )
