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

        with (
            patch("datus.models.openai_compatible.setup_tracing"),
            patch("datus.models.openai_compatible.LiteLLMAdapter"),
            patch("anthropic.Anthropic"),
            patch.dict("os.environ", {}, clear=True),
        ):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ClaudeModel(cfg)


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
