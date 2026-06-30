# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for Codex model."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.configuration.agent_config import ModelConfig
from datus.utils.exceptions import DatusException


def _mock_stream(output_text: str):
    """Create a mock stream that yields a response.completed event."""
    event = MagicMock()
    event.type = "response.completed"
    event.response.output_text = output_text
    return [event]


def _mock_streamed_result(final_output, turn_count=0):
    """Create a mock streaming result that appears already complete."""
    mock = MagicMock()
    mock.is_complete = True
    mock.final_output = final_output
    mock.turn_count = turn_count
    return mock


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

    def test_default_constants_present(self, model_config, mock_oauth):
        """Defaults are used when providers.yml / cache do not cover the slug."""
        from datus.models.codex_model import _CODEX_DEFAULT_CONTEXT_LENGTH, _CODEX_DEFAULT_MAX_TOKENS

        assert _CODEX_DEFAULT_CONTEXT_LENGTH == 192000
        assert _CODEX_DEFAULT_MAX_TOKENS == 16384


class TestCodexModelGenerate:
    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_string_prompt(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _mock_stream("Hello from Codex!")
        model._client = mock_client

        result = model.generate("Say hello")
        assert result == "Hello from Codex!"
        mock_client.responses.create.assert_called_once_with(
            model="gpt-5.3-codex",
            input=[{"role": "user", "content": "Say hello"}],
            store=False,
            stream=True,
        )

    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_list_prompt(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _mock_stream("Result")
        model._client = mock_client

        messages = [{"role": "user", "content": "Hi"}]
        result = model.generate(messages)
        assert result == "Result"

    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_401_retry(self, mock_oauth_cls, model_config):
        from openai import AuthenticationError

        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()

        # First call raises AuthenticationError, second succeeds
        auth_error = AuthenticationError(
            message="Unauthorized",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )
        mock_client.responses.create.side_effect = [
            auth_error,
            _mock_stream("Retried result"),
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
        mock_client.responses.create.return_value = _mock_stream(json.dumps({"sql": "SELECT 1"}))
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
        mock_client.responses.create.return_value = _mock_stream(json.dumps({"answer": 42}))
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
    def test_context_length_prefix_match_codex_mini_latest(self, mock_oauth_cls):
        """``codex-mini-latest`` should inherit ``codex-mini``'s 192K window from providers.yml."""
        import datus.models.openai_compatible as oc
        from datus.models.codex_model import CodexModel

        original_cache = oc._MODEL_SPECS_CACHE
        oc._MODEL_SPECS_CACHE = None  # force reload from providers.yml
        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(type="codex", api_key="", model="codex-mini-latest", auth_type="oauth")
        model = CodexModel(model_config=config)
        try:
            assert model.context_length() == 192000
            assert model.max_tokens() == 16384
        finally:
            oc._MODEL_SPECS_CACHE = original_cache

    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.cli.provider_model_catalog.load_cached_model_details", return_value=None)
    def test_codex_model_uses_openai_spec_via_prefix(self, _mock_cache, mock_oauth_cls):
        """A Codex slug like ``gpt-5.4-codex`` uses the OpenAI ``gpt-5`` spec via prefix matching."""
        import datus.models.openai_compatible as oc
        from datus.models.codex_model import CodexModel

        original_cache = oc._MODEL_SPECS_CACHE
        oc._MODEL_SPECS_CACHE = None
        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(type="codex", api_key="", model="gpt-5.4-codex", auth_type="oauth")
        model = CodexModel(model_config=config)
        try:
            assert model.context_length() == 1050000
        finally:
            oc._MODEL_SPECS_CACHE = original_cache

    @patch("datus.models.codex_model.OAuthManager")
    def test_o3_codex_uses_its_larger_max_tokens(self, mock_oauth_cls):
        """Direct match for ``o3-codex`` in providers.yml should yield its 100K max_tokens."""
        import datus.models.openai_compatible as oc
        from datus.models.codex_model import CodexModel

        original_cache = oc._MODEL_SPECS_CACHE
        oc._MODEL_SPECS_CACHE = None
        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(type="codex", api_key="", model="o3-codex", auth_type="oauth")
        model = CodexModel(model_config=config)
        try:
            assert model.context_length() == 200000
            assert model.max_tokens() == 100000
        finally:
            oc._MODEL_SPECS_CACHE = original_cache

    @patch("datus.models.codex_model.OAuthManager")
    def test_convert_prompt_to_input(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        assert CodexModel._convert_prompt_to_input("hello") == [{"role": "user", "content": "hello"}]
        assert CodexModel._convert_prompt_to_input([{"role": "user"}]) == [{"role": "user"}]
        assert CodexModel._convert_prompt_to_input(123) == [{"role": "user", "content": "123"}]


class TestCodexModelClientInit:
    @patch("datus.models.codex_model.OAuthManager")
    def test_get_client_lazy_init(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        assert model._client is None

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client = model._get_client()
            assert client is mock_openai.return_value
            assert model._client is client
            # Second call returns cached client
            client2 = model._get_client()
            assert client2 is client

    @patch("datus.models.codex_model.OAuthManager")
    def test_get_async_client_lazy_init(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        assert model._async_client is None

        with patch("openai.AsyncOpenAI") as mock_async:
            mock_async.return_value = MagicMock()
            client = model._get_async_client()
            assert client is mock_async.return_value
            assert model._async_client is client
            # Second call returns cached
            client2 = model._get_async_client()
            assert client2 is client

    @patch("datus.models.codex_model.OAuthManager")
    def test_get_responses_model(self, mock_oauth_cls, model_config):
        from agents.models.openai_responses import OpenAIResponsesModel

        from datus.models.codex_model import CodexModel, _CodexResponsesModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        with patch("openai.AsyncOpenAI") as mock_async:
            mock_async.return_value = MagicMock()
            resp_model = model._get_responses_model()
            assert isinstance(resp_model, _CodexResponsesModel)
            assert isinstance(resp_model, OpenAIResponsesModel)
            assert resp_model.model == model_config.model

    @patch("datus.models.codex_model.OAuthManager")
    def test_refresh_client_token_both_clients(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "new_tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._client = MagicMock()
        model._async_client = MagicMock()

        model._refresh_client_token()
        assert model._client.api_key == "new_tok"
        assert model._async_client.api_key == "new_tok"


class TestCodexModelGenerateNonAuthError:
    @patch("datus.models.codex_model.OAuthManager")
    def test_generate_reraises_non_auth_error(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = ValueError("some other error")
        model._client = mock_client

        with pytest.raises(DatusException, match="Codex generate failed"):
            model.generate("test")


class TestCodexModelGenerateWithTools:
    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    @patch("datus.models.codex_model.extract_sql_contexts")
    async def test_generate_with_tools_basic(
        self, mock_extract, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        # Mock the responses model
        with patch("agents.models.openai_responses.OpenAIResponsesModel") as mock_resp:
            mock_resp.return_value = MagicMock()

            # Mock MCP context manager
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            # Mock Runner.run_streamed result (Codex requires streaming)
            mock_runner.run_streamed.return_value = _mock_streamed_result("SQL result", 2)
            mock_extract.return_value = [{"sql": "SELECT 1"}]

            result = await model.generate_with_tools(prompt="Generate SQL", instruction="You are a SQL expert")

            assert result["content"] == "SQL result"
            assert result["model"] == "gpt-5.3-codex"
            assert result["sql_contexts"] == [{"sql": "SELECT 1"}]

    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    async def test_generate_with_tools_max_turns_exceeded(
        self, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from agents.exceptions import MaxTurnsExceeded

        from datus.models.codex_model import CodexModel
        from datus.utils.exceptions import DatusException

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with patch("agents.models.openai_responses.OpenAIResponsesModel"):
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_runner.run_streamed.side_effect = MaxTurnsExceeded("exceeded")

            with pytest.raises(DatusException, match="Maximum turns"):
                await model.generate_with_tools(prompt="test", max_turns=5)

    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    async def test_generate_with_tools_with_mcp_and_tools(
        self, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with (
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("datus.models.codex_model.extract_sql_contexts", return_value=[]),
        ):
            mock_server = MagicMock()
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={"db": mock_server})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_runner.run_streamed.return_value = _mock_streamed_result("done", 1)

            mock_tool = MagicMock()
            result = await model.generate_with_tools(
                prompt="test", tools=[mock_tool], mcp_servers={"db": MagicMock()}, hooks=MagicMock()
            )
            assert result["content"] == "done"


class TestCodexModelGenerateWithToolsStream:
    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    @patch("datus.models.codex_model.extract_sql_contexts")
    async def test_stream_basic_final_output(
        self, mock_extract, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with patch("agents.models.openai_responses.OpenAIResponsesModel"):
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            # Mock streamed result - already complete, no events
            mock_result = MagicMock()
            mock_result.is_complete = True
            mock_result.final_output = "Final answer"
            mock_runner.run_streamed.return_value = mock_result
            mock_extract.return_value = []

            actions = []
            async for action in model.generate_with_tools_stream(prompt="test"):
                actions.append(action)

            assert len(actions) == 1  # final action only
            assert actions[0].action_type == "final_response"
            assert "Final answer" in actions[0].output["raw_output"]

    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    @patch("datus.models.codex_model.extract_sql_contexts")
    async def test_stream_with_tool_call_events(
        self, mock_extract, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with patch("agents.models.openai_responses.OpenAIResponsesModel"):
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            # Build mock events
            tool_call_event = MagicMock()
            tool_call_event.type = "run_item_stream_event"
            tool_call_event.item.type = "tool_call_item"
            tool_call_event.item.raw_item.name = "execute_sql"
            tool_call_event.item.raw_item.call_id = "call_123"
            tool_call_event.item.raw_item.arguments = '{"sql": "SELECT 1"}'

            tool_output_event = MagicMock()
            tool_output_event.type = "run_item_stream_event"
            tool_output_event.item.type = "tool_call_output_item"
            tool_output_event.item.output = "1 row returned"
            tool_output_event.item.raw_item = {"call_id": "call_123"}

            mock_result = MagicMock()
            mock_result.final_output = "Done"

            # Use PropertyMock to control is_complete across iterations
            is_complete_values = iter([False, True])
            type(mock_result).is_complete = property(lambda self: next(is_complete_values))

            async def stream_events():
                yield tool_call_event
                yield tool_output_event

            mock_result.stream_events = stream_events

            mock_runner.run_streamed.return_value = mock_result
            mock_extract.return_value = []

            actions = []
            async for action in model.generate_with_tools_stream(prompt="test"):
                actions.append(action)

            # Should have: tool_call, tool_output, final
            assert len(actions) == 3
            assert actions[0].action_type == "execute_sql"
            assert str(actions[0].status) == "processing"
            assert actions[1].action_type == "execute_sql"
            assert str(actions[1].status) == "success"
            assert actions[2].action_type == "final_response"

    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    @patch("datus.models.codex_model.extract_sql_contexts")
    async def test_stream_with_raw_response_text(
        self, mock_extract, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with patch("agents.models.openai_responses.OpenAIResponsesModel"):
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            # Build a raw_response_event with message text
            text_part = MagicMock()
            text_part.text = "Here is the answer"
            raw_item = MagicMock()
            raw_item.type = "message"
            raw_item.content = [text_part]
            raw_data = MagicMock()
            raw_data.type = "response.output_item.done"
            raw_data.item = raw_item
            raw_event = MagicMock()
            raw_event.type = "raw_response_event"
            raw_event.data = raw_data

            mock_result = MagicMock()
            mock_result.final_output = "Here is the answer"

            is_complete_values = iter([False, True])
            type(mock_result).is_complete = property(lambda self: next(is_complete_values))

            async def stream_events():
                yield raw_event

            mock_result.stream_events = stream_events
            mock_runner.run_streamed.return_value = mock_result
            mock_extract.return_value = []

            actions = []
            async for action in model.generate_with_tools_stream(prompt="test"):
                actions.append(action)

            # Should have: assistant text + final
            assert len(actions) == 2
            assert str(actions[0].role) == "assistant"
            assert "Here is the answer" in actions[0].output["raw_output"]

    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    async def test_stream_with_tools_and_hooks(
        self, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with (
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("datus.models.codex_model.extract_sql_contexts", return_value=[]),
        ):
            mock_server = MagicMock()
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={"db": mock_server})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_result = MagicMock()
            mock_result.is_complete = True
            mock_result.final_output = "done"
            mock_runner.run_streamed.return_value = mock_result

            actions = []
            async for action in model.generate_with_tools_stream(
                prompt="test",
                tools=[MagicMock()],
                mcp_servers={"db": MagicMock()},
                hooks=MagicMock(),
            ):
                actions.append(action)

            assert len(actions) == 1
            # Verify agent was created with mcp_servers and tools
            agent_kwargs = mock_agent_cls.call_args[1]
            assert "mcp_servers" in agent_kwargs
            assert "tools" in agent_kwargs
            assert "hooks" in agent_kwargs


class TestCodexModelBaseUrl:
    @patch("datus.models.codex_model.OAuthManager")
    def test_base_url_from_config(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        model = CodexModel(model_config=model_config)
        assert model._base_url == "https://chatgpt.com/backend-api/codex"

    @patch("datus.models.codex_model.OAuthManager")
    def test_base_url_defaults_when_not_set(self, mock_oauth_cls):
        from datus.models.codex_model import CODEX_API_BASE_URL, CodexModel

        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(type="codex", api_key="", model="gpt-5.3-codex", auth_type="oauth")
        model = CodexModel(model_config=config)
        assert model._base_url == CODEX_API_BASE_URL

    @patch("datus.models.codex_model.OAuthManager")
    def test_custom_base_url(self, mock_oauth_cls):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(
            type="codex",
            api_key="",
            model="gpt-5.3-codex",
            base_url="https://my-proxy.example.com/codex",
            auth_type="oauth",
        )
        model = CodexModel(model_config=config)
        assert model._base_url == "https://my-proxy.example.com/codex"


class TestCodexSupportsBuiltinWebSearch:
    """Hosted ``web_search`` is gated on the official ChatGPT Codex host.

    A custom ``base_url`` points at a third-party relay that may not support the
    hosted tool, so it must fall back to the local Tavily backend.
    """

    @patch("datus.models.codex_model.OAuthManager")
    def test_official_codex_enables_web_search(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        model = CodexModel(model_config=model_config)
        assert model.supports_builtin_web_search() is True

    @patch("datus.models.codex_model.OAuthManager")
    def test_default_base_url_enables_web_search(self, mock_oauth_cls):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(type="codex", api_key="", model="gpt-5.3-codex", auth_type="oauth")
        model = CodexModel(model_config=config)
        assert model.supports_builtin_web_search() is True

    @patch("datus.models.codex_model.OAuthManager")
    def test_custom_proxy_disables_web_search(self, mock_oauth_cls):
        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(
            type="codex",
            api_key="",
            model="gpt-5.3-codex",
            base_url="https://my-proxy.example.com/codex",
            auth_type="oauth",
        )
        model = CodexModel(model_config=config)
        assert model.supports_builtin_web_search() is False


class TestCodexBuiltinWebSearchInjection:
    """The hosted ``WebSearchTool`` must only be injected when
    ``supports_builtin_web_search()`` is True — i.e. on the official ChatGPT
    Codex host. Custom relay hosts fall back to the local Tavily backend.
    """

    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    async def test_official_host_injects_web_search_tool(
        self, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from agents import WebSearchTool

        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with (
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("datus.models.codex_model.extract_sql_contexts", return_value=[]),
        ):
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_runner.run_streamed.return_value = _mock_streamed_result("done", 1)

            await model.generate_with_tools(prompt="test", builtin_web_tools={"web_search": True})

        injected = mock_agent_cls.call_args.kwargs.get("tools") or []
        assert any(isinstance(t, WebSearchTool) for t in injected)

    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    async def test_custom_proxy_does_not_inject_web_search_tool(
        self, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls
    ):
        from agents import WebSearchTool

        from datus.models.codex_model import CodexModel

        mock_oauth_cls.return_value = MagicMock()
        config = ModelConfig(
            type="codex",
            api_key="",
            model="gpt-5.3-codex",
            base_url="https://my-proxy.example.com/codex",
            auth_type="oauth",
        )
        model = CodexModel(model_config=config)
        model._async_client = MagicMock()

        with (
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("datus.models.codex_model.extract_sql_contexts", return_value=[]),
        ):
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_runner.run_streamed.return_value = _mock_streamed_result("done", 1)

            await model.generate_with_tools(prompt="test", builtin_web_tools={"web_search": True})

        injected = mock_agent_cls.call_args.kwargs.get("tools") or []
        assert not any(isinstance(t, WebSearchTool) for t in injected)


class TestCodexModelJsonOutput401Retry:
    @patch("datus.models.codex_model.OAuthManager")
    def test_json_output_401_retry(self, mock_oauth_cls, model_config):
        from openai import AuthenticationError

        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        mock_client = MagicMock()

        auth_error = AuthenticationError(
            message="Unauthorized",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )
        mock_client.responses.create.side_effect = [auth_error, _mock_stream(json.dumps({"sql": "SELECT 1"}))]
        model._client = mock_client

        result = model.generate_with_json_output("test")
        assert result == {"sql": "SELECT 1"}
        mock_oauth.refresh_tokens.assert_called_once()

    @patch("datus.models.codex_model.OAuthManager")
    def test_json_output_reraises_non_auth_error(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = ValueError("some error")
        model._client = mock_client

        with pytest.raises(DatusException, match="Codex generate_with_json_output failed"):
            model.generate_with_json_output("test")


class TestCodexModelToolsAuth401Retry:
    @pytest.mark.asyncio
    @patch("datus.models.codex_model.OAuthManager")
    @patch("datus.models.codex_model.multiple_mcp_servers")
    @patch("datus.models.codex_model.Runner")
    @patch("datus.models.codex_model.Agent")
    @patch("datus.models.codex_model.extract_sql_contexts")
    async def test_generate_with_tools_401_retry(
        self, mock_extract, mock_agent_cls, mock_runner, mock_mcp, mock_oauth_cls, model_config
    ):
        from openai import AuthenticationError

        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)
        model._async_client = MagicMock()

        with patch("agents.models.openai_responses.OpenAIResponsesModel"):
            mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
            mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

            auth_error = AuthenticationError(
                message="Unauthorized",
                response=MagicMock(status_code=401, headers={}),
                body=None,
            )
            mock_runner.run_streamed.side_effect = auth_error
            mock_extract.return_value = []

            # After review: no longer retries full run (tool side-effects risk).
            # Instead refreshes token and raises DatusException.
            from datus.utils.exceptions import DatusException

            with pytest.raises(DatusException, match="Authentication failed"):
                await model.generate_with_tools(prompt="test")

            mock_oauth.refresh_tokens.assert_called_once()


class TestCodexResponsesModelPatch:
    """Tests for _CodexResponsesModel.stream_response output-injection patch."""

    @pytest.mark.asyncio
    async def test_stream_response_injects_output_items(self, model_config):
        """Items from ResponseOutputItemDoneEvent are injected into ResponseCompletedEvent.output."""
        # Create a real function_call item
        from openai.types.responses import ResponseCompletedEvent, ResponseFunctionToolCall, ResponseOutputItemDoneEvent

        from datus.models.codex_model import _CodexResponsesModel

        tool_item = ResponseFunctionToolCall(
            id="fc_abc",
            call_id="call_abc",
            name="glob",
            arguments='{"pattern":"*"}',
            type="function_call",
            status="completed",
        )
        done_event = ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=tool_item,
            output_index=0,
            sequence_number=0,
        )

        # Build a completed event with empty output
        from openai.types.responses import Response

        mock_response = MagicMock(spec=Response)
        mock_response.output = []

        def _copy_response(update=None, **kw):
            new = MagicMock(spec=Response)
            new.output = (update or {}).get("output", mock_response.output)
            return new

        mock_response.model_copy = _copy_response

        mock_completed_event = MagicMock(spec=ResponseCompletedEvent)
        mock_completed_event.__class__ = ResponseCompletedEvent
        mock_completed_event.response = mock_response

        patched_events = []

        def _copy_event(update=None, **kw):
            new_evt = MagicMock(spec=ResponseCompletedEvent)
            new_evt.__class__ = ResponseCompletedEvent
            new_evt.response = (update or {}).get("response", mock_completed_event.response)
            patched_events.append(new_evt)
            return new_evt

        mock_completed_event.model_copy = _copy_event

        async def _fake_parent_stream(*args, **kwargs):
            yield done_event
            yield mock_completed_event

        instance = MagicMock(spec=_CodexResponsesModel)
        instance.__class__ = _CodexResponsesModel

        with patch.object(_CodexResponsesModel, "stream_response", new=_CodexResponsesModel.stream_response):
            with patch(
                "agents.models.openai_responses.OpenAIResponsesModel.stream_response",
                new=_fake_parent_stream,
            ):
                collected_yielded = []
                async for evt in _CodexResponsesModel.stream_response(instance):
                    collected_yielded.append(evt)

        # Should have yielded 2 events
        assert len(collected_yielded) == 2
        # First event: unchanged ResponseOutputItemDoneEvent
        assert isinstance(collected_yielded[0], ResponseOutputItemDoneEvent)
        # Second event: patched ResponseCompletedEvent with injected output
        assert isinstance(collected_yielded[1], ResponseCompletedEvent)
        assert len(patched_events) == 1
        assert patched_events[0].response.output == [tool_item]

    @pytest.mark.asyncio
    async def test_stream_response_excludes_reasoning_items(self, model_config):
        """Reference-only reasoning items (no ``encrypted_content``) are NOT
        injected into ResponseCompletedEvent.output — replaying them under
        store=False would be rejected by the server."""
        from openai.types.responses import (
            Response,
            ResponseCompletedEvent,
            ResponseFunctionToolCall,
            ResponseOutputItemDoneEvent,
            ResponseReasoningItem,
        )

        from datus.models.codex_model import _CodexResponsesModel

        reasoning_item = ResponseReasoningItem(
            id="rs_abc",
            type="reasoning",
            summary=[],
        )
        tool_item = ResponseFunctionToolCall(
            id="fc_abc",
            call_id="call_abc",
            name="glob",
            arguments="{}",
            type="function_call",
            status="completed",
        )
        done_reasoning = ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=reasoning_item,
            output_index=0,
            sequence_number=0,
        )
        done_tool = ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=tool_item,
            output_index=1,
            sequence_number=1,
        )

        mock_response = MagicMock(spec=Response)
        mock_response.output = []

        injected_outputs = []

        def _copy_response(update=None, **kw):
            new = MagicMock(spec=Response)
            new.output = (update or {}).get("output", [])
            injected_outputs.extend(new.output)
            return new

        mock_response.model_copy = _copy_response

        mock_completed = MagicMock(spec=ResponseCompletedEvent)
        mock_completed.__class__ = ResponseCompletedEvent
        mock_completed.response = mock_response
        mock_completed.model_copy = lambda update=None, **kw: MagicMock(
            spec=ResponseCompletedEvent, response=(update or {}).get("response", mock_response)
        )

        async def _fake_parent_stream(*args, **kwargs):
            yield done_reasoning
            yield done_tool
            yield mock_completed

        instance = MagicMock(spec=_CodexResponsesModel)
        instance.__class__ = _CodexResponsesModel

        with patch(
            "agents.models.openai_responses.OpenAIResponsesModel.stream_response",
            new=_fake_parent_stream,
        ):
            async for _ in _CodexResponsesModel.stream_response(instance):
                pass

        # Only the tool_item should be injected; reasoning_item must be excluded
        assert injected_outputs == [tool_item]
        assert reasoning_item not in injected_outputs

    @pytest.mark.asyncio
    async def test_stream_response_leaves_nonempty_output_unchanged(self, model_config):
        """ResponseCompletedEvent with existing output is not modified."""
        from openai.types.responses import ResponseCompletedEvent, ResponseFunctionToolCall, ResponseOutputItemDoneEvent

        from datus.models.codex_model import _CodexResponsesModel

        tool_item = ResponseFunctionToolCall(
            id="fc_xyz",
            call_id="call_xyz",
            name="grep",
            arguments="{}",
            type="function_call",
            status="completed",
        )

        from openai.types.responses import Response

        mock_response = MagicMock(spec=Response)
        mock_response.output = [tool_item]  # Already populated
        mock_response.model_copy = MagicMock(side_effect=AssertionError("should not be called"))

        mock_completed_event = MagicMock(spec=ResponseCompletedEvent)
        mock_completed_event.__class__ = ResponseCompletedEvent
        mock_completed_event.response = mock_response
        mock_completed_event.model_copy = MagicMock(side_effect=AssertionError("should not be called"))

        done_event = ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=tool_item,
            output_index=0,
            sequence_number=0,
        )

        async def _fake_parent_stream(*args, **kwargs):
            yield done_event
            yield mock_completed_event

        instance = MagicMock(spec=_CodexResponsesModel)
        instance.__class__ = _CodexResponsesModel

        with patch(
            "agents.models.openai_responses.OpenAIResponsesModel.stream_response",
            new=_fake_parent_stream,
        ):
            yielded = []
            async for evt in _CodexResponsesModel.stream_response(instance):
                yielded.append(evt)

        assert len(yielded) == 2
        # Completed event must be the exact same object (no copy made)
        assert yielded[1] is mock_completed_event

    @patch("datus.models.codex_model.OAuthManager")
    def test_get_responses_model_returns_codex_subclass(self, mock_oauth_cls, model_config):
        """_get_responses_model() returns _CodexResponsesModel, not plain OpenAIResponsesModel."""
        from agents.models.openai_responses import OpenAIResponsesModel

        from datus.models.codex_model import CodexModel, _CodexResponsesModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        with patch("openai.AsyncOpenAI"):
            model = CodexModel(model_config=model_config)
            responses_model = model._get_responses_model()

        assert type(responses_model) is _CodexResponsesModel
        assert issubclass(_CodexResponsesModel, OpenAIResponsesModel)
        assert responses_model.model == model_config.model


class TestCodexCacheIdentity:
    """Codex prompt caching is only honoured by the backend when the request
    mimics the official client: a real-timestamp UUIDv7 prompt_cache_key reused
    across the conversation, plus codex identity headers (originator /
    session_id / thread_id / chatgpt-account-id). Packet capture confirmed that
    with these the backend echoes our key (accepted) and caches ~84%, vs 0%
    without."""

    def _model(self, model_config, mock_oauth):
        from datus.models.codex_model import CodexModel

        m = CodexModel(model_config=model_config)
        m.current_node = None
        return m

    def test_generate_uuid7_is_v7_with_recent_timestamp(self, model_config, mock_oauth):
        import time
        import uuid

        from datus.models.codex_model import CodexModel

        key = CodexModel._generate_uuid7()
        u = uuid.UUID(key)
        assert u.version == 7
        embedded_ms = int.from_bytes(u.bytes[:6], "big")
        now_ms = int(time.time() * 1000)
        # Embedded timestamp must be sane/recent (within ~1 day), else the
        # backend rejects it as a malformed v7.
        assert abs(now_ms - embedded_ms) < 24 * 3600 * 1000

    def test_prompt_cache_key_stable_and_distinct(self, model_config, mock_oauth):
        model = self._model(model_config, mock_oauth)

        class _S1:
            session_id = "sess-1"

        class _S2:
            session_id = "sess-2"

        k1 = model._stable_prompt_cache_key("chat", _S1())
        # Same conversation → identical key reused for every call.
        assert k1 == model._stable_prompt_cache_key("chat", _S1())
        # Distinct conversation/agent → distinct key.
        assert k1 != model._stable_prompt_cache_key("chat", _S2())
        assert k1 != model._stable_prompt_cache_key("explore", _S1())

    def test_chatgpt_account_id_decoded_from_jwt(self, model_config, mock_oauth):
        import base64
        import json

        model = self._model(model_config, mock_oauth)
        payload = (
            base64.urlsafe_b64encode(
                json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-xyz"}}).encode()
            )
            .decode()
            .rstrip("=")
        )
        model._get_access_token = lambda: f"hdr.{payload}.sig"
        assert model._chatgpt_account_id() == "acct-xyz"

    def test_chatgpt_account_id_none_on_bad_token(self, model_config, mock_oauth):
        model = self._model(model_config, mock_oauth)
        model._get_access_token = lambda: "not-a-jwt"
        assert model._chatgpt_account_id() is None

    def test_model_settings_carry_identity_headers_and_key(self, model_config, mock_oauth):
        model = self._model(model_config, mock_oauth)
        model._get_access_token = lambda: "x.x.x"  # _chatgpt_account_id returns None

        class _Session:
            session_id = "sess-a"

        ms = model._codex_model_settings("chat", _Session())
        key = ms.extra_args["prompt_cache_key"]
        h = ms.extra_headers
        # The key is reused as session_id / thread_id / x-client-request-id.
        assert h["originator"] == "codex_exec"
        assert h["session_id"] == key
        assert h["thread_id"] == key
        assert h["x-client-request-id"] == key
        # client_metadata (installation id) is intentionally NOT sent — a live
        # capture confirmed the backend honours the key and caches ~93% without
        # it, so we keep the request minimal.
        assert ms.extra_body is None
        assert ms.store is False and ms.include_usage is True

    def test_model_settings_includes_account_id_when_available(self, model_config, mock_oauth):
        import base64
        import json

        model = self._model(model_config, mock_oauth)
        payload = (
            base64.urlsafe_b64encode(
                json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-42"}}).encode()
            )
            .decode()
            .rstrip("=")
        )
        model._get_access_token = lambda: f"h.{payload}.s"

        class _Session:
            session_id = "sess-b"

        ms = model._codex_model_settings("chat", _Session())
        assert ms.extra_headers["chatgpt-account-id"] == "acct-42"
