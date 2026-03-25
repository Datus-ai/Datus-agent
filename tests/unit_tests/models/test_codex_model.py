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
            assert client is not None
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
            assert client is not None
            assert model._async_client is client
            # Second call returns cached
            client2 = model._get_async_client()
            assert client2 is client

    @patch("datus.models.codex_model.OAuthManager")
    def test_get_responses_model(self, mock_oauth_cls, model_config):
        from datus.models.codex_model import CodexModel

        mock_oauth = MagicMock()
        mock_oauth.get_access_token.return_value = "tok"
        mock_oauth_cls.return_value = mock_oauth

        model = CodexModel(model_config=model_config)

        with (
            patch("openai.AsyncOpenAI") as mock_async,
            patch("agents.models.openai_responses.OpenAIResponsesModel") as mock_resp_model,
        ):
            mock_async.return_value = MagicMock()
            mock_resp_model.return_value = MagicMock()
            resp_model = model._get_responses_model()
            mock_resp_model.assert_called_once()
            assert resp_model is not None

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
