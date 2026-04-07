# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/openai_compatible.py.

CI-level: zero external dependencies. All LiteLLM / OpenAI SDK calls mocked.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.exceptions import ModelBehaviorError
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from datus.models.openai_compatible import OpenAICompatibleModel, classify_openai_compatible_error
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.exceptions import DatusException, ErrorCode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(
    model="gpt-4",
    model_type="openai",
    api_key="sk-test",
    base_url=None,
    temperature=None,
    top_p=None,
    enable_thinking=False,
    default_headers=None,
    save_llm_trace=False,
):
    cfg = MagicMock()
    cfg.model = model
    cfg.type = model_type
    cfg.api_key = api_key
    cfg.base_url = base_url
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.enable_thinking = enable_thinking
    cfg.default_headers = default_headers or {}
    cfg.max_retry = 3
    cfg.retry_interval = 0.0
    cfg.strict_json_schema = True
    cfg.save_llm_trace = save_llm_trace
    return cfg


def _make_model(model_config=None):
    """Create OpenAICompatibleModel with all I/O components mocked."""
    if model_config is None:
        model_config = _make_model_config()

    mock_litellm_adapter = MagicMock()
    mock_litellm_adapter.litellm_model_name = "openai/gpt-4"
    mock_litellm_adapter.provider = "openai"
    mock_litellm_adapter.is_thinking_model = False
    mock_litellm_adapter.get_agents_sdk_model.return_value = MagicMock()

    with (
        patch("datus.models.openai_compatible.setup_tracing"),
        patch("datus.models.openai_compatible.LiteLLMAdapter", return_value=mock_litellm_adapter),
    ):
        # Subclass to implement the abstract _get_api_key
        class _ConcreteModel(OpenAICompatibleModel):
            def _get_api_key(self):
                return self.model_config.api_key or "test-key"

        model = _ConcreteModel(model_config)
        model.litellm_adapter = mock_litellm_adapter
        return model


# ---------------------------------------------------------------------------
# classify_openai_compatible_error
# ---------------------------------------------------------------------------


class TestClassifyOpenAICompatibleError:
    def _make_api_error(self, message, status_code=400):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: message
        err.status_code = status_code
        return err

    def test_401_returns_authentication_error(self):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: "401 unauthorized"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_AUTHENTICATION_ERROR
        assert retryable is False

    def test_403_returns_permission_error(self):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: "403 forbidden"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_PERMISSION_ERROR
        assert retryable is False

    def test_404_returns_not_found(self):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: "404 not found"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_NOT_FOUND
        assert retryable is False

    def test_429_rate_limit_retryable(self):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: "429 rate limit exceeded"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_RATE_LIMIT
        assert retryable is True

    def test_quota_exceeded_not_retryable(self):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: "429 quota exceeded"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_QUOTA_EXCEEDED
        assert retryable is False

    def test_500_server_error_retryable(self):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: "500 internal server error"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_API_ERROR
        assert retryable is True

    def test_502_overloaded_retryable(self):
        err = MagicMock(spec=APIError)
        err.__str__ = lambda self: "502 overloaded"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_OVERLOADED
        assert retryable is True

    def test_rate_limit_error_class(self):
        err = MagicMock(spec=RateLimitError)
        err.__str__ = lambda self: "rate limit"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_RATE_LIMIT
        assert retryable is True

    def test_timeout_error_class(self):
        err = MagicMock(spec=APITimeoutError)
        err.__str__ = lambda self: "timeout"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_TIMEOUT_ERROR
        assert retryable is True

    def test_connection_error_class(self):
        err = MagicMock(spec=APIConnectionError)
        err.__str__ = lambda self: "connection error"
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_CONNECTION_ERROR
        assert retryable is True

    def test_unknown_exception_returns_request_failed(self):
        err = Exception("something weird")
        code, retryable = classify_openai_compatible_error(err)
        assert code == ErrorCode.MODEL_REQUEST_FAILED
        assert retryable is False


# ---------------------------------------------------------------------------
# OpenAICompatibleModel.__init__ / basic properties
# ---------------------------------------------------------------------------


class TestOpenAICompatibleModelInit:
    def test_model_name_set(self):
        model = _make_model()
        assert model.model_name == "gpt-4"

    def test_api_key_set(self):
        model = _make_model()
        assert model.api_key == "sk-test"

    def test_base_url_defaults_to_config(self):
        cfg = _make_model_config(base_url="https://custom.api.com/v1")
        model = _make_model(cfg)
        assert model.base_url == "https://custom.api.com/v1"

    def test_current_node_initially_none(self):
        model = _make_model()
        assert model.current_node is None

    def test_model_info_cache_initially_none(self):
        model = _make_model()
        assert model._model_info is None


# ---------------------------------------------------------------------------
# _setup_custom_json_encoder
# ---------------------------------------------------------------------------


class TestSetupCustomJsonEncoder:
    def test_does_not_raise(self):
        OpenAICompatibleModel._setup_custom_json_encoder()
        # Verify the encoder is installed: json._default_encoder should be our CustomJSONEncoder
        assert type(json._default_encoder).__name__ == "CustomJSONEncoder"

    def test_anyurl_serializable_after_setup(self):
        from pydantic import AnyUrl

        OpenAICompatibleModel._setup_custom_json_encoder()
        url = AnyUrl("https://example.com")
        # json.dumps must succeed directly — no try/except, that would hide failures
        encoded = json.dumps(url)
        assert "example.com" in encoded


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def _mock_litellm_response(self, content="Hello world"):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.choices[0].message.reasoning_content = None
        resp.choices[0].finish_reason = "stop"
        resp.model = "gpt-4"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        return resp

    def test_basic_generate_returns_content(self):
        model = _make_model()
        mock_resp = self._mock_litellm_response("Hello world")
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp):
            result = model.generate("Say hello")
        assert result == "Hello world"

    def test_generate_with_list_prompt(self):
        model = _make_model()
        mock_resp = self._mock_litellm_response("Response")
        messages = [{"role": "user", "content": "test"}]
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp) as mock_lit:
            result = model.generate(messages)
        assert result == "Response"
        call_kwargs = mock_lit.call_args[1]
        assert call_kwargs["messages"] == messages

    def test_temperature_from_kwargs(self):
        model = _make_model()
        mock_resp = self._mock_litellm_response("ok")
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp) as mock_lit:
            model.generate("prompt", temperature=0.5)
        call_kwargs = mock_lit.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_temperature_from_model_config(self):
        cfg = _make_model_config(temperature=0.3)
        model = _make_model(cfg)
        mock_resp = self._mock_litellm_response("ok")
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp) as mock_lit:
            model.generate("prompt")
        call_kwargs = mock_lit.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    def test_top_p_from_kwargs(self):
        model = _make_model()
        mock_resp = self._mock_litellm_response("ok")
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp) as mock_lit:
            model.generate("prompt", top_p=0.9)
        call_kwargs = mock_lit.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    def test_max_tokens_passed_through(self):
        model = _make_model()
        mock_resp = self._mock_litellm_response("ok")
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp) as mock_lit:
            model.generate("prompt", max_tokens=512)
        call_kwargs = mock_lit.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    def test_base_url_added_when_set(self):
        cfg = _make_model_config(base_url="https://myapi.com/v1")
        model = _make_model(cfg)
        mock_resp = self._mock_litellm_response("ok")
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp) as mock_lit:
            model.generate("prompt")
        call_kwargs = mock_lit.call_args[1]
        assert call_kwargs["api_base"] == "https://myapi.com/v1"

    def test_empty_content_returns_empty_string(self):
        model = _make_model()
        mock_resp = self._mock_litellm_response("")
        mock_resp.choices[0].message.content = None
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp):
            result = model.generate("prompt")
        assert result == ""

    def test_enable_thinking_uses_reasoning_content(self):
        cfg = _make_model_config(enable_thinking=True)
        model = _make_model(cfg)
        mock_resp = self._mock_litellm_response("")
        mock_resp.choices[0].message.content = ""
        mock_resp.choices[0].message.reasoning_content = "step by step reasoning"
        with patch("datus.models.openai_compatible.litellm.completion", return_value=mock_resp):
            result = model.generate("prompt")
        assert result == "step by step reasoning"


# ---------------------------------------------------------------------------
# generate_with_json_output
# ---------------------------------------------------------------------------


class TestGenerateWithJsonOutput:
    def test_valid_json_parsed(self):
        model = _make_model()
        with patch.object(model, "generate", return_value='{"key": "value"}'):
            result = model.generate_with_json_output("prompt")
        assert result == {"key": "value"}

    def test_json_in_response_extracted(self):
        model = _make_model()
        with patch.object(model, "generate", return_value='Here is the result: {"x": 1}'):
            result = model.generate_with_json_output("prompt")
        assert result == {"x": 1}

    def test_invalid_json_returns_error_dict(self):
        model = _make_model()
        with patch.object(model, "generate", return_value="not json at all"):
            result = model.generate_with_json_output("prompt")
        assert "error" in result
        assert "raw_response" in result

    def test_response_format_set_to_json(self):
        model = _make_model()
        with patch.object(model, "generate", return_value="{}") as mock_gen:
            model.generate_with_json_output("prompt")
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_enable_thinking_passed_through(self):
        model = _make_model()
        with patch.object(model, "generate", return_value='{"a": 1}') as mock_gen:
            model.generate_with_json_output("prompt", enable_thinking=True)
        # generate_with_json_output pops enable_thinking from kwargs and passes it as
        # the keyword argument enable_thinking to self.generate (see openai_compatible.py:406)
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("enable_thinking") is True


# ---------------------------------------------------------------------------
# _with_retry (sync)
# ---------------------------------------------------------------------------


class TestWithRetry:
    def test_succeeds_on_first_attempt(self):
        model = _make_model()
        result = model._with_retry(lambda: "ok", max_retries=2)
        assert result == "ok"

    def test_raises_datus_exception_on_non_retryable_api_error(self):
        model = _make_model()

        class _FakeAPIError(APIError):
            def __init__(self):
                pass  # avoid complex constructor

            def __str__(self):
                return "401 unauthorized"

        err = _FakeAPIError()

        def raise_it():
            raise err

        with pytest.raises(DatusException):
            model._with_retry(raise_it, max_retries=1)

    def test_raises_original_exception_on_unexpected_error(self):
        model = _make_model()

        def raise_it():
            raise ValueError("unexpected")

        with pytest.raises(ValueError, match="unexpected"):
            model._with_retry(raise_it, max_retries=1)

    def test_retry_on_retryable_error_succeeds(self):
        model = _make_model()
        call_count = [0]

        class _FakeRateLimit(RateLimitError):
            def __init__(self):
                pass

            def __str__(self):
                return "rate limit"

        def flaky():
            call_count[0] += 1
            if call_count[0] == 1:
                raise _FakeRateLimit()
            return "success"

        with patch("time.sleep"):
            result = model._with_retry(flaky, max_retries=2, base_delay=0.01)
        assert result == "success"
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# generate_with_tools (routing / basic)
# ---------------------------------------------------------------------------


class TestGenerateWithTools:
    @pytest.mark.asyncio
    async def test_returns_dict_with_content(self):
        model = _make_model()

        fake_internal_result = {
            "content": "done",
            "sql_contexts": [],
            "usage": {},
            "model": "gpt-4",
            "turns_used": 1,
            "final_output_length": 4,
        }

        with patch.object(
            model, "_generate_with_tools_internal", new_callable=AsyncMock, return_value=fake_internal_result
        ):
            result = await model.generate_with_tools(prompt="test", instruction="do something")

        assert result["content"] == "done"
        assert "model" in result
        assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_metadata_fields_added(self):
        model = _make_model()

        fake_result = {"content": "x", "sql_contexts": []}

        with patch.object(model, "_generate_with_tools_internal", new_callable=AsyncMock, return_value=fake_result):
            result = await model.generate_with_tools(
                prompt="query",
                instruction="system",
                max_turns=5,
            )

        assert result["max_turns"] == 5
        assert "tool_count" in result
        assert "mcp_server_count" in result


# --- Merged from test_openai_compatible_extended ---


# ---------------------------------------------------------------------------
# _with_retry_async
# ---------------------------------------------------------------------------


class TestWithRetryAsync:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        model = _make_model()

        async def op():
            return "result"

        result = await model._with_retry_async(op, max_retries=2)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_model_behavior_error_retried(self):
        model = _make_model()
        call_count = [0]

        async def op():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ModelBehaviorError("hallucinated tool")
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await model._with_retry_async(op, max_retries=2, base_delay=0.0)

        assert result == "ok"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_model_behavior_error_exhausted_raises(self):
        model = _make_model()

        async def op():
            raise ModelBehaviorError("always fails")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ModelBehaviorError):
                await model._with_retry_async(op, max_retries=1, base_delay=0.0)

    @pytest.mark.asyncio
    async def test_retryable_api_error_retried(self):
        model = _make_model()
        call_count = [0]

        class _FakeRateLimitError(RateLimitError):
            def __init__(self):
                pass

            def __str__(self):
                return "429 rate limit"

        async def op():
            call_count[0] += 1
            if call_count[0] == 1:
                raise _FakeRateLimitError()
            return "success"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await model._with_retry_async(op, max_retries=2, base_delay=0.0)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_non_retryable_api_error_raises_datus_exception(self):
        model = _make_model()

        class _FakeAPIError(APIError):
            def __init__(self):
                pass

            def __str__(self):
                return "401 unauthorized"

        async def op():
            raise _FakeAPIError()

        with pytest.raises(DatusException):
            await model._with_retry_async(op, max_retries=2, base_delay=0.0)

    @pytest.mark.asyncio
    async def test_unexpected_exception_propagates(self):
        model = _make_model()

        async def op():
            raise ValueError("unexpected")

        with pytest.raises(ValueError, match="unexpected"):
            await model._with_retry_async(op, max_retries=2, base_delay=0.0)


# ---------------------------------------------------------------------------
# _add_usage_to_action
# ---------------------------------------------------------------------------


class TestAddUsageToAction:
    def test_adds_usage_to_action_with_dict_output(self):
        model = _make_model()
        action = MagicMock(spec=ActionHistory)
        action.output = {"existing_key": "value"}
        usage = {"total_tokens": 100, "input_tokens": 80, "output_tokens": 20}
        model._add_usage_to_action(action, usage)
        assert action.output["usage"] == usage

    def test_adds_usage_when_output_is_none(self):
        model = _make_model()
        action = MagicMock(spec=ActionHistory)
        action.output = None
        usage = {"total_tokens": 50}
        model._add_usage_to_action(action, usage)
        assert action.output["usage"] == usage

    def test_adds_usage_when_output_is_non_dict(self):
        model = _make_model()
        action = MagicMock(spec=ActionHistory)
        action.output = "raw string output"
        usage = {"total_tokens": 30}
        model._add_usage_to_action(action, usage)
        assert action.output["usage"] == usage
        assert action.output["raw_output"] == "raw string output"


# ---------------------------------------------------------------------------
# _distribute_token_usage_to_actions
# ---------------------------------------------------------------------------


class TestDistributeTokenUsageToActions:
    def test_no_actions_does_nothing(self):
        model = _make_model()
        manager = ActionHistoryManager()
        usage = {"total_tokens": 100}
        model._distribute_token_usage_to_actions(manager, usage)  # should not raise

    def test_distributes_to_last_assistant_action(self):
        model = _make_model()
        manager = ActionHistoryManager()

        action1 = ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            messages="first",
            action_type="response",
            status=ActionStatus.SUCCESS,
        )
        action2 = ActionHistory(
            action_id="a2",
            role=ActionRole.ASSISTANT,
            messages="second",
            action_type="response",
            status=ActionStatus.SUCCESS,
        )
        manager.add_action(action1)
        manager.add_action(action2)

        usage = {"total_tokens": 200, "input_tokens": 150, "output_tokens": 50}
        model._distribute_token_usage_to_actions(manager, usage)

        # Only the last assistant action should have usage
        assert action2.output is not None
        assert isinstance(action2.output, dict)
        assert action2.output.get("usage") == usage

    def test_tool_actions_not_modified(self):
        model = _make_model()
        manager = ActionHistoryManager()

        tool_action = ActionHistory(
            action_id="t1",
            role=ActionRole.TOOL,
            messages="tool call",
            action_type="query",
            status=ActionStatus.SUCCESS,
            output={"result": "data"},
        )
        manager.add_action(tool_action)

        usage = {"total_tokens": 100}
        model._distribute_token_usage_to_actions(manager, usage)

        # Tool action should not be modified (no assistant action to add to)
        assert "usage" not in (tool_action.output or {})


# ---------------------------------------------------------------------------
# _extract_and_distribute_token_usage
# ---------------------------------------------------------------------------


class TestExtractAndDistributeTokenUsage:
    @pytest.mark.asyncio
    async def test_no_context_wrapper_logs_warning(self):
        model = _make_model()
        result = MagicMock(spec=[])  # no context_wrapper attribute
        manager = ActionHistoryManager()
        # Should not raise
        await model._extract_and_distribute_token_usage(result, manager)

    @pytest.mark.asyncio
    async def test_extracts_usage_from_context_wrapper(self):
        model = _make_model()

        usage = MagicMock()
        usage.requests = 1
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.total_tokens = 150
        usage.input_tokens_details = MagicMock()
        usage.input_tokens_details.cached_tokens = 10
        usage.output_tokens_details = MagicMock()
        usage.output_tokens_details.reasoning_tokens = 5

        context_wrapper = MagicMock()
        context_wrapper.usage = usage

        result = MagicMock()
        result.context_wrapper = context_wrapper

        manager = ActionHistoryManager()
        action = ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            messages="text",
            action_type="response",
            status=ActionStatus.SUCCESS,
        )
        manager.add_action(action)

        with patch.object(model, "context_length", return_value=128000):
            await model._extract_and_distribute_token_usage(result, manager)

        assert action.output is not None
        assert isinstance(action.output, dict)
        assert action.output["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        model = _make_model()
        result = MagicMock()
        result.context_wrapper = MagicMock()
        # Make usage access raise
        type(result.context_wrapper).usage = property(lambda self: (_ for _ in ()).throw(RuntimeError("bad")))

        manager = ActionHistoryManager()
        # Should not raise
        await model._extract_and_distribute_token_usage(result, manager)


# ---------------------------------------------------------------------------
# _format_tool_result_from_dict
# ---------------------------------------------------------------------------


class TestFormatToolResultFromDict:
    def setup_method(self):
        self.model = _make_model()

    def test_result_is_list(self):
        assert self.model._format_tool_result_from_dict({"result": [1, 2, 3]}) == "3 items"

    def test_result_is_int(self):
        assert self.model._format_tool_result_from_dict({"result": 42}) == "42 rows"

    def test_result_is_dict_with_count(self):
        assert self.model._format_tool_result_from_dict({"result": {"count": 7}}) == "7 items"

    def test_result_is_dict_without_count(self):
        assert self.model._format_tool_result_from_dict({"result": {"key": "val"}}) == "Success"

    def test_result_is_other_type(self):
        assert self.model._format_tool_result_from_dict({"result": "string"}) == "Success"

    def test_rows_field_int(self):
        assert self.model._format_tool_result_from_dict({"rows": 10}) == "10 rows"

    def test_rows_field_non_int(self):
        assert self.model._format_tool_result_from_dict({"rows": "many"}) == "Success"

    def test_items_field(self):
        assert self.model._format_tool_result_from_dict({"items": ["a", "b"]}) == "2 items"

    def test_success_field_only_true(self):
        assert self.model._format_tool_result_from_dict({"success": True}) == "Success"

    def test_success_field_only_false(self):
        assert self.model._format_tool_result_from_dict({"success": False}) == "Failed"

    def test_count_field(self):
        assert self.model._format_tool_result_from_dict({"count": 99}) == "99 items"

    def test_generic_dict(self):
        assert self.model._format_tool_result_from_dict({"anything": "value"}) == "Success"


# ---------------------------------------------------------------------------
# _format_tool_result (string version)
# ---------------------------------------------------------------------------


class TestFormatToolResult:
    def setup_method(self):
        self.model = _make_model()

    def test_empty_string_returns_empty_result(self):
        assert self.model._format_tool_result("") == "Empty result"

    def test_none_content_returns_empty_result(self):
        assert self.model._format_tool_result(None) == "Empty result"

    def test_json_dict_delegates_to_from_dict(self):
        result = self.model._format_tool_result('{"result": [1, 2]}')
        assert result == "2 items"

    def test_json_list(self):
        result = self.model._format_tool_result("[1, 2, 3]")
        assert result == "3 items"

    def test_json_scalar(self):
        result = self.model._format_tool_result('"hello"')
        assert "hello" in result

    def test_plain_text_short(self):
        result = self.model._format_tool_result("short text")
        assert "short text" in result

    def test_plain_text_long_truncated(self):
        long_text = "x" * 200
        result = self.model._format_tool_result(long_text)
        assert result.endswith("...")
        assert len(result) <= 103  # 100 + "..."


# ---------------------------------------------------------------------------
# model_specs / max_tokens / context_length
# ---------------------------------------------------------------------------


class TestModelSpecsAndTokenLimits:
    def test_exact_match_max_tokens(self):
        cfg = _make_model_config(model="gpt-4o")
        model = _make_model(cfg)
        assert model.max_tokens() == 16384

    def test_exact_match_context_length(self):
        cfg = _make_model_config(model="gpt-4o")
        model = _make_model(cfg)
        assert model.context_length() == 128000

    def test_prefix_match_max_tokens(self):
        # gpt-4o-mini should match gpt-4o prefix
        cfg = _make_model_config(model="gpt-4o-mini")
        model = _make_model(cfg)
        assert model.max_tokens() == 16384

    def test_prefix_match_context_length(self):
        cfg = _make_model_config(model="kimi-k2-0711-preview")
        model = _make_model(cfg)
        # Should match "kimi-k2" prefix
        assert model.context_length() == 256000

    def test_unknown_model_returns_none_for_max_tokens(self):
        cfg = _make_model_config(model="unknown-model-xyz")
        model = _make_model(cfg)
        assert model.max_tokens() is None

    def test_unknown_model_returns_none_for_context_length(self):
        cfg = _make_model_config(model="unknown-model-xyz")
        model = _make_model(cfg)
        assert model.context_length() is None

    def test_deepseek_chat_specs(self):
        cfg = _make_model_config(model="deepseek-chat")
        model = _make_model(cfg)
        assert model.max_tokens() == 8192
        assert model.context_length() == 65535

    def test_gemini_flash_specs(self):
        cfg = _make_model_config(model="gemini-2.5-flash")
        model = _make_model(cfg)
        assert model.context_length() == 1048576


# ---------------------------------------------------------------------------
# token_count
# ---------------------------------------------------------------------------


class TestTokenCount:
    def test_returns_litellm_count(self):
        model = _make_model()
        with patch("datus.models.openai_compatible.litellm.token_counter", return_value=42):
            count = model.token_count("hello world")
        assert count == 42

    def test_falls_back_to_approximation_on_error(self):
        model = _make_model()
        with patch("datus.models.openai_compatible.litellm.token_counter", side_effect=Exception("fail")):
            count = model.token_count("hello world")
        # Fallback: len(text) // 4 = 11 // 4 = 2
        assert count == 2


# ---------------------------------------------------------------------------
# _save_llm_trace
# ---------------------------------------------------------------------------


class TestSaveLlmTrace:
    def test_does_nothing_when_disabled(self, tmp_path):
        cfg = _make_model_config(save_llm_trace=False)
        model = _make_model(cfg)
        with patch("builtins.open") as mock_open:
            model._save_llm_trace("prompt", "response")
        # No file should have been opened — early return before any I/O
        mock_open.assert_not_called()

    def test_does_nothing_when_no_workflow_context(self, tmp_path):
        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)
        # No workflow/current_node attributes on the model
        with patch("builtins.open") as mock_open:
            model._save_llm_trace("prompt", "response")
        mock_open.assert_not_called()

    def test_does_nothing_when_workflow_is_none(self):
        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)
        model.workflow = None
        model.current_node = MagicMock()
        with patch("builtins.open") as mock_open:
            model._save_llm_trace("prompt", "response")
        mock_open.assert_not_called()

    def test_does_nothing_when_current_node_is_none(self):
        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)
        model.workflow = MagicMock()
        model.current_node = None
        with patch("builtins.open") as mock_open:
            model._save_llm_trace("prompt", "response")
        mock_open.assert_not_called()

    def test_saves_trace_file(self, tmp_path):
        import yaml as pyyaml

        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)

        mock_node = MagicMock()
        mock_node.id = "node_001"

        mock_task = MagicMock()
        mock_task.id = "task_001"

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task = mock_task

        model.workflow = mock_workflow
        model.current_node = mock_node

        model._save_llm_trace(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user question"},
            ],
            "SELECT 1",
            "reasoning here",
        )

        trace_file = tmp_path / "task_001" / "node_001.yml"
        assert trace_file.exists()

        with open(trace_file, "r") as f:
            data = pyyaml.safe_load(f)

        assert data["system_prompt"] == "system prompt"
        assert data["user_prompt"] == "user question"
        assert data["output_content"] == "SELECT 1"
        assert data["reason_content"] == "reasoning here"

    def test_saves_trace_with_string_prompt(self, tmp_path):
        import yaml as pyyaml

        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)

        mock_node = MagicMock()
        mock_node.id = "node_002"

        mock_task = MagicMock()
        mock_task.id = "task_002"

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task = mock_task

        model.workflow = mock_workflow
        model.current_node = mock_node

        model._save_llm_trace("direct string prompt", "answer")

        trace_file = tmp_path / "task_002" / "node_002.yml"
        assert trace_file.exists()

        with open(trace_file, "r") as f:
            data = pyyaml.safe_load(f)

        assert data["user_prompt"] == "direct string prompt"

    def test_handles_write_error_gracefully(self, tmp_path):
        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)

        mock_node = MagicMock()
        mock_node.id = "node_err"

        mock_task = MagicMock()
        mock_task.id = "task_err"

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task = mock_task

        model.workflow = mock_workflow
        model.current_node = mock_node

        with (
            patch("builtins.open", side_effect=OSError("permission denied")),
            patch("datus.models.openai_compatible.logger") as mock_logger,
        ):
            # Should not raise
            model._save_llm_trace("prompt", "response")

        # The error must be logged (logger.error is called in the except block)
        mock_logger.error.assert_called_once()
        logged_msg = mock_logger.error.call_args[0][0]
        assert "permission denied" in logged_msg

    def test_saves_trace_with_other_prompt_type(self, tmp_path):
        import yaml as pyyaml

        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)

        mock_node = MagicMock()
        mock_node.id = "node_003"

        mock_task = MagicMock()
        mock_task.id = "task_003"

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task = mock_task

        model.workflow = mock_workflow
        model.current_node = mock_node

        # Pass a non-string, non-list prompt
        model._save_llm_trace(12345, "response")

        trace_file = tmp_path / "task_003" / "node_003.yml"
        assert trace_file.exists()

        with open(trace_file, "r") as f:
            data = pyyaml.safe_load(f)
        assert data["user_prompt"] == "12345"

    def test_saves_trace_with_multiple_messages_same_role(self, tmp_path):
        import yaml as pyyaml

        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)

        mock_node = MagicMock()
        mock_node.id = "node_004"

        mock_task = MagicMock()
        mock_task.id = "task_004"

        mock_workflow = MagicMock()
        mock_workflow.global_config.trajectory_dir = str(tmp_path)
        mock_workflow.task = mock_task

        model.workflow = mock_workflow
        model.current_node = mock_node

        # Multiple user and system messages
        model._save_llm_trace(
            [
                {"role": "system", "content": "sys1"},
                {"role": "system", "content": "sys2"},
                {"role": "user", "content": "user1"},
                {"role": "assistant", "content": "asst"},  # should be skipped
                {"role": "user", "content": "user2"},
            ],
            "output",
        )

        trace_file = tmp_path / "task_004" / "node_004.yml"
        with open(trace_file, "r") as f:
            data = pyyaml.safe_load(f)

        assert "sys1" in data["system_prompt"]
        assert "sys2" in data["system_prompt"]
        assert "user1" in data["user_prompt"]
        assert "user2" in data["user_prompt"]


# ---------------------------------------------------------------------------
# generate_with_tools_stream (public method routing)
# ---------------------------------------------------------------------------


class TestGenerateWithToolsStream:
    @pytest.mark.asyncio
    async def test_yields_actions_from_internal(self):
        model = _make_model()

        async def _fake_internal(*args, **kwargs):
            yield ActionHistory(
                action_id="s1",
                role=ActionRole.ASSISTANT,
                messages="thinking",
                action_type="response",
                status=ActionStatus.SUCCESS,
            )

        with patch.object(model, "_generate_with_tools_stream_internal", side_effect=_fake_internal):
            actions = []
            async for a in model.generate_with_tools_stream(prompt="test"):
                actions.append(a)

        assert len(actions) == 1
        assert actions[0].action_id == "s1"

    @pytest.mark.asyncio
    async def test_creates_action_history_manager_if_none(self):
        model = _make_model()

        captured_manager = []

        async def _fake_internal(
            prompt,
            mcp,
            tools,
            instr,
            output_type,
            strict,
            max_turns,
            session,
            ahm,
            hooks,
            interrupt_controller,
            **kwargs,
        ):
            captured_manager.append(ahm)
            return
            yield  # make it an async generator

        with patch.object(model, "_generate_with_tools_stream_internal", side_effect=_fake_internal):
            async for _ in model.generate_with_tools_stream(prompt="test", action_history_manager=None):
                pass

        assert len(captured_manager) == 1
        assert isinstance(captured_manager[0], ActionHistoryManager)


# ---------------------------------------------------------------------------
# _build_agent
# ---------------------------------------------------------------------------


class TestBuildAgent:
    """Tests for _build_agent. We patch Agent to capture kwargs without SDK validation."""

    def _call_build_agent(self, model, **kwargs):
        defaults = {
            "instruction": "test",
            "output_type": str,
            "strict_json_schema": True,
            "connected_servers": {},
            "tools": None,
        }
        defaults.update(kwargs)
        with patch("datus.models.openai_compatible.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            model._build_agent(**defaults)
            return MockAgent, MockAgent.call_args

    def test_str_output_type_no_schema_wrapping(self):
        model = _make_model()
        _, call_args = self._call_build_agent(model, output_type=str)
        assert call_args[1]["output_type"] is str

    def test_structured_output_wraps_with_schema(self):
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            sql: str

        model = _make_model()
        _, call_args = self._call_build_agent(model, output_type=MyOutput)
        # Should be wrapped in AgentOutputSchema, not the raw type
        assert call_args[1]["output_type"] is not MyOutput

    def test_deepseek_adds_json_keyword_for_structured_output(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            x: int

        cfg = _make_model_config(model="deepseek-chat", model_type="deepseek")
        model = _make_model(cfg)
        model.litellm_adapter.provider = "deepseek"

        _, call_args = self._call_build_agent(model, instruction="Generate output", output_type=Out)
        assert "json" in call_args[1]["instructions"].lower()

    def test_deepseek_no_duplicate_json_keyword(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            x: int

        cfg = _make_model_config(model="deepseek-chat", model_type="deepseek")
        model = _make_model(cfg)
        model.litellm_adapter.provider = "deepseek"

        _, call_args = self._call_build_agent(model, instruction="Return valid JSON output", output_type=Out)
        assert call_args[1]["instructions"] == "Return valid JSON output"

    def test_default_headers_set_as_extra_headers(self):
        cfg = _make_model_config(default_headers={"X-Custom": "value"})
        model = _make_model(cfg)
        _, call_args = self._call_build_agent(model)
        ms = call_args[1]["model_settings"]
        assert ms.extra_headers == {"X-Custom": "value"}

    def test_thinking_model_gets_reasoning(self):
        model = _make_model()
        model.litellm_adapter.is_thinking_model = True
        _, call_args = self._call_build_agent(model)
        ms = call_args[1]["model_settings"]
        assert ms.reasoning is not None
        assert ms.reasoning.effort == "medium"

    def test_temperature_and_top_p_from_config(self):
        cfg = _make_model_config(temperature=0.5, top_p=0.9)
        model = _make_model(cfg)
        _, call_args = self._call_build_agent(model)
        ms = call_args[1]["model_settings"]
        assert ms.temperature == 0.5
        assert ms.top_p == 0.9

    def test_connected_servers_set_as_mcp_servers(self):
        model = _make_model()
        server1 = MagicMock()
        server2 = MagicMock()
        servers = {"s1": server1, "s2": server2}
        _, call_args = self._call_build_agent(model, connected_servers=servers)
        assert call_args[1]["mcp_servers"] == [server1, server2]

    def test_tools_passed_through(self):
        model = _make_model()
        mock_tool = MagicMock()
        _, call_args = self._call_build_agent(model, tools=[mock_tool])
        assert call_args[1]["tools"] == [mock_tool]


# ---------------------------------------------------------------------------
# _extract_usage_info
# ---------------------------------------------------------------------------


class TestExtractUsageInfo:
    def test_normal_usage(self):
        model = _make_model()
        usage = MagicMock()
        usage.requests = 2
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.total_tokens = 150
        usage.input_tokens_details = MagicMock()
        usage.input_tokens_details.cached_tokens = 20
        usage.output_tokens_details = MagicMock()
        usage.output_tokens_details.reasoning_tokens = 10

        with patch.object(model, "context_length", return_value=128000):
            info = model._extract_usage_info(usage)

        assert info["requests"] == 2
        assert info["input_tokens"] == 100
        assert info["output_tokens"] == 50
        assert info["total_tokens"] == 150
        assert info["cached_tokens"] == 20
        assert info["reasoning_tokens"] == 10
        assert info["cache_hit_rate"] == round(20 / 100, 3)
        assert info["context_usage_ratio"] == round(150 / 128000, 3)

    def test_zero_input_tokens_no_division_error(self):
        model = _make_model()
        usage = MagicMock()
        usage.requests = 0
        usage.input_tokens = 0
        usage.output_tokens = 0
        usage.total_tokens = 0
        usage.input_tokens_details = None
        usage.output_tokens_details = None

        with patch.object(model, "context_length", return_value=128000):
            info = model._extract_usage_info(usage)

        assert info["cache_hit_rate"] == 0
        assert info["cached_tokens"] == 0
        assert info["reasoning_tokens"] == 0

    def test_missing_details_attributes(self):
        model = _make_model()
        usage = MagicMock(spec=["requests", "input_tokens", "output_tokens", "total_tokens"])
        usage.requests = 1
        usage.input_tokens = 50
        usage.output_tokens = 25
        usage.total_tokens = 75

        with patch.object(model, "context_length", return_value=128000):
            info = model._extract_usage_info(usage)

        assert info["cached_tokens"] == 0
        assert info["reasoning_tokens"] == 0

    def test_unknown_model_context_length_none(self):
        cfg = _make_model_config(model="unknown-model-xyz")
        model = _make_model(cfg)
        usage = MagicMock()
        usage.requests = 1
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.total_tokens = 150
        usage.input_tokens_details = None
        usage.output_tokens_details = None

        with patch.object(model, "context_length", return_value=None):
            info = model._extract_usage_info(usage)

        assert info["context_usage_ratio"] == 0
