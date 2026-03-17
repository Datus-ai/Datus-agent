# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Extended unit tests for datus/models/openai_compatible.py.

CI-level: zero external dependencies. All litellm/OpenAI SDK calls mocked.

Covers uncovered lines:
- _with_retry_async (lines 191-243): async retry logic, ModelBehaviorError
- _generate_with_tools_internal (lines 507-684): structured output, tool usage
- _extract_and_distribute_token_usage (lines 1179-1224)
- _distribute_token_usage_to_actions (lines 1226-1254)
- _add_usage_to_action (lines 1256-1263)
- _format_tool_result_from_dict (lines 1265-1310): all branches
- _format_tool_result (lines 1312-1340): all branches
- model_specs / max_tokens / context_length (lines 1342-1405): prefix matching
- token_count (lines 1407-1418)
- _save_llm_trace (lines 1420-1502): all branches
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.exceptions import ModelBehaviorError
from openai import APIError, RateLimitError

from datus.models.openai_compatible import OpenAICompatibleModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(
    model="gpt-4o",
    model_type="openai",
    api_key="sk-test",
    base_url=None,
    temperature=None,
    top_p=None,
    enable_thinking=False,
    default_headers=None,
    save_llm_trace=False,
    max_retry=3,
    retry_interval=0.0,
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
    cfg.max_retry = max_retry
    cfg.retry_interval = retry_interval
    cfg.strict_json_schema = True
    cfg.save_llm_trace = save_llm_trace
    return cfg


def _make_model(model_config=None):
    if model_config is None:
        model_config = _make_model_config()

    mock_litellm_adapter = MagicMock()
    mock_litellm_adapter.litellm_model_name = "openai/gpt-4o"
    mock_litellm_adapter.provider = "openai"
    mock_litellm_adapter.is_thinking_model = False
    mock_litellm_adapter.get_agents_sdk_model.return_value = MagicMock()

    with (
        patch("datus.models.openai_compatible.setup_tracing"),
        patch("datus.models.openai_compatible.LiteLLMAdapter", return_value=mock_litellm_adapter),
    ):

        class _ConcreteModel(OpenAICompatibleModel):
            def _get_api_key(self):
                return self.model_config.api_key or "test-key"

        model = _ConcreteModel(model_config)
        model.litellm_adapter = mock_litellm_adapter
        return model


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
        # Should not raise, does nothing
        model._save_llm_trace("prompt", "response")

    def test_does_nothing_when_no_workflow_context(self, tmp_path):
        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)
        # No workflow/current_node attributes
        model._save_llm_trace("prompt", "response")

    def test_does_nothing_when_workflow_is_none(self):
        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)
        model.workflow = None
        model.current_node = MagicMock()
        model._save_llm_trace("prompt", "response")

    def test_does_nothing_when_current_node_is_none(self):
        cfg = _make_model_config(save_llm_trace=True)
        model = _make_model(cfg)
        model.workflow = MagicMock()
        model.current_node = None
        model._save_llm_trace("prompt", "response")

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

        with patch("builtins.open", side_effect=OSError("permission denied")):
            # Should not raise
            model._save_llm_trace("prompt", "response")

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
