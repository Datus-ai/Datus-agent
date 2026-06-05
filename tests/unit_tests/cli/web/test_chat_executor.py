# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/web/chat_executor.py — ChatExecutor."""

import asyncio
from unittest.mock import MagicMock

import pytest

from datus.cli.web import chat_executor
from datus.cli.web.chat_executor import ChatExecutor
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


def _make_action(
    role: ActionRole,
    status: ActionStatus,
    action_type: str = "test",
    messages: str = "",
    input_data: dict = None,
    output_data: dict = None,
) -> ActionHistory:
    import uuid

    return ActionHistory(
        action_id=str(uuid.uuid4()),
        role=role,
        messages=messages,
        action_type=action_type,
        input=input_data,
        output=output_data,
        status=status,
    )


@pytest.mark.ci
class TestChatExecutorFormatAction:
    """Test format_action_for_stream."""

    def test_tool_processing(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="list_tables",
            input_data={"function_name": "list_tables"},
        )
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert "list_tables" in result
        assert "\u27f3" in result  # ⟳

    def test_tool_success(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="read_query",
            input_data={"function_name": "read_query"},
            output_data={"result": "ok"},
        )
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert "read_query" in result
        assert "\u2713" in result  # ✓

    def test_tool_failed(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.FAILED,
            messages="read_query",
            input_data={"function_name": "read_query"},
        )
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert "read_query" in result
        assert "\u2717" in result  # ✗

    def test_assistant_thinking(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="I will query the database now",
        )
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert "Thinking:" in result
        assert "I will query the database now" in result

    def test_assistant_thinking_prefix_stripped(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="Thinking: I need to check something",
        )
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert "I need to check something" in result

    def test_assistant_empty_returns_empty(self):
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS, messages="")
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert result == ""

    def test_assistant_generic_thinking_skipped(self):
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS, messages="Thinking...")
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert result == ""

    def test_tool_with_result_preview(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="describe_table",
            input_data={"function_name": "describe_table"},
            output_data={"result": "columns: id, name, created_at"},
        )
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert "columns" in result

    def test_long_message_truncated(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="A" * 200,
        )
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert "..." in result
        assert len(result) < 200

    def test_other_role_returns_empty(self):
        action = _make_action(ActionRole.WORKFLOW, ActionStatus.SUCCESS, messages="workflow")
        executor = ChatExecutor()
        result = executor.format_action_for_stream(action)
        assert result == ""


@pytest.mark.ci
class TestChatExecutorExtractSqlAndResponse:
    """Test extract_sql_and_response."""

    def test_empty_actions(self):
        executor = ChatExecutor()
        sql, response = executor.extract_sql_and_response([], None)
        assert sql is None
        assert response is None

    def test_no_output(self):
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS)
        executor = ChatExecutor()
        sql, response = executor.extract_sql_and_response([action], None)
        assert sql is None
        assert response is None

    def test_extracts_sql_and_response(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"sql": "SELECT 1", "response": "Result is 1"},
        )
        executor = ChatExecutor()
        sql, response = executor.extract_sql_and_response([action], None)
        assert sql == "SELECT 1"
        assert response == "Result is 1"

    def test_none_response(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"sql": "SELECT 1", "response": None},
        )
        executor = ChatExecutor()
        sql, response = executor.extract_sql_and_response([action], None)
        assert sql == "SELECT 1"
        assert response is None

    def test_dict_response(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"sql": None, "response": {"raw_output": "hello"}},
        )
        executor = ChatExecutor()
        sql, response = executor.extract_sql_and_response([action], None)
        assert response == "hello"

    def test_non_string_response(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"sql": None, "response": 42},
        )
        executor = ChatExecutor()
        sql, response = executor.extract_sql_and_response([action], None)
        assert response == "42"


class _FakeInterruptController:
    def __init__(self):
        self._interrupted = False

    @property
    def is_interrupted(self):
        return self._interrupted

    def interrupt(self):
        self._interrupted = True


class _FakeNode:
    def __init__(self, actions=None, *, never_complete=False):
        self.actions = actions or []
        self.never_complete = never_complete
        self.interrupt_controller = _FakeInterruptController()
        self.interaction_broker = None
        self.closed = False
        self.input = None

    async def execute_stream_with_interactions(self, _actions):
        try:
            for action in self.actions:
                yield action
            if self.never_complete:
                await asyncio.sleep(3600)
        finally:
            self.closed = True


class _BackgroundTaskNode(_FakeNode):
    def __init__(self, actions=None, *, never_complete=False):
        super().__init__(actions, never_complete=never_complete)
        self.background_cancelled = False

    async def _background(self):
        try:
            await asyncio.sleep(3600)
        finally:
            self.background_cancelled = True

    async def execute_stream_with_interactions(self, actions):
        asyncio.create_task(self._background())
        async for action in super().execute_stream_with_interactions(actions):
            yield action


class _FakeChatCommands:
    def __init__(self, node):
        self.current_node = node

    def _should_create_new_node(self, _current_subagent):
        return False

    def create_node_input(self, *_args, **_kwargs):
        return object(), None


class _FakeCli:
    def __init__(self, node):
        self.chat_commands = _FakeChatCommands(node)
        self.at_completer = MagicMock()
        self.at_completer.parse_at_context.return_value = ([], [], [], None)
        self.actions = []


@pytest.mark.ci
class TestChatExecutorExecuteChatStream:
    """Test stream safety guards without real LLM calls."""

    def test_max_actions_interrupts_before_yielding_extra_action(self):
        first_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="thinking",
            messages="first",
        )
        extra_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="response",
            messages="extra",
        )
        node = _FakeNode([first_action, extra_action])

        results = list(
            ChatExecutor().execute_chat_stream("hello", _FakeCli(node), max_actions=1, stream_idle_timeout=1)
        )

        assert first_action in results
        assert extra_action not in results
        assert any(isinstance(item, str) and "stopped after 1 actions" in item for item in results)
        assert node.interrupt_controller.is_interrupted is True
        assert node.closed is True

    def test_idle_timeout_interrupts_and_closes_stream(self):
        node = _FakeNode(never_complete=True)

        results = list(
            ChatExecutor().execute_chat_stream("hello", _FakeCli(node), max_actions=10, stream_idle_timeout=0.01)
        )

        assert any(isinstance(item, str) and "timed out after" in item for item in results)
        assert node.interrupt_controller.is_interrupted is True
        assert node.closed is True

    def test_zero_idle_timeout_uses_plain_next_action(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="response",
            messages="done",
        )
        node = _FakeNode([action])

        results = list(
            ChatExecutor().execute_chat_stream("hello", _FakeCli(node), max_actions=10, stream_idle_timeout=0)
        )

        assert results == [action]
        assert node.closed is True

    def test_stream_can_finish_exactly_at_max_actions(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="response",
            messages="done",
        )
        node = _FakeNode([action])

        results = list(
            ChatExecutor().execute_chat_stream("hello", _FakeCli(node), max_actions=1, stream_idle_timeout=1)
        )

        assert results == [action]
        assert node.interrupt_controller.is_interrupted is False
        assert node.closed is True

    def test_cleanup_cancels_pending_loop_tasks(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="response",
            messages="done",
        )
        extra_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="thinking",
            messages="extra",
        )
        node = _BackgroundTaskNode([action, extra_action])

        list(ChatExecutor().execute_chat_stream("hello", _FakeCli(node), max_actions=1, stream_idle_timeout=1))

        assert node.closed is True
        assert node.background_cancelled is True


@pytest.mark.ci
class TestChatExecutorStreamConfig:
    def test_positive_float_env_parses_default_invalid_and_non_positive(self, monkeypatch):
        monkeypatch.delenv("DATUS_TEST_FLOAT", raising=False)
        assert chat_executor._positive_float_env("DATUS_TEST_FLOAT", 1.5) == 1.5

        monkeypatch.setenv("DATUS_TEST_FLOAT", "2.5")
        assert chat_executor._positive_float_env("DATUS_TEST_FLOAT", 1.5) == 2.5

        monkeypatch.setenv("DATUS_TEST_FLOAT", "invalid")
        assert chat_executor._positive_float_env("DATUS_TEST_FLOAT", 1.5) == 1.5

        monkeypatch.setenv("DATUS_TEST_FLOAT", "0")
        assert chat_executor._positive_float_env("DATUS_TEST_FLOAT", 1.5) == 1.5

    def test_positive_int_env_parses_default_invalid_and_non_positive(self, monkeypatch):
        monkeypatch.delenv("DATUS_TEST_INT", raising=False)
        assert chat_executor._positive_int_env("DATUS_TEST_INT", 7) == 7

        monkeypatch.setenv("DATUS_TEST_INT", "9")
        assert chat_executor._positive_int_env("DATUS_TEST_INT", 7) == 9

        monkeypatch.setenv("DATUS_TEST_INT", "invalid")
        assert chat_executor._positive_int_env("DATUS_TEST_INT", 7) == 7

        monkeypatch.setenv("DATUS_TEST_INT", "0")
        assert chat_executor._positive_int_env("DATUS_TEST_INT", 7) == 7
