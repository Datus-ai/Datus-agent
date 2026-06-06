# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/web/chat_executor.py — ChatExecutor."""

import pytest

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
