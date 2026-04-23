# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/schemas/action_content_builder.py
"""

from datetime import datetime, timedelta

import pytest

from datus.schemas.action_content_builder import (
    action_to_content,
    build_interaction_content,
    build_interaction_result_content,
    build_response_content,
    build_thinking_content,
    build_tool_call_content,
    build_tool_result_content,
    build_user_content,
    extract_function,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


def _make_action(
    role=ActionRole.ASSISTANT,
    status=ActionStatus.SUCCESS,
    action_type="test",
    input_data=None,
    output_data=None,
    messages="",
    action_id="test-action-1",
    depth=0,
):
    return ActionHistory(
        action_id=action_id,
        role=role,
        messages=messages,
        action_type=action_type,
        input=input_data,
        output=output_data,
        status=status,
        depth=depth,
    )


class TestExtractFunction:
    def test_basic(self):
        action = _make_action(input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}})
        name, args = extract_function(action)
        assert name == "read_query"
        assert args == {"sql": "SELECT 1"}

    def test_string_arguments(self):
        action = _make_action(input_data={"function_name": "fn", "arguments": '{"key": "val"}'})
        name, args = extract_function(action)
        assert name == "fn"
        assert args == {"key": "val"}

    def test_non_dict_input(self):
        action = _make_action(input_data="not a dict")
        name, args = extract_function(action)
        assert name == "unknown"
        assert args == {}

    def test_invalid_json_arguments(self):
        action = _make_action(input_data={"function_name": "fn", "arguments": "not json"})
        name, args = extract_function(action)
        assert name == "fn"
        assert args == {}

    def test_non_dict_arguments(self):
        action = _make_action(input_data={"function_name": "fn", "arguments": 42})
        name, args = extract_function(action)
        assert name == "fn"
        assert args == {}


class TestBuildToolCallContent:
    def test_basic(self):
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.PROCESSING,
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}},
        )
        contents = build_tool_call_content(action)
        assert len(contents) == 1
        assert contents[0].type == "call-tool"
        assert contents[0].payload["toolName"] == "read_query"
        assert contents[0].payload["callToolId"] == "test-action-1"


class TestBuildToolResultContent:
    def test_basic(self):
        start = datetime.now()
        end = start + timedelta(seconds=1.5)
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.SUCCESS,
            action_id="complete_tool-1",
            input_data={"function_name": "read_query", "arguments": {}},
            output_data={"summary": "ok", "raw_output": {"rows": 5}},
        )
        action.start_time = start
        action.end_time = end
        contents = build_tool_result_content(action)
        assert len(contents) == 1
        assert contents[0].type == "call-tool-result"
        assert contents[0].payload["callToolId"] == "tool-1"
        assert contents[0].payload["toolName"] == "read_query"
        assert contents[0].payload["duration"] == pytest.approx(1.5, abs=0.1)

    def test_non_dict_output(self):
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.SUCCESS,
            action_id="complete_tool-2",
            input_data={"function_name": "fn", "arguments": {}},
            output_data="raw string output",
        )
        contents = build_tool_result_content(action)
        assert len(contents) == 1
        assert contents[0].type == "call-tool-result"
        assert contents[0].payload["result"] == "raw string output"
        assert contents[0].payload["shortDesc"] == ""

    def test_none_output(self):
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.SUCCESS,
            action_id="complete_tool-3",
            input_data={"function_name": "fn", "arguments": {}},
            output_data=None,
        )
        contents = build_tool_result_content(action)
        assert len(contents) == 1
        assert contents[0].payload["result"] is None


class TestBuildThinkingContent:
    def test_llm_generation(self):
        action = _make_action(action_type="llm_generation", messages="thinking about it")
        contents = build_thinking_content(action)
        assert len(contents) == 1
        assert contents[0].type == "thinking"
        assert contents[0].payload["content"] == "thinking about it"

    def test_output_with_response(self):
        action = _make_action(output_data={"response": "some response"})
        contents = build_thinking_content(action)
        assert contents is not None
        assert len(contents) >= 1

    def test_fallback_to_messages(self):
        action = _make_action(messages="fallback", output_data={})
        contents = build_thinking_content(action)
        assert contents[0].type == "thinking"
        assert contents[0].payload["content"] == "fallback"


class TestBuildInteractionContent:
    def test_with_choices(self):
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            input_data={"events": [{"content": "Pick one", "choices": {"y": "Yes", "n": "No"}}]},
        )
        contents = build_interaction_content(action)
        assert len(contents) == 1
        assert contents[0].type == "user-interaction"
        assert contents[0].payload["content"] == "Pick one"
        assert contents[0].payload["interactionKey"] == "test-action-1"
        assert len(contents[0].payload["options"]) == 2

    def test_without_choices(self):
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            input_data={"events": [{"content": "Enter something"}]},
        )
        contents = build_interaction_content(action)
        assert contents[0].payload["options"] is None

    def test_with_broker_format_single(self):
        """Broker canonical format: single event with content + choices."""
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            input_data={"events": [{"content": "Pick one", "choices": {"y": "Yes", "n": "No"}}]},
        )
        contents = build_interaction_content(action)
        assert contents[0].payload["content"] == "Pick one"
        assert len(contents[0].payload["options"]) == 2

    def test_with_broker_format_batch(self):
        """Broker canonical format: multiple events joined into numbered list."""
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            input_data={
                "events": [
                    {"content": "Question A?", "choices": {"y": "Yes"}},
                    {"content": "Question B?", "choices": {}},
                ]
            },
        )
        contents = build_interaction_content(action)
        content_text = contents[0].payload["content"]
        assert "1. Question A?" in content_text
        assert "2. Question B?" in content_text


class TestBuildInteractionResultContent:
    def test_with_content(self):
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.SUCCESS,
            output_data={"content": "Result text", "user_choice": "y"},
        )
        contents = build_interaction_result_content(action)
        assert contents is not None
        assert contents[0].type == "markdown"
        assert contents[0].payload["content"] == "Result text"

    def test_empty_content(self):
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.SUCCESS,
            output_data={"content": ""},
        )
        assert build_interaction_result_content(action) is None


class TestBuildResponseContent:
    def test_with_sql(self):
        action = _make_action(output_data={"sql": "SELECT 1", "response": "Here is the query"})
        contents = build_response_content(action)
        assert len(contents) == 2
        assert contents[0].type == "code"
        assert contents[0].payload["content"] == "SELECT 1"
        assert contents[1].type == "markdown"

    def test_without_sql(self):
        action = _make_action(output_data={"response": "Just text"})
        contents = build_response_content(action)
        assert len(contents) == 1
        assert contents[0].type == "markdown"


class TestBuildUserContent:
    def test_basic(self):
        action = _make_action(role=ActionRole.USER, input_data={"user_message": "hello"})
        contents = build_user_content(action)
        assert contents[0].type == "markdown"
        assert contents[0].payload["content"] == "hello"


class TestActionToContent:
    def test_tool_processing(self):
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.PROCESSING,
            input_data={"function_name": "fn", "arguments": {}},
        )
        contents = action_to_content(action)
        assert contents is not None
        assert contents[0].type == "call-tool"

    def test_tool_success(self):
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.SUCCESS,
            input_data={"function_name": "fn", "arguments": {}},
            output_data={"summary": "", "raw_output": "ok"},
        )
        contents = action_to_content(action)
        assert contents is not None
        assert contents[0].type == "call-tool-result"

    def test_user_role_skipped(self):
        action = _make_action(role=ActionRole.USER)
        assert action_to_content(action) is None

    def test_final_response_skipped(self):
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="chat_response",
        )
        assert action_to_content(action) is None

    def test_interaction_processing(self):
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            input_data={"content": "Choose"},
        )
        contents = action_to_content(action)
        assert contents is not None
        assert contents[0].type == "user-interaction"

    def test_interaction_success_with_content(self):
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.SUCCESS,
            output_data={"content": "done"},
        )
        contents = action_to_content(action)
        assert contents is not None

    def test_interaction_success_empty(self):
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.SUCCESS,
            output_data={"content": ""},
        )
        assert action_to_content(action) is None

    def test_assistant_thinking(self):
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="llm_generation",
            messages="thinking...",
        )
        contents = action_to_content(action)
        assert contents is not None
        assert contents[0].type == "thinking"
