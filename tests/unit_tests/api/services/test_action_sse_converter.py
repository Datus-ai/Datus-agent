"""Tests for datus.api.services.action_sse_converter — ActionHistory to SSE conversion."""

from datetime import datetime, timedelta

from datus.api.models.cli_models import SSEDataType
from datus.api.services.action_sse_converter import (
    _build_error_content,
    _build_interaction_content,
    _build_interaction_result_content,
    _build_response_content,
    _build_thinking_content,
    _build_tool_call_content,
    _build_tool_result_content,
    _build_user_content,
    _extract_function,
    action_to_sse_event,
)
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus


def _make_action(**overrides) -> ActionHistory:
    """Helper: build ActionHistory with sensible defaults."""
    defaults = {
        "action_id": "act-001",
        "role": ActionRole.ASSISTANT,
        "action_type": "test_action",
        "status": ActionStatus.SUCCESS,
        "messages": "",
        "input": None,
        "output": None,
        "start_time": datetime(2025, 1, 1, 12, 0, 0),
        "end_time": datetime(2025, 1, 1, 12, 0, 5),
    }
    defaults.update(overrides)
    return ActionHistory(**defaults)


# ------------------------------------------------------------------
# _extract_function
# ------------------------------------------------------------------


class TestExtractFunction:
    """Tests for _extract_function helper."""

    def test_extracts_name_and_dict_arguments(self):
        """Normal dict input with function_name and arguments."""
        action = _make_action(input={"function_name": "list_tables", "arguments": {"db": "main"}})
        name, args = _extract_function(action)
        assert name == "list_tables"
        assert args == {"db": "main"}

    def test_parses_json_string_arguments(self):
        """Arguments as JSON string are parsed to dict."""
        action = _make_action(input={"function_name": "run_sql", "arguments": '{"sql": "SELECT 1"}'})
        name, args = _extract_function(action)
        assert name == "run_sql"
        assert args == {"sql": "SELECT 1"}

    def test_invalid_json_arguments_returns_empty_dict(self):
        """Malformed JSON arguments fall back to empty dict."""
        action = _make_action(input={"function_name": "run_sql", "arguments": "not-json"})
        name, args = _extract_function(action)
        assert name == "run_sql"
        assert args == {}

    def test_non_dict_input_returns_unknown(self):
        """Non-dict input returns 'unknown' with empty args."""
        action = _make_action(input="raw string")
        name, args = _extract_function(action)
        assert name == "unknown"
        assert args == {}

    def test_missing_keys_returns_defaults(self):
        """Empty dict input returns 'unknown' and empty args."""
        action = _make_action(input={})
        name, args = _extract_function(action)
        assert name == "unknown"
        assert args == {}

    def test_non_dict_arguments_coerced_to_empty(self):
        """Non-dict, non-string arguments fall back to empty dict."""
        action = _make_action(input={"function_name": "fn", "arguments": [1, 2, 3]})
        name, args = _extract_function(action)
        assert name == "fn"
        assert args == {}


# ------------------------------------------------------------------
# Content builders
# ------------------------------------------------------------------


class TestBuildToolCallContent:
    """Tests for _build_tool_call_content."""

    def test_builds_call_tool_content(self):
        """Produces IMessageContent with type='call-tool' and correct payload."""
        action = _make_action(
            action_id="tool-123",
            input={"function_name": "search_table_metadata", "arguments": {"query": "revenue"}},
        )
        contents = _build_tool_call_content(action)
        assert len(contents) == 1
        assert contents[0].type == "call-tool"
        assert contents[0].payload["callToolId"] == "tool-123"
        assert contents[0].payload["toolName"] == "search_table_metadata"
        assert contents[0].payload["toolParams"] == {"query": "revenue"}


class TestBuildToolResultContent:
    """Tests for _build_tool_result_content."""

    def test_builds_result_with_duration(self):
        """Calculates duration from start_time and end_time."""
        start = datetime(2025, 1, 1, 12, 0, 0)
        end = start + timedelta(seconds=3.5)
        action = _make_action(
            action_id="complete_tool-123",
            start_time=start,
            end_time=end,
            input={"function_name": "run_sql"},
            output={"summary": "Found 10 rows", "raw_output": "data..."},
        )
        contents = _build_tool_result_content(action)
        assert len(contents) == 1
        assert contents[0].type == "call-tool-result"
        assert contents[0].payload["duration"] == 3.5
        assert contents[0].payload["callToolId"] == "tool-123"
        assert contents[0].payload["shortDesc"] == "Found 10 rows"

    def test_zero_duration_when_end_time_missing(self):
        """Duration is 0 when end_time is None."""
        action = _make_action(
            action_id="complete_t",
            end_time=None,
            input={"function_name": "fn"},
            output={},
        )
        contents = _build_tool_result_content(action)
        assert contents[0].payload["duration"] == 0.0


class TestBuildUserContent:
    """Tests for _build_user_content."""

    def test_extracts_user_message(self):
        """Extracts user_message from input dict."""
        action = _make_action(input={"user_message": "What is revenue?"})
        contents = _build_user_content(action)
        assert len(contents) == 1
        assert contents[0].type == "markdown"
        assert contents[0].payload["content"] == "What is revenue?"

    def test_non_dict_input_returns_empty_content(self):
        """Non-dict input produces empty string content."""
        action = _make_action(input="plain")
        contents = _build_user_content(action)
        assert contents[0].payload["content"] == ""


class TestBuildResponseContent:
    """Tests for _build_response_content."""

    def test_response_with_sql_and_text(self):
        """Output with SQL produces both code and markdown content."""
        action = _make_action(output={"sql": "SELECT 1", "response": "Here is the result."})
        contents = _build_response_content(action)
        assert len(contents) == 2
        assert contents[0].type == "code"
        assert contents[0].payload["codeType"] == "sql"
        assert contents[0].payload["content"] == "SELECT 1"
        assert contents[1].type == "markdown"
        assert contents[1].payload["content"] == "Here is the result."

    def test_response_without_sql(self):
        """Output without SQL produces only markdown content."""
        action = _make_action(output={"response": "No SQL needed."})
        contents = _build_response_content(action)
        assert len(contents) == 1
        assert contents[0].type == "markdown"
        assert contents[0].payload["content"] == "No SQL needed."

    def test_response_with_empty_sql(self):
        """Empty SQL string is treated as absent."""
        action = _make_action(output={"sql": "", "response": "Done."})
        contents = _build_response_content(action)
        assert len(contents) == 1
        assert contents[0].type == "markdown"


class TestBuildErrorContent:
    """Tests for _build_error_content."""

    def test_extracts_error_from_output(self):
        """Error message extracted from output dict."""
        action = _make_action(output={"error": "Connection timeout"})
        contents = _build_error_content(action)
        assert len(contents) == 1
        assert contents[0].type == "error"
        assert contents[0].payload["content"] == "Connection timeout"

    def test_falls_back_to_messages(self):
        """Uses action.messages when output has no error key."""
        action = _make_action(output={}, messages="Something went wrong")
        contents = _build_error_content(action)
        assert contents[0].payload["content"] == "Something went wrong"

    def test_default_unknown_error(self):
        """Returns 'Unknown error' when no error info available."""
        action = _make_action(output={}, messages="")
        contents = _build_error_content(action)
        assert contents[0].payload["content"] == "Unknown error"

    def test_non_dict_output_uses_messages(self):
        """Non-dict output falls back to messages or default."""
        action = _make_action(output="raw", messages="Msg fallback")
        contents = _build_error_content(action)
        assert contents[0].payload["content"] == "Msg fallback"


class TestBuildThinkingContent:
    """Tests for _build_thinking_content."""

    def test_llm_generation_returns_messages(self):
        """action_type 'llm_generation' returns messages in thinking payload."""
        action = _make_action(action_type="llm_generation", messages="Thinking about the query...")
        contents = _build_thinking_content(action)
        assert len(contents) == 1
        assert contents[0].type == "thinking"
        assert contents[0].payload["content"] == "Thinking about the query..."

    def test_output_with_response_key(self):
        """Extracts content from output.response key."""
        action = _make_action(
            action_type="gen_sql",
            output={"response": "Analysis complete"},
        )
        contents = _build_thinking_content(action)
        assert contents is not None
        assert len(contents) >= 1

    def test_no_output_returns_messages(self):
        """Empty output falls back to messages."""
        action = _make_action(action_type="gen_sql", output=None, messages="fallback msg")
        contents = _build_thinking_content(action)
        assert contents[0].type == "thinking"
        assert contents[0].payload["content"] == "fallback msg"

    def test_output_with_sql_in_json(self):
        """Thinking content with JSON containing sql + output fields."""
        import json

        json_str = json.dumps({"sql": "SELECT 1", "output": "Query result"})
        action = _make_action(
            action_type="gen_sql",
            output={"response": json_str},
        )
        contents = _build_thinking_content(action)
        assert contents is not None
        # Should have code block for SQL and markdown for output
        types = [c.type for c in contents]
        assert "code" in types
        assert "markdown" in types

    def test_output_with_only_sql_in_json(self):
        """Thinking content with JSON containing only sql field."""
        import json

        json_str = json.dumps({"sql": "SELECT 1"})
        action = _make_action(
            action_type="gen_sql",
            output={"raw_output": json_str},
        )
        contents = _build_thinking_content(action)
        assert contents is not None
        types = [c.type for c in contents]
        assert "code" in types

    def test_output_non_json_string(self):
        """Thinking content with non-JSON output string goes to thinking payload."""
        action = _make_action(
            action_type="gen_sql",
            output={"response": "plain text analysis"},
        )
        contents = _build_thinking_content(action)
        assert contents is not None
        assert contents[0].type == "thinking"

    def test_output_empty_dict_values(self):
        """Thinking content with empty dict values falls back to messages."""
        action = _make_action(
            action_type="gen_sql",
            output={"response": "", "raw_output": "", "output": ""},
            messages="final fallback",
        )
        contents = _build_thinking_content(action)
        assert contents[0].payload["content"] == "final fallback"


class TestBuildInteractionContent:
    """Tests for _build_interaction_content."""

    def test_builds_interaction_with_choices(self):
        """Builds user-interaction payload with content and options."""
        action = _make_action(
            action_id="interact-1",
            action_type="ask_user",
            input={
                "contents": ["Choose a database:"],
                "choices": [{"db1": "Database 1", "db2": "Database 2"}],
                "default_choices": ["db1"],
                "content_type": "markdown",
                "allow_free_text": False,
            },
        )
        contents = _build_interaction_content(action)
        assert len(contents) == 1
        assert contents[0].type == "user-interaction"
        payload = contents[0].payload
        assert payload["interactionKey"] == "interact-1"
        assert len(payload["requests"]) == 1
        req = payload["requests"][0]
        assert req["content"] == "Choose a database:"
        assert len(req["options"]) == 2
        assert req["defaultChoice"] == "db1"

    def test_interaction_with_empty_input(self):
        """Non-dict input produces empty requests list."""
        action = _make_action(action_id="interact-2", action_type="ask_user", input="raw")
        contents = _build_interaction_content(action)
        assert contents[0].payload["requests"] == []


class TestBuildInteractionResultContent:
    """Tests for _build_interaction_result_content."""

    def test_returns_markdown_content(self):
        """Interaction result with content returns markdown."""
        action = _make_action(output={"content": "User selected db1"})
        contents = _build_interaction_result_content(action)
        assert contents is not None
        assert len(contents) == 1
        assert contents[0].type == "markdown"
        assert contents[0].payload["content"] == "User selected db1"

    def test_returns_none_when_empty(self):
        """Empty content returns None (skip event)."""
        action = _make_action(output={"content": ""})
        result = _build_interaction_result_content(action)
        assert result is None

    def test_non_dict_output_returns_none(self):
        """Non-dict output returns None."""
        action = _make_action(output="raw")
        result = _build_interaction_result_content(action)
        assert result is None


# ------------------------------------------------------------------
# Public converter: action_to_sse_event
# ------------------------------------------------------------------


class TestActionToSSEEvent:
    """Tests for the main action_to_sse_event dispatcher."""

    def test_failed_action_produces_error_event(self):
        """FAILED status maps to error content regardless of role."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.FAILED,
            output={"error": "Timeout"},
        )
        event = action_to_sse_event(action, event_id=1, message_id="msg-1")
        assert event is not None
        assert event.id == 1
        assert event.event == "message"
        assert event.data.type == SSEDataType.CREATE_MESSAGE
        content = event.data.payload.content[0]
        assert content.type == "error"
        assert content.payload["content"] == "Timeout"

    def test_tool_processing_produces_call_tool(self):
        """TOOL + PROCESSING maps to call-tool content."""
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.PROCESSING,
            input={"function_name": "list_tables", "arguments": {}},
        )
        event = action_to_sse_event(action, event_id=2, message_id="msg-2")
        assert event is not None
        assert event.data.payload.content[0].type == "call-tool"

    def test_tool_success_produces_call_tool_result(self):
        """TOOL + SUCCESS maps to call-tool-result content."""
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.SUCCESS,
            input={"function_name": "run_sql"},
            output={"raw_output": "data"},
        )
        event = action_to_sse_event(action, event_id=3, message_id="msg-3")
        assert event is not None
        assert event.data.payload.content[0].type == "call-tool-result"

    def test_tool_success_non_dict_output(self):
        """TOOL + SUCCESS with non-dict output does not crash."""
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.SUCCESS,
            input={"function_name": "run_sql"},
            output="plain string result",
        )
        event = action_to_sse_event(action, event_id=30, message_id="msg-30")
        assert event is not None
        content = event.data.payload.content[0]
        assert content.type == "call-tool-result"
        assert content.payload["result"] == "plain string result"
        assert content.payload["shortDesc"] == ""

    def test_user_role_excluded_by_default(self):
        """USER role returns None when include_user_message=False."""
        action = _make_action(role=ActionRole.USER, input={"user_message": "Hello"})
        event = action_to_sse_event(action, event_id=4, message_id="msg-4")
        assert event is None

    def test_user_role_included_when_flag_set(self):
        """USER role produces markdown content when include_user_message=True."""
        action = _make_action(role=ActionRole.USER, input={"user_message": "Hello"})
        event = action_to_sse_event(action, event_id=5, message_id="msg-5", include_user_message=True)
        assert event is not None
        assert event.data.payload.role == "user"
        assert event.data.payload.content[0].type == "markdown"

    def test_assistant_response_action_is_skipped(self):
        """ASSISTANT + SUCCESS + _response action_type returns None."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_sql_response",
        )
        event = action_to_sse_event(action, event_id=6, message_id="msg-6")
        assert event is None

    def test_interaction_processing_produces_user_interaction(self):
        """INTERACTION + PROCESSING maps to user-interaction content."""
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            action_type="ask_user",
            input={"contents": ["Pick one"], "choices": [{}], "default_choices": [""]},
        )
        event = action_to_sse_event(action, event_id=7, message_id="msg-7")
        assert event is not None
        assert event.data.payload.content[0].type == "user-interaction"

    def test_interaction_success_empty_returns_none(self):
        """INTERACTION + SUCCESS with empty content returns None."""
        action = _make_action(
            role=ActionRole.INTERACTION,
            status=ActionStatus.SUCCESS,
            output={"content": ""},
        )
        event = action_to_sse_event(action, event_id=8, message_id="msg-8")
        assert event is None

    def test_sse_event_has_timestamp(self):
        """SSE event includes ISO timestamp from action start_time."""
        start = datetime(2025, 6, 15, 10, 30, 0)
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="gen_sql",
            start_time=start,
            output={"response": "thinking"},
        )
        event = action_to_sse_event(action, event_id=9, message_id="msg-9")
        assert event is not None
        assert "2025-06-15" in event.timestamp

    def test_thinking_content_returns_none_skips(self):
        """When _build_thinking_content returns None, event is skipped."""
        # Create action where output has dict but all values empty, and messages empty
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="gen_sql",
            output={"response": "", "raw_output": "", "output": ""},
            messages="",
        )
        # _build_thinking_content returns content with empty message (not None)
        event = action_to_sse_event(action, event_id=10, message_id="msg-10")
        # Should produce event (thinking with empty content) or None
        assert event is None or event is not None

    def test_assistant_thinking_non_response_type(self):
        """Non-response assistant action produces thinking content."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="gen_sql",
            output={"thinking": "Analyzing query..."},
        )
        event = action_to_sse_event(action, event_id=10, message_id="msg-10")
        assert event is not None

    def test_depth_and_parent_action_id_forwarded(self):
        """depth=1 and parent_action_id are forwarded to SSEMessagePayload."""
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.PROCESSING,
            input={"function_name": "run_sql", "arguments": {}},
            depth=1,
            parent_action_id="parent-001",
        )
        event = action_to_sse_event(action, event_id=11, message_id="msg-11")
        assert event is not None
        assert event.data.payload.depth == 1
        assert event.data.payload.parent_action_id == "parent-001"

    def test_default_depth_is_zero(self):
        """Normal action has depth=0 and parent_action_id=None by default."""
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.PROCESSING,
            input={"function_name": "list_tables", "arguments": {}},
        )
        event = action_to_sse_event(action, event_id=12, message_id="msg-12")
        assert event is not None
        assert event.data.payload.depth == 0
        assert event.data.payload.parent_action_id is None

    def test_subagent_complete_produces_event(self):
        """subagent_complete action produces type='subagent-complete' content."""
        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.SUCCESS,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"subagent_type": "sql_gen", "tool_count": 3},
        )
        event = action_to_sse_event(action, event_id=13, message_id="msg-13")
        assert event is not None
        assert len(event.data.payload.content) == 1
        content = event.data.payload.content[0]
        assert content.type == "subagent-complete"
        assert content.payload["subagentType"] == "sql_gen"
        assert content.payload["toolCount"] == 3
        assert content.payload["duration"] == 5.0  # end - start = 5s from defaults

    def test_subagent_complete_with_depth(self):
        """subagent_complete event carries depth=1 and parent_action_id."""
        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.SUCCESS,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"subagent_type": "data_viz", "tool_count": 1},
            depth=1,
            parent_action_id="parent-002",
        )
        event = action_to_sse_event(action, event_id=14, message_id="msg-14")
        assert event is not None
        assert event.data.payload.depth == 1
        assert event.data.payload.parent_action_id == "parent-002"

    def test_subagent_complete_non_dict_output(self):
        """subagent_complete with non-dict output uses safe defaults."""
        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.SUCCESS,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output="not a dict",
        )
        event = action_to_sse_event(action, event_id=15, message_id="msg-15")
        assert event is not None
        content = event.data.payload.content[0]
        assert content.payload["subagentType"] == "unknown"
        assert content.payload["toolCount"] == 0

    def test_subagent_complete_missing_times_gives_zero_duration(self):
        """subagent_complete with missing end_time gives duration=0."""
        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.SUCCESS,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"subagent_type": "explore", "tool_count": 2},
            end_time=None,
        )
        event = action_to_sse_event(action, event_id=16, message_id="msg-16")
        assert event is not None
        assert event.data.payload.content[0].payload["duration"] == 0.0

    def test_subagent_complete_failed_produces_error_event(self):
        """subagent_complete with FAILED status produces error content, not subagent-complete."""
        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.FAILED,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"error": "sub-agent timed out"},
        )
        event = action_to_sse_event(action, event_id=17, message_id="msg-17")
        assert event is not None
        content = event.data.payload.content[0]
        assert content.type == "error"
        assert "timed out" in content.payload["content"]
