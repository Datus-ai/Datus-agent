"""Tests for datus.api.services.action_sse_converter — ActionHistory to SSE conversion."""

from datetime import datetime, timedelta

from datus.api.models.cli_models import IMessageContent, SSEDataType, SSEEvent
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
from datus.utils.text_utils import LITELLM_EMPTY_PLACEHOLDER


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


def _assert_content_list(contents):
    assert isinstance(contents, list)
    assert isinstance(contents[0], IMessageContent)
    return contents


def _assert_sse_event(event):
    assert isinstance(event, SSEEvent)
    return event


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
        assert contents[0].payload["result"] == {"success": 1, "result": "data..."}

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

    def test_failed_tool_includes_error_from_output(self):
        """Failed tool action includes an error field from output.error."""
        action = _make_action(
            action_id="complete_tool-77",
            status=ActionStatus.FAILED,
            input={"function_name": "run_sql"},
            output={"error": "boom"},
        )
        contents = _build_tool_result_content(action)
        assert contents[0].type == "call-tool-result"
        assert contents[0].payload["error"] == "boom"
        assert contents[0].payload["result"] == {"success": 0, "result": None, "error": "boom"}

    def test_failed_tool_falls_back_to_messages(self):
        """Failed tool with no output.error uses action.messages as the error field."""
        action = _make_action(
            action_id="complete_tool-88",
            status=ActionStatus.FAILED,
            input={"function_name": "run_sql"},
            output={},
            messages="kaboom",
        )
        contents = _build_tool_result_content(action)
        assert contents[0].payload["error"] == "kaboom"
        assert contents[0].payload["result"] == {"success": 0, "result": {}, "error": "kaboom"}

    def test_successful_tool_omits_error_field(self):
        """Successful tool actions never carry an error field."""
        action = _make_action(
            action_id="complete_tool-89",
            status=ActionStatus.SUCCESS,
            input={"function_name": "run_sql"},
            output={"raw_output": "data"},
        )
        contents = _build_tool_result_content(action)
        assert "error" not in contents[0].payload
        assert contents[0].payload["result"] == {"success": 1, "result": "data"}

    def test_successful_func_tool_envelope_is_normalized(self):
        """Successful FuncToolResult-shaped output keeps the frontend success envelope."""
        tables = [{"type": "table", "name": "orders"}]
        action = _make_action(
            action_id="complete_tool-90",
            status=ActionStatus.SUCCESS,
            input={"function_name": "list_tables"},
            output={
                "summary": "1 table: orders",
                "raw_output": {"success": 1, "error": None, "result": tables},
            },
        )

        contents = _build_tool_result_content(action)

        assert contents[0].payload["result"] == {"success": 1, "result": tables}
        assert "error" not in contents[0].payload

    def test_non_envelope_result_key_is_preserved(self):
        """Ordinary tool output with a result key is wrapped, not destructively unwrapped."""
        raw_output = {"result": [{"name": "orders"}], "metadata": {"source": "catalog"}}
        action = _make_action(
            action_id="complete_tool-93",
            status=ActionStatus.SUCCESS,
            input={"function_name": "inspect_catalog"},
            output={"raw_output": raw_output},
        )

        contents = _build_tool_result_content(action)

        assert contents[0].payload["result"] == {"success": 1, "result": raw_output}

    def test_successful_json_func_tool_envelope_is_normalized(self):
        """JSON-string FuncToolResult output is parsed before sending to the web UI."""
        action = _make_action(
            action_id="complete_tool-91",
            status=ActionStatus.SUCCESS,
            input={"function_name": "list_tables"},
            output={"raw_output": '{"success": 1, "error": null, "result": [{"name": "orders"}]}'},
        )

        contents = _build_tool_result_content(action)

        assert contents[0].payload["result"] == {"success": 1, "result": [{"name": "orders"}]}

    def test_failed_func_tool_envelope_uses_nested_error(self):
        """Failed FuncToolResult-shaped output exposes the nested error to the card."""
        action = _make_action(
            action_id="complete_tool-92",
            status=ActionStatus.FAILED,
            input={"function_name": "read_query"},
            output={
                "summary": "Failed",
                "raw_output": {"success": 0, "error": "syntax error", "result": None},
            },
        )

        contents = _build_tool_result_content(action)

        assert contents[0].payload["error"] == "syntax error"
        assert contents[0].payload["result"] == {"success": 0, "result": None, "error": "syntax error"}


class TestBuildSubagentCompleteContent:
    """Tests for _build_subagent_complete_content."""

    def test_success_omits_error_field(self):
        """Successful sub-agent completion never carries an error field."""
        from datus.api.services.action_sse_converter import _build_subagent_complete_content

        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.SUCCESS,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"subagent_type": "explore", "tool_count": 5},
        )
        contents = _build_subagent_complete_content(action)
        assert contents[0].type == "subagent-complete"
        assert contents[0].payload["subagentType"] == "explore"
        assert contents[0].payload["toolCount"] == 5
        assert "error" not in contents[0].payload

    def test_failed_includes_error_from_output(self):
        """Failed sub-agent completion includes an error field from output.error."""
        from datus.api.services.action_sse_converter import _build_subagent_complete_content

        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.FAILED,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"subagent_type": "explore", "tool_count": 1, "error": "timeout"},
        )
        contents = _build_subagent_complete_content(action)
        assert contents[0].payload["error"] == "timeout"

    def test_failed_falls_back_to_messages(self):
        """Failed sub-agent completion with no output.error uses action.messages."""
        from datus.api.services.action_sse_converter import _build_subagent_complete_content

        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.FAILED,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"subagent_type": "explore"},
            messages="cancelled",
        )
        contents = _build_subagent_complete_content(action)
        assert contents[0].payload["error"] == "cancelled"


class TestBuildArtifactContent:
    """Tests for _build_artifact_content — visual report/dashboard completion cards."""

    def test_report_create_mode_emits_full_payload(self):
        """A finished gen_visual_report run carries every field the frontend card needs."""
        from datus.api.services.action_sse_converter import _build_artifact_content

        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={
                "artifact_kind": "report",
                "artifact_mode": "new",
                "report_slug": "q3_revenue",
                "name": "Q3 Revenue Report",
                "description": "Quarterly revenue breakdown.",
                "created_at": "2026-05-16T10:00:00Z",
                "response": "Generated a five-chart report covering Q3 revenue trends.",
            },
        )
        contents = _build_artifact_content(action)
        assert isinstance(contents, list)
        assert len(contents) == 1
        assert contents[0].type == "artifact"
        payload = contents[0].payload
        assert payload["slug"] == "q3_revenue"
        assert payload["kind"] == "report"
        assert payload["mode"] == "new"
        assert payload["name"] == "Q3 Revenue Report"
        assert payload["description"] == "Quarterly revenue breakdown."
        assert payload["created_at"] == "2026-05-16T10:00:00Z"
        assert payload["preview_summary"] == "Generated a five-chart report covering Q3 revenue trends."

    def test_dashboard_edit_mode_uses_dashboard_slug(self):
        """gen_visual_dashboard runs surface dashboard_slug as the artifact slug."""
        from datus.api.services.action_sse_converter import _build_artifact_content

        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_dashboard_response",
            output={
                "artifact_kind": "dashboard",
                "artifact_mode": "edit",
                "dashboard_slug": "sales_overview",
                "name": "Sales Overview",
                "description": None,
                "created_at": "2026-05-10T08:00:00Z",
                "response": "Updated the sales overview dashboard.",
            },
        )
        contents = _build_artifact_content(action)
        assert isinstance(contents, list)
        payload = contents[0].payload
        assert payload["slug"] == "sales_overview"
        assert payload["kind"] == "dashboard"
        assert payload["mode"] == "edit"
        assert payload["description"] is None

    def test_missing_slug_returns_none(self):
        """No slug means the run never bound an artifact — fall back to the regular response path."""
        from datus.api.services.action_sse_converter import _build_artifact_content

        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={"artifact_kind": "report", "report_slug": None, "response": "no-op"},
        )
        assert _build_artifact_content(action) is None

    def test_missing_kind_returns_none(self):
        """A response action without artifact_kind is not a visual-artifact response — caller falls back."""
        from datus.api.services.action_sse_converter import _build_artifact_content

        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="chat_response",
            output={"report_slug": "x", "response": "..."},
        )
        assert _build_artifact_content(action) is None

    def test_unknown_kind_returns_none(self):
        """Kinds outside {'report','dashboard'} are rejected so the frontend never sees a card it can't open."""
        from datus.api.services.action_sse_converter import _build_artifact_content

        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={"artifact_kind": "notebook", "report_slug": "x", "response": "..."},
        )
        assert _build_artifact_content(action) is None

    def test_long_response_truncated_with_ellipsis(self):
        """Preview summary is capped to keep the card readable."""
        from datus.api.services.action_sse_converter import _ARTIFACT_PREVIEW_LIMIT, _build_artifact_content

        long_text = "x" * (_ARTIFACT_PREVIEW_LIMIT + 50)
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={
                "artifact_kind": "report",
                "report_slug": "x",
                "response": long_text,
            },
        )
        contents = _build_artifact_content(action)
        assert isinstance(contents, list)
        preview = contents[0].payload["preview_summary"]
        assert preview.endswith("…")
        assert len(preview) <= _ARTIFACT_PREVIEW_LIMIT + 1

    def test_empty_response_makes_preview_none(self):
        """An empty LLM response should yield ``preview_summary=None`` rather than an empty string."""
        from datus.api.services.action_sse_converter import _build_artifact_content

        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={
                "artifact_kind": "report",
                "report_slug": "x",
                "response": "   ",
            },
        )
        contents = _build_artifact_content(action)
        assert isinstance(contents, list)
        assert contents[0].payload["preview_summary"] is None

    def test_non_dict_output_returns_none(self):
        """Defensive: action.output may be a string in legacy paths — must not crash."""
        from datus.api.services.action_sse_converter import _build_artifact_content

        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output="plain text",
        )
        assert _build_artifact_content(action) is None


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
        contents = _assert_content_list(contents)
        assert contents[0].type == "thinking"
        assert contents[0].payload["content"] == "Analysis complete"

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
        contents = _assert_content_list(contents)
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
        contents = _assert_content_list(contents)
        types = [c.type for c in contents]
        assert "code" in types

    def test_output_non_json_string(self):
        """Thinking content with non-JSON output string goes to thinking payload."""
        action = _make_action(
            action_type="gen_sql",
            output={"response": "plain text analysis"},
        )
        contents = _build_thinking_content(action)
        contents = _assert_content_list(contents)
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

    def test_placeholder_in_output_returns_none(self):
        """LiteLLM sanitizer placeholder in output is filtered out."""
        action = _make_action(
            action_type="gen_sql",
            output={"raw_output": LITELLM_EMPTY_PLACEHOLDER},
            messages="",
        )
        contents = _build_thinking_content(action)
        assert contents is None

    def test_placeholder_in_messages_returns_none(self):
        """LiteLLM sanitizer placeholder in messages fallback is filtered out."""
        action = _make_action(
            action_type="gen_sql",
            output=None,
            messages=LITELLM_EMPTY_PLACEHOLDER,
        )
        contents = _build_thinking_content(action)
        assert contents is None

    def test_placeholder_in_llm_generation_returns_none(self):
        """LiteLLM sanitizer placeholder in llm_generation messages is filtered out."""
        action = _make_action(
            action_type="llm_generation",
            messages=LITELLM_EMPTY_PLACEHOLDER,
        )
        contents = _build_thinking_content(action)
        assert contents is None


class TestBuildInteractionContent:
    """Tests for _build_interaction_content."""

    def test_builds_interaction_with_choices(self):
        """Builds user-interaction payload with content and options."""
        action = _make_action(
            action_id="interact-1",
            action_type="ask_user",
            input={
                "events": [
                    {
                        "content": "Choose a database:",
                        "choices": {"db1": "Database 1", "db2": "Database 2"},
                        "default_choice": "db1",
                        "content_type": "markdown",
                        "allow_free_text": False,
                    }
                ]
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

    def test_multi_select_field_present(self):
        """multiSelect field is included in SSE payload when multi_select is True on the event."""
        action = _make_action(
            action_id="interact-3",
            action_type="ask_user",
            input={
                "events": [
                    {
                        "content": "Pick databases:",
                        "choices": {"db1": "DB 1", "db2": "DB 2"},
                        "multi_select": True,
                    }
                ]
            },
        )
        contents = _build_interaction_content(action)
        req = contents[0].payload["requests"][0]
        assert req["multiSelect"] is True

    def test_multi_select_defaults_to_false(self):
        """multiSelect defaults to False when multi_select is not set on the event."""
        action = _make_action(
            action_id="interact-4",
            action_type="ask_user",
            input={
                "events": [
                    {
                        "content": "Pick one:",
                        "choices": {"a": "A"},
                        "default_choice": "a",
                    }
                ]
            },
        )
        contents = _build_interaction_content(action)
        req = contents[0].payload["requests"][0]
        assert req["multiSelect"] is False

    def test_multi_select_batch_mixed(self):
        """Batch with mixed multi_select values across events."""
        action = _make_action(
            action_id="interact-5",
            action_type="ask_user",
            input={
                "events": [
                    {"content": "Single select:", "choices": {"a": "A"}, "default_choice": "a"},
                    {"content": "Multi select:", "choices": {"b": "B", "c": "C"}, "multi_select": True},
                    {"content": "No flag:", "choices": {"d": "D"}},
                ]
            },
        )
        contents = _build_interaction_content(action)
        requests = contents[0].payload["requests"]
        assert len(requests) == 3
        assert requests[0]["multiSelect"] is False
        assert requests[1]["multiSelect"] is True
        assert requests[2]["multiSelect"] is False


class TestBuildInteractionResultContent:
    """Tests for _build_interaction_result_content."""

    def test_returns_markdown_content(self):
        """Interaction result with content returns markdown."""
        action = _make_action(output={"content": "User selected db1"})
        contents = _build_interaction_result_content(action)
        contents = _assert_content_list(contents)
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

    def test_assistant_failed_action_produces_error_event(self):
        """ASSISTANT + FAILED falls through to the catch-all error branch."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.FAILED,
            output={"error": "Timeout"},
        )
        event = action_to_sse_event(action, event_id=1, message_id="msg-1")
        event = _assert_sse_event(event)
        assert event.id == 1
        assert event.event == "message"
        assert event.data.type == SSEDataType.CREATE_MESSAGE
        content = event.data.payload.content[0]
        assert content.type == "error"
        assert content.payload["content"] == "Timeout"

    def test_tool_failed_produces_call_tool_result_with_error(self):
        """TOOL + FAILED stays on the call-tool-result channel and adds an error field.

        This preserves the callToolId / toolName pairing so the frontend can mark the
        original call-tool card as failed instead of emitting an orphaned error event.
        """
        action = _make_action(
            action_id="complete_tool-99",
            role=ActionRole.TOOL,
            status=ActionStatus.FAILED,
            input={"function_name": "run_sql", "arguments": {"sql": "SELECT 1"}},
            output={"error": "syntax error", "summary": "failed"},
        )
        event = action_to_sse_event(action, event_id=40, message_id="msg-40")
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "call-tool-result"
        assert content.payload["callToolId"] == "tool-99"
        assert content.payload["toolName"] == "run_sql"
        assert content.payload["error"] == "syntax error"

    def test_tool_failed_without_error_falls_back_to_messages(self):
        """TOOL + FAILED with no output.error uses action.messages for the error field."""
        action = _make_action(
            action_id="complete_tool-100",
            role=ActionRole.TOOL,
            status=ActionStatus.FAILED,
            input={"function_name": "run_sql"},
            output={},
            messages="connection lost",
        )
        event = action_to_sse_event(action, event_id=41, message_id="msg-41")
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "call-tool-result"
        assert content.payload["error"] == "connection lost"

    def test_tool_processing_produces_call_tool(self):
        """TOOL + PROCESSING maps to call-tool content."""
        action = _make_action(
            role=ActionRole.TOOL,
            status=ActionStatus.PROCESSING,
            input={"function_name": "list_tables", "arguments": {}},
        )
        event = action_to_sse_event(action, event_id=2, message_id="msg-2")
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "call-tool-result"
        assert content.payload["result"] == {"success": 1, "result": "plain string result"}
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
        event = _assert_sse_event(event)
        assert event.data.payload.role == "user"
        assert event.data.payload.content[0].type == "markdown"

    def test_user_insert_action_always_emitted(self):
        """``user_insert`` actions represent text the user typed mid-run
        (TUI / API ``/insert``) and must reach the SSE client regardless
        of ``include_user_message`` — that flag gates the initial-request
        echo, which is unrelated to live mid-run injections."""
        action = _make_action(
            role=ActionRole.USER,
            action_type="user_insert",
            status=ActionStatus.SUCCESS,
            input={"user_message": "顺便统计行数", "source": "mid_run_insert"},
            output={"user_message": "顺便统计行数"},
        )
        # No include_user_message flag — default False; must still emit.
        event = action_to_sse_event(action, event_id=6, message_id="msg-6")
        event = _assert_sse_event(event)
        assert event.data.payload.role == "user"
        assert event.data.payload.content[0].type == "markdown"
        assert event.data.payload.content[0].payload["content"] == "顺便统计行数"

    def test_visual_report_response_emits_artifact_without_final_response_flag(self):
        """gen_visual_report completion fires an artifact event even when include_final_response=False.

        The artifact card is not a substitute for the streamed markdown response —
        the LLM's text has already gone through earlier action events — so the
        dispatcher must not gate it behind the regular ``include_final_response``
        toggle.
        """
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={
                "artifact_kind": "report",
                "artifact_mode": "new",
                "report_slug": "q3_revenue",
                "name": "Q3 Revenue",
                "description": "",
                "response": "Done.",
            },
        )
        event = action_to_sse_event(action, event_id=200, message_id="msg-200")
        event = _assert_sse_event(event)
        assert event.data.type == SSEDataType.CREATE_MESSAGE
        content = event.data.payload.content[0]
        assert content.type == "artifact"
        assert content.payload["slug"] == "q3_revenue"
        assert content.payload["kind"] == "report"
        assert content.payload["mode"] == "new"

    def test_visual_dashboard_response_emits_artifact(self):
        """gen_visual_dashboard completion emits artifact card with kind='dashboard'."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_dashboard_response",
            output={
                "artifact_kind": "dashboard",
                "artifact_mode": "edit",
                "dashboard_slug": "sales_overview",
                "name": "Sales Overview",
                "description": "Quarterly sales view.",
                "response": "Updated.",
            },
        )
        event = action_to_sse_event(action, event_id=201, message_id="msg-201")
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "artifact"
        assert content.payload["slug"] == "sales_overview"
        assert content.payload["kind"] == "dashboard"

    def test_artifact_response_with_missing_slug_falls_back_to_markdown_when_requested(self):
        """Malformed artifact payloads must not suppress the assistant's prose.

        ``artifact_kind`` is set but ``report_slug`` is missing — the artifact
        builder returns None. Instead of dropping the wrapper event entirely,
        the converter falls back to the standard ``_response`` markdown
        handler so history tooling (``include_final_response=True``) still
        surfaces the LLM's response text.
        """
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={
                "artifact_kind": "report",
                "report_slug": None,
                "response": "I got partway but never bound a report.",
            },
        )
        event = action_to_sse_event(action, event_id=203, message_id="msg-203", include_final_response=True)
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "markdown"
        assert content.payload["content"] == "I got partway but never bound a report."

    def test_artifact_response_with_missing_slug_drops_event_in_streaming_mode(self):
        """In streaming chat mode (``include_final_response=False``) the
        malformed-artifact fallback honors the same gating as a normal
        wrapper response — no markdown bubble is emitted because the LLM's
        prose was already streamed via earlier response actions."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_visual_report_response",
            output={
                "artifact_kind": "report",
                "report_slug": None,
                "response": "stream already covered this",
            },
        )
        event = action_to_sse_event(action, event_id=204, message_id="msg-204")
        assert event is None

    def test_failed_visual_report_response_falls_through_to_error(self):
        """A FAILED gen_visual_report_response is not an artifact event — must surface as error.

        The base node emits ``_response`` actions with ``status=FAILED`` when
        validate_render never succeeded; in that case the output still carries
        ``artifact_kind`` but no usable slug. The dispatcher's FAILED catch-all
        should win so the frontend renders the failure, not a broken card.
        """
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.FAILED,
            action_type="gen_visual_report_response",
            output={
                "artifact_kind": "report",
                "report_slug": None,
                "error": "validate_render never returned success",
            },
        )
        event = action_to_sse_event(action, event_id=202, message_id="msg-202")
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "error"
        assert "validate_render" in content.payload["content"]

    def test_assistant_response_action_is_skipped(self):
        """ASSISTANT + SUCCESS + _response action_type returns None."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="gen_sql_response",
        )
        event = action_to_sse_event(action, event_id=6, message_id="msg-6")
        assert event is None

    def test_assistant_response_action_included_when_requested(self):
        """Wrapper final response can be emitted when no plain assistant response was streamed."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="chat_response",
            output={"response": "1 table: orders"},
        )
        event = action_to_sse_event(action, event_id=6, message_id="msg-6", include_final_response=True)
        event = _assert_sse_event(event)
        assert event.data.payload.content[0].type == "markdown"
        assert event.data.payload.content[0].payload == {"content": "1 table: orders"}

    def test_plain_assistant_response_renders_as_markdown(self):
        """Completed model response actions must not be mislabeled as thinking."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="response",
            output={"raw_output": "Hello from the model", "is_thinking": False},
        )
        event = action_to_sse_event(action, event_id=6, message_id="msg-6")
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "markdown"
        assert content.payload == {"content": "Hello from the model"}

    def test_empty_assistant_response_action_still_skipped_when_requested(self):
        """Empty wrapper responses do not create blank assistant bubbles."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="chat_response",
            output={"response": ""},
        )
        event = action_to_sse_event(action, event_id=6, message_id="msg-6", include_final_response=True)
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
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
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
        assert event is None

    def test_assistant_thinking_non_response_type(self):
        """Non-response assistant action produces thinking content."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="gen_sql",
            output={"thinking": "Analyzing query..."},
        )
        event = action_to_sse_event(action, event_id=10, message_id="msg-10")
        event = _assert_sse_event(event)
        assert event.data.payload.content[0].payload["content"] == "Analyzing query..."

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
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
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
        event = _assert_sse_event(event)
        assert event.data.payload.content[0].payload["duration"] == 0.0

    def test_thinking_delta_first_creates_message(self):
        """First thinking_delta uses CREATE_MESSAGE SSE type."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="thinking_delta",
            output={"delta": "Hello ", "accumulated": "Hello "},
        )
        event = action_to_sse_event(action, event_id=20, message_id="msg-20", stream_thinking=True, is_first_delta=True)
        event = _assert_sse_event(event)
        assert event.event == "message"
        assert event.data.type == SSEDataType.CREATE_MESSAGE
        content = event.data.payload.content[0]
        assert content.type == "thinking"
        assert content.payload["content"] == "Hello "

    def test_thinking_delta_subsequent_appends_message(self):
        """Subsequent thinking_delta uses APPEND_MESSAGE SSE type."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="thinking_delta",
            output={"delta": "world"},
        )
        event = action_to_sse_event(
            action, event_id=21, message_id="msg-21", stream_thinking=True, is_first_delta=False
        )
        event = _assert_sse_event(event)
        assert event.event == "message"
        assert event.data.type == SSEDataType.APPEND_MESSAGE
        content = event.data.payload.content[0]
        assert content.type == "thinking"
        assert content.payload["content"] == "world"

    def test_thinking_delta_skipped_when_stream_disabled(self):
        """thinking_delta returns None when stream_thinking=False (default)."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="thinking_delta",
            output={"delta": "Hello "},
        )
        event = action_to_sse_event(action, event_id=20, message_id="msg-20")
        assert event is None

    def test_thinking_delta_with_empty_delta(self):
        """thinking_delta with empty delta string still produces event when enabled."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="thinking_delta",
            output={"delta": "", "accumulated": ""},
        )
        event = action_to_sse_event(action, event_id=22, message_id="msg-22", stream_thinking=True)
        event = _assert_sse_event(event)
        assert event.event == "message"
        assert event.data.type == SSEDataType.CREATE_MESSAGE  # is_first_delta defaults to True
        assert event.data.payload.content[0].payload["content"] == ""

    def test_thinking_delta_non_dict_output(self):
        """thinking_delta with non-dict output defaults to empty delta."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="thinking_delta",
            output="raw string",
        )
        event = action_to_sse_event(action, event_id=23, message_id="msg-23", stream_thinking=True)
        event = _assert_sse_event(event)
        assert event.event == "message"
        assert event.data.payload.content[0].payload["content"] == ""

    def test_thinking_delta_with_depth(self):
        """thinking_delta carries depth and parent_action_id."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.PROCESSING,
            action_type="thinking_delta",
            output={"delta": "chunk"},
            depth=1,
            parent_action_id="parent-003",
        )
        event = action_to_sse_event(action, event_id=24, message_id="msg-24", stream_thinking=True)
        event = _assert_sse_event(event)
        assert event.event == "message"
        assert event.data.payload.depth == 1
        assert event.data.payload.parent_action_id == "parent-003"

    def test_thinking_response_update_message(self):
        """Complete thinking response with is_update=True uses UPDATE_MESSAGE."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="response",
            output={"is_thinking": True, "thinking": "Full thinking content"},
        )
        event = action_to_sse_event(action, event_id=30, message_id="msg-30", stream_thinking=True, is_update=True)
        event = _assert_sse_event(event)
        assert event.data.type == SSEDataType.UPDATE_MESSAGE

    def test_thinking_response_default_create_message(self):
        """Complete thinking response with is_update=False (default) uses CREATE_MESSAGE."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="response",
            output={"is_thinking": True, "thinking": "Full thinking content"},
        )
        event = action_to_sse_event(action, event_id=31, message_id="msg-31", stream_thinking=True)
        event = _assert_sse_event(event)
        assert event.data.type == SSEDataType.CREATE_MESSAGE

    def test_finalize_progress_first_emit_is_create_message(self):
        """Stage 1 finalize_progress arrives before the bubble exists → CREATE_MESSAGE."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="finalize_progress",
            output={"stage": 1, "text": "Generating insights and follow-up questions..."},
        )
        event = action_to_sse_event(action, event_id=40, message_id="msg-40", is_update=False)
        event = _assert_sse_event(event)
        assert event.data.type == SSEDataType.CREATE_MESSAGE
        content = event.data.payload.content[0]
        assert content.type == "markdown"
        assert content.payload["content"] == "Generating insights and follow-up questions..."

    def test_finalize_progress_subsequent_emit_is_update_message(self):
        """Stages 2 and 3 reuse the same action_id → caller passes is_update=True → UPDATE_MESSAGE."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="finalize_progress",
            output={"stage": 2, "text": "Refining analysis intent..."},
        )
        event = action_to_sse_event(action, event_id=41, message_id="msg-40", is_update=True)
        event = _assert_sse_event(event)
        # Same message_id as the stage-1 emission, but UPDATE so the chat
        # panel replaces the bubble's body in place.
        assert event.data.type == SSEDataType.UPDATE_MESSAGE
        assert event.data.payload.message_id == "msg-40"
        assert event.data.payload.content[0].payload["content"] == "Refining analysis intent..."

    def test_finalize_progress_empty_text_skipped(self):
        """Defensive: a finalize_progress with no text drops the event rather than emitting a blank bubble."""
        action = _make_action(
            role=ActionRole.ASSISTANT,
            status=ActionStatus.SUCCESS,
            action_type="finalize_progress",
            output={"stage": 1, "text": ""},
        )
        event = action_to_sse_event(action, event_id=42, message_id="msg-42")
        assert event is None

    def test_subagent_complete_failed_produces_subagent_complete_with_error(self):
        """subagent_complete + FAILED stays on the subagent-complete channel and adds an error field.

        This keeps the subagentType / duration metadata so the frontend can mark the
        original subagent card as failed instead of emitting an orphaned error event.
        """
        action = _make_action(
            role=ActionRole.SYSTEM,
            status=ActionStatus.FAILED,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            output={"error": "sub-agent timed out", "subagent_type": "explore", "tool_count": 3},
        )
        event = action_to_sse_event(action, event_id=17, message_id="msg-17")
        event = _assert_sse_event(event)
        content = event.data.payload.content[0]
        assert content.type == "subagent-complete"
        assert content.payload["subagentType"] == "explore"
        assert content.payload["toolCount"] == 3
        assert content.payload["error"] == "sub-agent timed out"
