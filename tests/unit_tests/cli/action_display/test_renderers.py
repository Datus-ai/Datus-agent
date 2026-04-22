# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/action_display/renderers.py (ActionRenderer)."""

import uuid
from datetime import datetime, timedelta
from io import StringIO

import pytest
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from datus.cli.action_display.renderers import ActionContentGenerator, ActionRenderer, _get_assistant_content
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.text_utils import LITELLM_EMPTY_PLACEHOLDER


def _make_action(
    role: ActionRole,
    status: ActionStatus,
    depth: int = 0,
    action_type: str = "test",
    messages: str = "",
    input_data: dict = None,
    output_data: dict = None,
    start_time: datetime = None,
    end_time: datetime = None,
    parent_action_id: str = None,
) -> ActionHistory:
    return ActionHistory(
        action_id=str(uuid.uuid4()),
        role=role,
        messages=messages,
        action_type=action_type,
        input=input_data,
        output=output_data,
        status=status,
        start_time=start_time or datetime.now(),
        end_time=end_time,
        depth=depth,
        parent_action_id=parent_action_id,
    )


def _renderer():
    return ActionRenderer(ActionContentGenerator(enable_truncation=True))


def _plain(renderables):
    """Extract plain text from a list of renderables."""
    parts = []
    for r in renderables:
        if isinstance(r, Text):
            parts.append(r.plain)
        elif isinstance(r, Markdown):
            buf = StringIO()
            Console(file=buf, no_color=True, width=120).print(r)
            parts.append(buf.getvalue().strip())
        else:
            parts.append(str(r))
    return "\n".join(parts)


# ── render_subagent_header ─────────────────────────────────────────


@pytest.mark.ci
class TestRenderSubagentHeader:
    """Test subagent header rendering."""

    def test_compact_with_description(self):
        """Compact mode shows description in header."""
        action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages="User: What is total revenue?",
            input_data={"_task_description": "Generate SQL for revenue"},
        )
        result = _renderer().render_subagent_header(action, verbose=False)
        text = _plain(result)
        assert "gen_sql" in text
        assert "Generate SQL for revenue" in text

    def test_compact_without_description(self):
        """Compact mode shows truncated prompt when no description."""
        action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages="User: What is total revenue?",
        )
        result = _renderer().render_subagent_header(action, verbose=False)
        text = _plain(result)
        assert "gen_sql" in text
        assert "What is total revenue?" in text

    def test_verbose_shows_full_prompt(self):
        """Verbose mode shows full prompt line."""
        action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages="User: What is total revenue for Q4?",
        )
        result = _renderer().render_subagent_header(action, verbose=True)
        text = _plain(result)
        assert "prompt:" in text
        assert "What is total revenue for Q4?" in text

    def test_long_prompt_truncated_compact(self):
        """Long prompt is truncated in compact mode."""
        long_prompt = "User: " + "A" * 300
        action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages=long_prompt,
        )
        result = _renderer().render_subagent_header(action, verbose=False)
        text = _plain(result)
        assert " ... " in text

    def test_no_goal_shows_type_only(self):
        """When no prompt or description, show type only."""
        action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages="",
        )
        result = _renderer().render_subagent_header(action, verbose=False)
        text = _plain(result)
        assert "gen_sql" in text


# ── render_subagent_action ─────────────────────────────────────────


@pytest.mark.ci
class TestRenderSubagentAction:
    """Test subagent action rendering."""

    def test_tool_compact(self):
        """TOOL action renders with function name and status."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            messages="list_tables",
            input_data={"function_name": "list_tables"},
            start_time=now,
            end_time=now + timedelta(seconds=1.5),
        )
        result = _renderer().render_subagent_action(action, verbose=False)
        text = _plain(result)
        assert "list_tables" in text
        assert "\u2713" in text
        assert "1.5s" in text

    def test_tool_verbose_with_args_and_output(self):
        """Verbose TOOL action shows arguments and output."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            messages="read_query",
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}},
            output_data={"raw_output": '{"success": true, "text": "ok"}'},
            start_time=now,
            end_time=now + timedelta(seconds=0.5),
        )
        result = _renderer().render_subagent_action(action, verbose=True)
        text = _plain(result)
        assert "read_query" in text
        assert "sql" in text
        assert "SELECT 1" in text

    def test_assistant_action(self):
        """ASSISTANT action renders message content."""
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            depth=1,
            messages="I will query the database",
        )
        result = _renderer().render_subagent_action(action, verbose=False)
        text = _plain(result)
        assert "I will query the database" in text

    def test_user_action_skipped(self):
        """USER actions return empty list (prompt already in header)."""
        action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            messages="User: query",
        )
        result = _renderer().render_subagent_action(action, verbose=False)
        assert result == []

    def test_other_role(self):
        """Other roles render with label."""
        action = _make_action(
            ActionRole.WORKFLOW,
            ActionStatus.SUCCESS,
            depth=1,
            messages="workflow step",
        )
        result = _renderer().render_subagent_action(action, verbose=False)
        text = _plain(result)
        assert "workflow step" in text


# ── render_subagent_done ──────────────────────────────────────────


@pytest.mark.ci
class TestRenderSubagentDone:
    """Test subagent Done summary rendering."""

    def test_done_with_duration(self):
        """Done line shows tool count and duration."""
        now = datetime.now()
        end_action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS, end_time=now + timedelta(seconds=3))
        result = _renderer().render_subagent_done(5, now, end_action)
        text = result.plain
        assert "Done" in text
        assert "5 tool uses" in text
        assert "3.0s" in text

    def test_done_without_start_time(self):
        """Done line works when start_time is None."""
        end_action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS, end_time=datetime.now())
        result = _renderer().render_subagent_done(2, None, end_action)
        text = result.plain
        assert "Done" in text
        assert "2 tool uses" in text


# ── render_subagent_collapsed ────────────────────────────────────


@pytest.mark.ci
class TestRenderSubagentCollapsed:
    """Test collapsed subagent group rendering."""

    def test_collapsed_format(self):
        """Collapsed group shows header and Done line."""
        now = datetime.now()
        first = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages="User: revenue query",
        )
        end = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS, end_time=now + timedelta(seconds=5))
        result = _renderer().render_subagent_collapsed(first, 3, now, end)
        assert len(result) == 2
        text = _plain(result)
        assert "gen_sql" in text
        assert "Done" in text
        assert "3 tool uses" in text
        assert "\u2713" in text


# ── render_main_action ────────────────────────────────────────────


@pytest.mark.ci
class TestRenderMainAction:
    """Test main-agent action rendering."""

    def test_tool_action(self):
        """Regular TOOL action renders with inline completed format."""
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="list_tables",
            input_data={"function_name": "list_tables"},
        )
        result = _renderer().render_main_action(action, verbose=False)
        text = _plain(result)
        assert "list_tables" in text

    def test_task_tool_renders_as_subagent(self):
        """TOOL(task) action renders as subagent summary."""
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="task",
            input_data={"function_name": "task", "type": "gen_sql", "prompt": "revenue query"},
            output_data={"raw_output": '{"success": 1, "result": {"response": "done"}}'},
        )
        result = _renderer().render_main_action(action, verbose=False)
        text = _plain(result)
        assert "gen_sql" in text

    def test_assistant_action_markdown(self):
        """ASSISTANT action renders as Markdown."""
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="Here is the answer",
        )
        result = _renderer().render_main_action(action, verbose=False)
        assert len(result) == 1
        assert isinstance(result[0], Markdown)

    def test_user_action(self):
        """USER action renders with Datus> prefix."""
        action = _make_action(
            ActionRole.USER,
            ActionStatus.SUCCESS,
            messages="User: my question",
        )
        result = _renderer().render_main_action(action, verbose=False)
        text = _plain(result)
        assert "Datus>" in text
        assert "my question" in text


# ── render_task_tool_as_subagent ──────────────────────────────────


@pytest.mark.ci
class TestRenderTaskToolAsSubagent:
    """Test task tool as subagent rendering."""

    def test_compact_with_preview(self):
        """Compact mode shows output preview."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="task",
            input_data={"function_name": "task", "type": "gen_sql", "prompt": "get revenue"},
            output_data={"raw_output": '{"success": 1, "result": {"response": "SQL generated"}}'},
            start_time=now,
            end_time=now + timedelta(seconds=2),
        )
        result = _renderer().render_task_tool_as_subagent(action, verbose=False)
        text = _plain(result)
        assert "gen_sql" in text
        assert "result" in text

    def test_verbose_with_response(self):
        """Verbose mode shows full response."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="task",
            input_data={"function_name": "task", "type": "gen_sql", "prompt": "get revenue"},
            output_data={"raw_output": '{"success": 1, "result": {"response": "SELECT SUM(revenue) FROM sales"}}'},
            start_time=now,
            end_time=now + timedelta(seconds=2),
        )
        result = _renderer().render_task_tool_as_subagent(action, verbose=True)
        text = _plain(result)
        assert "gen_sql" in text
        assert "response:" in text
        assert "SELECT SUM(revenue) FROM sales" in text

    def test_wrapped_arguments_format(self):
        """Type is extracted from arguments JSON string (resume/session_loader format)."""
        import json

        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="task",
            input_data={
                "function_name": "task",
                "arguments": json.dumps({"type": "gen_sql", "prompt": "get revenue", "description": "Revenue query"}),
            },
            output_data={"raw_output": '{"success": 1, "result": {"response": "done"}}'},
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        result = _renderer().render_task_tool_as_subagent(action, verbose=False)
        text = _plain(result)
        assert "gen_sql" in text
        assert "subagent" not in text.lower() or "gen_sql" in text

    def test_wrapped_arguments_dict_format(self):
        """Type is extracted from arguments when already parsed as dict."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="task",
            input_data={
                "function_name": "task",
                "arguments": {"type": "explore", "prompt": "check schema"},
            },
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        result = _renderer().render_task_tool_as_subagent(action, verbose=False)
        text = _plain(result)
        assert "explore" in text


# ── render_processing ─────────────────────────────────────────────


@pytest.mark.ci
class TestRenderProcessing:
    """Test PROCESSING tool blinking frame rendering."""

    def test_processing_frame(self):
        """PROCESSING tool shows blinking frame."""
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="list_tables",
            input_data={"function_name": "list_tables"},
        )
        result = _renderer().render_processing(action, "\u25cb")
        assert isinstance(result, Text)
        assert "list_tables" in result.plain


# ── render_user_header / render_separator ──────────────────────────


@pytest.mark.ci
class TestUtilityRenderables:
    """Test utility renderables."""

    def test_user_header(self):
        text = _renderer().render_user_header("What is revenue?")
        assert "Datus>" in text.plain
        assert "What is revenue?" in text.plain

    def test_separator(self):
        text = _renderer().render_separator()
        assert "\u2500" in text.plain


# ── print_renderables ─────────────────────────────────────────────


@pytest.mark.ci
class TestPrintRenderables:
    """Test print_renderables convenience method."""

    def test_prints_to_console(self):
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        renderables = [Text("line1"), Text("line2")]
        ActionRenderer.print_renderables(console, renderables)
        output = buf.getvalue()
        assert "line1" in output
        assert "line2" in output


# ── _extract_subagent_response ────────────────────────────────────


@pytest.mark.ci
class TestExtractSubagentResponse:
    """Test response extraction from task tool output."""

    def test_extracts_from_nested_result(self):
        output = {"raw_output": '{"success": 1, "result": {"response": "hello"}}'}
        assert ActionRenderer._extract_subagent_response(output) == "hello"

    def test_extracts_from_flat_dict(self):
        output = {"response": "flat hello"}
        assert ActionRenderer._extract_subagent_response(output) == "flat hello"

    def test_returns_empty_for_invalid(self):
        assert ActionRenderer._extract_subagent_response({}) == ""

    def test_returns_empty_for_non_json_string(self):
        output = {"raw_output": "not json"}
        assert ActionRenderer._extract_subagent_response(output) == ""


# ── render_subagent_response ──────────────────────────────────────


@pytest.mark.ci
class TestRenderSubagentResponse:
    """Test subagent response rendering."""

    def test_renders_response_lines(self):
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": '{"success": 1, "result": {"response": "line1\\nline2"}}'},
        )
        result = _renderer().render_subagent_response(action)
        text = _plain(result)
        assert "response:" in text
        assert "line1" in text
        assert "line2" in text

    def test_no_output_returns_empty(self):
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        result = _renderer().render_subagent_response(action)
        assert result == []


# ── Verbose markup rendering ──────────────────────────────────────


@pytest.mark.ci
class TestVerboseMarkupRendering:
    """Test that verbose mode uses Rich markup in Text objects."""

    def test_main_tool_verbose_all_text_objects(self):
        """Verbose _render_main_tool returns only Text objects."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="read_query",
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}},
            start_time=now,
            end_time=now + timedelta(seconds=0.3),
        )
        result = _renderer()._render_main_tool(action, verbose=True)
        assert len(result) >= 2
        for r in result:
            assert isinstance(r, Text)

    def test_sql_args_highlighted(self):
        """SQL arguments rendered with bright_cyan markup."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="read_query",
            input_data={
                "function_name": "read_query",
                "arguments": {"sql": "SELECT SUM(revenue) FROM sales WHERE year = 2024"},
            },
            start_time=now,
            end_time=now + timedelta(seconds=0.3),
        )
        result = _renderer()._render_main_tool(action, verbose=True)
        text = _plain(result)
        assert "SELECT SUM(revenue) FROM sales" in text
        assert "sql" in text

    def test_args_key_bold(self):
        """Argument keys are rendered with bold markup."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="custom_tool",
            input_data={"function_name": "custom_tool", "arguments": {"param": "value"}},
            start_time=now,
            end_time=now + timedelta(seconds=0.1),
        )
        result = _renderer()._render_main_tool(action, verbose=True)
        text = _plain(result)
        assert "param" in text
        assert "value" in text

    def test_subagent_verbose_text_objects(self):
        """Verbose render_subagent_action returns Text objects with markup."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            messages="describe_table",
            input_data={"function_name": "describe_table", "arguments": {"table_name": "sales"}},
            output_data={"raw_output": '{"success": true, "result": {"columns": [{"name": "id", "type": "INT"}]}}'},
            start_time=now,
            end_time=now + timedelta(seconds=0.2),
        )
        result = _renderer().render_subagent_action(action, verbose=True)
        for r in result:
            assert isinstance(r, Text)
        text = _plain(result)
        assert "columns" in text
        assert "id" in text

    def test_edit_file_verbose_diff_colors(self):
        """edit_file verbose shows diff with old/new lines."""
        import json

        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="edit_file",
            input_data={
                "function_name": "edit_file",
                "arguments": {
                    "path": "src/main.py",
                    "edits": [{"oldText": "old_line", "newText": "new_line"}],
                },
            },
            output_data={"raw_output": json.dumps({"success": True, "result": "Applied 1 edit"})},
            start_time=now,
            end_time=now + timedelta(seconds=0.2),
        )
        result = _renderer()._render_main_tool(action, verbose=True)
        text = _plain(result)
        assert "- old_line" in text
        assert "+ new_line" in text

    def test_error_output_in_verbose(self):
        """Error output shows error text in verbose mode."""
        import json

        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.FAILED,
            messages="read_query",
            input_data={"function_name": "read_query", "arguments": {"sql": "INVALID SQL"}},
            output_data={"raw_output": json.dumps({"success": False, "error": "Syntax error near INVALID"})},
            start_time=now,
            end_time=now + timedelta(seconds=0.1),
        )
        result = _renderer()._render_main_tool(action, verbose=True)
        text = _plain(result)
        assert "Syntax error near INVALID" in text
        assert "error" in text

    def test_output_with_markup_keys_bold(self):
        """Output keys rendered bold in verbose mode, metadata (success/error) skipped."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="list_tables",
            input_data={"function_name": "list_tables"},
            output_data={"raw_output": '{"success": true, "text": "ok"}'},
            start_time=now,
            end_time=now + timedelta(seconds=0.5),
        )
        result = _renderer()._render_main_tool(action, verbose=True)
        assert len(result) >= 2
        text = _plain(result)
        # success metadata should be stripped in verbose output
        assert "success" not in text
        assert "ok" in text

    def test_inline_expanded_strips_markup(self):
        """format_inline_expanded strips markup tags for plain text output."""
        now = datetime.now()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="custom_tool",
            input_data={"function_name": "custom_tool", "arguments": {"key": "val"}},
            output_data={"raw_output": '{"result": "ok"}'},
            start_time=now,
            end_time=now + timedelta(seconds=0.1),
        )
        renderer = _renderer()
        lines = renderer.cg.format_inline_expanded(action)
        for line in lines:
            assert "[bold]" not in line
            assert "[/bold]" not in line


# ── render_interaction (batch) ──────────────────────────────────


@pytest.mark.ci
class TestRenderBatchInteractionRequest:
    """Test batch interaction request rendering (contents/choices format)."""

    def test_single_question_shows_legacy_header(self):
        """Single question renders with legacy Interaction Request header."""
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            action_type="request_choice",
            input_data={
                "contents": ["Which DB?"],
                "choices": [{"1": "MySQL", "2": "PG"}],
                "content_type": "markdown",
            },
        )
        result = _renderer().render_interaction_request(action, verbose=False)
        text = _plain(result)
        assert "Interaction Request" in text
        assert "Which DB?" in text

    def test_multi_question_shows_brief_header(self):
        """Multiple questions render brief header only (no detailed listing)."""
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            action_type="request_batch",
            input_data={
                "contents": ["Which DB?", "Time range?"],
                "choices": [{"1": "MySQL", "2": "PG"}, {}],
            },
        )
        result = _renderer().render_interaction_request(action, verbose=False)
        text = _plain(result)
        assert "Agent Questions (2 questions)" in text
        # Individual questions are NOT listed in the overview
        assert "MySQL / PG" not in text

    def test_empty_contents_fallback(self):
        """Empty contents list renders header only."""
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            action_type="request_choice",
            input_data={"contents": []},
        )
        result = _renderer().render_interaction_request(action, verbose=False)
        text = _plain(result)
        assert "Interaction Request" in text

    def test_single_question_markdown_content(self):
        """Single question with markdown content_type renders as Markdown."""
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            action_type="request_choice",
            input_data={"contents": ["Legacy question"], "choices": [{}], "content_type": "markdown"},
        )
        result = _renderer().render_interaction_request(action, verbose=False)
        text = _plain(result)
        assert "Interaction Request" in text
        assert "Legacy question" in text


@pytest.mark.ci
class TestRenderBatchInteractionSuccess:
    """Test batch interaction success rendering (contents/choices format)."""

    def test_success_with_contents_and_answers(self):
        """SUCCESS renders answers matched to contents."""
        import json

        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.SUCCESS,
            action_type="request_batch",
            input_data={"contents": ["Which DB?", "Time range?"]},
            output_data={"user_choice": json.dumps(["MySQL", "Last 7 days"])},
        )
        result = _renderer().render_interaction_success(action, verbose=False)
        text = _plain(result)
        assert "Answers submitted (2/2)" in text
        assert "Which DB?" in text
        assert "MySQL" in text
        assert "Time range?" in text
        assert "Last 7 days" in text

    def test_success_non_json_user_choice(self):
        """SUCCESS with non-JSON user_choice wraps as single answer."""
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.SUCCESS,
            action_type="request_batch",
            input_data={"contents": ["Q1?", "Q2?"]},
            output_data={"user_choice": "plain text"},
        )
        result = _renderer().render_interaction_success(action, verbose=False)
        text = _plain(result)
        assert "Answers submitted (1/2)" in text
        assert "plain text" in text

    def test_success_multi_question_non_list_json_falls_back(self):
        """SUCCESS with multiple questions and non-list JSON falls back to raw string."""
        import json

        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.SUCCESS,
            action_type="request_batch",
            input_data={"contents": ["Q1?", "Q2?"]},
            output_data={"user_choice": json.dumps("just a string")},
        )
        result = _renderer().render_interaction_success(action, verbose=False)
        text = _plain(result)
        assert "Answers submitted (1/2)" in text

    def test_success_truncates_long_question(self):
        """Long question text is truncated to 40 chars."""
        import json

        long_q = "A" * 60
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.SUCCESS,
            action_type="request_batch",
            input_data={"contents": [long_q, "Q2?"]},
            output_data={"user_choice": json.dumps(["yes", "no"])},
        )
        result = _renderer().render_interaction_success(action, verbose=False)
        text = _plain(result)
        assert "..." in text

    def test_single_question_success_not_routed_to_batch(self):
        """Single-question SUCCESS renders as single choice (Selected: y)."""
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.SUCCESS,
            action_type="request_choice",
            input_data={"contents": ["Sync?"]},
            output_data={"user_choice": "y", "content": "Saved", "content_type": "markdown"},
        )
        result = _renderer().render_interaction_success(action, verbose=False)
        text = _plain(result)
        assert "Selected: y" in text


class TestGetAssistantContentPlaceholderFiltering:
    """Verify _get_assistant_content strips LiteLLM sanitizer placeholder."""

    def test_placeholder_in_raw_output_returns_empty(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            output_data={"raw_output": LITELLM_EMPTY_PLACEHOLDER},
        )
        assert _get_assistant_content(action) == ""

    def test_placeholder_in_messages_returns_empty(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages=LITELLM_EMPTY_PLACEHOLDER,
        )
        assert _get_assistant_content(action) == ""

    def test_normal_raw_output_unchanged(self):
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            output_data={"raw_output": "SELECT * FROM users"},
        )
        assert _get_assistant_content(action) == "SELECT * FROM users"
