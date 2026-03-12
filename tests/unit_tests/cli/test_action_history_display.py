# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/action_history_display.py.

Tests cover:
- Sub-agent group header printed on first depth>0 action
- Sub-agent display updates tool_count for TOOL actions
- Sub-agent group ends with Done summary when depth returns to 0
- Flush correctly ends an active sub-agent group
- Normal depth=0 flow unchanged
- Multiple sub-agent groups handled sequentially
- Task SUCCESS action skipped after Done line
- ActionContentGenerator: format_streaming_action, format_inline_completed,
  format_inline_expanded, _format_tool_output_verbose, generate_streaming_content,
  _create_result_table, format_data, get_data_summary, _get_tool_args_preview,
  _get_tool_output_preview, format_inline_processing, get_role_color
- ActionHistoryDisplay: format_action_summary, format_action_detail,
  _extract_subagent_response, _render_subagent_response, _render_task_tool_as_subagent,
  display_action_list, display_final_action_history, _get_data_summary_with_full_sql,
  render_action_history (skip patterns)
- InlineStreamingContext: context manager, _flush_remaining_actions, _print_completed_action,
  stop_display, restart_display, toggle_verbose
- create_action_display factory
"""

from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.markdown import Markdown

from datus.cli.action_history_display import (
    ActionContentGenerator,
    ActionHistoryDisplay,
    BaseActionContentGenerator,
    InlineStreamingContext,
    _get_assistant_content,
    _truncate_middle,
    create_action_display,
)
from datus.schemas.action_history import (
    SUBAGENT_COMPLETE_ACTION_TYPE,
    ActionHistory,
    ActionRole,
    ActionStatus,
)


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
    action_id: str = None,
    parent_action_id: str = None,
) -> ActionHistory:
    """Helper to create ActionHistory instances for testing."""
    import uuid

    return ActionHistory(
        action_id=action_id or str(uuid.uuid4()),
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


@pytest.mark.ci
class TestSubAgentGroupStart:
    """depth>0 action triggers group header print."""

    def test_subagent_group_start(self):
        """First depth>0 action prints SubAgent header and sets _subagent_groups."""
        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)

        # Pre-set internal state to simulate mid-processing
        ctx._processed_index = 0
        ctx._tick = 0

        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages="User: What is the total revenue?",
        )
        actions.append(first_action)

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            ctx._process_actions()

        # Header should contain subagent type
        header_text = "\n".join(printed)
        assert "gen_sql(What is the total revenue?)" in header_text

        # Group state should be set
        assert len(ctx._subagent_groups) == 1
        group = list(ctx._subagent_groups.values())[0]
        assert group["subagent_type"] == "gen_sql"
        assert group["tool_count"] == 0

    def test_subagent_prompt_truncated_middle(self):
        """Long prompt is truncated in the middle."""
        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        long_prompt = "User: " + "A" * 300
        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_sql",
            messages=long_prompt,
        )
        actions.append(first_action)

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            ctx._process_actions()

        header_text = "\n".join(printed)
        # Should contain " ... " truncation marker
        assert " ... " in header_text
        # Total displayed prompt should be <= 120 chars
        # Find the prompt line (second line after header)
        prompt_lines = [p for p in printed if " ... " in p]
        assert len(prompt_lines) == 1


@pytest.mark.ci
class TestTruncateMiddle:
    """_truncate_middle static method tests."""

    def test_short_text_unchanged(self):
        """Text shorter than max_len is returned unchanged."""
        result = InlineStreamingContext._truncate_middle("hello world", max_len=120)
        assert result == "hello world"

    def test_long_text_truncated(self):
        """Text longer than max_len is truncated in the middle."""
        text = "A" * 200
        result = InlineStreamingContext._truncate_middle(text, max_len=120)
        assert len(result) <= 120
        assert " ... " in result
        assert result.startswith("A")
        assert result.endswith("A")

    def test_exact_boundary(self):
        """Text exactly at max_len is not truncated."""
        text = "B" * 120
        result = InlineStreamingContext._truncate_middle(text, max_len=120)
        assert result == text
        assert " ... " not in result


@pytest.mark.ci
class TestSubAgentDisplayUpdates:
    """depth>0 TOOL actions increment tool_count and show args."""

    def test_tool_count_increments(self):
        """Each depth>0 TOOL action increments the group tool_count."""
        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        # First action starts the group
        actions.append(_make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql"))
        # Two TOOL actions with messages containing args (same format as main agent)
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="describe_table",
                messages="Tool call: describe_table('users')",
                input_data={"function_name": "describe_table"},
            )
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="read_query",
                messages="Tool call: read_query('SELECT * FROM users')",
                input_data={"function_name": "read_query"},
            )
        )

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        assert len(ctx._subagent_groups) == 1
        group = list(ctx._subagent_groups.values())[0]
        assert group["tool_count"] == 2

        # Verify that printed output contains tool messages with args
        all_output = "\n".join(printed)
        assert "read_query" in all_output
        assert "SELECT * FROM users" in all_output

    def test_non_tool_action_no_increment(self):
        """ASSISTANT depth>0 action does not increment tool_count."""
        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        actions.append(_make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql"))
        actions.append(_make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS, depth=1, action_type="gen_sql"))

        with patch.object(display.console, "print"):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        group = list(ctx._subagent_groups.values())[0]
        assert group["tool_count"] == 0


@pytest.mark.ci
class TestSubAgentGroupEnd:
    """depth returns to 0 → Done summary printed."""

    def test_done_summary_printed(self):
        """When depth=0 action follows depth>0 group, Done line is printed."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=5.2)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        # Sub-agent group
        actions.append(
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0)
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="describe_table",
                input_data={"function_name": "describe_table"},
                start_time=t0,
            )
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="read_query",
                input_data={"function_name": "read_query"},
                start_time=t0,
            )
        )
        # depth=0 task result ends the group
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                messages="task result",
                end_time=t1,
            )
        )

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        # Group should be cleared
        assert len(ctx._subagent_groups) == 0

        # Done summary should contain tool count and duration
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1
        assert "2 tool uses" in done_lines[0]
        assert "5.2s" in done_lines[0]


@pytest.mark.ci
class TestSubAgentFlushOnExit:
    """Flush correctly ends an active sub-agent group."""

    def test_flush_ends_active_group(self):
        """_flush_remaining_actions ends sub-agent group if active."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        # Start a group via _process_actions
        actions.append(
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0)
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="describe_table",
                input_data={"function_name": "describe_table"},
                start_time=t0,
            )
        )

        with patch.object(display.console, "print"):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        assert len(ctx._subagent_groups) == 1

        # Now flush remaining (simulating __exit__)
        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            ctx._flush_remaining_actions()

        assert len(ctx._subagent_groups) == 0
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1
        assert "1 tool uses" in done_lines[0]


@pytest.mark.ci
class TestNoSubAgentNormalFlow:
    """depth=0 actions maintain existing behavior."""

    def test_normal_completed_action(self):
        """depth=0 completed actions are printed normally."""
        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="search_table",
                messages="search_table(...)",
                input_data={"function_name": "search_table"},
            )
        )

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            ctx._process_actions()

        assert len(ctx._subagent_groups) == 0
        assert len(printed) > 0

    def test_processing_tool_pauses(self):
        """depth=0 PROCESSING TOOL pauses _process_actions (returns without advancing)."""
        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                depth=0,
                action_type="search_table",
                messages="search_table(...)",
                input_data={"function_name": "search_table"},
            )
        )

        with patch.object(display.console, "print"):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        # Index should NOT advance past PROCESSING
        assert ctx._processed_index == 0


@pytest.mark.ci
class TestMultipleSubAgentGroups:
    """Multiple sequential sub-agent groups are handled correctly."""

    def test_two_groups(self):
        """Two sub-agent groups produce two headers and two Done lines."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        t2 = t1 + timedelta(seconds=2)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        # Group 1
        actions.append(
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0)
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
            )
        )
        # End group 1
        actions.append(_make_action(ActionRole.TOOL, ActionStatus.SUCCESS, depth=0, action_type="task", end_time=t1))
        # Group 2
        actions.append(
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="fix_sql", start_time=t1)
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "read_query"},
                start_time=t1,
            )
        )
        # End group 2
        actions.append(_make_action(ActionRole.TOOL, ActionStatus.SUCCESS, depth=0, action_type="task", end_time=t2))

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        headers = [line for line in printed if "\u23fa gen_sql" in line or "\u23fa fix_sql" in line]
        dones = [line for line in printed if "Done" in line]

        assert len(headers) == 2
        assert "gen_sql" in headers[0]
        assert "fix_sql" in headers[1]
        assert len(dones) == 2


@pytest.mark.ci
class TestTaskSuccessSkippedAfterDone:
    """The depth=0 task SUCCESS following a sub-agent group is not printed as a normal action."""

    def test_task_success_not_double_printed(self):
        """The task SUCCESS action that ends a sub-agent group should not produce a normal action line."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        actions.append(
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0)
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                messages="task result",
                end_time=t1,
            )
        )

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        # Should have header + Done, but NOT a normal "task result" action line
        normal_lines = [
            line for line in printed if "task result" in line and "Done" not in line and "gen_sql" not in line
        ]
        assert len(normal_lines) == 0

        # Done line should exist
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1


@pytest.mark.ci
class TestModuleLevelTruncateMiddle:
    """Module-level _truncate_middle function tests."""

    def test_short_text_unchanged(self):
        assert _truncate_middle("hello", max_len=120) == "hello"

    def test_long_text_truncated(self):
        text = "X" * 200
        result = _truncate_middle(text, max_len=120)
        assert len(result) <= 120
        assert " ... " in result

    def test_delegates_same_as_staticmethod(self):
        text = "Y" * 300
        assert InlineStreamingContext._truncate_middle(text, 50) == _truncate_middle(text, 50)


@pytest.mark.ci
class TestRenderActionHistory:
    """Tests for ActionHistoryDisplay.render_action_history() — the unified renderer."""

    @staticmethod
    def _stringify_arg(arg):
        """Convert a print argument to string, extracting markup from Markdown objects."""
        if isinstance(arg, Markdown):
            return arg.markup
        return str(arg)

    def _collect_prints(self, display, actions, verbose=False):
        """Helper: call render_action_history and capture all console.print calls."""
        printed = []
        with patch.object(
            display.console, "print", side_effect=lambda *a, **kw: printed.append(self._stringify_arg(a[0]))
        ):
            display.render_action_history(actions, verbose=verbose)
        return printed

    # -- empty / skip tests --

    def test_empty_actions(self):
        """Empty action list prints 'No actions to display'."""
        display = ActionHistoryDisplay()
        printed = self._collect_prints(display, [])
        assert len(printed) == 1
        assert "No actions to display" in printed[0]

    def test_skip_interaction(self):
        """INTERACTION actions are skipped entirely."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.PROCESSING, messages="Choose an option"),
        ]
        printed = self._collect_prints(display, actions)
        # All actions skipped — nothing printed
        assert len(printed) == 0

    def test_skip_processing_tool(self):
        """TOOL actions with PROCESSING status are skipped."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                messages="describe_table",
                input_data={"function_name": "describe_table"},
            ),
        ]
        printed = self._collect_prints(display, actions)
        # All actions skipped — nothing printed
        assert len(printed) == 0

    # -- user prompt rendering --

    def test_user_action_rendered(self):
        """USER action at depth=0 renders user prompt with Datus> prefix."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                messages="User: how many tables are there?",
                action_type="chat_interaction",
            ),
        ]
        printed = self._collect_prints(display, actions)
        assert len(printed) == 1
        assert "Datus>" in printed[0]
        assert "how many tables are there?" in printed[0]

    def test_user_action_rendered_without_prefix(self):
        """USER action without 'User: ' prefix still renders correctly."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                messages="some direct message",
                action_type="chat_interaction",
            ),
        ]
        printed = self._collect_prints(display, actions)
        assert len(printed) == 1
        assert "Datus>" in printed[0]
        assert "some direct message" in printed[0]

    # -- main agent rendering --

    def test_main_action_compact(self):
        """depth=0 SUCCESS TOOL rendered via format_inline_completed in compact mode."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="describe_table(users)",
                input_data={"function_name": "describe_table"},
            ),
        ]
        printed = self._collect_prints(display, actions, verbose=False)
        assert len(printed) >= 1
        combined = "\n".join(printed)
        assert "describe_table" in combined

    def test_main_action_verbose(self):
        """depth=0 SUCCESS TOOL rendered via format_inline_expanded in verbose mode."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="describe_table(users)",
                input_data={"function_name": "describe_table", "arguments": {"table": "users"}},
            ),
        ]
        printed = self._collect_prints(display, actions, verbose=True)
        combined = "\n".join(printed)
        assert "describe_table" in combined
        # Verbose shows arguments
        assert "table" in combined
        assert "users" in combined

    def test_standalone_task_tool_rendered_as_subagent(self):
        """depth=0 TOOL with function_name='task' (resume case) renders as subagent summary."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3.5)
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="task result",
                input_data={"function_name": "task", "type": "gen_sql", "prompt": "What is total revenue?"},
                start_time=t0,
                end_time=t1,
            ),
        ]
        actions[0].output = {"success": 1, "sql": "SELECT SUM(revenue) FROM orders"}

        printed = self._collect_prints(display, actions, verbose=False)
        combined = "\n".join(printed)
        # Should render as subagent header + result
        assert "gen_sql(What is total revenue?)" in combined
        assert "✓" in combined

    def test_standalone_task_tool_verbose(self):
        """Standalone task tool in verbose mode shows response from raw_output."""
        import json

        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                input_data={"function_name": "task", "type": "gen_sql", "prompt": "Get revenue"},
            ),
        ]
        # Actual action output has raw_output containing FuncToolResult serialization
        actions[0].output = {
            "success": True,
            "raw_output": json.dumps(
                {"success": 1, "error": None, "result": {"response": "SELECT SUM(revenue) FROM orders"}}
            ),
            "summary": "✓ Success",
        }

        printed = self._collect_prints(display, actions, verbose=True)
        combined = "\n".join(printed)
        assert "\u23fa gen_sql" in combined
        assert "SELECT SUM(revenue)" in combined

    def test_task_tool_after_subagent_group_still_skipped(self):
        """Task tool following a depth>0 subagent group is still skipped (Done covers it)."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)
        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task", "type": "gen_sql", "prompt": "test"},
                end_time=t1,
            ),
        ]
        printed = self._collect_prints(display, actions)
        # Should NOT have two subagent headers — only one from the depth>0 group
        headers = [line for line in printed if "\u23fa gen_sql" in line]
        assert len(headers) == 1
        # Done line should exist
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1

    # -- subagent grouping --

    def test_subagent_group_header_and_actions(self):
        """Subagent group renders header + action lines."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: What is total revenue?",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="describe_table(orders)",
                input_data={"function_name": "describe_table"},
            ),
            # End group with a depth=0 task action
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=datetime(2025, 1, 1, 12, 0, 5),
            ),
        ]
        # Set start_time on first action
        actions[0].start_time = datetime(2025, 1, 1, 12, 0, 0)

        printed = self._collect_prints(display, actions, verbose=False)
        combined = "\n".join(printed)

        # Header
        assert "gen_sql(What is total revenue?)" in combined
        # Tool line
        assert "describe_table" in combined
        assert "✓" in combined
        # Done summary
        assert "Done" in combined
        assert "1 tool uses" in combined

    def test_subagent_verbose_shows_args_and_output(self):
        """In verbose mode, subagent tool actions show full arguments and output."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: query",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="read_query",
                input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}},
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=datetime(2025, 1, 1, 12, 0, 3),
            ),
        ]
        actions[0].start_time = datetime(2025, 1, 1, 12, 0, 0)

        printed = self._collect_prints(display, actions, verbose=True)
        combined = "\n".join(printed)

        # Arguments visible in verbose
        assert "sql" in combined
        assert "SELECT 1" in combined

    def test_subagent_done_with_duration(self):
        """Done summary line includes duration."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=7.3)

        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=t1,
            ),
        ]

        printed = self._collect_prints(display, actions)
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1
        assert "7.3s" in done_lines[0]
        assert "1 tool uses" in done_lines[0]

    # -- multiple groups --

    def test_multiple_subagent_groups(self):
        """Two sequential subagent groups produce two headers and two Done lines."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        t2 = t1 + timedelta(seconds=4)

        display = ActionHistoryDisplay()
        actions = [
            # Group 1
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=t1,
            ),
            # Group 2
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="fix_sql", start_time=t1),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "read_query"},
                start_time=t1,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=t2,
            ),
        ]

        printed = self._collect_prints(display, actions)
        headers = [line for line in printed if "\u23fa gen_sql" in line or "\u23fa fix_sql" in line]
        dones = [line for line in printed if "Done" in line]

        assert len(headers) == 2
        assert "gen_sql" in headers[0]
        assert "fix_sql" in headers[1]
        assert len(dones) == 2

    # -- unclosed group --

    def test_unclosed_subagent_group(self):
        """If actions end mid-subagent, a partial Done line is still printed."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)

        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
            ),
            # No depth=0 action follows — group stays open
        ]

        printed = self._collect_prints(display, actions)
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1
        assert "1 tool uses" in done_lines[0]

    def test_unclosed_subagent_no_done_when_partial_disabled(self):
        """With show_partial_done=False, unclosed group does NOT print Done."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)

        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql", start_time=t0),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
            ),
        ]

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            display.render_action_history(actions, verbose=False, show_partial_done=False)
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 0

    # -- compact truncation vs verbose no-truncation --

    def test_compact_truncates_subagent_prompt(self):
        """In compact mode, long subagent prompts are truncated."""
        display = ActionHistoryDisplay()
        long_prompt = "User: " + "Z" * 300
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages=long_prompt,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=datetime(2025, 1, 1, 12, 0, 1),
            ),
        ]
        actions[0].start_time = datetime(2025, 1, 1, 12, 0, 0)

        printed = self._collect_prints(display, actions, verbose=False)
        combined = "\n".join(printed)
        assert " ... " in combined

    def test_verbose_does_not_truncate_subagent_prompt(self):
        """In verbose mode, long subagent prompts are NOT truncated."""
        display = ActionHistoryDisplay()
        long_prompt = "User: " + "Z" * 300
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages=long_prompt,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=datetime(2025, 1, 1, 12, 0, 1),
            ),
        ]
        actions[0].start_time = datetime(2025, 1, 1, 12, 0, 0)

        printed = self._collect_prints(display, actions, verbose=True)
        combined = "\n".join(printed)
        # Full 300-char string should be present, no truncation marker
        assert "Z" * 300 in combined
        assert " ... " not in combined

    # -- assistant action in subagent --

    def test_subagent_assistant_action(self):
        """ASSISTANT actions in subagent group render with ⏺ prefix and Markdown from raw_output."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: test",
            ),
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                depth=1,
                messages="short fallback",
                output_data={"raw_output": "Thinking about the query..."},
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=datetime(2025, 1, 1, 12, 0, 1),
            ),
        ]
        actions[0].start_time = datetime(2025, 1, 1, 12, 0, 0)

        printed = self._collect_prints(display, actions)
        combined = "\n".join(printed)
        # Should show ⏺ prefix with subagent indentation and raw_output content
        assert "⏺" in combined
        assert "💬" in combined
        assert "Thinking about the query" in combined


# ── SubAgent complete action display ──────────────────────────────


@pytest.mark.ci
class TestSubAgentCompleteAction:
    """Tests for subagent_complete action closing groups."""

    def test_complete_action_closes_group_in_streaming(self):
        """A subagent_complete action closes the corresponding group in InlineStreamingContext."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=4.5)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        call_id = "parent_call_1"

        # Sub-agent group with parent_action_id
        actions.append(
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: What is total revenue?",
                start_time=t0,
                parent_action_id=call_id,
            )
        )
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="describe_table",
                input_data={"function_name": "describe_table"},
                start_time=t0,
                parent_action_id=call_id,
            )
        )
        # subagent_complete action closes the group
        actions.append(
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id,
            )
        )

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        # Group should be cleared
        assert len(ctx._subagent_groups) == 0
        assert call_id in ctx._completed_group_ids

        # Done summary should contain tool count and duration
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1
        assert "1 tool uses" in done_lines[0]
        assert "4.5s" in done_lines[0]

    def test_complete_action_closes_group_in_batch(self):
        """A subagent_complete action closes the corresponding group in render_action_history."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3.0)

        display = ActionHistoryDisplay()
        call_id = "parent_call_batch"

        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: query",
                start_time=t0,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id,
            ),
        ]

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            display.render_action_history(actions, verbose=False)

        combined = "\n".join(printed)
        assert "gen_sql(query)" in combined
        assert "Done" in combined
        assert "1 tool uses" in combined


# ── Parallel sub-agent groups ─────────────────────────────────────


@pytest.mark.ci
class TestParallelSubAgentGroups:
    """Tests for multiple parallel sub-agent groups with different parent_action_ids."""

    def test_two_interleaved_groups_streaming(self):
        """Two interleaved sub-agent groups (different parent_action_ids) each produce correct output."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        t2 = t0 + timedelta(seconds=5)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        call_id_a = "call_a"
        call_id_b = "call_b"

        # Group A starts
        actions.append(
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: Revenue query",
                start_time=t0,
                parent_action_id=call_id_a,
            )
        )
        # Group B starts (interleaved)
        actions.append(
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="fix_sql",
                messages="User: Fix query",
                start_time=t0,
                parent_action_id=call_id_b,
            )
        )
        # Group A tool
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="describe_table",
                input_data={"function_name": "describe_table"},
                start_time=t0,
                parent_action_id=call_id_a,
            )
        )
        # Group B tool
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="read_query",
                input_data={"function_name": "read_query"},
                start_time=t0,
                parent_action_id=call_id_b,
            )
        )
        # Group A completes
        actions.append(
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id_a,
            )
        )
        # Group B completes
        actions.append(
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t2,
                parent_action_id=call_id_b,
            )
        )

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            with patch("datus.cli.action_history_display.Live"):
                ctx._process_actions()

        # Both groups should be closed
        assert len(ctx._subagent_groups) == 0
        assert call_id_a in ctx._completed_group_ids
        assert call_id_b in ctx._completed_group_ids

        # Two headers and two Done lines
        headers = [line for line in printed if "\u23fa gen_sql" in line or "\u23fa fix_sql" in line]
        dones = [line for line in printed if "Done" in line]
        assert len(headers) == 2
        assert len(dones) == 2

        # Each Done should show 1 tool use
        for done in dones:
            assert "1 tool uses" in done

    def test_two_interleaved_groups_batch(self):
        """Two interleaved sub-agent groups render correctly in batch mode."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)
        t2 = t0 + timedelta(seconds=4)

        display = ActionHistoryDisplay()
        call_id_a = "call_batch_a"
        call_id_b = "call_batch_b"

        actions = [
            # Group A
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                start_time=t0,
                parent_action_id=call_id_a,
            ),
            # Group B (interleaved)
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="fix_sql",
                start_time=t0,
                parent_action_id=call_id_b,
            ),
            # Group A tool
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
                parent_action_id=call_id_a,
            ),
            # Group B tool
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "read_query"},
                start_time=t0,
                parent_action_id=call_id_b,
            ),
            # Group A complete
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id_a,
            ),
            # Group B complete
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t2,
                parent_action_id=call_id_b,
            ),
        ]

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            display.render_action_history(actions, verbose=False)

        headers = [line for line in printed if "\u23fa gen_sql" in line or "\u23fa fix_sql" in line]
        dones = [line for line in printed if "Done" in line]

        assert len(headers) == 2
        assert "gen_sql" in headers[0]
        assert "fix_sql" in headers[1]
        assert len(dones) == 2

    def test_task_tool_skipped_after_complete_with_assistant_in_between(self):
        """TOOL(task) is skipped even when ASSISTANT action appears between complete and task."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=5)

        display = ActionHistoryDisplay()
        call_id = "call_between"

        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: query",
                start_time=t0,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
                parent_action_id=call_id,
            ),
            # subagent_complete
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id,
            ),
            # ASSISTANT action between complete and TOOL(task)
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                depth=0,
                messages="thinking",
                output_data={"raw_output": "I'll analyze this."},
            ),
            # depth=0 TOOL(task) — should be skipped
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=t1,
            ),
        ]

        printed = []
        with patch.object(
            display.console,
            "print",
            side_effect=lambda *a, **kw: printed.append(str(a[0])),
        ):
            display.render_action_history(actions, verbose=False)

        # Should NOT have a standalone subagent header line
        standalone = [line for line in printed if "\u23fa subagent" in line]
        assert len(standalone) == 0

        # Should have one group header and one Done
        headers = [line for line in printed if "\u23fa gen_sql" in line]
        dones = [line for line in printed if "Done" in line]
        assert len(headers) == 1
        assert len(dones) == 1


# ── Description display ───────────────────────────────────────────


@pytest.mark.ci
class TestDescriptionDisplay:
    """Tests for description display in compact vs verbose modes."""

    @staticmethod
    def _stringify_arg(arg):
        if isinstance(arg, Markdown):
            return arg.markup
        return str(arg)

    def _collect_prints(self, display, actions, verbose=False):
        printed = []
        with patch.object(
            display.console, "print", side_effect=lambda *a, **kw: printed.append(self._stringify_arg(a[0]))
        ):
            display.render_action_history(actions, verbose=verbose)
        return printed

    # -- batch/redraw path: _render_subagent_header --

    def test_compact_with_description_shows_goal_label(self):
        """In compact mode, when _task_description is present, show 'goal:' label."""
        display = ActionHistoryDisplay()
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="gen_sql",
                messages="User: Generate a complex SQL query that joins orders with customers and products",
                input_data={"_task_description": "Generate monthly sales report"},
                parent_action_id="call_1",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="describe_table",
                input_data={"function_name": "describe_table"},
                parent_action_id="call_1",
                start_time=t0,
                end_time=t1,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                output_data={"subagent_type": "gen_sql", "tool_count": 1},
                parent_action_id="call_1",
                start_time=t0,
                end_time=t1,
            ),
        ]
        printed = self._collect_prints(display, actions, verbose=False)
        combined = "\n".join(printed)
        assert "gen_sql(Generate monthly sales report)" in combined
        # Should NOT show the full prompt in compact mode
        assert "prompt:" not in combined

    def test_verbose_with_description_shows_full_prompt(self):
        """In verbose mode, even when description is present, show full prompt."""
        display = ActionHistoryDisplay()
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)
        long_prompt = "Generate a complex SQL query that joins orders with customers and products"
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="gen_sql",
                messages=f"User: {long_prompt}",
                input_data={"_task_description": "Generate monthly sales report"},
                parent_action_id="call_1",
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                output_data={"subagent_type": "gen_sql", "tool_count": 0},
                parent_action_id="call_1",
                start_time=t0,
                end_time=t1,
            ),
        ]
        printed = self._collect_prints(display, actions, verbose=True)
        combined = "\n".join(printed)
        assert "prompt:" in combined
        assert long_prompt in combined
        # Should NOT show goal label in verbose mode
        assert "goal:" not in combined

    def test_compact_without_description_falls_back_to_truncated_prompt(self):
        """In compact mode without description, fall back to truncated prompt."""
        display = ActionHistoryDisplay()
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="gen_sql",
                messages="User: What is the total revenue?",
                parent_action_id="call_1",
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                output_data={"subagent_type": "gen_sql", "tool_count": 0},
                parent_action_id="call_1",
                start_time=t0,
                end_time=t1,
            ),
        ]
        printed = self._collect_prints(display, actions, verbose=False)
        combined = "\n".join(printed)
        assert "gen_sql(What is the total revenue?)" in combined
        assert "goal:" not in combined
        assert "prompt:" not in combined

    # -- standalone task tool path: _render_task_tool_as_subagent --

    def test_standalone_task_compact_with_description(self):
        """Standalone task tool in compact mode shows description with 'goal:' label."""
        display = ActionHistoryDisplay()
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="task result",
                input_data={
                    "function_name": "task",
                    "type": "gen_sql",
                    "prompt": "Generate a very long and detailed SQL query",
                    "description": "Generate sales report",
                },
                start_time=t0,
                end_time=t1,
            ),
        ]
        printed = self._collect_prints(display, actions, verbose=False)
        combined = "\n".join(printed)
        assert "gen_sql(Generate sales report)" in combined
        assert "prompt:" not in combined

    def test_standalone_task_verbose_with_description(self):
        """Standalone task tool in verbose mode shows full prompt even with description."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                input_data={
                    "function_name": "task",
                    "type": "gen_sql",
                    "prompt": "Generate a very long and detailed SQL query",
                    "description": "Generate sales report",
                },
            ),
        ]
        actions[0].output = {"success": 1, "sql": "SELECT 1"}
        printed = self._collect_prints(display, actions, verbose=True)
        combined = "\n".join(printed)
        assert "prompt:" in combined
        assert "Generate a very long and detailed SQL query" in combined
        assert "goal:" not in combined

    def test_standalone_task_compact_without_description(self):
        """Standalone task tool in compact mode without description falls back to prompt."""
        display = ActionHistoryDisplay()
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="task result",
                input_data={
                    "function_name": "task",
                    "type": "gen_sql",
                    "prompt": "What is total revenue?",
                },
                start_time=t0,
                end_time=t1,
            ),
        ]
        printed = self._collect_prints(display, actions, verbose=False)
        combined = "\n".join(printed)
        assert "gen_sql(What is total revenue?)" in combined
        assert "goal:" not in combined
        assert "prompt:" not in combined


@pytest.mark.ci
class TestRenderMultiTurnHistory:
    """Tests for render_multi_turn_history."""

    def test_empty_turns(self):
        """Empty list renders nothing without error."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        display.render_multi_turn_history([], verbose=False)
        output = buf.getvalue()
        assert output == "" or output.strip() == ""

    def test_single_turn(self):
        """Single turn renders user message header and actions."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="tool result",
                input_data={"function_name": "read_query"},
            ),
        ]
        display.render_multi_turn_history([("Hello world", actions)], verbose=False)
        output = buf.getvalue()
        assert "Datus>" in output
        assert "Hello world" in output

    def test_multi_turns(self):
        """Multiple turns each render their own user message header."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions1 = [
            _make_action(
                ActionRole.TOOL, ActionStatus.SUCCESS, messages="result1", input_data={"function_name": "read_query"}
            )
        ]
        actions2 = [
            _make_action(
                ActionRole.TOOL, ActionStatus.SUCCESS, messages="result2", input_data={"function_name": "list_tables"}
            )
        ]
        turns = [("Question 1", actions1), ("Question 2", actions2)]

        display.render_multi_turn_history(turns, verbose=False)
        output = buf.getvalue()
        assert "Question 1" in output
        assert "Question 2" in output
        # Should have separator lines
        assert "\u2500" * 40 in output

    def test_long_user_message_not_truncated(self):
        """Long user message is shown in full (not middle-truncated) in the header."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        long_msg = "A" * 200
        actions = [_make_action(ActionRole.TOOL, ActionStatus.SUCCESS, messages="result")]
        display.render_multi_turn_history([(long_msg, actions)], verbose=False)
        output = buf.getvalue()
        # Rich wraps long lines, so check no truncation marker
        assert " ... " not in output


@pytest.mark.ci
class TestReprintHistoryWithTurns:
    """Tests for _reprint_history with history_turns prefix."""

    def test_reprint_with_history_turns(self):
        """_reprint_history renders history turns before current actions."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        history_actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="prev result",
                input_data={"function_name": "read_query"},
            )
        ]
        current_actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="cur result",
                input_data={"function_name": "list_tables"},
            )
        ]

        ctx = InlineStreamingContext(
            current_actions,
            display,
            history_turns=[("Previous question", history_actions)],
            current_user_message="Current question",
        )
        ctx._processed_index = 1  # All current actions processed
        ctx._verbose = False
        ctx._reprint_history()

        output = buf.getvalue()
        assert "Previous question" in output
        assert "Current question" in output

    def test_reprint_without_history_turns(self):
        """_reprint_history works without history_turns (backward compat)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        current_actions = [
            _make_action(
                ActionRole.TOOL, ActionStatus.SUCCESS, messages="cur result", input_data={"function_name": "read_query"}
            )
        ]
        ctx = InlineStreamingContext(current_actions, display)
        ctx._processed_index = 1
        ctx._verbose = False
        ctx._reprint_history()

        output = buf.getvalue()
        # Should not crash, should render current actions
        assert output is not None


# ── _get_assistant_content helper ──────────────────────────────────


@pytest.mark.ci
class TestGetAssistantContent:
    """Tests for the module-level _get_assistant_content function."""

    def test_returns_raw_output_when_present(self):
        """Prefers output.raw_output over messages."""
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="fallback message",
            output_data={"raw_output": "preferred content"},
        )
        assert _get_assistant_content(action) == "preferred content"

    def test_returns_messages_when_no_raw_output(self):
        """Falls back to messages when raw_output is empty."""
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="my message",
            output_data={"raw_output": ""},
        )
        assert _get_assistant_content(action) == "my message"

    def test_returns_messages_when_output_not_dict(self):
        """Falls back to messages when output is not a dict."""
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="my message",
        )
        action.output = "string output"
        assert _get_assistant_content(action) == "my message"

    def test_returns_empty_string_when_no_messages_and_no_output(self):
        """Returns empty string when both messages and output are empty."""
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="",
        )
        assert _get_assistant_content(action) == ""


# ── BaseActionContentGenerator ─────────────────────────────────────


@pytest.mark.ci
class TestBaseActionContentGenerator:
    """Tests for BaseActionContentGenerator._get_action_dot and get_status_icon."""

    def test_get_action_dot_tool(self):
        """TOOL role returns tool emoji."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        assert gen._get_action_dot(action) == "🔧"

    def test_get_action_dot_assistant(self):
        """ASSISTANT role returns chat emoji."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS)
        assert gen._get_action_dot(action) == "💬"

    def test_get_action_dot_workflow_uses_status(self):
        """WORKFLOW role uses status-based dot."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.WORKFLOW, ActionStatus.SUCCESS)
        assert gen._get_action_dot(action) == "🟢"

    def test_get_action_dot_user_failed(self):
        """USER role with FAILED status returns red dot."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.USER, ActionStatus.FAILED)
        assert gen._get_action_dot(action) == "🔴"

    def test_get_action_dot_system_processing(self):
        """SYSTEM role with PROCESSING status returns yellow dot."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.SYSTEM, ActionStatus.PROCESSING)
        assert gen._get_action_dot(action) == "🟡"

    def test_get_status_icon_processing(self):
        """PROCESSING status returns hourglass icon."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.TOOL, ActionStatus.PROCESSING)
        assert gen.get_status_icon(action) == "⏳"

    def test_get_status_icon_success(self):
        """SUCCESS status returns check mark icon."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        assert gen.get_status_icon(action) == "✅"

    def test_get_status_icon_failed(self):
        """FAILED status returns X icon."""
        gen = BaseActionContentGenerator()
        action = _make_action(ActionRole.TOOL, ActionStatus.FAILED)
        assert gen.get_status_icon(action) == "❌"


# ── ActionContentGenerator ─────────────────────────────────────────


@pytest.mark.ci
class TestActionContentGeneratorFormatStreaming:
    """Tests for ActionContentGenerator.format_streaming_action."""

    def test_tool_processing_shows_only_messages(self):
        """PROCESSING tool shows tool emoji and messages, no status suffix."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="describe_table(school_name=...)",
        )
        result = gen.format_streaming_action(action)
        assert "🔧" in result
        assert "describe_table" in result
        assert "✓" not in result
        assert "✗" not in result

    def test_tool_success_with_duration(self):
        """SUCCESS tool shows check mark and duration."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2.5)
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="read_query(SELECT 1)",
            start_time=t0,
            end_time=t1,
        )
        result = gen.format_streaming_action(action)
        assert "✓" in result
        assert "2.5s" in result

    def test_tool_failed_shows_cross(self):
        """FAILED tool shows cross mark."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.FAILED,
            messages="read_query(bad)",
        )
        result = gen.format_streaming_action(action)
        assert "✗" in result

    def test_tool_success_with_output_preview(self):
        """SUCCESS tool with output shows preview on next line."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="list_tables()",
            input_data={"function_name": "list_tables"},
            output_data={"raw_output": '{"result": [{"name": "t1"}, {"name": "t2"}]}'},
        )
        result = gen.format_streaming_action(action)
        assert "2 tables" in result

    def test_non_tool_role_shows_dot_and_messages(self):
        """Non-TOOL roles show role-appropriate dot and messages."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS, messages="thinking about query")
        result = gen.format_streaming_action(action)
        assert "💬" in result
        assert "thinking about query" in result


@pytest.mark.ci
class TestActionContentGeneratorFormatInlineCompleted:
    """Tests for ActionContentGenerator.format_inline_completed."""

    def test_assistant_with_raw_output(self):
        """ASSISTANT action shows raw_output content."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="fallback",
            output_data={"raw_output": "Generated SQL query"},
        )
        lines = gen.format_inline_completed(action)
        assert len(lines) == 1
        assert "💬" in lines[0]
        assert "Generated SQL query" in lines[0]

    def test_tool_success_has_green_dot(self):
        """SUCCESS TOOL action gets green status dot."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="read_query(SELECT 1)",
        )
        lines = gen.format_inline_completed(action)
        assert len(lines) == 1
        assert "[green]⏺[/green]" in lines[0]

    def test_tool_failed_has_red_dot(self):
        """FAILED TOOL action gets red status dot."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.FAILED,
            messages="read_query(bad SQL)",
        )
        lines = gen.format_inline_completed(action)
        assert len(lines) == 1
        assert "[red]⏺[/red]" in lines[0]

    def test_workflow_role(self):
        """WORKFLOW action shows yellow dot and messages."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.WORKFLOW, ActionStatus.SUCCESS, messages="Executing workflow step")
        lines = gen.format_inline_completed(action)
        assert len(lines) == 1
        assert "🟡" in lines[0]
        assert "Executing workflow step" in lines[0]

    def test_system_role(self):
        """SYSTEM action shows purple dot and messages."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.SYSTEM, ActionStatus.SUCCESS, messages="System init")
        lines = gen.format_inline_completed(action)
        assert len(lines) == 1
        assert "🟣" in lines[0]
        assert "System init" in lines[0]

    def test_interaction_role_returns_empty(self):
        """INTERACTION actions return empty list (handled elsewhere)."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="user input")
        lines = gen.format_inline_completed(action)
        assert lines == []


@pytest.mark.ci
class TestActionContentGeneratorFormatInlineExpanded:
    """Tests for ActionContentGenerator.format_inline_expanded (verbose mode)."""

    def test_tool_expanded_with_dict_args(self):
        """Verbose TOOL shows function name, args dict keys, and output."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=1.5)
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="read_query",
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1", "limit": 10}},
            output_data={"raw_output": '{"result": [{"col": "val"}]}'},
            start_time=t0,
            end_time=t1,
        )
        lines = gen.format_inline_expanded(action)
        assert any("read_query" in line for line in lines)
        assert any("sql: SELECT 1" in line for line in lines)
        assert any("limit: 10" in line for line in lines)
        assert any("✓" in line for line in lines)
        assert any("1.5s" in line for line in lines)

    def test_tool_expanded_with_non_dict_args(self):
        """Verbose TOOL with non-dict arguments shows 'args:' prefix."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "search", "arguments": "plain string arg"},
        )
        lines = gen.format_inline_expanded(action)
        assert any("args: plain string arg" in line for line in lines)

    def test_tool_expanded_no_args(self):
        """Verbose TOOL with no arguments shows just function name and status."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.FAILED,
            input_data={"function_name": "broken_tool"},
        )
        lines = gen.format_inline_expanded(action)
        assert any("broken_tool" in line for line in lines)
        assert any("✗" in line for line in lines)

    def test_assistant_expanded(self):
        """Verbose ASSISTANT shows content from raw_output."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="fallback",
            output_data={"raw_output": "Full expanded assistant content"},
        )
        lines = gen.format_inline_expanded(action)
        assert any("Full expanded assistant content" in line for line in lines)

    def test_workflow_expanded(self):
        """Verbose WORKFLOW shows messages."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.WORKFLOW, ActionStatus.SUCCESS, messages="Step 1: fetch data")
        lines = gen.format_inline_expanded(action)
        assert any("Step 1: fetch data" in line for line in lines)

    def test_system_expanded(self):
        """Verbose SYSTEM shows messages."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.SYSTEM, ActionStatus.SUCCESS, messages="System ready")
        lines = gen.format_inline_expanded(action)
        assert any("System ready" in line for line in lines)


@pytest.mark.ci
class TestFormatToolOutputVerbose:
    """Tests for ActionContentGenerator._format_tool_output_verbose."""

    def test_empty_output(self):
        """Empty output returns no lines."""
        gen = ActionContentGenerator()
        assert gen._format_tool_output_verbose(None) == []
        assert gen._format_tool_output_verbose("") == []
        assert gen._format_tool_output_verbose({}) == []

    def test_string_json_parsed(self):
        """JSON string is parsed into dict entries."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose('{"key": "value"}')
        assert any("key: value" in line for line in lines)

    def test_invalid_json_string(self):
        """Non-JSON string is shown as-is."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose("plain text output")
        assert any("output: plain text output" in line for line in lines)

    def test_non_dict_non_string(self):
        """Non-dict, non-string data is shown with output prefix."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose([1, 2, 3])
        assert any("output:" in line for line in lines)

    def test_dict_with_raw_output_string(self):
        """Dict with raw_output string that is valid JSON."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose({"raw_output": '{"foo": "bar"}'})
        assert any("foo: bar" in line for line in lines)

    def test_dict_with_raw_output_invalid_json(self):
        """Dict with raw_output string that is not valid JSON."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose({"raw_output": "not json"})
        assert any("output: not json" in line for line in lines)

    def test_dict_with_multiline_values(self):
        """Dict values containing newlines are split into continuation lines."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose({"raw_output": {"sql": "SELECT\n  col\nFROM t"}})
        assert any("sql:" in line for line in lines)
        assert any("SELECT" in line for line in lines)
        assert any("FROM t" in line for line in lines)

    def test_dict_raw_output_non_dict(self):
        """Dict with raw_output that resolves to a non-dict after parsing."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose({"raw_output": "[1, 2, 3]"})
        assert any("output:" in line for line in lines)

    def test_custom_indent(self):
        """Custom indent is applied to output lines."""
        gen = ActionContentGenerator()
        lines = gen._format_tool_output_verbose({"raw_output": {"key": "val"}}, indent=">>")
        assert lines[0].startswith(">>")


@pytest.mark.ci
class TestFormatInlineProcessing:
    """Tests for ActionContentGenerator.format_inline_processing."""

    def test_processing_with_function_name(self):
        """Shows function name with blinking frame."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="describe_table",
            input_data={"function_name": "describe_table"},
        )
        result = gen.format_inline_processing(action, "●")
        assert "●" in result
        assert "🔧" in result
        assert "describe_table" in result

    def test_processing_without_input(self):
        """Falls back to messages when input is None."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="some_tool",
        )
        result = gen.format_inline_processing(action, "○")
        assert "some_tool..." in result


@pytest.mark.ci
class TestGetRoleColor:
    """Tests for ActionContentGenerator.get_role_color."""

    def test_known_roles(self):
        """Known roles return their assigned colors."""
        gen = ActionContentGenerator()
        assert gen.get_role_color(ActionRole.SYSTEM) == "bright_magenta"
        assert gen.get_role_color(ActionRole.ASSISTANT) == "bright_blue"
        assert gen.get_role_color(ActionRole.USER) == "bright_green"
        assert gen.get_role_color(ActionRole.TOOL) == "bright_cyan"
        assert gen.get_role_color(ActionRole.WORKFLOW) == "bright_yellow"


@pytest.mark.ci
class TestGenerateStreamingContent:
    """Tests for ActionContentGenerator.generate_streaming_content."""

    def test_empty_actions(self):
        """Empty action list returns waiting message."""
        gen = ActionContentGenerator()
        result = gen.generate_streaming_content([])
        assert "Waiting for actions" in result

    def test_truncation_enabled_uses_simple_text(self):
        """With truncation enabled, returns plain text format."""
        gen = ActionContentGenerator(enable_truncation=True)
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="thinking",
        )
        result = gen.generate_streaming_content([action])
        assert isinstance(result, str)
        assert "thinking" in result

    def test_simple_text_skips_processing_tools(self):
        """_generate_simple_text_content skips PROCESSING TOOL actions."""
        gen = ActionContentGenerator(enable_truncation=True)
        actions = [
            _make_action(ActionRole.TOOL, ActionStatus.PROCESSING, messages="loading..."),
            _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS, messages="done thinking"),
        ]
        result = gen.generate_streaming_content(actions)
        assert "loading" not in result
        assert "done thinking" in result

    def test_truncation_disabled_uses_rich_panel(self):
        """With truncation disabled, returns rich Group object."""
        gen = ActionContentGenerator(enable_truncation=False)
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS, messages="test")
        result = gen.generate_streaming_content([action])
        # Should be a rich Group (not str)
        assert not isinstance(result, str)

    def test_rich_panel_content_empty(self):
        """Rich panel with no displayable actions returns dim message."""
        gen = ActionContentGenerator(enable_truncation=False)
        # All actions are INTERACTION (filtered out in practice, but _generate_rich_panel_content
        # would show them since it doesn't filter)
        result = gen._generate_rich_panel_content([])
        assert "No actions to display" in str(result)


@pytest.mark.ci
class TestCreateResultTable:
    """Tests for ActionContentGenerator._create_result_table."""

    def test_no_output_returns_none(self):
        """Action with no output returns None."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        assert gen._create_result_table(action) is None

    def test_string_json_output_parsed(self):
        """JSON string output is parsed into table."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": '{"result": [{"name": "t1", "type": "table"}, {"name": "t2", "type": "view"}]}'},
        )
        table = gen._create_result_table(action)
        assert table is not None

    def test_invalid_json_string_returns_none(self):
        """Invalid JSON string output returns None."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        action.output = "not json"
        assert gen._create_result_table(action) is None

    def test_non_dict_output_returns_none(self):
        """Non-dict output returns None."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        action.output = [1, 2, 3]
        assert gen._create_result_table(action) is None

    def test_text_field_parsed_as_json_array(self):
        """Text field containing JSON array creates a table."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": {"text": '[{"col1": "a", "col2": "b"}]'}},
        )
        table = gen._create_result_table(action)
        assert table is not None

    def test_empty_list_returns_none(self):
        """Empty result list returns None."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": {"result": []}},
        )
        assert gen._create_result_table(action) is None

    def test_non_dict_items_returns_none(self):
        """List of non-dict items returns None."""
        gen = ActionContentGenerator()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": {"result": ["a", "b"]}},
        )
        assert gen._create_result_table(action) is None

    def test_long_values_truncated_in_table(self):
        """Values longer than 50 chars are truncated in the table."""
        gen = ActionContentGenerator()
        long_val = "x" * 100
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": {"result": [{"field": long_val}]}},
        )
        table = gen._create_result_table(action)
        assert table is not None

    def test_more_than_10_rows_shows_summary(self):
        """More than 10 items shows summary row."""
        gen = ActionContentGenerator()
        items = [{"id": str(i)} for i in range(15)]
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": {"result": items}},
        )
        table = gen._create_result_table(action)
        assert table is not None
        # Table should have 11 rows (10 data + 1 summary)
        assert len(table.rows) == 11


@pytest.mark.ci
class TestFormatData:
    """Tests for ActionContentGenerator.format_data."""

    def test_dict_with_sql_key_not_truncated(self):
        """SQL-related keys are never truncated."""
        gen = ActionContentGenerator(enable_truncation=True)
        long_sql = "SELECT " + "col, " * 100
        result = gen.format_data({"sql_query": long_sql})
        assert long_sql in result

    def test_dict_with_long_value_truncated(self):
        """Long non-SQL values are truncated with truncation enabled."""
        gen = ActionContentGenerator(enable_truncation=True)
        result = gen.format_data({"description": "a" * 200})
        assert "..." in result
        assert len(result.split("description:")[1].strip()) < 200

    def test_dict_without_truncation(self):
        """With truncation disabled, long values are preserved."""
        gen = ActionContentGenerator(enable_truncation=False)
        long_val = "a" * 200
        result = gen.format_data({"description": long_val})
        assert long_val in result

    def test_string_truncated(self):
        """Long string input is truncated with truncation enabled."""
        gen = ActionContentGenerator(enable_truncation=True)
        long_str = "x" * 200
        result = gen.format_data(long_str)
        assert len(result) < 200
        assert result.endswith("...")

    def test_string_not_truncated(self):
        """Short string is returned as-is."""
        gen = ActionContentGenerator(enable_truncation=True)
        result = gen.format_data("short string")
        assert result == "short string"

    def test_other_types(self):
        """Non-dict, non-string data is converted to string."""
        gen = ActionContentGenerator()
        result = gen.format_data(42)
        assert result == "42"


@pytest.mark.ci
class TestGetDataSummary:
    """Tests for ActionContentGenerator.get_data_summary."""

    def test_dict_with_success_and_sql(self):
        """Dict with success=True and sql_query shows SQL."""
        gen = ActionContentGenerator()
        result = gen.get_data_summary({"success": True, "sql_query": "SELECT 1"})
        assert "✅" in result
        assert "SQL: SELECT 1" in result

    def test_dict_with_success_no_sql(self):
        """Dict with success but no sql shows field count."""
        gen = ActionContentGenerator()
        result = gen.get_data_summary({"success": True, "data": "value"})
        assert "✅" in result
        assert "2 fields" in result

    def test_dict_with_failure(self):
        """Dict with success=False shows failure icon."""
        gen = ActionContentGenerator()
        result = gen.get_data_summary({"success": False})
        assert "❌" in result

    def test_dict_without_success_key(self):
        """Dict without 'success' key shows field count."""
        gen = ActionContentGenerator()
        result = gen.get_data_summary({"a": 1, "b": 2})
        assert "2 fields" in result

    def test_long_string_truncated(self):
        """Long string is truncated to 30 chars."""
        gen = ActionContentGenerator(enable_truncation=True)
        result = gen.get_data_summary("a" * 50)
        assert len(result) <= 34  # 30 + "..."
        assert result.endswith("...")

    def test_short_string(self):
        """Short string is returned as-is."""
        gen = ActionContentGenerator()
        result = gen.get_data_summary("hello")
        assert result == "hello"

    def test_other_types(self):
        """Non-dict, non-string data is converted to string."""
        gen = ActionContentGenerator(enable_truncation=True)
        result = gen.get_data_summary(12345)
        assert "12345" in result

    def test_long_sql_truncated(self):
        """Long SQL query is truncated with truncation enabled."""
        gen = ActionContentGenerator(enable_truncation=True)
        long_sql = "SELECT " + "x" * 300
        result = gen.get_data_summary({"success": True, "sql_query": long_sql})
        assert "..." in result


@pytest.mark.ci
class TestGetToolArgsPreview:
    """Tests for ActionContentGenerator._get_tool_args_preview."""

    def test_with_query_argument(self):
        """Shows query argument value."""
        gen = ActionContentGenerator()
        result = gen._get_tool_args_preview({"arguments": {"query": "SELECT 1"}})
        assert "query='SELECT 1'" in result

    def test_with_other_dict_arg(self):
        """Shows first key-value pair for non-query args."""
        gen = ActionContentGenerator()
        result = gen._get_tool_args_preview({"arguments": {"table_name": "users"}})
        assert "table_name='users'" in result

    def test_with_long_query_truncated(self):
        """Long query is truncated with truncation enabled."""
        gen = ActionContentGenerator(enable_truncation=True)
        long_q = "a" * 300
        result = gen._get_tool_args_preview({"arguments": {"query": long_q}})
        assert "..." in result

    def test_with_non_dict_args(self):
        """Non-dict arguments are shown as quoted string."""
        gen = ActionContentGenerator()
        result = gen._get_tool_args_preview({"arguments": "plain string"})
        assert "'plain string'" in result

    def test_empty_arguments(self):
        """Empty or missing arguments returns empty string."""
        gen = ActionContentGenerator()
        assert gen._get_tool_args_preview({}) == ""
        assert gen._get_tool_args_preview({"arguments": None}) == ""
        assert gen._get_tool_args_preview({"arguments": {}}) == ""

    def test_long_non_dict_args_truncated(self):
        """Long non-dict args are truncated."""
        gen = ActionContentGenerator(enable_truncation=True)
        result = gen._get_tool_args_preview({"arguments": "x" * 100})
        assert "..." in result

    def test_long_value_truncated(self):
        """Long first value in dict args is truncated."""
        gen = ActionContentGenerator(enable_truncation=True)
        result = gen._get_tool_args_preview({"arguments": {"key": "y" * 100}})
        assert "..." in result


@pytest.mark.ci
class TestGetToolOutputPreview:
    """Tests for ActionContentGenerator._get_tool_output_preview."""

    def test_empty_output(self):
        """Empty output returns empty string."""
        gen = ActionContentGenerator()
        assert gen._get_tool_output_preview(None) == ""
        assert gen._get_tool_output_preview({}) == ""

    def test_string_json_parsed(self):
        """JSON string output is parsed correctly."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview('{"result": [{"id": 1}, {"id": 2}]}')
        assert "2 items" in result

    def test_invalid_json_string(self):
        """Invalid JSON string returns preview unavailable."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview("not json")
        assert "preview unavailable" in result

    def test_non_dict_output(self):
        """Non-dict output returns preview unavailable."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview([1, 2])
        assert "preview unavailable" in result

    def test_list_tables_function(self):
        """list_tables function shows table count."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": [{"name": "t1"}, {"name": "t2"}]}'},
            function_name="list_tables",
        )
        assert "2 tables" in result

    def test_describe_table_function(self):
        """describe_table function shows column count."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": [{"col": "a"}, {"col": "b"}, {"col": "c"}]}'},
            function_name="describe_table",
        )
        assert "3 columns" in result

    def test_error_output(self):
        """Failed output with error message shows failure."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"success": false, "error": "connection timeout"}'},
        )
        assert "Failed" in result
        assert "connection timeout" in result

    def test_error_output_long_error_truncated(self):
        """Long error message is truncated to 50 chars."""
        gen = ActionContentGenerator()
        long_error = "e" * 100
        result = gen._get_tool_output_preview(
            {"raw_output": f'{{"success": false, "error": "{long_error}"}}'},
        )
        assert "..." in result

    def test_error_without_message(self):
        """Failed output without error message shows generic failure."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"success": false}'},
        )
        assert "Failed" in result

    def test_text_field_plain_text(self):
        """Plain text in 'text' field shown as preview."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": {"text": "Some plain text result"}},
        )
        assert "Some plain text result" in result

    def test_text_field_long_truncated(self):
        """Long plain text is truncated."""
        gen = ActionContentGenerator(enable_truncation=True)
        result = gen._get_tool_output_preview(
            {"raw_output": {"text": "x" * 100}},
        )
        assert "..." in result

    def test_read_query_with_original_rows(self):
        """read_query function shows row count from original_rows."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": {"original_rows": 42}}'},
            function_name="read_query",
        )
        assert "42 rows" in result

    def test_search_table_function(self):
        """search_table function shows metadata and sample counts."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": {"metadata": [{"t": 1}, {"t": 2}], "sample_data": [{"r": 1}]}}'},
            function_name="search_table",
        )
        assert "2 tables" in result
        assert "1 sample rows" in result

    def test_search_metrics_function(self):
        """search_metrics function shows metrics count."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": {"m1": "v1", "m2": "v2"}}'},
            function_name="search_metrics",
        )
        assert "metrics" in result

    def test_search_reference_sql_function(self):
        """search_reference_sql function shows SQL count."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": {"sql1": "v1"}}'},
            function_name="search_reference_sql",
        )
        assert "reference SQLs" in result

    def test_search_external_knowledge_function(self):
        """search_external_knowledge function shows knowledge count."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": {"k1": "v1"}}'},
            function_name="search_external_knowledge",
        )
        assert "extensions of knowledge" in result

    def test_search_documents_function(self):
        """search_documents function shows document count."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview(
            {"raw_output": '{"result": {"d1": "v1"}}'},
            function_name="search_documents",
        )
        assert "documents" in result

    def test_generic_success_fallback(self):
        """Generic success output without items shows 'Success'."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview({"success": True, "raw_output": '{"noitems": true}'})
        assert "Success" in result

    def test_generic_failure_fallback(self):
        """Generic failure output shows 'Failed'."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview({"success": False, "raw_output": '{"noitems": true}'})
        assert "Failed" in result

    def test_generic_completed_fallback(self):
        """No success key and no items returns 'Completed'."""
        gen = ActionContentGenerator()
        result = gen._get_tool_output_preview({"raw_output": '{"noitems": true}'})
        assert "Completed" in result


# ── ActionHistoryDisplay methods ───────────────────────────────────


@pytest.mark.ci
class TestFormatActionSummary:
    """Tests for ActionHistoryDisplay.format_action_summary."""

    def test_summary_contains_icon_and_message(self):
        """Summary includes status icon and action message."""
        display = ActionHistoryDisplay()
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS, messages="query executed")
        result = display.format_action_summary(action)
        assert "✅" in result
        assert "query executed" in result
        assert "bright_cyan" in result

    def test_summary_failed_action(self):
        """Failed action shows failure icon."""
        display = ActionHistoryDisplay()
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.FAILED, messages="generation failed")
        result = display.format_action_summary(action)
        assert "❌" in result


@pytest.mark.ci
class TestFormatActionDetail:
    """Tests for ActionHistoryDisplay.format_action_detail."""

    def test_detail_panel_with_input_output(self):
        """Detail panel includes input and output data."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3.14)
        display = ActionHistoryDisplay()
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            messages="read_query",
            input_data={"sql": "SELECT 1"},
            output_data={"result": "ok"},
            start_time=t0,
            end_time=t1,
        )
        panel = display.format_action_detail(action)
        # Panel should be a rich Panel object
        from rich.panel import Panel

        assert isinstance(panel, Panel)

    def test_detail_panel_without_input_output(self):
        """Detail panel works without input/output."""
        display = ActionHistoryDisplay()
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="thinking",
        )
        panel = display.format_action_detail(action)
        from rich.panel import Panel

        assert isinstance(panel, Panel)


@pytest.mark.ci
class TestExtractSubagentResponse:
    """Tests for ActionHistoryDisplay._extract_subagent_response."""

    def test_extract_from_json_string_raw_output(self):
        """Extracts response from JSON-encoded raw_output."""
        result = ActionHistoryDisplay._extract_subagent_response(
            {"raw_output": '{"success": 1, "result": {"response": "The total is 42"}}'}
        )
        assert result == "The total is 42"

    def test_extract_from_dict_raw_output(self):
        """Extracts response from dict raw_output."""
        result = ActionHistoryDisplay._extract_subagent_response(
            {"raw_output": {"success": 1, "result": {"response": "Hello"}}}
        )
        assert result == "Hello"

    def test_extract_flat_response(self):
        """Extracts response from flat dict format."""
        result = ActionHistoryDisplay._extract_subagent_response({"response": "Direct response"})
        assert result == "Direct response"

    def test_invalid_json_returns_empty(self):
        """Invalid JSON in raw_output returns empty string."""
        result = ActionHistoryDisplay._extract_subagent_response({"raw_output": "not json"})
        assert result == ""

    def test_no_response_key(self):
        """Missing response key returns empty string."""
        result = ActionHistoryDisplay._extract_subagent_response({"raw_output": {"other": "data"}})
        assert result == ""

    def test_no_raw_output(self):
        """Dict without raw_output falls back to direct access."""
        result = ActionHistoryDisplay._extract_subagent_response({"some_key": "val"})
        assert result == ""


@pytest.mark.ci
class TestRenderSubagentResponse:
    """Tests for ActionHistoryDisplay._render_subagent_response."""

    def test_renders_single_line_response(self):
        """Single-line response is rendered with 'response:' label."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": '{"success": 1, "result": {"response": "Answer is 42"}}'},
        )
        display._render_subagent_response(action)
        output = buf.getvalue()
        assert "response:" in output
        assert "Answer is 42" in output

    def test_renders_multiline_response(self):
        """Multi-line response renders each line."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            output_data={"raw_output": '{"success": 1, "result": {"response": "Line 1\\nLine 2"}}'},
        )
        display._render_subagent_response(action)
        output = buf.getvalue()
        assert "response:" in output
        assert "Line 1" in output
        assert "Line 2" in output

    def test_skips_when_no_output(self):
        """Does nothing when output is None."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        display._render_subagent_response(action)
        assert buf.getvalue() == ""

    def test_skips_when_output_not_dict(self):
        """Does nothing when output is not a dict."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS)
        action.output = "string output"
        display._render_subagent_response(action)
        assert buf.getvalue() == ""


@pytest.mark.ci
class TestRenderTaskToolAsSubagent:
    """Tests for ActionHistoryDisplay._render_task_tool_as_subagent."""

    def test_compact_mode_with_output(self):
        """Compact mode shows header with result summary."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=5)
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "task", "type": "gen_sql", "prompt": "query"},
            output_data={"raw_output": '{"success": true}'},
            start_time=t0,
            end_time=t1,
        )
        display._render_task_tool_as_subagent(action, verbose=False)
        output = buf.getvalue()
        assert "gen_sql" in output
        assert "✓" in output
        assert "5.0s" in output

    def test_verbose_mode_with_response(self):
        """Verbose mode shows full prompt and response."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "task", "type": "fix_sql", "prompt": "Fix the query"},
            output_data={"raw_output": '{"success": 1, "result": {"response": "Fixed output"}}'},
        )
        display._render_task_tool_as_subagent(action, verbose=True)
        output = buf.getvalue()
        assert "fix_sql" in output
        assert "prompt:" in output
        assert "Fix the query" in output
        assert "response:" in output
        assert "Fixed output" in output

    def test_failed_status(self):
        """Failed task tool shows cross mark."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.FAILED,
            input_data={"function_name": "task", "type": "gen_sql"},
            output_data={"raw_output": '{"success": false}'},
        )
        display._render_task_tool_as_subagent(action, verbose=False)
        output = buf.getvalue()
        assert "✗" in output


@pytest.mark.ci
class TestRenderMainAction:
    """Tests for ActionHistoryDisplay._render_main_action."""

    def test_user_action_strips_prefix(self):
        """USER action strips 'User: ' prefix."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(ActionRole.USER, ActionStatus.SUCCESS, messages="User: What is revenue?")
        display._render_main_action(action, verbose=False)
        output = buf.getvalue()
        assert "Datus>" in output
        assert "What is revenue?" in output

    def test_user_action_without_prefix(self):
        """USER action without 'User: ' prefix shows message directly."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(ActionRole.USER, ActionStatus.SUCCESS, messages="Direct message")
        display._render_main_action(action, verbose=False)
        output = buf.getvalue()
        assert "Direct message" in output

    def test_assistant_action_with_content(self):
        """ASSISTANT action renders content with Markdown."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="some thinking",
            output_data={"raw_output": "Here is the result"},
        )
        display._render_main_action(action, verbose=False)
        output = buf.getvalue()
        assert "Here is the result" in output

    def test_assistant_response_skipped(self):
        """ASSISTANT with _response action_type is skipped by render_action_history."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="chat_response",
            messages="should be skipped",
        )
        display.render_action_history([action])
        output = buf.getvalue()
        assert "should be skipped" not in output

    def test_verbose_mode_uses_expanded_format(self):
        """Verbose mode delegates to format_inline_expanded."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.WORKFLOW,
            ActionStatus.SUCCESS,
            messages="workflow step",
        )
        display._render_main_action(action, verbose=True)
        output = buf.getvalue()
        assert "workflow step" in output


@pytest.mark.ci
class TestDisplayActionList:
    """Tests for ActionHistoryDisplay.display_action_list."""

    def test_empty_list(self):
        """Empty action list shows 'No actions' message."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        display.display_action_list([])
        output = buf.getvalue()
        assert "No actions to display" in output

    def test_with_actions(self):
        """Non-empty list delegates to render_action_history."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="list_tables()",
                input_data={"function_name": "list_tables"},
            )
        ]
        display.display_action_list(actions)
        output = buf.getvalue()
        assert "list_tables" in output


@pytest.mark.ci
class TestDisplayFinalActionHistory:
    """Tests for ActionHistoryDisplay.display_final_action_history."""

    def test_empty_list(self):
        """Empty list shows 'No actions' message."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        display.display_final_action_history([])
        output = buf.getvalue()
        assert "No actions to display" in output

    def test_with_dict_input_output(self):
        """Actions with dict input/output are rendered in tree format."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2.5)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="read_query",
                input_data={"function_name": "read_query", "sql": "SELECT 1"},
                output_data={"result": "ok", "rows": 5},
                start_time=t0,
                end_time=t1,
            )
        ]
        display.display_final_action_history(actions)
        output = buf.getvalue()
        assert "Action History" in output
        assert "read_query" in output

    def test_with_non_dict_input_output(self):
        """Actions with non-dict input/output are rendered as strings."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="thinking",
        )
        action.input = "plain string input"
        action.output = "plain string output"
        display.display_final_action_history([action])
        output = buf.getvalue()
        assert "Input:" in output
        assert "Output:" in output


@pytest.mark.ci
class TestGetDataSummaryWithFullSql:
    """Tests for ActionHistoryDisplay._get_data_summary_with_full_sql."""

    def test_dict_with_success_and_sql(self):
        """Shows full SQL without truncation."""
        display = ActionHistoryDisplay()
        long_sql = "SELECT " + "x" * 500
        result = display._get_data_summary_with_full_sql({"success": True, "sql_query": long_sql})
        assert long_sql in result
        assert "✅" in result

    def test_dict_with_success_no_sql(self):
        """Shows field count when no SQL."""
        display = ActionHistoryDisplay()
        result = display._get_data_summary_with_full_sql({"success": True, "data": 1})
        assert "2 fields" in result

    def test_dict_with_failure(self):
        """Shows failure icon."""
        display = ActionHistoryDisplay()
        result = display._get_data_summary_with_full_sql({"success": False})
        assert "❌" in result

    def test_dict_without_success(self):
        """Shows field count without success key."""
        display = ActionHistoryDisplay()
        result = display._get_data_summary_with_full_sql({"a": 1, "b": 2, "c": 3})
        assert "3 fields" in result

    def test_string(self):
        """String is returned as-is."""
        display = ActionHistoryDisplay()
        assert display._get_data_summary_with_full_sql("hello") == "hello"

    def test_other_types(self):
        """Other types are converted to string."""
        display = ActionHistoryDisplay()
        assert display._get_data_summary_with_full_sql(42) == "42"


# ── render_action_history skip patterns ───────────────────────────


@pytest.mark.ci
class TestRenderActionHistorySkipPatterns:
    """Tests for skip conditions in render_action_history."""

    def test_skips_interaction_actions(self):
        """INTERACTION actions are skipped."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="user input request"),
            _make_action(ActionRole.WORKFLOW, ActionStatus.SUCCESS, messages="should appear"),
        ]
        display.render_action_history(actions)
        output = buf.getvalue()
        assert "user input request" not in output
        assert "should appear" in output

    def test_skips_processing_tools(self):
        """PROCESSING TOOL actions are skipped."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        actions = [
            _make_action(ActionRole.TOOL, ActionStatus.PROCESSING, messages="loading..."),
            _make_action(ActionRole.TOOL, ActionStatus.SUCCESS, messages="done"),
        ]
        display.render_action_history(actions)
        output = buf.getvalue()
        assert "loading" not in output

    def test_skip_task_tools_reset_on_non_task_tool(self):
        """skip_task_tools flag resets when a non-task TOOL action appears."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        call_id = "call_1"
        actions = [
            # Subagent group
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                start_time=t0,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id,
            ),
            # task tool should be skipped
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=t1,
            ),
            # non-task tool should NOT be skipped (resets the flag)
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="read_query(SELECT 1)",
                input_data={"function_name": "read_query"},
            ),
        ]
        display.render_action_history(actions)
        output = buf.getvalue()
        assert "read_query" in output

    def test_verbose_shows_subagent_response_for_task_tool(self):
        """In verbose mode, task tool after subagent_complete shows response."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=2)
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        call_id = "call_verbose"
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                start_time=t0,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                output_data={"raw_output": '{"success": 1, "result": {"response": "SQL generated"}}'},
                end_time=t1,
            ),
        ]
        display.render_action_history(actions, verbose=True)
        output = buf.getvalue()
        assert "response:" in output
        assert "SQL generated" in output


@pytest.mark.ci
class TestRenderSubagentOtherRoles:
    """Tests for _render_subagent_action with non-TOOL/ASSISTANT/USER roles."""

    def test_other_role_in_subagent_compact(self):
        """WORKFLOW action in subagent group is shown with truncation in compact mode."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.WORKFLOW,
            ActionStatus.SUCCESS,
            depth=1,
            messages="executing step 1",
        )
        display._render_subagent_action(action, verbose=False)
        output = buf.getvalue()
        assert "executing step 1" in output

    def test_other_role_long_message_truncated(self):
        """Long messages in non-TOOL/ASSISTANT roles are truncated in compact mode."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        long_msg = "x" * 300
        action = _make_action(
            ActionRole.SYSTEM,
            ActionStatus.SUCCESS,
            depth=1,
            messages=long_msg,
        )
        display._render_subagent_action(action, verbose=False)
        output = buf.getvalue()
        assert " ... " in output

    def test_verbose_tool_with_output(self):
        """Verbose mode for TOOL actions in subagent shows full output."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}},
            output_data={"raw_output": '{"result": "ok"}'},
        )
        display._render_subagent_action(action, verbose=True)
        output = buf.getvalue()
        assert "sql: SELECT 1" in output


# ── InlineStreamingContext ─────────────────────────────────────────


@pytest.mark.ci
class TestInlineStreamingContextProperties:
    """Tests for InlineStreamingContext properties and simple methods."""

    def test_live_property_initially_none(self):
        """live property is None before any PROCESSING tool."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        assert ctx.live is None

    def test_stop_display(self):
        """stop_display sets _paused to True."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._paused = False
        ctx.stop_display()
        assert ctx._paused is True

    def test_restart_display(self):
        """restart_display sets _paused to False."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._paused = True
        ctx.restart_display()
        assert ctx._paused is False

    def test_toggle_verbose(self):
        """toggle_verbose sets the event."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        assert not ctx._verbose_toggle_event.is_set()
        ctx.toggle_verbose()
        assert ctx._verbose_toggle_event.is_set()

    def test_recreate_live_display_calls_restart(self):
        """recreate_live_display is a compatibility shim for restart_display."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._paused = True
        ctx.recreate_live_display()
        assert ctx._paused is False


@pytest.mark.ci
class TestInlineStreamingContextFlush:
    """Tests for InlineStreamingContext._flush_remaining_actions."""

    def test_flush_skips_interaction(self):
        """Flush skips INTERACTION actions."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="skip me"),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._flush_remaining_actions()
        assert ctx._processed_index == 1

    def test_flush_skips_processing_tools(self):
        """Flush skips PROCESSING TOOL actions."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.TOOL, ActionStatus.PROCESSING, messages="loading"),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._flush_remaining_actions()
        assert ctx._processed_index == 1

    def test_flush_handles_subagent_complete(self):
        """Flush closes groups via subagent_complete."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        call_id = "flush_call"
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                start_time=t0,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "describe_table"},
                start_time=t0,
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                start_time=t0,
                end_time=t1,
                parent_action_id=call_id,
            ),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._flush_remaining_actions()

        output = buf.getvalue()
        assert "Done" in output
        assert ctx._processed_index == 3

    def test_flush_closes_unclosed_groups(self):
        """Flush closes groups that never received subagent_complete."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                start_time=t0,
                parent_action_id="orphan",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                input_data={"function_name": "read_query"},
                start_time=t0,
                parent_action_id="orphan",
            ),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._flush_remaining_actions()

        output = buf.getvalue()
        assert "Done" in output
        assert "1 tool uses" in output

    def test_flush_depth0_completed_action(self):
        """Flush prints normal depth=0 completed actions."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(ActionRole.WORKFLOW, ActionStatus.SUCCESS, messages="step done"),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._flush_remaining_actions()

        output = buf.getvalue()
        assert "step done" in output


@pytest.mark.ci
class TestInlineStreamingContextProcess:
    """Tests for InlineStreamingContext._process_actions specific branches."""

    def test_process_skips_interaction(self):
        """INTERACTION actions are skipped during processing."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="input request"),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0
        ctx._process_actions()
        assert ctx._processed_index == 1

    def test_process_skips_depth1_processing_tools(self):
        """PROCESSING tools in subagent groups are skipped."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                depth=1,
                input_data={"function_name": "read_query"},
                parent_action_id="call1",
            ),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        printed = []
        with patch.object(display.console, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
            ctx._process_actions()

        assert ctx._processed_index == 1

    def test_print_completed_action_skips_task_tool(self):
        """_print_completed_action skips 'task' function tool calls."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "task"},
        )
        ctx._print_completed_action(action)
        assert buf.getvalue() == ""

    def test_print_completed_action_assistant(self):
        """_print_completed_action renders ASSISTANT with Markdown."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)

        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            messages="my thought",
            output_data={"raw_output": "Result content"},
        )
        ctx._print_completed_action(action)
        output = buf.getvalue()
        assert "Result content" in output

    def test_print_completed_verbose(self):
        """_print_completed_action in verbose mode uses expanded format."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)
        ctx._verbose = True

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}},
        )
        ctx._print_completed_action(action)
        output = buf.getvalue()
        assert "sql: SELECT 1" in output

    def test_print_completed_empty_lines_skipped(self):
        """_print_completed_action does nothing for roles that return empty lines."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)

        action = _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="skip")
        ctx._print_completed_action(action)
        assert buf.getvalue() == ""


@pytest.mark.ci
class TestStopAndStartLive:
    """Tests for ActionHistoryDisplay.stop_live and restart_live."""

    def test_stop_live_no_context(self):
        """stop_live does nothing when no current context."""
        display = ActionHistoryDisplay()
        display.stop_live()  # Should not raise

    def test_restart_live_no_context(self):
        """restart_live does nothing when no current context."""
        display = ActionHistoryDisplay()
        display.restart_live()  # Should not raise

    def test_stop_live_with_context(self):
        """stop_live calls stop_display on current context."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        display._current_context = ctx
        display.stop_live()
        assert ctx._paused is True

    def test_restart_live_with_context(self):
        """restart_live calls restart_display on current context."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._paused = True
        display._current_context = ctx
        display.restart_live()
        assert ctx._paused is False


# ── create_action_display factory ──────────────────────────────────


@pytest.mark.ci
class TestCreateActionDisplay:
    """Tests for the create_action_display factory function."""

    def test_default_params(self):
        """Creates display with default console and truncation enabled."""
        display = create_action_display()
        assert isinstance(display, ActionHistoryDisplay)
        assert display.enable_truncation is True

    def test_custom_params(self):
        """Creates display with custom console and truncation disabled."""
        buf = StringIO()
        console = Console(file=buf)
        display = create_action_display(console=console, enable_truncation=False)
        assert display.console is console
        assert display.enable_truncation is False


# ── InlineStreamingContext: _update_subagent_display branches ──────


@pytest.mark.ci
class TestUpdateSubagentDisplayBranches:
    """Tests for InlineStreamingContext._update_subagent_display edge cases."""

    def test_verbose_tool_with_non_dict_args(self):
        """Verbose tool with non-dict arguments shows 'args:' label."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)
        ctx._verbose = True
        ctx._subagent_groups = {"g1": {"start_time": datetime.now(), "tool_count": 0, "subagent_type": "gen_sql"}}

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            input_data={"function_name": "search", "arguments": "plain text"},
            parent_action_id="g1",
        )
        ctx._update_subagent_display(action, group_key="g1")
        output = buf.getvalue()
        assert "args: plain text" in output
        assert ctx._subagent_groups["g1"]["tool_count"] == 1

    def test_other_role_verbose(self):
        """Non-TOOL/ASSISTANT/USER role in verbose mode shows full message."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)
        ctx._verbose = True
        ctx._subagent_groups = {"g2": {"start_time": datetime.now(), "tool_count": 0, "subagent_type": "fix_sql"}}

        action = _make_action(
            ActionRole.SYSTEM,
            ActionStatus.SUCCESS,
            depth=1,
            messages="system message in subagent",
            parent_action_id="g2",
        )
        ctx._update_subagent_display(action, group_key="g2")
        output = buf.getvalue()
        assert "system message in subagent" in output

    def test_other_role_compact_truncated(self):
        """Non-TOOL/ASSISTANT/USER role in compact mode truncates long messages."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)
        ctx._verbose = False
        ctx._subagent_groups = {"g3": {"start_time": datetime.now(), "tool_count": 0, "subagent_type": "gen_sql"}}

        long_msg = "z" * 300
        action = _make_action(
            ActionRole.WORKFLOW,
            ActionStatus.SUCCESS,
            depth=1,
            messages=long_msg,
            parent_action_id="g3",
        )
        ctx._update_subagent_display(action, group_key="g3")
        output = buf.getvalue()
        assert " ... " in output

    def test_assistant_empty_content_skips(self):
        """ASSISTANT with empty content is not printed."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)
        ctx._subagent_groups = {"g4": {"start_time": datetime.now(), "tool_count": 0, "subagent_type": "gen_sql"}}

        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            depth=1,
            messages="",
            parent_action_id="g4",
        )
        ctx._update_subagent_display(action, group_key="g4")
        assert buf.getvalue() == ""


@pytest.mark.ci
class TestEndSubagentGroupByKey:
    """Tests for InlineStreamingContext._end_subagent_group_by_key."""

    def test_unknown_group_key_noop(self):
        """Ending a group with unknown key does nothing."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._end_subagent_group_by_key("nonexistent", _make_action(ActionRole.SYSTEM, ActionStatus.SUCCESS))
        # Should not crash

    def test_none_group_key_not_added_to_completed(self):
        """None group key is not added to _completed_group_ids."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._subagent_groups = {None: {"start_time": datetime.now(), "tool_count": 1, "subagent_type": "gen_sql"}}

        end_action = _make_action(ActionRole.SYSTEM, ActionStatus.SUCCESS, end_time=datetime.now())
        ctx._end_subagent_group_by_key(None, end_action)
        assert None not in ctx._completed_group_ids
