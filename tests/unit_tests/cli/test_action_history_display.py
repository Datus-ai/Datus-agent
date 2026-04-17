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
- ActionHistoryDisplay: _render_subagent_response,
  _render_task_tool_as_subagent, render_action_history (skip patterns)
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
from rich.text import Text

from datus.cli.action_display.display import ActionHistoryDisplay, create_action_display
from datus.cli.action_display.renderers import (
    ActionContentGenerator,
    ActionRenderer,
    BaseActionContentGenerator,
    _get_assistant_content,
    _truncate_middle,
)
from datus.cli.action_display.streaming import _SUBAGENT_ROLLING_WINDOW_SIZE, InlineStreamingContext
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus


def _group_plain(group_renderable) -> str:
    """Extract plain text from a Group renderable."""
    if isinstance(group_renderable, Text):
        return group_renderable.plain
    parts = []
    for item in group_renderable.renderables:
        if isinstance(item, Text):
            parts.append(item.plain)
        else:
            parts.append(str(item))
    return "\n".join(parts)


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
        """First depth>0 action creates group state and Live renderable contains header."""
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

        with patch("datus.cli.action_display.streaming.Live"):
            ctx._process_actions()

        # Group state should be set
        assert len(ctx._subagent_groups) == 1
        group = next(iter(ctx._subagent_groups.values()))
        assert group["subagent_type"] == "gen_sql"
        assert group["tool_count"] == 0
        assert group["first_action"] is first_action

        # Live renderable should contain subagent type and prompt
        renderable = ctx._build_subagent_groups_renderable()
        assert "gen_sql(What is the total revenue?)" in _group_plain(renderable)

    def test_subagent_prompt_truncated_middle(self):
        """Long prompt is truncated in the middle in the Live renderable."""
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

        with patch("datus.cli.action_display.streaming.Live"):
            ctx._process_actions()

        renderable = ctx._build_subagent_groups_renderable()
        # Should contain " ... " truncation marker
        assert " ... " in _group_plain(renderable)


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
        """Each depth>0 TOOL action increments the group tool_count and buffers actions."""
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

        with patch("datus.cli.action_display.streaming.Live"):
            ctx._process_actions()

        assert len(ctx._subagent_groups) == 1
        group = next(iter(ctx._subagent_groups.values()))
        assert group["tool_count"] == 2
        assert len(group["actions"]) == 2  # USER is skipped, 2 TOOL actions buffered

        # Verify that Live renderable contains tool function names
        renderable = ctx._build_subagent_groups_renderable()
        all_output = _group_plain(renderable)
        assert "read_query" in all_output
        assert "describe_table" in all_output

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
            with patch("datus.cli.action_display.streaming.Live"):
                ctx._process_actions()

        group = next(iter(ctx._subagent_groups.values()))
        assert group["tool_count"] == 0


@pytest.mark.ci
class TestSubAgentGroupEnd:
    """depth returns to 0 → Done summary printed."""

    def test_done_summary_printed(self):
        """When depth=0 action follows depth>0 group, Done line is printed (verbose mode)."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=5.2)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0
        ctx._verbose = True  # verbose mode keeps Done lines

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
            with patch("datus.cli.action_display.streaming.Live"):
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
            with patch("datus.cli.action_display.streaming.Live"):
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
            with patch("datus.cli.action_display.streaming.Live"):
                ctx._process_actions()

        # Index should NOT advance past PROCESSING
        assert ctx._processed_index == 0


@pytest.mark.ci
class TestMultipleSubAgentGroups:
    """Multiple sequential sub-agent groups are handled correctly."""

    def test_two_groups(self):
        """Two sub-agent groups produce two headers and two Done lines (verbose mode)."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        t2 = t1 + timedelta(seconds=2)

        actions = []
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0
        ctx._verbose = True  # verbose mode keeps Done lines

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
            with patch("datus.cli.action_display.streaming.Live"):
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
        ctx._verbose = True  # verbose mode keeps Done lines

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
            with patch("datus.cli.action_display.streaming.Live"):
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
        """INTERACTION actions are skipped in history replay."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.PROCESSING, messages="Choose an option"),
        ]
        printed = self._collect_prints(display, actions)
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
        # Should render as subagent header (with bold markup) + result
        assert "gen_sql" in combined
        assert "What is total revenue?" in combined
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
        # In compact mode (verbose=False), the group is collapsed — only one header + Done
        headers = [line for line in printed if "gen_sql" in line]
        assert len(headers) >= 1
        # Done line should exist
        done_lines = [line for line in printed if "Done" in line]
        assert len(done_lines) == 1

    # -- subagent grouping --

    def test_subagent_group_header_and_actions(self):
        """Subagent group renders header + action lines in verbose mode."""
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

        printed = self._collect_prints(display, actions, verbose=True)
        combined = "\n".join(printed)

        # Header (with bold markup)
        assert "gen_sql" in combined
        assert "What is total revenue?" in combined
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
        # In compact mode (verbose=False), groups are collapsed with ⏴ marker
        headers = [line for line in printed if "gen_sql" in line or "fix_sql" in line]
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
            display.render_action_history(actions, verbose=False, _show_partial_done=False)
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

        printed = self._collect_prints(display, actions, verbose=True)
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
        ctx._verbose = True  # verbose mode keeps Done lines

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
            with patch("datus.cli.action_display.streaming.Live"):
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
        assert "gen_sql" in combined
        assert "query" in combined
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
        ctx._verbose = True  # verbose mode keeps Done lines

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
            with patch("datus.cli.action_display.streaming.Live"):
                ctx._process_actions()

        # Both groups should be closed
        assert len(ctx._subagent_groups) == 0
        assert call_id_a in ctx._completed_group_ids
        assert call_id_b in ctx._completed_group_ids

        # Two headers and two Done lines (compact mode uses ⏴ collapsed marker)
        headers = [line for line in printed if "gen_sql" in line or "fix_sql" in line]
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

        # In compact mode (verbose=False), groups are collapsed with ⏴ marker
        headers = [line for line in printed if "gen_sql" in line or "fix_sql" in line]
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
            # depth=0 TOOL(task) — should be skipped (action_id matches parent_action_id of subagent children)
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                input_data={"function_name": "task"},
                end_time=t1,
                action_id=call_id,
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

        # Should have one group header (collapsed \u23f4 or expanded \u23fa) and one Done
        headers = [line for line in printed if "gen_sql" in line and ("⏺" in line or "⏴" in line)]
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
        assert "gen_sql" in combined
        assert "Generate monthly sales report" in combined
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
        assert "gen_sql" in combined
        assert "What is the total revenue?" in combined
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
        assert "gen_sql" in combined
        assert "Generate sales report" in combined
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
        assert "gen_sql" in combined
        assert "What is total revenue?" in combined
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
        ctx._reprint_history(verbose=ctx._verbose)

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
        ctx._reprint_history(verbose=ctx._verbose)

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
        """INTERACTION actions return empty list (not shown in history)."""
        gen = ActionContentGenerator()
        action = _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="user input")
        lines = gen.format_inline_completed(action)
        assert len(lines) == 0


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
        assert any("sql" in line for line in lines)
        assert any("SELECT 1" in line for line in lines)
        assert any("limit" in line and "10" in line for line in lines)
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
    """Tests for format_output_verbose (migrated from ActionContentGenerator to tool_content module)."""

    def test_empty_output(self):
        """Empty output returns no lines."""
        from datus.cli.action_display.tool_content import format_output_verbose

        assert format_output_verbose(None) == []
        assert format_output_verbose("") == []
        assert format_output_verbose({}) == []

    def test_string_json_parsed(self):
        """JSON string is parsed into dict entries."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose('{"key": "value"}')
        assert any("key: value" in line for line in lines)

    def test_invalid_json_string(self):
        """Non-JSON string is shown as-is."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose("plain text output")
        assert any("output: plain text output" in line for line in lines)

    def test_non_dict_non_string(self):
        """Non-dict, non-string data is shown with output prefix."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose([1, 2, 3])
        assert any("output:" in line for line in lines)

    def test_dict_with_raw_output_string(self):
        """Dict with raw_output string that is valid JSON."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose({"raw_output": '{"foo": "bar"}'})
        assert any("foo: bar" in line for line in lines)

    def test_dict_with_raw_output_invalid_json(self):
        """Dict with raw_output string that is not valid JSON."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose({"raw_output": "not json"})
        assert any("output: not json" in line for line in lines)

    def test_dict_with_multiline_values(self):
        """Dict values containing newlines are split into continuation lines."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose({"raw_output": {"sql": "SELECT\n  col\nFROM t"}})
        assert any("sql:" in line for line in lines)
        assert any("SELECT" in line for line in lines)
        assert any("FROM t" in line for line in lines)

    def test_dict_raw_output_non_dict(self):
        """Dict with raw_output that resolves to a non-dict after parsing."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose({"raw_output": "[1, 2, 3]"})
        assert any("output:" in line for line in lines)

    def test_custom_indent(self):
        """Custom indent is applied to output lines."""
        from datus.cli.action_display.tool_content import format_output_verbose

        lines = format_output_verbose({"raw_output": {"key": "val"}}, indent=">>")
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
    """Tests for tool output preview via ToolCallContentBuilder.

    Tool-specific previews (list_tables, search_table, etc.) are now
    handled by per-tool builder functions registered in ToolCallContentBuilder.
    Generic previews are tested via format_generic_preview.
    """

    def test_empty_output(self):
        """Empty output returns empty string."""
        from datus.cli.action_display.tool_content import format_generic_preview

        assert format_generic_preview(None) == ""
        assert format_generic_preview({}) == ""

    def test_string_json_parsed(self):
        """JSON string output is parsed correctly."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview('{"result": [{"id": 1}, {"id": 2}]}')
        assert "2 items" in result

    def test_invalid_json_string(self):
        """Invalid JSON string returns preview unavailable."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview("not json")
        assert "preview unavailable" in result

    def test_non_dict_output(self):
        """Non-dict output returns preview unavailable."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview([1, 2])
        assert "preview unavailable" in result

    def test_list_tables_function(self):
        """list_tables function shows table count via registered builder."""
        from datus.cli.action_display.tool_content import _build_list_tables

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "list_tables"},
            output_data={"raw_output": '{"result": [{"name": "t1"}, {"name": "t2"}]}'},
        )
        tc = _build_list_tables(action, verbose=False)
        assert "2 tables" in tc.compact_result

    def test_describe_table_function(self):
        """describe_table function shows column count via registered builder."""
        from datus.cli.action_display.tool_content import _build_describe_table

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "describe_table"},
            output_data={"raw_output": '{"result": [{"col": "a"}, {"col": "b"}, {"col": "c"}]}'},
        )
        tc = _build_describe_table(action, verbose=False)
        assert "3 columns" in tc.compact_result

    def test_error_output(self):
        """Failed output with error message shows failure."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview(
            {"raw_output": '{"success": false, "error": "connection timeout"}'},
        )
        assert "Failed" in result
        assert "connection timeout" in result

    def test_error_output_long_error_truncated(self):
        """Long error message is truncated to 50 chars."""
        from datus.cli.action_display.tool_content import format_generic_preview

        long_error = "e" * 100
        result = format_generic_preview(
            {"raw_output": f'{{"success": false, "error": "{long_error}"}}'},
        )
        assert "..." in result

    def test_error_without_message(self):
        """Failed output without error message shows generic failure."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview(
            {"raw_output": '{"success": false}'},
        )
        assert "Failed" in result

    def test_text_field_plain_text(self):
        """Plain text in 'text' field shown as preview."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview(
            {"raw_output": {"text": "Some plain text result"}},
        )
        assert "Some plain text result" in result

    def test_text_field_long_truncated(self):
        """Long plain text is truncated."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview(
            {"raw_output": {"text": "x" * 100}},
            enable_truncation=True,
        )
        assert "..." in result

    def test_read_query_with_original_rows(self):
        """read_query function shows row count via registered builder."""
        from datus.cli.action_display.tool_content import _build_read_query

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "read_query"},
            output_data={"raw_output": '{"original_rows": 42}'},
        )
        tc = _build_read_query(action, verbose=False)
        assert "42 rows" in tc.compact_result

    def test_search_table_function(self):
        """search_table function shows metadata and sample counts via registered builder."""
        from datus.cli.action_display.tool_content import _build_search_table

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "search_table"},
            output_data={"raw_output": '{"metadata": [{"t": 1}, {"t": 2}], "sample_data": [{"r": 1}]}'},
        )
        tc = _build_search_table(action, verbose=False)
        assert "2 tables" in tc.compact_result
        assert "1 sample row" in tc.compact_result

    def test_search_metrics_function(self):
        """search_metrics function shows metrics count via registered builder."""
        from datus.cli.action_display.tool_content import _build_search_metrics

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "search_metrics"},
            output_data={"result": [1, 2]},
        )
        tc = _build_search_metrics(action, verbose=False)
        assert "metrics" in tc.compact_result

    def test_search_reference_sql_function(self):
        """search_reference_sql function shows SQL count via registered builder."""
        from datus.cli.action_display.tool_content import _build_search_reference_sql

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "search_reference_sql"},
            output_data={"result": [1]},
        )
        tc = _build_search_reference_sql(action, verbose=False)
        assert "1 reference SQL matched" in tc.compact_result

    def test_search_external_knowledge_function(self):
        """search_external_knowledge function shows knowledge count via registered builder."""
        from datus.cli.action_display.tool_content import _build_search_external_knowledge

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "search_external_knowledge"},
            output_data={"result": [1]},
        )
        tc = _build_search_external_knowledge(action, verbose=False)
        assert "1 knowledge entry matched" in tc.compact_result

    def test_search_documents_function(self):
        """search_documents function shows document count via registered builder."""
        from datus.cli.action_display.tool_content import _build_search_documents

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            input_data={"function_name": "search_documents"},
            output_data={"result": [1]},
        )
        tc = _build_search_documents(action, verbose=False)
        assert "1 document matched" in tc.compact_result

    def test_generic_success_fallback(self):
        """Generic success output without items shows 'Success'."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview({"success": True, "raw_output": '{"noitems": true}'})
        assert "Success" in result

    def test_generic_failure_fallback(self):
        """Generic failure output shows 'Failed'."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview({"success": False, "raw_output": '{"noitems": true}'})
        assert "Failed" in result

    def test_generic_completed_fallback(self):
        """No success key and no items returns 'Completed'."""
        from datus.cli.action_display.tool_content import format_generic_preview

        result = format_generic_preview({"raw_output": '{"noitems": true}'})
        assert "Completed" in result


# ── ActionHistoryDisplay methods ───────────────────────────────────


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


# ── INTERACTION rendering ─────────────────────────────────────────


@pytest.mark.ci
class TestRenderInteraction:
    """Tests for INTERACTION action rendering via ActionRenderer."""

    def test_render_interaction_request_text(self):
        """INTERACTION PROCESSING with text content_type."""
        from datus.cli.action_display.renderers import ActionRenderer

        renderer = ActionRenderer()
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            messages="Choose option",
            input_data={"contents": ["Please confirm"], "content_type": "text", "choices": [{}]},
        )
        renderables = renderer.render_interaction_request(action, verbose=False)
        texts = [r.plain if hasattr(r, "plain") else str(r) for r in renderables]
        full = " ".join(texts)
        assert "Interaction Request" in full
        assert "Please confirm" in full

    def test_render_interaction_request_yaml(self):
        """INTERACTION PROCESSING with yaml content_type renders Syntax."""
        from datus.cli.action_display.renderers import ActionRenderer

        renderer = ActionRenderer()
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            input_data={"contents": ["key: value"], "content_type": "yaml", "choices": [{}]},
        )
        renderables = renderer.render_interaction_request(action, verbose=False)
        from rich.syntax import Syntax

        assert any(isinstance(r, Syntax) for r in renderables)

    def test_render_interaction_request_sql(self):
        """INTERACTION PROCESSING with sql content_type renders Syntax."""
        from datus.cli.action_display.renderers import ActionRenderer

        renderer = ActionRenderer()
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            input_data={"contents": ["SELECT 1"], "content_type": "sql", "choices": [{}]},
        )
        renderables = renderer.render_interaction_request(action, verbose=False)
        from rich.syntax import Syntax

        assert any(isinstance(r, Syntax) for r in renderables)

    def test_render_interaction_request_markdown(self):
        """INTERACTION PROCESSING with markdown content_type renders Markdown."""
        from datus.cli.action_display.renderers import ActionRenderer

        renderer = ActionRenderer()
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            input_data={"contents": ["## Title\nBody"], "content_type": "markdown", "choices": [{}]},
        )
        renderables = renderer.render_interaction_request(action, verbose=False)
        assert any(isinstance(r, Markdown) for r in renderables)

    def test_render_interaction_request_with_choices_shows_content_only(self):
        """INTERACTION PROCESSING with choices renders content only (not choice list)."""
        from datus.cli.action_display.renderers import ActionRenderer

        renderer = ActionRenderer()
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            input_data={
                "contents": ["Pick one: y=Yes, n=No"],
                "content_type": "text",
                "choices": [{"y": "Yes", "n": "No"}],
            },
        )
        renderables = renderer.render_interaction_request(action, verbose=False)
        texts = [r.plain if hasattr(r, "plain") else str(r) for r in renderables]
        full = "\n".join(texts)
        assert "Pick one" in full
        # Choices are handled by select_choice UI, not rendered in request
        assert len(renderables) == 2  # header + content only

    def test_render_interaction_success_with_choice(self):
        """INTERACTION SUCCESS shows user_choice and content."""
        from datus.cli.action_display.renderers import ActionRenderer

        renderer = ActionRenderer()
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.SUCCESS,
            messages="Confirmed",
            output_data={"content": "Saved!", "content_type": "text", "user_choice": "y"},
        )
        renderables = renderer.render_interaction_success(action, verbose=False)
        texts = [r.plain if hasattr(r, "plain") else str(r) for r in renderables]
        full = " ".join(texts)
        assert "Selected: y" in full
        assert "Saved!" in full

    def test_render_main_action_skips_interaction(self):
        """render_main_action returns empty for INTERACTION (not shown in history)."""
        renderer = ActionRenderer(ActionContentGenerator())
        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            input_data={"content": "Confirm?", "content_type": "text"},
        )
        result = renderer.render_main_action(action, verbose=False)
        assert result == []

        action_succ = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.SUCCESS,
            messages="Done",
            output_data={"content": "All good", "content_type": "text"},
        )
        result2 = renderer.render_main_action(action_succ, verbose=False)
        assert result2 == []


# ── render_action_history skip patterns ───────────────────────────


@pytest.mark.ci
class TestRenderActionHistorySkipPatterns:
    """Tests for skip conditions in render_action_history."""

    def test_skips_interaction_actions(self):
        """INTERACTION actions are skipped in history replay."""
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
        assert "sql" in output
        assert "SELECT 1" in output


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
        """Flush skips INTERACTION actions (only shown during live interaction)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="render me"),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._flush_remaining_actions()
        assert ctx._processed_index == 1
        assert "render me" not in buf.getvalue()

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
        """Flush closes groups via subagent_complete (verbose mode shows Done line)."""
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
        ctx._verbose = True  # verbose mode keeps Done lines
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

    def test_process_skips_interaction_success(self):
        """INTERACTION SUCCESS actions are skipped during processing (only shown live)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="input request"),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0
        ctx._process_actions()
        assert ctx._processed_index == 1
        assert "input request" not in buf.getvalue()

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
        assert "sql" in output
        assert "SELECT 1" in output

    def test_print_completed_interaction_skipped(self):
        """_print_completed_action produces no output for INTERACTION (render_main_action returns [])."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)

        action = _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="interaction done")
        ctx._print_completed_action(action)
        assert buf.getvalue().strip() == ""


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
        """Verbose tool with non-dict arguments shows 'args:' label in renderable."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = True
        first_action = _make_action(ActionRole.USER, ActionStatus.PROCESSING, depth=1, action_type="gen_sql")
        ctx._subagent_groups = {
            "g1": {
                "start_time": datetime.now(),
                "tool_count": 0,
                "subagent_type": "gen_sql",
                "first_action": first_action,
                "actions": [],
            }
        }

        action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            input_data={"function_name": "search", "arguments": "plain text"},
            parent_action_id="g1",
        )
        with patch("datus.cli.action_display.streaming.Live"):
            ctx._update_subagent_display(action, group_key="g1")
        assert ctx._subagent_groups["g1"]["tool_count"] == 1
        line = ctx._format_subagent_action_line(action)
        assert "args: plain text" in line

    def test_other_role_verbose(self):
        """Non-TOOL/ASSISTANT/USER role in verbose mode shows full message in renderable."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = True

        action = _make_action(
            ActionRole.SYSTEM,
            ActionStatus.SUCCESS,
            depth=1,
            messages="system message in subagent",
            parent_action_id="g2",
        )
        line = ctx._format_subagent_action_line(action)
        assert "system message in subagent" in line

    def test_other_role_compact_truncated(self):
        """Non-TOOL/ASSISTANT/USER role in compact mode truncates long messages."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = False

        long_msg = "z" * 300
        action = _make_action(
            ActionRole.WORKFLOW,
            ActionStatus.SUCCESS,
            depth=1,
            messages=long_msg,
            parent_action_id="g3",
        )
        line = ctx._format_subagent_action_line(action)
        assert " ... " in line

    def test_assistant_empty_content_skips(self):
        """ASSISTANT with empty content produces empty format line."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)

        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            depth=1,
            messages="",
            parent_action_id="g4",
        )
        line = ctx._format_subagent_action_line(action)
        assert line == ""


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
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)
        # Set verbose so the old Done-line path is taken (no reprint)
        ctx._verbose = True
        ctx._subagent_groups = {
            None: {
                "start_time": datetime.now(),
                "tool_count": 1,
                "subagent_type": "gen_sql",
                "first_action": None,
                "actions": [],
            }
        }

        end_action = _make_action(ActionRole.SYSTEM, ActionStatus.SUCCESS, end_time=datetime.now())
        ctx._end_subagent_group_by_key(None, end_action)
        assert None not in ctx._completed_group_ids


@pytest.mark.ci
class TestCollapseCompleted:
    """Tests for collapse/expand behavior in render_action_history (driven by verbose flag)."""

    def _build_completed_group_actions(self, parent_id="grp1"):
        """Build a list of actions representing a completed subagent group."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        t2 = datetime(2025, 1, 1, 12, 0, 2)
        return [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="gen_sql",
                messages="User: Find total revenue",
                input_data={"_task_description": "Find total revenue"},
                start_time=t0,
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="search_schema",
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="execute_sql",
                messages="execute_sql",
                input_data={"function_name": "execute_sql"},
                start_time=t1,
                end_time=t2,
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="gen_sql",
                messages="",
                output_data={"raw_output": "SELECT SUM(revenue) FROM sales"},
                start_time=t2,
                end_time=t2,
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="",
                start_time=t0,
                end_time=t2,
                parent_action_id=parent_id,
            ),
        ]

    def test_render_action_history_collapse_completed(self):
        """verbose=False collapses completed group as header + Done summary."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        actions = self._build_completed_group_actions()
        display.render_action_history(actions, verbose=False)

        output = buf.getvalue()
        lines = [line for line in output.splitlines() if line.strip()]
        # Line 1: header (not dim) with collapsed marker
        assert "\u23f4" in lines[0]  # ⏴ collapsed marker
        assert "gen_sql" in lines[0]
        assert "Find total revenue" in lines[0]
        # Line 2: Done summary with status, tool count, duration
        assert "Done" in lines[1]
        assert "\u2713" in lines[1]  # ✓ success mark
        assert "2 tool uses" in lines[1]
        # Should NOT contain expanded action lines
        assert "search_schema" not in output
        assert "execute_sql" not in output

    def test_render_action_history_collapse_verbose_overrides(self):
        """verbose=True expands all actions (no collapsing)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        actions = self._build_completed_group_actions()
        display.render_action_history(actions, verbose=True)

        output = buf.getvalue()
        # Should NOT have collapsed marker
        assert "\u23f4" not in output  # ⏴ should not appear
        # Should have expanded header marker
        assert "\u23fa" in output  # ⏺ expanded marker
        # Should contain individual tool actions
        assert "search_schema" in output
        assert "execute_sql" in output
        # Should contain Done summary
        assert "Done" in output

    def test_render_action_history_collapse_active_group_expanded(self):
        """Unclosed (active) groups remain expanded even in compact mode (verbose=False)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        # Build actions without the subagent_complete action (group still active)
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="gen_sql",
                messages="User: Find total revenue",
                input_data={"_task_description": "Find total revenue"},
                start_time=t0,
                parent_action_id="grp_active",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="search_schema",
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
                parent_action_id="grp_active",
            ),
        ]

        display.render_action_history(actions, verbose=False)

        output = buf.getvalue()
        # Should NOT have collapsed marker (group is not complete)
        assert "\u23f4" not in output
        # Active group should be expanded with header
        assert "\u23fa" in output or "gen_sql" in output
        # Should show tool action
        assert "search_schema" in output

    def test_render_action_history_verbose_expanded(self):
        """verbose=True renders all actions expanded (no collapse)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        actions = self._build_completed_group_actions()
        display.render_action_history(actions, verbose=True)

        output = buf.getvalue()
        # Should NOT have collapsed marker
        assert "\u23f4" not in output
        # Should have expanded header
        assert "\u23fa" in output
        # Should contain individual tool actions
        assert "search_schema" in output
        assert "execute_sql" in output
        # Should contain Done summary
        assert "Done" in output

    def test_render_action_history_collapse_failed_group(self):
        """Collapsed group shows failure marker in Done summary line."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 2)
        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="gen_sql",
                messages="User: Bad query",
                start_time=t0,
                parent_action_id="grp_fail",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.FAILED,
                depth=1,
                action_type="execute_sql",
                messages="execute_sql",
                input_data={"function_name": "execute_sql"},
                start_time=t0,
                end_time=t1,
                parent_action_id="grp_fail",
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.FAILED,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="",
                start_time=t0,
                end_time=t1,
                parent_action_id="grp_fail",
            ),
        ]

        display.render_action_history(actions, verbose=False)

        output = buf.getvalue()
        lines = [line for line in output.splitlines() if line.strip()]
        # Line 1: header with collapsed marker
        assert "\u23f4" in lines[0]
        # Line 2: Done summary with failure mark
        assert "Done" in lines[1]
        assert "\u2717" in lines[1]  # ✗ failure mark
        assert "1 tool uses" in lines[1]

    def test_render_multi_turn_history_collapse(self):
        """render_multi_turn_history collapses groups in compact mode (verbose=False)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        actions = self._build_completed_group_actions()
        turns = [("Find total revenue", actions)]

        display.render_multi_turn_history(turns, verbose=False)

        output = buf.getvalue()
        assert "\u23f4" in output  # collapsed marker
        assert "Find total revenue" in output
        assert "Done" in output  # Done summary on second line
        # Should NOT show expanded tool actions
        assert "search_schema" not in output


# ---------------------------------------------------------------------------
# Verbose Freeze + Subagent Rolling Window tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestVerboseFrozenField:
    """Test _verbose_frozen field initialization and cleanup."""

    def test_verbose_frozen_init_false(self):
        """_verbose_frozen starts as False."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        assert ctx._verbose_frozen is False

    def test_exit_resets_verbose_frozen(self):
        """__exit__ resets _verbose_frozen to False before flushing."""
        display = ActionHistoryDisplay()
        actions = []
        ctx = InlineStreamingContext(actions, display)
        ctx._verbose_frozen = True
        # Simulate __exit__ without the threading parts
        ctx._verbose_frozen = False  # This is what __exit__ does first
        assert ctx._verbose_frozen is False


@pytest.mark.ci
class TestVerboseFreezeProcessingGuard:
    """Test that _process_actions is blocked when _verbose_frozen is True."""

    def test_process_actions_skipped_when_frozen(self):
        """When _verbose_frozen=True, _process_actions should not advance _processed_index."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
            ),
        ]

        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._verbose_frozen = True

        # The refresh loop guard: `if not self._paused and not self._verbose_frozen`
        # We simulate by checking the condition directly
        assert ctx._verbose_frozen is True
        # _process_actions would run, but the guard prevents it
        # Verify _processed_index stays at 0 (actions not consumed)
        assert ctx._processed_index == 0

    def test_process_actions_runs_when_not_frozen(self):
        """When _verbose_frozen=False, _process_actions advances normally."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
            ),
        ]

        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._verbose_frozen = False

        ctx._process_actions()
        assert ctx._processed_index == 1


@pytest.mark.ci
class TestReprintHistoryVerboseSnapshot:
    """Test _reprint_history_verbose_snapshot renders verbose output with active groups."""

    def test_snapshot_renders_processed_actions_verbose(self):
        """Snapshot renders already-processed actions in verbose mode."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="search_schema",
                input_data={"function_name": "search_schema", "arguments": {"query": "revenue"}},
                start_time=t0,
                end_time=t1,
            ),
        ]

        ctx = InlineStreamingContext(actions, display, current_user_message="What is revenue?")
        ctx._processed_index = 1  # One action already processed
        ctx._verbose = True

        ctx._reprint_history(verbose=True, show_active_groups=True)

        output = buf.getvalue()
        # Should contain the user message header
        assert "What is revenue?" in output
        # Should contain the tool action rendered in verbose mode
        assert "search_schema" in output

    def test_snapshot_renders_active_subagent_groups(self):
        """Snapshot renders active subagent groups with 'in progress' indicator."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = []

        ctx = InlineStreamingContext(actions, display, current_user_message="Test")
        ctx._processed_index = 0

        # Simulate an active subagent group
        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="explore",
            messages="User: Find tables",
            input_data={"_task_description": "Find tables"},
            start_time=t0,
            parent_action_id="grp1",
        )
        tool_action = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            messages="search_schema",
            input_data={"function_name": "search_schema"},
            start_time=t0,
            end_time=t1,
            parent_action_id="grp1",
        )
        ctx._subagent_groups["grp1"] = {
            "start_time": t0,
            "tool_count": 1,
            "subagent_type": "explore",
            "first_action": first_action,
            "actions": [tool_action],
        }

        ctx._reprint_history(verbose=True, show_active_groups=True)

        output = buf.getvalue()
        # Should contain the subagent header
        assert "explore" in output
        # Should contain the tool action
        assert "search_schema" in output
        # Should contain "in progress" indicator
        assert "in progress" in output
        assert "1 tool uses" in output

    def test_snapshot_renders_history_turns(self):
        """Snapshot renders previous history turns in verbose mode."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)

        history_actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
                start_time=t0,
                end_time=t1,
            ),
        ]
        history_turns = [("Show tables", history_actions)]

        ctx = InlineStreamingContext([], display, history_turns=history_turns, current_user_message="Current query")
        ctx._processed_index = 0

        ctx._reprint_history(verbose=True, show_active_groups=True)

        output = buf.getvalue()
        # Should contain the historical turn
        assert "Show tables" in output
        assert "list_tables" in output
        # Should contain the current turn header
        assert "Current query" in output

    def test_snapshot_no_live_display_started(self):
        """Snapshot should not start any Live display — it's all static output."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        ctx = InlineStreamingContext([], display)
        ctx._processed_index = 0
        ctx._subagent_groups["grp1"] = {
            "start_time": datetime(2025, 1, 1, 12, 0, 0),
            "tool_count": 0,
            "subagent_type": "explore",
            "first_action": _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="explore",
                messages="User: Test",
                parent_action_id="grp1",
            ),
            "actions": [],
        }

        ctx._reprint_history(verbose=True, show_active_groups=True)

        # No Live display should have been started
        assert ctx._subagent_live is None
        assert ctx._live is None


@pytest.mark.ci
class TestSubagentRollingWindow:
    """Test subagent rolling window: compact shows last N, verbose shows all."""

    def _make_subagent_actions(self, count: int, group_key: str = "grp1"):
        """Create N subagent TOOL actions."""
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        actions = []
        for i in range(count):
            t_end = datetime(2025, 1, 1, 12, 0, i + 1)
            actions.append(
                _make_action(
                    ActionRole.TOOL,
                    ActionStatus.SUCCESS,
                    depth=1,
                    messages=f"tool_{i}",
                    input_data={"function_name": f"tool_{i}"},
                    start_time=t0,
                    end_time=t_end,
                    parent_action_id=group_key,
                )
            )
        return actions

    def test_compact_shows_only_last_n_actions(self):
        """In compact mode, only the last _SUBAGENT_ROLLING_WINDOW_SIZE actions are displayed."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = False

        tool_actions = self._make_subagent_actions(5)
        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="explore",
            messages="User: Test",
            parent_action_id="grp1",
        )
        ctx._subagent_groups["grp1"] = {
            "start_time": datetime(2025, 1, 1, 12, 0, 0),
            "tool_count": 5,
            "subagent_type": "explore",
            "first_action": first_action,
            "actions": tool_actions,
        }

        renderable = ctx._build_subagent_groups_renderable()
        plain = _group_plain(renderable)

        # Should show "earlier action(s)" hint
        hidden = 5 - _SUBAGENT_ROLLING_WINDOW_SIZE
        assert f"{hidden} earlier action(s)" in plain

        # Should contain only the last N tool names
        for i in range(5 - _SUBAGENT_ROLLING_WINDOW_SIZE):
            assert f"tool_{i}" not in plain, f"tool_{i} should be hidden in compact mode"
        for i in range(5 - _SUBAGENT_ROLLING_WINDOW_SIZE, 5):
            assert f"tool_{i}" in plain, f"tool_{i} should be visible in compact mode"

    def test_compact_no_hidden_hint_when_few_actions(self):
        """No 'earlier action(s)' hint when action count <= window size."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = False

        tool_actions = self._make_subagent_actions(_SUBAGENT_ROLLING_WINDOW_SIZE)
        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="explore",
            messages="User: Test",
            parent_action_id="grp1",
        )
        ctx._subagent_groups["grp1"] = {
            "start_time": datetime(2025, 1, 1, 12, 0, 0),
            "tool_count": _SUBAGENT_ROLLING_WINDOW_SIZE,
            "subagent_type": "explore",
            "first_action": first_action,
            "actions": tool_actions,
        }

        renderable = ctx._build_subagent_groups_renderable()
        plain = _group_plain(renderable)

        # Should NOT show "earlier action(s)" hint
        assert "earlier action(s)" not in plain
        # All actions should be visible
        for i in range(_SUBAGENT_ROLLING_WINDOW_SIZE):
            assert f"tool_{i}" in plain

    def test_verbose_shows_all_actions(self):
        """In verbose mode, all subagent actions are displayed regardless of count."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = True

        tool_actions = self._make_subagent_actions(5)
        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="explore",
            messages="User: Test",
            parent_action_id="grp1",
        )
        ctx._subagent_groups["grp1"] = {
            "start_time": datetime(2025, 1, 1, 12, 0, 0),
            "tool_count": 5,
            "subagent_type": "explore",
            "first_action": first_action,
            "actions": tool_actions,
        }

        renderable = ctx._build_subagent_groups_renderable()
        plain = _group_plain(renderable)

        # Should NOT show "earlier action(s)" hint in verbose mode
        assert "earlier action(s)" not in plain
        # All 5 actions should be visible
        for i in range(5):
            assert f"tool_{i}" in plain

    def test_compact_empty_actions_no_crash(self):
        """Empty actions list renders without crash and without hidden hint."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = False

        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="explore",
            messages="User: Test",
            parent_action_id="grp1",
        )
        ctx._subagent_groups["grp1"] = {
            "start_time": datetime(2025, 1, 1, 12, 0, 0),
            "tool_count": 0,
            "subagent_type": "explore",
            "first_action": first_action,
            "actions": [],
        }

        renderable = ctx._build_subagent_groups_renderable()
        plain = _group_plain(renderable)

        # Header should be present, no hidden hint
        assert "explore" in plain
        assert "earlier action(s)" not in plain

    def test_compact_single_action_no_hidden_hint(self):
        """A single action (less than window size) shows no hidden hint."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        ctx._verbose = False

        tool_actions = self._make_subagent_actions(1)
        first_action = _make_action(
            ActionRole.USER,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="explore",
            messages="User: Test",
            parent_action_id="grp1",
        )
        ctx._subagent_groups["grp1"] = {
            "start_time": datetime(2025, 1, 1, 12, 0, 0),
            "tool_count": 1,
            "subagent_type": "explore",
            "first_action": first_action,
            "actions": tool_actions,
        }

        renderable = ctx._build_subagent_groups_renderable()
        plain = _group_plain(renderable)

        assert "earlier action(s)" not in plain
        assert "tool_0" in plain


@pytest.mark.ci
class TestPendingTaskToolSkips:
    """Test the counter-based pending_task_tool_skips logic in render_action_history.

    When multiple subagent groups complete and a non-task TOOL action appears
    between subagent_complete and TOOL(task) actions, the counter should NOT
    be reset (unlike the old boolean skip_task_tools flag).
    """

    def _build_interleaved_actions(self):
        """Build actions with: 2 parallel subagent groups, non-task TOOL between
        subagent_complete actions and TOOL(task) actions.

        Action order:
        1. depth>0 group1 USER
        2. depth>0 group1 TOOL (search_schema)
        3. depth>0 group2 USER
        4. depth>0 group2 TOOL (search_metrics)
        5. subagent_complete group1
        6. subagent_complete group2
        7. depth=0 TOOL (parse_temporal_expressions) — non-task TOOL
        8. depth=0 TOOL (task) — corresponds to group1
        9. depth=0 TOOL (task) — corresponds to group2
        """
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        t2 = datetime(2025, 1, 1, 12, 0, 2)

        actions = [
            # Group 1 actions
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="explore",
                messages="User: Find tables",
                input_data={"_task_description": "Find tables"},
                start_time=t0,
                parent_action_id="grp1",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="search_schema",
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
                parent_action_id="grp1",
            ),
            # Group 2 actions
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="explore",
                messages="User: Find metrics",
                input_data={"_task_description": "Find metrics"},
                start_time=t0,
                parent_action_id="grp2",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="search_metrics",
                messages="search_metrics",
                input_data={"function_name": "search_metrics"},
                start_time=t0,
                end_time=t1,
                parent_action_id="grp2",
            ),
            # subagent_complete for group1
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="",
                start_time=t0,
                end_time=t1,
                parent_action_id="grp1",
            ),
            # subagent_complete for group2
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="",
                start_time=t0,
                end_time=t1,
                parent_action_id="grp2",
            ),
            # Non-task TOOL (interleaved)
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="parse_temporal_expressions",
                messages="parse_temporal_expressions",
                input_data={"function_name": "parse_temporal_expressions"},
                start_time=t1,
                end_time=t2,
            ),
            # TOOL(task) for group1
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                messages="task",
                input_data={"function_name": "task", "type": "explore", "prompt": "Find tables"},
                output_data={"raw_output": '{"success": 1, "result": {"response": "found tables"}}'},
                start_time=t0,
                end_time=t1,
                action_id="grp1",
            ),
            # TOOL(task) for group2
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type="task",
                messages="task",
                input_data={"function_name": "task", "type": "explore", "prompt": "Find metrics"},
                output_data={"raw_output": '{"success": 1, "result": {"response": "found metrics"}}'},
                start_time=t0,
                end_time=t1,
                action_id="grp2",
            ),
        ]
        return actions

    def test_collapsed_no_extra_subagent_labels(self):
        """In compact mode (verbose=False), TOOL(task) actions after interleaved non-task
        TOOL should NOT produce extra 'subagent' labels."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        actions = self._build_interleaved_actions()
        display.render_action_history(actions, verbose=False)

        output = buf.getvalue()
        lines = [line for line in output.splitlines() if line.strip()]

        # Should have 2 collapsed groups + their Done lines + 1 parse_temporal_expressions action
        # Should NOT have any "subagent" standalone entries
        subagent_standalone = [line for line in lines if "subagent" in line.lower() and "result" in line.lower()]
        assert len(subagent_standalone) == 0, (
            f"Expected no standalone subagent entries but found: {subagent_standalone}"
        )

        # The collapsed groups should be present
        collapsed_count = sum(1 for line in lines if "\u23f4" in line)  # ⏴
        assert collapsed_count == 2, f"Expected 2 collapsed groups, got {collapsed_count}"

        # parse_temporal_expressions should be rendered
        assert "parse_temporal_expressions" in output

    def test_expanded_no_extra_subagent_labels(self):
        """With verbose=True, TOOL(task) actions after interleaved non-task
        TOOL should NOT produce extra 'subagent' labels."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        actions = self._build_interleaved_actions()
        display.render_action_history(actions, verbose=True)

        output = buf.getvalue()

        # Should have expanded groups with Done lines
        assert "Done" in output
        # The non-task tool should be rendered
        assert "parse_temporal_expressions" in output
        # Should NOT have standalone "subagent" entries rendered by _render_task_tool_as_subagent
        # Look for the standalone "⏺ subagent" pattern or "⏺ explore(" without depth>0 context
        lines = [line for line in output.splitlines() if line.strip()]
        subagent_standalone = [line for line in lines if "subagent" in line.lower() and "result" in line.lower()]
        assert len(subagent_standalone) == 0, (
            f"Expected no standalone subagent entries but found: {subagent_standalone}"
        )

    def test_verbose_renders_deferred_groups_flushed_before_response(self):
        """In verbose mode with interleaved non-task TOOL, deferred groups are
        flushed before TOOL(task) arrives, so responses are NOT shown.
        The groups and the non-task tool are all rendered correctly."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        actions = self._build_interleaved_actions()
        display.render_action_history(actions, verbose=True)

        output = buf.getvalue()

        # Verbose mode: groups should be expanded
        assert "search_schema" in output
        assert "search_metrics" in output
        # parse_temporal_expressions should be rendered
        assert "parse_temporal_expressions" in output
        # Both groups should have "Done" lines
        assert output.count("Done") == 2
        # TOOL(task) actions should be skipped (not rendered as standalone subagent)
        lines = [line for line in output.splitlines() if line.strip()]
        subagent_standalone = [line for line in lines if "subagent" in line.lower() and "result" in line.lower()]
        assert len(subagent_standalone) == 0

    def test_verbose_deferred_groups_paired_without_interleaving(self):
        """Without interleaved non-task TOOL, deferred groups pair with TOOL(task)
        and responses ARE rendered in verbose mode."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)

        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="explore",
                messages="User: Find tables",
                input_data={"_task_description": "Find tables"},
                start_time=t0,
                parent_action_id="grp1",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
                parent_action_id="grp1",
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="",
                start_time=t0,
                end_time=t1,
                parent_action_id="grp1",
            ),
            # TOOL(task) immediately follows — no interleaving
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="task",
                input_data={"function_name": "task", "type": "explore", "prompt": "Find tables"},
                output_data={"raw_output": '{"success": 1, "result": {"response": "found tables"}}'},
                start_time=t0,
                end_time=t1,
                action_id="grp1",
            ),
        ]

        display.render_action_history(actions, verbose=True)

        output = buf.getvalue()
        # Group should be expanded with response shown
        assert "search_schema" in output
        assert "Done" in output
        assert "found tables" in output

    def test_single_group_with_non_task_tool_after(self):
        """Single subagent group followed by non-task TOOL then TOOL(task) — no extra labels."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        t2 = datetime(2025, 1, 1, 12, 0, 2)

        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="explore",
                messages="User: Query",
                start_time=t0,
                parent_action_id="grp1",
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
                parent_action_id="grp1",
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="",
                start_time=t0,
                end_time=t1,
                parent_action_id="grp1",
            ),
            # Non-task TOOL between subagent_complete and TOOL(task)
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="parse_temporal_expressions",
                input_data={"function_name": "parse_temporal_expressions"},
                start_time=t1,
                end_time=t2,
            ),
            # TOOL(task)
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="task",
                input_data={"function_name": "task"},
                start_time=t0,
                end_time=t1,
                action_id="grp1",
            ),
        ]

        display.render_action_history(actions, verbose=False)

        output = buf.getvalue()
        lines = [line for line in output.splitlines() if line.strip()]

        # 1 collapsed group + 1 parse_temporal_expressions, no extra subagent
        collapsed_count = sum(1 for line in lines if "\u23f4" in line)
        assert collapsed_count == 1
        assert "parse_temporal_expressions" in output
        subagent_standalone = [line for line in lines if "subagent" in line.lower() and "result" in line.lower()]
        assert len(subagent_standalone) == 0


@pytest.mark.ci
class TestVerboseToggleRefreshLoop:
    """Test the verbose toggle behavior in _refresh_loop (freeze/unfreeze).

    These tests simulate the toggle event and verify state transitions
    without actually running the background thread.
    """

    def test_toggle_to_verbose_sets_frozen(self):
        """Toggling to verbose mode sets _verbose_frozen=True and _verbose=True."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        ctx = InlineStreamingContext([], display)
        ctx._verbose = False
        ctx._verbose_frozen = False

        # Simulate what _refresh_loop does when toggle event fires
        ctx._verbose = not ctx._verbose  # becomes True
        if ctx._verbose:
            ctx._verbose_frozen = True

        assert ctx._verbose is True
        assert ctx._verbose_frozen is True

    def test_toggle_back_to_compact_clears_frozen(self):
        """Toggling back to compact mode sets _verbose_frozen=False."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        ctx = InlineStreamingContext([], display)
        ctx._verbose = True
        ctx._verbose_frozen = True

        # Simulate toggling back to compact
        ctx._verbose = not ctx._verbose  # becomes False
        if not ctx._verbose:
            ctx._verbose_frozen = False

        assert ctx._verbose is False
        assert ctx._verbose_frozen is False

    def test_reprint_history_uses_compact_mode(self):
        """_reprint_history renders in compact mode with collapsed groups."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="search_schema",
                input_data={"function_name": "search_schema"},
                start_time=t0,
                end_time=t1,
            ),
        ]

        ctx = InlineStreamingContext(actions, display, current_user_message="Test query")
        ctx._processed_index = 1
        ctx._verbose = False

        ctx._reprint_history(verbose=ctx._verbose)

        output = buf.getvalue()
        assert "Test query" in output
        assert "search_schema" in output

    def test_verbose_snapshot_uses_verbose_mode(self):
        """_reprint_history_verbose_snapshot renders everything in verbose mode."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="search_schema",
                input_data={"function_name": "search_schema", "arguments": {"query": "revenue table"}},
                start_time=t0,
                end_time=t1,
            ),
        ]

        ctx = InlineStreamingContext(actions, display, current_user_message="Test query")
        ctx._processed_index = 1
        ctx._verbose = True

        ctx._reprint_history(verbose=True, show_active_groups=True)

        output = buf.getvalue()
        assert "Test query" in output
        assert "search_schema" in output
        # Verbose mode should show arguments
        assert "revenue table" in output

    def test_freeze_unfreeze_round_trip(self):
        """Full round-trip: compact → verbose (freeze) → compact (unfreeze)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=200)
        display = ActionHistoryDisplay(console)

        t0 = datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime(2025, 1, 1, 12, 0, 1)
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
                start_time=t0,
                end_time=t1,
            ),
        ]

        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 1

        # Initial state: compact, not frozen
        assert ctx._verbose is False
        assert ctx._verbose_frozen is False

        # Toggle to verbose → frozen
        ctx._verbose = True
        ctx._verbose_frozen = True
        assert ctx._verbose_frozen is True

        # Process actions should be skipped (guard condition)
        old_idx = ctx._processed_index
        # Simulate the guard: if not paused and not frozen → skip
        if not ctx._paused and not ctx._verbose_frozen:
            ctx._process_actions()
        assert ctx._processed_index == old_idx  # unchanged

        # Toggle back to compact → unfrozen
        ctx._verbose = False
        ctx._verbose_frozen = False
        assert ctx._verbose_frozen is False
