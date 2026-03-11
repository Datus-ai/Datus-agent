# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/action_history_display.py — SubAgent group display.

Tests cover:
- Sub-agent group header printed on first depth>0 action
- Sub-agent display updates tool_count for TOOL actions
- Sub-agent group ends with Done summary when depth returns to 0
- Flush correctly ends an active sub-agent group
- Normal depth=0 flow unchanged
- Multiple sub-agent groups handled sequentially
- Task SUCCESS action skipped after Done line
"""

from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.markdown import Markdown

from datus.cli.action_history_display import ActionHistoryDisplay, InlineStreamingContext, _truncate_middle
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
