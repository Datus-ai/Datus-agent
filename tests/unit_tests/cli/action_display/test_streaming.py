# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/action_display/streaming.py — sync mode and unified reprint."""

import asyncio
import uuid
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from datus.cli.action_display.display import ActionHistoryDisplay
from datus.cli.action_display.streaming import InlineStreamingContext
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus


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


# ── Sync mode basic ──────────────────────────────────────────────


@pytest.mark.ci
class TestSyncMode:
    """Test run_sync processes all actions."""

    def test_sync_processes_all_actions(self):
        """run_sync processes all completed actions."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
            ),
            _make_action(ActionRole.ASSISTANT, ActionStatus.SUCCESS, messages="Here is the result"),
        ]

        ctx = InlineStreamingContext(actions, display, sync_mode=True)
        ctx.run_sync()

        output = buf.getvalue()
        assert "list_tables" in output
        assert "Here is the result" in output
        assert ctx._processed_index == 2

    def test_sync_skips_processing_tools(self):
        """Sync mode skips PROCESSING entries."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
            ),
        ]

        ctx = InlineStreamingContext(actions, display, sync_mode=True)
        ctx.run_sync()

        output = buf.getvalue()
        assert ctx._processed_index == 2
        # Only one output line for list_tables (the SUCCESS one)
        assert output.count("list_tables") >= 1

    def test_sync_skips_interaction_actions(self):
        """Sync mode skips INTERACTION actions (only shown during live interaction)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(ActionRole.INTERACTION, ActionStatus.SUCCESS, messages="Confirm?"),
            _make_action(ActionRole.TOOL, ActionStatus.SUCCESS, messages="done", input_data={"function_name": "done"}),
        ]

        ctx = InlineStreamingContext(actions, display, sync_mode=True)
        ctx.run_sync()
        assert ctx._processed_index == 2
        output = buf.getvalue()
        assert "Confirm?" not in output


# ── Sync mode subagent groups ─────────────────────────────────────


@pytest.mark.ci
class TestSyncModeSubagentGroups:
    """Test sync mode handles subagent groups correctly."""

    def test_sync_subagent_group(self):
        """Sync mode renders a complete subagent group."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        parent_id = "parent-123"

        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: revenue?",
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="read_query",
                input_data={"function_name": "read_query"},
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="complete",
                parent_action_id=parent_id,
            ),
        ]

        ctx = InlineStreamingContext(actions, display, sync_mode=True)
        ctx.run_sync()

        output = buf.getvalue()
        assert "gen_sql" in output
        assert "Done" in output
        assert "2 tool uses" in output

    def test_sync_skips_processing_in_subagent(self):
        """Sync mode skips PROCESSING tools inside subagent groups."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        parent_id = "parent-456"

        actions = [
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: test",
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                depth=1,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
                parent_action_id=parent_id,
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=0,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="complete",
                parent_action_id=parent_id,
            ),
        ]

        ctx = InlineStreamingContext(actions, display, sync_mode=True)
        ctx.run_sync()

        output = buf.getvalue()
        assert "1 tool uses" in output  # Only the SUCCESS one counted


# ── Unified reprint ──────────────────────────────────────────────


@pytest.mark.ci
class TestUnifiedReprint:
    """Test the unified _reprint_history method."""

    def test_reprint_compact_mode(self):
        """Reprint in compact mode with collapsed groups."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(
                ActionRole.TOOL, ActionStatus.SUCCESS, messages="result", input_data={"function_name": "read_query"}
            ),
        ]

        ctx = InlineStreamingContext(
            actions,
            display,
            history_turns=[
                (
                    "Previous question",
                    [
                        _make_action(
                            ActionRole.TOOL,
                            ActionStatus.SUCCESS,
                            messages="prev",
                            input_data={"function_name": "list_tables"},
                        ),
                    ],
                )
            ],
            current_user_message="Current question",
        )
        ctx._processed_index = 1
        ctx._verbose = False

        ctx._reprint_history(verbose=False)

        output = buf.getvalue()
        assert "Previous question" in output
        assert "Current question" in output

    def test_reprint_verbose_mode_with_active_groups(self):
        """Reprint in verbose mode shows active groups with in-progress indicator."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(
                ActionRole.TOOL, ActionStatus.SUCCESS, messages="done", input_data={"function_name": "list_tables"}
            ),
        ]

        ctx = InlineStreamingContext(actions, display, current_user_message="My question")
        ctx._processed_index = 1
        ctx._verbose = True

        # Simulate an active subagent group
        ctx._subagent_groups["active-group"] = {
            "start_time": datetime.now() - timedelta(seconds=5),
            "tool_count": 3,
            "subagent_type": "gen_sql",
            "first_action": _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                depth=1,
                action_type="gen_sql",
                messages="User: active query",
                parent_action_id="active-group",
            ),
            "actions": [],
        }

        ctx._reprint_history(verbose=True, show_active_groups=True)

        output = buf.getvalue()
        assert "in progress" in output
        assert "3 tool uses" in output


# ── INTERACTION handling in streaming ─────────────────────────────


@pytest.mark.ci
class TestStreamingInteractionProcessing:
    """Tests for INTERACTION action handling in streaming context."""

    def test_process_interaction_processing_calls_input_collector(self):
        """INTERACTION PROCESSING calls input_collector and submits to broker."""
        import threading

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        action = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            messages="Choose an option",
            action_type="request_choice",
            input_data={
                "content": "Pick one",
                "content_type": "text",
                "choices": {"a": "Option A", "b": "Option B"},
                "default_choice": "a",
            },
        )
        actions = [action]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0

        # Track broker.submit calls using a real event loop running in a thread
        submit_calls = []

        async def fake_submit(action_id, user_input):
            submit_calls.append((action_id, user_input))

        mock_broker = MagicMock()
        mock_broker.submit = fake_submit
        ctx._broker = mock_broker

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        ctx._event_loop = loop

        # Mock input collector
        ctx._input_collector = MagicMock(return_value="a")

        try:
            ctx._process_actions()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=2)
            loop.close()

        assert ctx._processed_index == 1
        ctx._input_collector.assert_called_once_with(action, console)
        assert len(submit_calls) == 1
        assert submit_calls[0] == (action.action_id, "a")

    def test_process_interaction_success_skipped(self):
        """INTERACTION SUCCESS is skipped (not shown after live interaction)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(
                ActionRole.INTERACTION,
                ActionStatus.SUCCESS,
                messages="Saved successfully",
                output_data={"content": "Done!", "content_type": "text"},
            ),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0
        ctx._process_actions()

        assert ctx._processed_index == 1
        output = buf.getvalue()
        assert "Done!" not in output

    def test_process_interaction_processing_without_collector_skips(self):
        """INTERACTION PROCESSING without input_collector is skipped."""
        display = ActionHistoryDisplay()
        actions = [
            _make_action(
                ActionRole.INTERACTION,
                ActionStatus.PROCESSING,
                messages="Choose",
                input_data={"choices": {"a": "A"}},
            ),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._tick = 0
        ctx._input_collector = None

        ctx._process_actions()
        # Without input_collector, PROCESSING falls through to else branch
        assert ctx._processed_index == 1

    def test_flush_skips_interaction(self):
        """Flush skips INTERACTION actions (only shown during live interaction)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            _make_action(
                ActionRole.INTERACTION,
                ActionStatus.SUCCESS,
                messages="interaction result",
                output_data={"content": "Synced!", "content_type": "text"},
            ),
        ]
        ctx = InlineStreamingContext(actions, display)
        ctx._processed_index = 0
        ctx._flush_remaining_actions()

        assert ctx._processed_index == 1
        assert "Synced!" not in buf.getvalue()

    def test_set_event_loop_and_input_collector(self):
        """set_event_loop and set_input_collector store values correctly."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)

        loop = asyncio.new_event_loop()
        collector = MagicMock()
        try:
            ctx.set_event_loop(loop)
            ctx.set_input_collector(collector)
            assert ctx._event_loop is loop
            assert ctx._input_collector is collector
        finally:
            loop.close()

    def test_set_clear_header_callback(self):
        """set_clear_header_callback stores and clears the banner reprint hook."""
        display = ActionHistoryDisplay()
        ctx = InlineStreamingContext([], display)
        assert ctx._clear_header_callback is None

        callback = MagicMock()
        ctx.set_clear_header_callback(callback)
        assert ctx._clear_header_callback is callback

        ctx.set_clear_header_callback(None)
        assert ctx._clear_header_callback is None

    def test_history_reprint_skips_interaction(self):
        """Ctrl+O reprint skips INTERACTION actions (only shown during live interaction)."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        history_actions = [
            _make_action(
                ActionRole.INTERACTION,
                ActionStatus.SUCCESS,
                messages="Previous interaction",
                output_data={"content": "Confirmed", "content_type": "text"},
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="list_tables",
                input_data={"function_name": "list_tables"},
            ),
        ]

        ctx = InlineStreamingContext(
            [],
            display,
            history_turns=[("Previous question", history_actions)],
            current_user_message="Current question",
        )
        ctx._processed_index = 0
        ctx._reprint_history(verbose=False)

        output = buf.getvalue()
        assert "Previous question" in output
        assert "Confirmed" not in output
