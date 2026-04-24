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


# ── TUI path (LiveDisplayState injected) ─────────────────────────


@pytest.mark.ci
class TestTuiPath:
    """When ``live_state`` is injected, rolling-window updates go through it
    instead of Rich ``Live``."""

    def _make_subagent_tool_action(self, parent_id: str, name: str) -> ActionHistory:
        return _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="task_tool",
            messages=name,
            input_data={"function_name": name},
            parent_action_id=parent_id,
        )

    def test_subagent_block_has_header_and_tool_tail(self):
        """Rolling window pins header + tool tail for a single subagent."""
        from datus.cli.tui.live_display_state import TOOL_LINES_PER_GROUP, LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)

        ctx = InlineStreamingContext([], display, live_state=live_state)
        parent_id = "parent-tui"
        first = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_metrics",
            messages="task",
            parent_action_id=parent_id,
        )
        ctx._start_subagent_group(first, group_key=parent_id)

        for i in range(4):
            ctx._update_subagent_display(
                self._make_subagent_tool_action(parent_id, f"tool_{i}"),
                group_key=parent_id,
            )

        snap = live_state.snapshot()
        assert live_state.is_active() is True
        # Header + at most TOOL_LINES_PER_GROUP tool rows.
        assert live_state.line_count() == 1 + TOOL_LINES_PER_GROUP
        # Header line splits into "⏺ name" (cyan) + "(goal)" (default).
        header_segments = snap[0].segments
        styles = [style for style, _ in header_segments]
        assert "class:subagent-header-live" in styles
        name_text = "".join(txt for style, txt in header_segments if style == "class:subagent-header-live")
        assert "gen_metrics" in name_text
        # Tool tail comes after the header and shows the most-recent tools.
        last_line_text = "".join(seg for _, seg in snap[-1].segments)
        assert "tool_3" in last_line_text

    def test_parallel_subagents_each_have_own_header_and_tools(self):
        """Each active subagent renders as its own header + tool block."""
        from datus.cli.tui.live_display_state import TOOL_LINES_PER_GROUP, LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        ctx = InlineStreamingContext([], display, live_state=live_state)

        for parent_id, label in (("parent-A", "alpha"), ("parent-B", "beta")):
            first = _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                depth=1,
                action_type=label,
                messages="task",
                parent_action_id=parent_id,
            )
            ctx._start_subagent_group(first, group_key=parent_id)
            for i in range(3):
                ctx._update_subagent_display(
                    self._make_subagent_tool_action(parent_id, f"{label}_tool_{i}"),
                    group_key=parent_id,
                )

        snap = live_state.snapshot()
        # Each group = 1 header + TOOL_LINES_PER_GROUP tool rows; two groups.
        assert live_state.line_count() == 2 * (1 + TOOL_LINES_PER_GROUP)
        # Block order must be header1, tools1..., header2, tools2... — NOT
        # header1, header2, tools1, tools2.
        block_size = 1 + TOOL_LINES_PER_GROUP

        def _header_name(line) -> str:
            return "".join(txt for style, txt in line.segments if style == "class:subagent-header-live")

        assert "alpha" in _header_name(snap[0])
        assert "beta" in _header_name(snap[block_size])
        # alpha tool tail lands in rows 1..block_size-1 (immediately after header1).
        alpha_tail = " ".join("".join(seg for _, seg in line.segments) for line in snap[1:block_size])
        assert "alpha_tool_2" in alpha_tail
        assert "beta_tool_2" not in alpha_tail

    def test_end_subagent_group_clears_live_state_in_tui_mode(self):
        """When the group ends, the pinned region clears (no reprint-with-collapse)."""
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        ctx = InlineStreamingContext([], display, live_state=live_state)

        parent_id = "parent-end"
        first = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_metrics",
            messages="task",
            parent_action_id=parent_id,
        )
        ctx._start_subagent_group(first, group_key=parent_id)
        ctx._update_subagent_display(self._make_subagent_tool_action(parent_id, "tool_end"), group_key=parent_id)
        assert live_state.is_active() is True

        end_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            parent_action_id=parent_id,
            end_time=datetime.now() + timedelta(seconds=3),
        )
        ctx._end_subagent_group_by_key(parent_id, end_action)
        assert live_state.is_active() is False

    def test_tui_mode_does_not_create_rich_live(self):
        """Confirm Rich ``Live`` is never instantiated on the TUI path."""
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        ctx = InlineStreamingContext([], display, live_state=live_state)

        proc_action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="slow_tool",
            input_data={"function_name": "slow_tool"},
        )
        ctx._update_processing_live(proc_action)
        assert ctx._live is None
        assert ctx._subagent_live is None
        assert live_state.is_active() is True


# ── Streaming markdown (thinking_delta in TUI mode) ─────────────────


@pytest.mark.ci
class TestStreamingMarkdown:
    """Cover the thinking_delta → pinned region + scrollback pipeline."""

    def _make_delta(self, delta: str, action_id: str = "stream-1") -> ActionHistory:
        return _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.PROCESSING,
            action_type="thinking_delta",
            input_data={},
            output_data={"delta": delta, "accumulated": delta},
            action_id=action_id,
        )

    def _make_ctx(self):
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        ctx = InlineStreamingContext([], display, live_state=live_state)
        return ctx, live_state, buf

    def test_non_tui_mode_has_no_markdown_buffer(self):
        """Without a LiveDisplayState, streaming markdown stays disabled."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)
        assert ctx._markdown_buffer is None
        assert ctx._tui_mode is False

    def test_tui_mode_provisions_markdown_buffer(self):
        ctx, _live_state, _buf = self._make_ctx()
        assert ctx._markdown_buffer is not None
        assert ctx._markdown_stream_has_streamed is False

    def test_delta_populates_pinned_region(self):
        """A partial delta with no blank-line boundary stays in the tail."""
        ctx, live_state, buf = self._make_ctx()

        ctx._handle_thinking_delta(self._make_delta("hello "))
        ctx._handle_thinking_delta(self._make_delta("world"))

        assert ctx._markdown_buffer.get_tail() == "hello world"
        assert ctx._markdown_stream_has_streamed is True
        assert live_state.is_active() is True
        # Nothing stable yet, so the scrollback console remains empty.
        assert "hello" not in buf.getvalue()

    def test_paragraph_boundary_streams_to_scrollback(self):
        """Markdown segment end (``\\n\\n``) commits the prefix immediately.

        Long bodies must stream upward into the scrollback as soon as a
        paragraph closes, so the pinned region stays within the live
        budget. The outer ``_display_markdown_response`` is blocked from
        repainting because ``has_streamed_response`` is latched as soon
        as any segment has been handed off.
        """
        ctx, live_state, buf = self._make_ctx()

        ctx._handle_thinking_delta(self._make_delta("para one\n\nhead of two"))

        # The closed paragraph landed in the scrollback right away.
        assert buf.getvalue().count("para one") == 1
        # The unfinished second paragraph rides live in the pinned region.
        assert ctx._markdown_buffer.get_tail() == "head of two"
        assert live_state.is_active() is True
        # Spill latch is set so ``_render_final_response`` will skip the
        # one-shot repaint of the full body.
        assert ctx.has_streamed_response is True

        # Finalize drains the residual tail exactly once — no duplicate
        # of the already-committed paragraph.
        ctx._finalize_markdown_stream()
        assert ctx._markdown_buffer.has_tail() is False
        assert buf.getvalue().count("para one") == 1
        assert buf.getvalue().count("head of two") == 1

    def test_response_action_dedupes_when_stream_active(self):
        """Paired terminal action must not duplicate the streamed body.

        A delta that closes a paragraph with ``\\n\\n`` already lands in
        the scrollback during the stream; the paired response action
        arrives later and must be suppressed by ``render_main_action``
        so the body doesn't appear twice.
        """
        ctx, _live_state, buf = self._make_ctx()

        ctx._handle_thinking_delta(self._make_delta("streamed body\n\n", action_id="stream-xyz"))
        # Paragraph boundary already committed the body to the scrollback.
        assert buf.getvalue().count("streamed body") == 1
        assert "stream-xyz" in ctx._markdown_active_stream_ids

        # Terminal response arrives with the plain ``action_type="response"``
        # shape emitted by openai_compatible.py after
        # ``response.content_part.done``.
        response_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="response",
            messages="streamed body",
            output_data={"raw_output": "streamed body"},
            action_id="stream-xyz",
        )
        ctx._print_completed_action(response_action)

        # The body lands in the scrollback exactly once — via finalize.
        assert buf.getvalue().count("streamed body") == 1
        assert ctx.has_streamed_response is True
        # Consumed id is retained for future reprint de-dup; active set cleared.
        assert "stream-xyz" in ctx._markdown_stream_consumed_ids
        assert "stream-xyz" not in ctx._markdown_active_stream_ids
        assert ctx._markdown_stream_has_streamed is False

    def test_response_action_without_stream_renders_normally(self):
        """If no delta was seen (action_id not in active streams), render normally."""
        ctx, _live_state, buf = self._make_ctx()
        assert not ctx._markdown_active_stream_ids

        response_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="chat_response",
            messages="fresh body",
            output_data={"raw_output": "fresh body"},
            action_id="resp-never-streamed",
        )
        ctx._print_completed_action(response_action)
        # Markdown render path fired → body lands on the console.
        assert "fresh body" in buf.getvalue()
        assert "resp-never-streamed" not in ctx._markdown_stream_consumed_ids

    def test_finalize_flushes_leftover_tail(self):
        """Finalize flushes any pending tail even without a blank-line boundary."""
        ctx, live_state, buf = self._make_ctx()
        ctx._handle_thinking_delta(self._make_delta("tail only"))
        assert ctx._markdown_buffer.has_tail()

        ctx._finalize_markdown_stream()

        assert ctx._markdown_buffer.has_tail() is False
        assert ctx._markdown_stream_has_streamed is False
        assert live_state.is_active() is False
        assert "tail only" in buf.getvalue()

    def test_repaint_priority_processing_over_markdown(self):
        """Running tool blink takes the pinned region over markdown tail."""
        ctx, live_state, _buf = self._make_ctx()

        ctx._handle_thinking_delta(self._make_delta("md tail"))
        md_snap = live_state.snapshot()
        assert md_snap  # markdown tail occupying the region

        proc = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="busy_tool",
            input_data={"function_name": "busy_tool"},
        )
        ctx._update_processing_live(proc)
        proc_snap = live_state.snapshot()
        # One pinned line for the blinking processing tool.
        assert len(proc_snap) == 1
        assert any(style == "class:processing-live" for style, _ in proc_snap[0].segments)

        # Stop processing → markdown tail reclaims the region.
        ctx._stop_processing_live()
        restored = live_state.snapshot()
        assert restored  # still painted with the tail
        assert ctx._markdown_buffer.get_tail() == "md tail"

    def test_repaint_priority_subagent_over_markdown(self):
        """Active subagent block takes priority over streaming markdown tail."""
        ctx, live_state, _buf = self._make_ctx()

        ctx._handle_thinking_delta(self._make_delta("tail text"))
        assert live_state.is_active() is True

        parent_id = "parent-mix"
        first = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_metrics",
            messages="task",
            parent_action_id=parent_id,
        )
        ctx._start_subagent_group(first, group_key=parent_id)
        snap = live_state.snapshot()
        # At least one line is the subagent header (styled accordingly).
        has_subagent_header = any(
            any(style == "class:subagent-header-live" for style, _ in line.segments) for line in snap
        )
        assert has_subagent_header

    def test_wrapper_response_after_stream_is_dropped(self):
        """Regression: agent-node wrapper re-emissions must not re-render body.

        ``chat_agentic_node`` often emits a ``chat_response`` action *after*
        the underlying ``openai_compatible`` ``"response"`` action — same
        turn, same body, different id. The first one drives finalize +
        dedup; the wrapper falls into the ``_turn_finalized`` /
        ``_stream_body_finalized`` branch and is silently dropped.
        """
        ctx, _live_state, buf = self._make_ctx()
        ctx._handle_thinking_delta(self._make_delta("final body text\n\n", action_id="stream-1"))
        # Paragraph boundary already committed the body during the stream.
        assert buf.getvalue().count("final body text") == 1

        # First completion — the openai_compatible "response" (plain type) —
        # triggers finalize; the residual tail is empty so no extra print.
        ctx._print_completed_action(
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="response",
                messages="final body text",
                output_data={"raw_output": "final body text"},
                action_id="stream-1",
            )
        )
        assert ctx._turn_finalized is True
        assert buf.getvalue().count("final body text") == 1

        # Agent-node wrapper emission — same turn, *different* id, typical
        # ``*_response`` action_type. Would previously re-render the body.
        ctx._print_completed_action(
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="chat_response",
                messages="final body text",
                output_data={"raw_output": "final body text"},
                action_id="wrapper-xyz",
            )
        )
        assert buf.getvalue().count("final body text") == 1
        assert "wrapper-xyz" in ctx._markdown_stream_consumed_ids

    def test_unterminated_tail_with_paired_response_prints_once(self):
        """Regression: last paragraph without trailing ``\\n\\n`` must not duplicate.

        A paragraph that closes mid-stream commits straight to the
        scrollback; whatever follows without a trailing ``\\n\\n`` rides in
        the tail until ``_finalize_markdown_stream`` drains it. The
        combined output must contain each span exactly once.
        """
        ctx, _live_state, buf = self._make_ctx()
        stream_id = "stream-unterminated"

        ctx._handle_thinking_delta(self._make_delta("Please let me know\n\n", action_id=stream_id))
        ctx._handle_thinking_delta(self._make_delta("what you need!", action_id=stream_id))
        # Closed paragraph is already in the scrollback; unfinished tail
        # still rides live in the pinned region.
        assert ctx._markdown_buffer.get_tail() == "what you need!"
        assert buf.getvalue().count("Please let me know") == 1
        assert "what you need!" not in buf.getvalue()

        # Paired response arrives with the same id and plain "response" type.
        response_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="response",
            messages="",
            output_data={"raw_output": "Please let me know\n\nwhat you need!"},
            action_id=stream_id,
        )
        ctx._print_completed_action(response_action)
        output = buf.getvalue()
        # Both parts of the body appear exactly once across the whole
        # scrollback — mid-stream commit for paragraph #1, finalize for
        # the unterminated tail of paragraph #2.
        assert output.count("what you need!") == 1
        assert output.count("Please let me know") == 1

    def test_incomplete_table_fallback_scope(self):
        """Fallback is narrow: only header / header+separator trigger plain text.

        Rich can already draw a correct box table from 3+ pipe rows, so we
        *stop* falling back once the first body row arrives and let the
        live pinned region show the real box-drawing rendering mid-stream.
        """
        from datus.cli.action_display.streaming import _tail_has_incomplete_table

        # Only the header → fall back (Rich would lose the single pipe line).
        assert _tail_has_incomplete_table("| a | b |") is True
        # Header + separator → still fall back (Rich draws a broken grid).
        assert _tail_has_incomplete_table("| a | b |\n| - | - |") is True
        # Header + separator + body row → Rich can draw this correctly, so
        # we hand it over even mid-stream for the "live markdown" feel.
        assert _tail_has_incomplete_table("| a | b |\n| - | - |\n| 1 | 2 |\n") is False
        # Closed with blank line → stable segment, not a fallback decision.
        assert _tail_has_incomplete_table("| a | b |\n| - | - |\n| 1 | 2 |\n\n") is False
        assert _tail_has_incomplete_table("plain prose\n") is False
        assert _tail_has_incomplete_table("") is False

        ctx, live_state, _buf = self._make_ctx()
        # 3-row table tail — no fallback; pinned region carries Rich's
        # ANSI output. We don't assert on exact glyphs (ANSI color codes
        # vary) but the numeric values must remain visible.
        ctx._handle_thinking_delta(self._make_delta("| x | y |\n| - | - |\n| 1 | 2 |"))
        assert ctx._markdown_buffer.has_tail()
        assert live_state.is_active() is True
        snap = live_state.snapshot()
        joined = "\n".join("".join(txt for _, txt in line.segments) for line in snap)
        assert "1" in joined and "2" in joined

    def test_verbose_frozen_forces_clear(self):
        """Frozen verbose snapshot owns the screen — pinned region must clear."""
        ctx, live_state, _buf = self._make_ctx()
        ctx._handle_thinking_delta(self._make_delta("tail text"))
        assert live_state.is_active() is True

        ctx._verbose_frozen = True
        ctx._repaint_live()
        assert live_state.is_active() is False

    def test_process_deltas_detects_clear_and_finalizes(self):
        """``_process_deltas`` must finalize when the caller clears the queue.

        The chat_commands pipeline calls ``streaming_deltas.clear()`` at
        every message boundary. The streaming context — running on the
        refresh daemon — should notice the queue shrank, flush the
        accumulator to the scrollback, and reset its cursor so follow-up
        deltas (next message) start fresh.
        """
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        deltas: list[ActionHistory] = []
        ctx = InlineStreamingContext([], display, live_state=live_state, streaming_deltas=deltas)

        # Push two deltas through the public queue + run the pump.
        deltas.append(self._make_delta("part a "))
        deltas.append(self._make_delta("part b"))
        ctx._process_deltas()
        assert ctx._delta_processed_index == 2
        assert ctx._markdown_buffer.get_tail() == "part a part b"
        assert "part a" not in buf.getvalue()

        # Caller signals message boundary by clearing the queue.
        deltas.clear()
        ctx._process_deltas()

        # Body landed in the scrollback exactly once; cursor reset.
        assert buf.getvalue().count("part a part b") == 1
        assert ctx._delta_processed_index == 0
        assert ctx._markdown_buffer.has_tail() is False
        assert ctx.has_streamed_response is True

    def test_has_streamed_response_is_false_without_deltas(self):
        """Contexts that never painted a delta must not latch the flag.

        Otherwise the outer ``_render_final_response`` would skip the
        ``_display_markdown_response`` step on *every* turn, including
        non-streaming providers that rely on it for their final output.
        """
        ctx, _live_state, _buf = self._make_ctx()
        assert ctx.has_streamed_response is False
        # Finalize with an empty buffer must not set the latch either.
        ctx._finalize_markdown_stream()
        assert ctx.has_streamed_response is False
