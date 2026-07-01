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
from datus.cli.action_display.markdown_stream import MarkdownStreamBuffer
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
            # task PROCESSING anchors the group (Path A). Required for
            # depth>0 actions below to slot in — orphan depth>0 actions
            # are warned and dropped under the cleaned-up contract.
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                action_id=parent_id,
                action_type="task",
                messages="task(gen_sql, revenue?)",
                input_data={"type": "gen_sql", "prompt": "revenue?"},
            ),
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

    def test_sync_outer_task_processing_starts_subagent_group(self):
        """Outer ``task`` PROCESSING action seeds the subagent group with
        the correct ``input["type"]`` label, so depth=1 inner actions
        slot under it and the rendered header is ``gen_sql_summary``,
        not the literal ``task``.

        Regression guard: before the streaming fix, sync replay simply
        skipped TOOL PROCESSING entries (incl. the outer task) and the
        first depth=1 inner action created the group with whatever
        action_type it carried — losing the user-meaningful label.
        """
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        call_id = "call-task-1"

        actions = [
            # 1. outer task PROCESSING — should seed the group
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                action_id=call_id,
                action_type="task",
                messages="task(gen_sql_summary, orders.sql)",
                input_data={
                    "type": "gen_sql_summary",
                    "function_name": "task",
                    "_task_description": "orders.sql",
                },
            ),
            # 2. first inner depth=1 action parents itself to call_id
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="read_query",
                messages="read_query",
                input_data={"function_name": "read_query"},
                parent_action_id=call_id,
            ),
            # 3. subagent_complete closes the group
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="complete",
                output_data={"subagent_type": "gen_sql_summary", "tool_count": 1},
                parent_action_id=call_id,
            ),
        ]

        ctx = InlineStreamingContext(actions, display, sync_mode=True)
        ctx.run_sync()

        output = buf.getvalue()
        # Header label comes from input["type"], NOT the literal "task".
        assert "gen_sql_summary" in output
        assert "orders.sql" in output
        # No bare "○ 🔧 task" / standalone "task(" line — the outer became
        # the group header instead of a flat PROCESSING tool line.
        assert "Done" in output
        assert "1 tool uses" in output

    def test_sync_skips_processing_in_subagent(self):
        """Sync mode skips PROCESSING tools inside subagent groups."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        parent_id = "parent-456"

        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                action_id=parent_id,
                action_type="task",
                messages="task(gen_sql, test)",
                input_data={"type": "gen_sql", "prompt": "test"},
            ),
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


@pytest.mark.ci
class TestSubagentTokenCounter:
    """The pinned subagent header / collapsed Done line surface the subagent's
    cumulative token total (+ cached) fed by depth>0 ``token_usage`` actions."""

    def _ctx(self, actions=None):
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        return InlineStreamingContext(actions or [], display, sync_mode=True)

    def _usage_action(self, parent_id, input_tokens, output_tokens, cached):
        return _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="token_usage",
            output_data={
                "cumulative": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached,
                }
            },
            parent_action_id=parent_id,
        )

    def test_apply_subagent_usage_folds_totals(self):
        group = {"token_input": 0, "token_output": 0, "token_cached": 0, "actions": []}
        consumed = InlineStreamingContext._apply_subagent_usage(group, self._usage_action("g", 12603, 2048, 8192))
        assert consumed is True
        assert group["token_input"] == 12603
        assert group["token_output"] == 2048
        assert group["token_cached"] == 8192

    def test_apply_subagent_usage_ignores_non_usage(self):
        group = {"token_input": 5, "token_output": 3, "token_cached": 1}
        action = _make_action(ActionRole.TOOL, ActionStatus.SUCCESS, depth=1, action_type="read_query")
        assert InlineStreamingContext._apply_subagent_usage(group, action) is False
        # Non-usage action must not perturb the counters.
        assert group["token_input"] == 5
        assert group["token_output"] == 3
        assert group["token_cached"] == 1

    def test_usage_action_updates_group_without_buffering_a_row(self):
        """A token_usage action must update the group's counters but never be
        buffered into ``actions`` (which would draw a bogus tool row), and must
        not bump the tool count."""
        ctx = self._ctx()
        gid = "call-1"
        first = _make_action(
            ActionRole.TOOL, ActionStatus.PROCESSING, action_id=gid, action_type="task", input_data={"type": "gen_sql"}
        )
        ctx._start_subagent_group_sync(first, gid)
        ctx._update_subagent_display_sync(self._usage_action(gid, 9000, 1500, 4000), gid)
        group = ctx._subagent_groups[gid]
        assert group["token_input"] == 9000
        assert group["token_output"] == 1500
        assert group["token_cached"] == 4000
        assert group["actions"] == []  # usage never buffered as a render row
        assert group["tool_count"] == 0  # usage is not a tool

    def test_header_segments_include_token_counter(self):
        first = _make_action(
            ActionRole.TOOL, ActionStatus.PROCESSING, action_id="g", action_type="task", input_data={"type": "gen_sql"}
        )
        segments = InlineStreamingContext._build_subagent_header_segments(
            first, token_input=12603, token_output=2048, token_cached=8192
        )
        joined = "".join(text for _style, text in segments)
        assert "gen_sql" in joined
        assert "↑12K(8.0K) ↓2.0K" in joined

    def test_header_segments_omit_counter_when_no_tokens(self):
        first = _make_action(
            ActionRole.TOOL, ActionStatus.PROCESSING, action_id="g", action_type="task", input_data={"type": "gen_sql"}
        )
        segments = InlineStreamingContext._build_subagent_header_segments(first)
        joined = "".join(text for _style, text in segments)
        assert "↑" not in joined
        assert "↓" not in joined

    def test_token_suffix_formatting(self):
        assert InlineStreamingContext._subagent_token_suffix(0, 0, 0) == ""
        assert InlineStreamingContext._subagent_token_suffix(2048, 512, 0) == " · ↑2.0K ↓0.5K"
        assert InlineStreamingContext._subagent_token_suffix(12603, 2048, 8192) == " · ↑12K(8.0K) ↓2.0K"


@pytest.mark.ci
class TestPathAGate:
    """Regression coverage for the task-PROCESSING anchor contract.

    The cleaned-up grouping model in
    :class:`datus.cli.action_display.streaming.InlineStreamingContext`
    requires every subagent group to be seeded by a depth=0 ``task``
    PROCESSING action; depth>0 actions slot in via ``parent_action_id``
    matching the seed's ``action_id``. The original bug was that
    Path A's gate only accepted the *direct* input layout, so
    model-emitted task actions (wrapped layout) silently fell through
    to a now-removed depth>0 fallback path and produced double-wrapped
    headers like ``⏺ load_skill(Tool call: load_skill(...))``.
    """

    def _build_ctx(self, actions):
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext(actions, display, sync_mode=True)
        return ctx, buf

    def test_wrapped_layout_task_processing_anchors_group(self):
        """Model adapters emit task with ``input={"function_name": "task", "arguments": {...}}``.

        The gate must accept this layout and anchor the group with the
        subagent_type extracted from ``arguments.type`` — not bypass it
        and let the first depth>0 child action seed a misnamed group.
        Direct regression for the user-reported ``load_skill`` mis-render.
        """
        call_id = "call-wrapped-1"
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                action_id=call_id,
                action_type="task",
                messages='Tool call: task(\'{"type": "gen_table"}...\')',
                input_data={
                    "function_name": "task",
                    "arguments": {
                        "type": "gen_table",
                        "prompt": "Build a sales rollup",
                        "description": "sales rollup",
                    },
                },
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="load_skill",
                messages='Tool call: load_skill(\'{"skill_name": "gen-table"}\')',
                input_data={"function_name": "load_skill", "arguments": {"skill_name": "gen-table"}},
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="complete",
                output_data={"subagent_type": "gen_table", "tool_count": 1},
                parent_action_id=call_id,
            ),
        ]
        ctx, buf = self._build_ctx(actions)
        ctx.run_sync()
        output = buf.getvalue()

        # Header pulls subagent_type from arguments.type, NOT from the
        # inner load_skill action.
        assert "gen_table" in output
        assert "sales rollup" in output
        # The inner tool name must NOT appear in the header position —
        # i.e. there is no ``load_skill(Tool call: ...)`` double-wrap.
        assert "Tool call: load_skill" not in output
        assert "1 tool uses" in output

    def test_direct_layout_task_processing_anchors_group(self):
        """Bootstrap path builds task actions with the direct layout
        (``input={"type": ..., "prompt": ..., "description": ...}``).
        The gate must accept this layout identically to the wrapped case.
        """
        call_id = "call-direct-1"
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                action_id=call_id,
                action_type="task",
                messages="task(gen_sql_summary)",
                input_data={
                    "type": "gen_sql_summary",
                    "prompt": "summarise orders.sql",
                    "description": "orders.sql",
                },
            ),
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="read_query",
                messages="read_query",
                input_data={"function_name": "read_query"},
                parent_action_id=call_id,
            ),
            _make_action(
                ActionRole.SYSTEM,
                ActionStatus.SUCCESS,
                depth=1,
                action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
                messages="complete",
                parent_action_id=call_id,
            ),
        ]
        ctx, buf = self._build_ctx(actions)
        ctx.run_sync()
        output = buf.getvalue()
        assert "gen_sql_summary" in output
        assert "orders.sql" in output

    def test_orphan_depth_gt_zero_action_is_warned_and_dropped(self, caplog):
        """A depth>0 action without a preceding task PROCESSING anchor
        must be warned about and dropped from the visible scrollback —
        creating an on-the-fly group from such an action is what produced
        the original double-wrap regression.
        """
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                depth=1,
                action_type="load_skill",
                messages="load_skill('gen-table')",
                input_data={"function_name": "load_skill", "arguments": {"skill_name": "gen-table"}},
                parent_action_id="no-anchor-call-id",
            ),
        ]
        ctx, buf = self._build_ctx(actions)
        with caplog.at_level("WARNING", logger="datus.cli.action_display.streaming"):
            ctx.run_sync()
        output = buf.getvalue()

        # Action is dropped from rendered output.
        assert "load_skill" not in output
        assert "Done" not in output
        # No group was synthesised.
        assert ctx._subagent_groups == {}
        # Warning fires with the diagnostic kwargs.
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warning_records, "expected an orphan-depth>0 warning"
        msg = warning_records[-1].getMessage()
        assert "orphan depth>0 action" in msg
        assert "parent_action_id=no-anchor-call-id" in msg
        assert "action_type=load_skill" in msg

    @pytest.mark.parametrize(
        "input_data,expected_type,expected_goal",
        [
            (
                {"type": "explore", "description": "schema discovery"},
                "explore",
                "schema discovery",
            ),
            (
                {
                    "function_name": "task",
                    "arguments": {"type": "explore", "description": "schema discovery"},
                },
                "explore",
                "schema discovery",
            ),
            (
                {
                    "function_name": "task",
                    "arguments": '{"type": "explore", "description": "schema discovery"}',
                },
                "explore",
                "schema discovery",
            ),
        ],
        ids=["direct", "wrapped_dict", "wrapped_json_string"],
    )
    def test_tui_live_header_segments_use_parsed_task_type(self, input_data, expected_type, expected_goal):
        """TUI pinned-region header (``_build_subagent_header_segments``)
        must canonicalise both layouts through :func:`parse_task_tool_input`,
        matching ``ActionRenderer.render_subagent_header``. Regression for
        the user-reported live header rendering as
        ``⏺ task(Tool call: task('...'))`` instead of ``⏺ explore(...)`` —
        the segment builder previously only handled the direct layout.
        """
        first_action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=0,
            action_type="task",
            messages='Tool call: task(\'{"type": "explore"...}...\')',
            input_data=input_data,
        )
        segments = InlineStreamingContext._build_subagent_header_segments(first_action)
        # Segment 0 carries the cyan name; segment 1 (if present) carries
        # the goal. Together they form the visible ⏺ header.
        rendered = "".join(text for _style, text in segments)
        assert f"\u23fa {expected_type}" in rendered
        assert f"({expected_goal})" in rendered
        # The model adapter's "Tool call: task(..." messages must NOT leak
        # into the goal — that was the original double-wrap symptom.
        assert "Tool call:" not in rendered


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

    def test_reprint_preserves_user_insert_actions(self):
        """Ctrl+O verbose toggle reprints history from ``self.actions``.

        The initial USER request (depth=0, original action_type) is
        dropped because it's already echoed by the user-header at the top
        of the turn. But ``user_insert`` actions — text the user typed
        mid-run via TUI / API — must survive the reprint, otherwise the
        verbose snapshot would silently lose the user's own mid-stream
        contributions.
        """
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        actions = [
            # 1) Initial request — should NOT appear in reprint body
            # (the user-header at the top already shows it).
            _make_action(
                ActionRole.USER,
                ActionStatus.PROCESSING,
                action_type="chat_agentic_node_request",
                messages="User: original question",
                input_data={"user_message": "original question"},
            ),
            # 2) A tool call mid-turn.
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                messages="ok",
                input_data={"function_name": "list_tables"},
            ),
            # 3) The mid-run user injection that must survive Ctrl+O.
            _make_action(
                ActionRole.USER,
                ActionStatus.SUCCESS,
                action_type="user_insert",
                messages="also count the rows",
                input_data={"user_message": "also count the rows", "source": "mid_run_insert"},
                output_data={"user_message": "also count the rows"},
            ),
        ]

        ctx = InlineStreamingContext(actions, display, current_user_message="original question")
        ctx._processed_index = len(actions)
        ctx._verbose = True

        ctx._reprint_history(verbose=True)

        output = buf.getvalue()
        # The verbose reprint must include the mid-run injection ...
        assert "also count the rows" in output
        # ... but not echo the initial request twice (its text only shows
        # up once, in the user header at the top — which we also emit).
        assert output.count("original question") == 1

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

    def test_processing_frame_restored_after_ask_approval(self):
        """A tool pinned as the running frame is restored after an ASK approval.

        Regression: ``bash`` (bash) is ASK-gated, so a permission
        INTERACTION fires while the tool's PROCESSING frame is pinned. The
        interaction handler clears the frame to draw the prompt; without a
        restore the (possibly long) bash then runs with a blank pinned region
        and only appears on completion.
        """
        import threading

        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)

        interaction = _make_action(
            ActionRole.INTERACTION,
            ActionStatus.PROCESSING,
            messages="Allow bash?",
            action_type="request_choice",
            input_data={"content": "Allow?", "choices": {"y": "Yes", "n": "No"}, "default_choice": "y"},
        )
        ctx = InlineStreamingContext([interaction], display, live_state=live_state)
        ctx._processed_index = 0
        ctx._tick = 0

        # Simulate the bash tool already pinned as the running frame.
        bash_action = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            messages="Tool call: bash",
            action_type="bash",
            input_data={"function_name": "bash", "arguments": {"command": "sleep 5"}},
        )
        ctx._processing_action = bash_action

        async def fake_submit(action_id, user_input):
            return None

        mock_broker = MagicMock()
        mock_broker.submit = fake_submit
        ctx._broker = mock_broker

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        ctx._event_loop = loop
        ctx._input_collector = MagicMock(return_value="y")

        try:
            ctx._process_actions()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=2)
            loop.close()

        # Frame restored to the same bash tool so live progress keeps showing.
        assert ctx._processing_action is bash_action

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
        # first_action is the outer task PROCESSING under the cleaned-up
        # grouping contract — the subagent_type comes from ``input["type"]``,
        # not from a depth=1 inner action's ``action_type``.
        first = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=0,
            action_id=parent_id,
            action_type="task",
            messages="task(gen_metrics)",
            input_data={"type": "gen_metrics", "prompt": "compute base metrics"},
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
            # Each subagent group is seeded by its own outer task
            # PROCESSING action — same shape as production Path A.
            first = _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                depth=0,
                action_id=parent_id,
                action_type="task",
                messages=f"task({label})",
                input_data={"type": label, "prompt": f"{label} prompt"},
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


# ── Tool PROCESSING lifecycle (async path) ──────────────────────────


@pytest.mark.ci
class TestToolProcessingLifecycle:
    """Cover the async ``_process_actions`` path for TOOL PROCESSING entries,
    both at depth=0 (top-level) and depth>0 (inside subagent groups)."""

    def test_top_level_processing_pins_frame_and_clears_on_success(self):
        """A depth=0 TOOL PROCESSING pins a frame; the paired SUCCESS clears it."""
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)

        call_id = "tool-call-xyz"
        actions = [
            _make_action(
                ActionRole.TOOL,
                ActionStatus.PROCESSING,
                action_id=call_id,
                messages="db_describe",
                input_data={"function_name": "db_describe", "arguments": {"table": "orders"}},
            ),
        ]
        ctx = InlineStreamingContext(actions, display, live_state=live_state)

        # First pass: PROCESSING entry -> pinned frame; index advances so the
        # paired SUCCESS action (same action_id) can be consumed on arrival.
        ctx._process_actions()
        assert ctx._processing_action is actions[0]
        assert live_state.is_active() is True
        assert ctx._processed_index == 1

        # SUCCESS arrives with the same action_id; streaming advances and clears pinned.
        actions.append(
            _make_action(
                ActionRole.TOOL,
                ActionStatus.SUCCESS,
                action_id=call_id,
                messages="db_describe",
                input_data={"function_name": "db_describe", "arguments": {"table": "orders"}},
                output_data={"raw_output": {"rows": []}},
                end_time=datetime.now() + timedelta(seconds=1),
            )
        )
        ctx._process_actions()
        assert ctx._processing_action is None
        assert live_state.is_active() is False
        assert ctx._processed_index == 2

    def test_subagent_processing_sets_group_slot_and_paints_blink_row(self):
        """A depth>0 TOOL PROCESSING sets ``processing_action`` on its group
        and is rendered as an extra blinking row at the tail of the group."""
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        parent_id = "parent-proc"

        # Existing subagent group already open with one completed tool.
        ctx = InlineStreamingContext([], display, live_state=live_state)
        seed = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="gen_metrics",
            messages="task",
            input_data={"function_name": "db_sample"},
            parent_action_id=parent_id,
            end_time=datetime.now(),
        )
        ctx._start_subagent_group(seed, group_key=parent_id)
        ctx._update_subagent_display(seed, group_key=parent_id)

        # A new tool is now running inside the same subagent.
        call_id = "inner-call"
        processing = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_metrics",
            action_id=call_id,
            messages="db_describe",
            input_data={"function_name": "db_describe", "arguments": {"table": "orders"}},
            parent_action_id=parent_id,
        )
        ctx.actions.append(processing)
        ctx._process_actions()

        assert ctx._subagent_groups[parent_id]["processing_action"] is processing
        assert ctx._processing_action is None  # Top-level pinned must not be hijacked.

        lines = ctx._build_subagent_live_lines()
        plain = [" ".join(seg for _, seg in line.segments) for line in lines]
        joined = " | ".join(plain)
        assert "db_describe" in joined
        # Explicitly assert the blink row uses the processing-live style.
        styles = {style for line in lines for style, _ in line.segments}
        assert "class:processing-live" in styles

        # Paired SUCCESS arrives -> pending slot clears, completed tool joins the list.
        success = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="gen_metrics",
            action_id=call_id,
            messages="db_describe",
            input_data={"function_name": "db_describe", "arguments": {"table": "orders"}},
            output_data={"raw_output": {"rows": 10}},
            parent_action_id=parent_id,
            end_time=datetime.now() + timedelta(seconds=1),
        )
        ctx.actions.append(success)
        ctx._process_actions()

        group = ctx._subagent_groups[parent_id]
        assert group["processing_action"] is None
        assert any(a.action_id == call_id and a.status == ActionStatus.SUCCESS for a in group["actions"])
        lines_after = ctx._build_subagent_live_lines()
        styles_after = {style for line in lines_after for style, _ in line.segments}
        assert "class:processing-live" not in styles_after

    def test_subagent_block_height_stays_two_when_processing_active(self):
        """With N completed tools + running row, the block is exactly
        ``1 header + TOOL_LINES_PER_GROUP`` rows (running row occupies a slot)."""
        from datus.cli.tui.live_display_state import TOOL_LINES_PER_GROUP, LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        ctx = InlineStreamingContext([], display, live_state=live_state)

        parent_id = "parent-height"
        first = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_metrics",
            parent_action_id=parent_id,
        )
        ctx._start_subagent_group(first, group_key=parent_id)
        for i in range(4):
            ctx._update_subagent_display(
                _make_action(
                    ActionRole.TOOL,
                    ActionStatus.SUCCESS,
                    depth=1,
                    action_type="gen_metrics",
                    input_data={"function_name": f"tool_{i}"},
                    parent_action_id=parent_id,
                    end_time=datetime.now(),
                ),
                group_key=parent_id,
            )
        ctx._subagent_groups[parent_id]["processing_action"] = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_metrics",
            input_data={"function_name": "db_describe", "arguments": {"table": "orders"}},
            parent_action_id=parent_id,
        )

        lines = ctx._build_subagent_live_lines()
        assert len(lines) == 1 + TOOL_LINES_PER_GROUP  # header + exactly 2 content rows
        # Last content row is the running indicator (processing-live style).
        assert any(style == "class:processing-live" for style, _ in lines[-1].segments)
        # Previous content row is the most recent completed tool.
        prev_text = "".join(seg for _, seg in lines[-2].segments)
        assert "tool_3" in prev_text

    def test_top_level_event_appends_blank_line_separator(self):
        """Each independent depth=0 action prints a trailing blank line so
        adjacent events are visually separated in the scrollback."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)

        success = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=0,
            input_data={"function_name": "list_tables"},
            end_time=datetime.now() + timedelta(milliseconds=200),
        )
        ctx._print_completed_action(success)
        output = buf.getvalue()
        # The tool row renders as `<status-dot> 🔧 ... \n  └─ …\n` and we now
        # append an extra blank line at the end.
        assert output.endswith("\n\n")

    def test_subagent_renderable_includes_processing_row(self):
        """Non-TUI Rich ``Live`` path also appends the blinking row per group."""
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)
        ctx = InlineStreamingContext([], display)

        parent_id = "parent-rich"
        first = _make_action(
            ActionRole.TOOL,
            ActionStatus.SUCCESS,
            depth=1,
            action_type="gen_metrics",
            input_data={"function_name": "list_tables"},
            parent_action_id=parent_id,
            end_time=datetime.now(),
        )
        ctx._start_subagent_group(first, group_key=parent_id)
        ctx._update_subagent_display(first, group_key=parent_id)
        ctx._subagent_groups[parent_id]["processing_action"] = _make_action(
            ActionRole.TOOL,
            ActionStatus.PROCESSING,
            depth=1,
            action_type="gen_metrics",
            input_data={"function_name": "db_describe", "arguments": {"table": "orders"}},
            parent_action_id=parent_id,
        )
        group = ctx._build_subagent_groups_renderable()
        rendered = " | ".join(str(r) for r in getattr(group, "renderables", []))
        assert "db_describe" in rendered


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
        assert isinstance(ctx._markdown_buffer, MarkdownStreamBuffer)
        assert ctx._markdown_stream_has_streamed is False

    def test_delta_populates_pinned_region(self):
        """A partial delta with no blank-line boundary stays in the tail."""
        ctx, live_state, buf = self._make_ctx()

        ctx._handle_thinking_delta(self._make_delta("hello "))
        ctx._handle_thinking_delta(self._make_delta("world"))

        assert ctx._markdown_buffer.get_tail() == "hello world"
        assert ctx._markdown_stream_has_streamed is True
        assert live_state.is_active() is True
        joined = "\n".join("".join(txt for _, txt in line.segments) for line in live_state.snapshot())
        assert "hello world" in joined
        assert "⏺" not in joined
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
        assert "⏺" not in buf.getvalue()
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
        assert "⏺" not in buf.getvalue()

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
        assert "⏺" not in buf.getvalue()

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
        assert any(style == "class:processing-live-top" for style, _ in proc_snap[0].segments)

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

    def test_plan_preview_renders_after_streamed_preamble(self):
        """Regression: ``broker.send`` plan_preview must NOT be swallowed by the same-turn dedup latch.

        Reproduces the original report: LLM streams a "let me confirm" preamble,
        then calls ``confirm_plan`` which pushes the plan markdown via
        ``broker.send``. That action is also ASSISTANT/SUCCESS/depth=0 in TUI
        mode, so the dedupe condition would have dropped it without the
        ``action_type != "plan_preview"`` carve-out.
        """
        ctx, _live_state, buf = self._make_ctx()
        # Step 1 — LLM streams preamble, finalize sets ``_turn_finalized``.
        ctx._handle_thinking_delta(self._make_delta("Let me confirm the plan.\n\n", action_id="stream-1"))
        ctx._print_completed_action(
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="response",
                messages="Let me confirm the plan.",
                output_data={"raw_output": "Let me confirm the plan."},
                action_id="stream-1",
            )
        )
        assert ctx._turn_finalized is True

        # Step 2 — ``broker.send`` pushes the plan preview content.
        plan_md = "# Final Plan\n\nStep A\nStep B"
        ctx._print_completed_action(
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="plan_preview",
                messages=plan_md,
                input_data={"content": plan_md, "content_type": "markdown"},
                output_data={"content": plan_md, "content_type": "markdown"},
                action_id="preview-1",
            )
        )

        # The plan body must actually land in the scrollback this time.
        assert "Final Plan" in buf.getvalue()
        assert "Step A" in buf.getvalue()
        assert "Step B" in buf.getvalue()

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

    def test_process_deltas_finalizes_on_terminal_response(self):
        """Per-message bucket: terminal response triggers finalize + bucket drop.

        chat_commands routes deltas into ``streaming_deltas[action_id]``
        and never clears the dict. The streaming context drains each
        bucket forward via a per-bucket cursor and finalizes the body
        when the matching terminal ``response`` action is processed by
        :meth:`_print_completed_action`. Replaces the legacy
        ``streaming_deltas.clear()`` boundary signal.
        """
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        deltas: dict[str, list[ActionHistory]] = {}
        ctx = InlineStreamingContext([], display, live_state=live_state, streaming_deltas=deltas)

        stream_id = "stream-msg-1"
        deltas.setdefault(stream_id, []).append(self._make_delta("part a ", action_id=stream_id))
        deltas.setdefault(stream_id, []).append(self._make_delta("part b", action_id=stream_id))
        ctx._process_deltas()

        assert ctx._delta_cursors[stream_id] == 2
        assert ctx._active_message_id == stream_id
        assert ctx._markdown_buffer.get_tail() == "part a part b"
        assert "part a" not in buf.getvalue()

        # Terminal response arrives — drains any trailing deltas (none
        # here), finalizes the tail, drops the bucket.
        response_action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="response",
            messages="",
            output_data={"raw_output": "part a part b"},
            action_id=stream_id,
        )
        ctx._print_completed_action(response_action)

        assert buf.getvalue().count("part a part b") == 1
        assert stream_id not in deltas
        assert stream_id not in ctx._delta_cursors
        assert ctx._active_message_id is None
        assert ctx._markdown_buffer.has_tail() is False
        assert ctx.has_streamed_response is True

    def test_process_deltas_handles_concurrent_buckets(self):
        """Mid-tick race: A's deltas + terminal land while B's deltas accumulate.

        Producer can append to a new bucket B before the consumer
        processes A's terminal. The drain must pick up both buckets in
        insertion order, finalize A's tail when switching to B, and
        leave B intact for its own terminal-response handling.
        """
        from datus.cli.tui.live_display_state import LiveDisplayState

        buf = StringIO()
        console = Console(file=buf, no_color=True)
        live_state = LiveDisplayState()
        display = ActionHistoryDisplay(console, live_state=live_state)
        deltas: dict[str, list[ActionHistory]] = {}
        ctx = InlineStreamingContext([], display, live_state=live_state, streaming_deltas=deltas)

        # Bucket A — message #1.
        a_id = "stream-A"
        deltas.setdefault(a_id, []).append(self._make_delta("alpha\n\n", action_id=a_id))
        # Bucket B — message #2 starts before A's terminal is observed.
        b_id = "stream-B"
        deltas.setdefault(b_id, []).append(self._make_delta("beta", action_id=b_id))

        ctx._process_deltas()

        # Both buckets advanced their cursors; switch from A → B
        # finalized A's tail (paragraph closed, so already in scrollback)
        # and now active is B.
        assert ctx._delta_cursors[a_id] == 1
        assert ctx._delta_cursors[b_id] == 1
        assert ctx._active_message_id == b_id
        assert "alpha" in buf.getvalue()

        # Terminal A arrives later; bucket A is already fully drained,
        # so the dedup branch just drops the bucket. The active body
        # the user is still watching is B's.
        ctx._print_completed_action(
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="response",
                output_data={"raw_output": "alpha"},
                action_id=a_id,
            )
        )
        assert a_id not in deltas
        # Terminal B arrives and finalizes its own tail.
        ctx._print_completed_action(
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="response",
                output_data={"raw_output": "beta"},
                action_id=b_id,
            )
        )
        assert b_id not in deltas
        assert "beta" in buf.getvalue()
        assert buf.getvalue().count("alpha") == 1
        assert buf.getvalue().count("beta") == 1

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


@pytest.mark.ci
class TestCompactRendering:
    """compact_progress / compact_summary rendering: in-progress hint vs.
    clear-screen + summary panel."""

    def _ctx(self, buf):
        console = Console(file=buf, no_color=True, width=100)
        display = ActionHistoryDisplay(console)
        return InlineStreamingContext([], display, sync_mode=True), console

    def test_compact_summary_clears_then_renders_panel(self):
        buf = StringIO()
        ctx, console = self._ctx(buf)
        cleared = []
        console.clear = lambda *a, **k: cleared.append(True)
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="compact_summary",
            output_data={"summary": "Recap body", "summary_token": 5, "history_jsonl": "/h.jsonl"},
        )
        ctx._render_compact_action(action)
        assert cleared  # screen was cleared before printing the panel
        out = buf.getvalue()
        assert "Recap body" in out
        assert "Context compacted" in out

    def test_compact_summary_prefers_clear_screen_callback_in_tui(self):
        buf = StringIO()
        ctx, console, _live_state = self._ctx_tui(buf)
        cb = MagicMock()
        ctx._clear_screen_callback = cb
        console.clear = MagicMock()  # must NOT be used when a TUI callback exists
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="compact_summary",
            output_data={"summary": "X"},
        )
        ctx._render_compact_action(action)
        cb.assert_called_once()
        console.clear.assert_not_called()

    def _ctx_tui(self, buf):
        console = Console(file=buf, no_color=True, width=100)
        display = ActionHistoryDisplay(console)
        live_state = MagicMock()
        ctx = InlineStreamingContext([], display, sync_mode=True, live_state=live_state)
        return ctx, console, live_state

    def test_compact_progress_uses_pinned_region_not_scrollback(self):
        """compact_progress must NOT append to the committed scrollback; it sets
        a pinned-region flag so repeated majors overwrite a single line instead
        of stacking."""
        buf = StringIO()
        ctx, console, live_state = self._ctx_tui(buf)
        console.clear = MagicMock()
        action = _make_action(ActionRole.ASSISTANT, ActionStatus.PROCESSING, action_type="compact_progress")
        ctx._render_compact_action(action)
        assert ctx._compact_in_progress is True
        console.clear.assert_not_called()
        assert buf.getvalue() == ""  # nothing appended to scrollback
        live_state.set_lines.assert_called()  # pinned hint drawn via _repaint_live

    def test_repeated_compact_progress_does_not_stack_in_scrollback(self):
        buf = StringIO()
        ctx, _console, _live_state = self._ctx_tui(buf)
        a = _make_action(ActionRole.ASSISTANT, ActionStatus.PROCESSING, action_type="compact_progress")
        ctx._render_compact_action(a)
        ctx._render_compact_action(a)
        assert buf.getvalue() == ""  # two progress hints, still nothing stacked
        assert ctx._compact_in_progress is True

    def test_repaint_live_draws_single_compact_line_when_in_progress(self):
        buf = StringIO()
        ctx, _console, live_state = self._ctx_tui(buf)
        ctx._compact_in_progress = True
        ctx._repaint_live()
        live_state.set_lines.assert_called_once()
        lines = live_state.set_lines.call_args.args[0]
        assert len(lines) == 1  # exactly one pinned line — never stacks

    def test_compact_summary_clears_progress_flag(self):
        buf = StringIO()
        ctx, _console, _live_state = self._ctx_tui(buf)
        ctx._compact_in_progress = True
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="compact_summary",
            output_data={"summary": "Recap"},
        )
        ctx._render_compact_action(action)
        assert ctx._compact_in_progress is False

    def test_compact_summary_empty_clears_flag_without_clearing_screen(self):
        """A failed/empty summary (no summary text) must drop the pinned hint
        but NOT wipe the screen or draw a panel."""
        buf = StringIO()
        ctx, console, _live = self._ctx_tui(buf)
        console.clear = MagicMock()
        ctx._compact_in_progress = True
        action = _make_action(
            ActionRole.ASSISTANT,
            ActionStatus.SUCCESS,
            action_type="compact_summary",
            output_data={"summary": ""},
        )
        ctx._render_compact_action(action)
        assert ctx._compact_in_progress is False
        console.clear.assert_not_called()
        assert buf.getvalue() == ""  # no panel printed

    def test_process_actions_dispatches_compact_summary_to_renderer(self):
        buf = StringIO()
        ctx, _console = self._ctx(buf)
        ctx._render_compact_action = MagicMock()
        ctx.actions = [
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.PROCESSING,
                action_type="compact_progress",
            ),
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="compact_summary",
                output_data={"summary": "X"},
            ),
        ]
        ctx._processed_index = 0
        ctx._process_actions()
        # Both the in-progress hint and the final summary route to the renderer.
        assert ctx._render_compact_action.call_count == 2
        assert ctx._processed_index == 2

    def test_flush_remaining_dispatches_compact_summary_to_renderer(self):
        buf = StringIO()
        ctx, _console = self._ctx(buf)
        ctx._render_compact_action = MagicMock()
        ctx.actions = [
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.PROCESSING,
                action_type="compact_progress",
            ),
            _make_action(
                ActionRole.ASSISTANT,
                ActionStatus.SUCCESS,
                action_type="compact_summary",
                output_data={"summary": "X"},
            ),
        ]
        ctx._processed_index = 0
        ctx._flush_remaining_actions()
        # Both the in-progress hint and the final summary route to the renderer.
        assert ctx._render_compact_action.call_count == 2
