# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Integration test: per-LLM-call ``event: usage`` SSE events.

Covers the producer→converter contract end-to-end so a regression that drops
the mid-turn usage updates surfaces here even when the unit tests for each
layer still pass individually.

Scenario: one user turn consists of 3 LLM calls (typical tool-use loop).
After the turn:

* the SSE stream must carry at least 3 ``event: "usage"`` events
* the deltas of those events must sum to the final turn total
* the final ``event: "end"`` payload total must match the last cumulative
  ``event: "usage"`` payload total
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from agents.extensions.memory import AdvancedSQLiteSession

from datus.agent.node.token_usage_hook import TokenUsageHook
from datus.api.models.cli_models import (
    SSEEndData,
    SSEEvent,
    SSEUsageData,
)
from datus.api.services.action_sse_converter import action_to_sse_event
from datus.models.session_manager import SessionManager
from datus.schemas.action_history import ActionHistoryManager
from datus.utils.time_utils import now_utc_iso


@pytest.fixture
def session_manager(tmp_path):
    """Real ``SessionManager`` backed by an on-disk SQLite store so the
    persistence side of the contract is exercised too."""
    sm = SessionManager(session_dir=str(tmp_path / "sessions"))
    yield sm
    sm.close_all_sessions()


def _bootstrap_session(session_manager: SessionManager, session_id: str) -> str:
    db_path = os.path.join(session_manager.session_dir, f"{session_id}.db")
    AdvancedSQLiteSession(session_id=session_id, db_path=db_path, create_tables=True)
    return db_path


def _fake_node(session_manager: SessionManager, session_id: str) -> MagicMock:
    """Stand-in for ``AgenticNode`` exposing only what the hook reads."""
    node = MagicMock()
    node.session_id = session_id
    node.session_manager = session_manager
    node.context_length = 200_000
    node._current_action_history = ActionHistoryManager()
    # Real ``ActionBus`` would require an event loop; for this test we only
    # care about the manager-side fan-out + converter mapping.
    node.action_bus = None
    node.actions = []
    node.running_turn_usage = None
    node._notify_status_dirty = MagicMock()
    return node


@pytest.mark.asyncio
async def test_three_llm_calls_emit_three_usage_events_aligned_with_end(session_manager):
    """Drive 3 sequential ``emit_manual`` calls (one per LLM round-trip),
    convert every queued ``token_usage`` action to SSE, then synthesise the
    ``end`` event from the persisted running snapshot. Verify counts and
    alignment."""
    session_id = "chat_session_int"
    _bootstrap_session(session_manager, session_id)
    node = _fake_node(session_manager, session_id)
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)

    # Three LLM calls, monotonically increasing cumulative usage.
    cumulatives = [
        {
            "requests": 1,
            "input_tokens": 500,
            "output_tokens": 100,
            "total_tokens": 600,
            "cached_tokens": 50,
            "last_call_input_tokens": 500,
        },
        {
            "requests": 2,
            "input_tokens": 1100,
            "output_tokens": 250,
            "total_tokens": 1350,
            "cached_tokens": 50,
            "last_call_input_tokens": 600,
        },
        {
            "requests": 3,
            "input_tokens": 1700,
            "output_tokens": 450,
            "total_tokens": 2150,
            "cached_tokens": 50,
            "last_call_input_tokens": 600,
        },
    ]
    for cumulative in cumulatives:
        await hook.emit_manual(cumulative)

    # Drain the action manager and convert each action to an SSE event,
    # mirroring what ``ChatTaskManager`` does in its main loop.
    actions = node._current_action_history.get_actions()
    usage_actions = [a for a in actions if a.action_type == "token_usage"]
    assert len(usage_actions) == 3, "one usage action per LLM call"

    usage_events: list[SSEEvent] = []
    for idx, action in enumerate(usage_actions, start=1):
        event = action_to_sse_event(action, event_id=idx, message_id=action.action_id)
        # The converter MUST produce a fully-populated SSE envelope — a bare
        # ``is not None`` check would let a converter regression that strips
        # the event id silently pass.
        assert event.event == "usage"
        assert event.id == idx
        assert isinstance(event.data, SSEUsageData)
        # Main-agent usage is depth=0 with no parent and carries the producing
        # node's session, so the API can treat it as the top-level meter and
        # tell it apart from forwarded sub-agent usage (depth>0).
        assert event.data.depth == 0
        assert event.data.parent_action_id is None
        assert event.data.llm_session_id == session_id
        usage_events.append(event)

    # Cumulative on the last usage event matches the final LLM call's total.
    last_usage = usage_events[-1].data
    assert last_usage.total_tokens == 2150
    assert last_usage.requests == 3

    # Deltas across all three usage events sum to the turn cumulative total.
    delta_total = sum(ev.data.delta.total_tokens for ev in usage_events)
    assert delta_total == 2150

    delta_input = sum(ev.data.delta.input_tokens for ev in usage_events)
    delta_output = sum(ev.data.delta.output_tokens for ev in usage_events)
    assert delta_input == 1700
    assert delta_output == 450

    # Each individual delta must match the expected per-call increment, not
    # just sum to the cumulative total — a regression that reported all usage
    # in the first event (or mis-attributed tokens across calls) would still
    # pass the aggregate checks above but fail here.
    expected_deltas = [
        (600, 500, 100),  # Call 1: first call, full cumulative
        (750, 600, 150),  # Call 2: 1350-600, 1100-500, 250-100
        (800, 600, 200),  # Call 3: 2150-1350, 1700-1100, 450-250
    ]
    for idx, (exp_total, exp_input, exp_output) in enumerate(expected_deltas):
        assert usage_events[idx].data.delta.total_tokens == exp_total
        assert usage_events[idx].data.delta.input_tokens == exp_input
        assert usage_events[idx].data.delta.output_tokens == exp_output

    # Each per-call delta must be non-negative — UI consumers display this
    # to the user and a negative delta would be confusing nonsense.
    assert all(ev.data.delta.total_tokens >= 0 for ev in usage_events)

    # The running snapshot in the side table reflects the last cumulative
    # so a CLI status bar reading mid-turn (or resume) sees the live total.
    running = session_manager.get_running_turn_usage(session_id)
    assert isinstance(running, dict)
    assert running["cumulative"]["total_tokens"] == 2150
    assert running["cumulative"]["requests"] == 3
    assert running["context_length"] == 200_000

    # Synthesise the ``end`` event as ``ChatTaskManager`` does at turn end
    # (after ``store_run_usage`` writes the persisted row).
    end_data = SSEEndData(
        session_id=session_id,
        llm_session_id=session_id,
        total_events=len(usage_events) + 1,
        action_count=len(actions),
        duration=0.5,
        requests=last_usage.requests,
        input_tokens=last_usage.input_tokens,
        output_tokens=last_usage.output_tokens,
        total_tokens=last_usage.total_tokens,
        cached_tokens=last_usage.cached_tokens,
        session_total_tokens=last_usage.last_call_input_tokens,
        context_length=last_usage.context_length,
    )
    end_event = SSEEvent(
        id=len(usage_events) + 1,
        event="end",
        data=end_data,
        timestamp=now_utc_iso(),
    )

    # Wire-level invariant: ``end.total_tokens`` == final ``usage.total_tokens``.
    assert end_event.data.total_tokens == usage_events[-1].data.total_tokens
    assert end_event.data.requests == usage_events[-1].data.requests
    assert end_event.data.session_total_tokens == usage_events[-1].data.last_call_input_tokens


@pytest.mark.asyncio
async def test_usage_event_is_not_confused_with_message_event(session_manager):
    """Defensive: the per-call usage event must never produce
    ``SSEMessageData`` — the chat-task-manager dedup logic only fires on
    message events and would silently drop a misclassified usage event."""
    session_id = "chat_session_mismatch"
    _bootstrap_session(session_manager, session_id)
    node = _fake_node(session_manager, session_id)
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)
    await hook.emit_manual({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    actions = node._current_action_history.get_actions()
    [action] = [a for a in actions if a.action_type == "token_usage"]
    event = action_to_sse_event(action, event_id=1, message_id=action.action_id)
    assert event.event == "usage"
    # Directly assert the expected type (mirrors the main test) rather than a
    # double-negative on a ``type`` attribute that ``SSEUsageData`` never has.
    assert isinstance(event.data, SSEUsageData)


@pytest.mark.asyncio
async def test_running_snapshot_cleared_does_not_strand_data(session_manager):
    """After ``clear_running_turn_usage`` (called at turn end) the side-table
    row must disappear so the next status-bar refresh stops counting the
    just-finished turn on top of the persisted ``turn_usage`` totals."""
    session_id = "chat_session_clear"
    _bootstrap_session(session_manager, session_id)
    node = _fake_node(session_manager, session_id)
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)
    await hook.emit_manual({"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})
    # Sanity-check the pre-clear state by reading the actual cumulative —
    # a bare ``is not None`` would let a regression that persists an empty
    # dict slip through.
    pre = session_manager.get_running_turn_usage(session_id)
    assert pre["cumulative"]["total_tokens"] == 10

    session_manager.clear_running_turn_usage(session_id)
    assert session_manager.get_running_turn_usage(session_id) is None

    detailed = session_manager.get_detailed_usage(session_id)
    assert detailed["running"] is None
    # No persisted turn rows either, so total stays at 0.
    assert detailed["total"]["total_tokens"] == 0
