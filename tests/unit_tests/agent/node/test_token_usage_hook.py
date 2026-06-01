# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``TokenUsageHook``.

The hook is a thin adapter between the SDK ``on_llm_end`` callback and
Datus's mid-turn streaming pipeline. The tests below cover the public
contract:

* per-call delta is computed against the prior cumulative snapshot
* the node's running snapshot and the side-table are updated atomically
* a ``token_usage`` action lands in the action manager *and* the bus
* ``on_start`` / ``on_handoff`` reset the delta baseline
* manual fan-in (``emit_manual``) reuses the same publish path
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from datus.agent.node.token_usage_hook import TokenUsageHook, _delta
from datus.schemas.action_history import ActionRole, ActionStatus


class _FakeManager:
    def __init__(self) -> None:
        self.added: List[Any] = []

    def add_action(self, action: Any) -> None:
        self.added.append(action)


class _FakeBus:
    def __init__(self) -> None:
        self.published: List[Any] = []

    def put(self, action: Any) -> None:
        self.published.append(action)


def _fake_node(
    cumulative_sequence: List[Dict[str, Any]],
) -> Tuple[MagicMock, _FakeManager, _FakeBus, MagicMock, MagicMock]:
    """Build a stub node whose ``model._extract_usage_info`` walks a script.

    ``cumulative_sequence`` lists dicts to return on successive calls so
    each test can exercise specific accounting scenarios without a real
    SDK Usage object.
    """

    cursor: Dict[str, int] = {"i": 0}

    def _extract(_usage: Any) -> Dict[str, Any]:
        i = cursor["i"]
        cursor["i"] = min(i + 1, len(cumulative_sequence) - 1) if cumulative_sequence else 0
        return cumulative_sequence[i] if cumulative_sequence else {}

    manager = _FakeManager()
    bus = _FakeBus()
    session_manager = MagicMock()
    session_manager.upsert_running_turn_usage = MagicMock()

    node = MagicMock()
    node.model = MagicMock()
    node.model._extract_usage_info = _extract
    node.context_length = 200_000
    node._current_action_history = manager
    node.action_bus = bus
    node.session_id = "chat_session_abc"
    node.session_manager = session_manager
    node.actions = []
    node.running_turn_usage = None

    notify = MagicMock()
    node._notify_status_dirty = notify
    return node, manager, bus, session_manager, notify


def test_delta_handles_missing_and_decreasing_fields():
    """Delta must clamp to ``0`` so a missing prior frame, or an upstream
    accounting glitch that rewinds the counter, never reports negative
    consumption to the UI."""
    out = _delta({"input_tokens": 100, "total_tokens": 150}, None)
    assert out["input_tokens"] == 100
    assert out["output_tokens"] == 0  # missing key
    assert out["total_tokens"] == 150

    out = _delta({"total_tokens": 50}, {"total_tokens": 80})
    assert out["total_tokens"] == 0  # never negative


@pytest.mark.asyncio
async def test_on_llm_end_first_call_emits_full_delta_and_persists():
    """The first ``on_llm_end`` after ``on_start`` reports the entire call as
    delta and writes a row to ``running_turn_usage`` so the status bar and
    resume both see partial progress immediately."""
    node, manager, bus, sm, notify = _fake_node(
        [
            {
                "requests": 1,
                "input_tokens": 800,
                "output_tokens": 200,
                "total_tokens": 1000,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "last_call_input_tokens": 800,
            }
        ]
    )
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)
    await hook.on_llm_end(SimpleNamespace(usage=object()), None, None)

    # one usage action enqueued into both the manager and the bus
    assert len(manager.added) == 1
    assert len(bus.published) == 1
    action = manager.added[0]
    assert action.action_type == "token_usage"
    assert action.role == ActionRole.ASSISTANT
    assert action.status == ActionStatus.SUCCESS
    assert action.output["cumulative"]["total_tokens"] == 1000
    assert action.output["delta"]["total_tokens"] == 1000
    assert action.output["context_length"] == 200_000
    assert action.output["last_call_input_tokens"] == 800

    # persistence: the running row reflects the cumulative snapshot
    sm.upsert_running_turn_usage.assert_called_once()
    kwargs = sm.upsert_running_turn_usage.call_args.kwargs
    assert kwargs["session_id"] == "chat_session_abc"
    assert kwargs["context_length"] == 200_000
    assert kwargs["cumulative"]["total_tokens"] == 1000

    # context-window occupancy persisted to the on-disk session_state via the
    # node, using the call's real context window (``last_call_input_tokens``).
    node.persist_context_state.assert_called_once_with(800, 200_000)

    # node snapshot populated so the status bar's next render sees it
    snapshot = node.running_turn_usage
    assert snapshot.total_tokens == 1000
    assert snapshot.session_total_tokens == 800
    assert snapshot.context_length == 200_000

    notify.assert_called_once()


@pytest.mark.asyncio
async def test_on_llm_end_second_call_reports_only_incremental_delta():
    """Successive calls during one turn must report only the LLM call's
    contribution as ``delta`` while ``cumulative`` keeps the running total."""
    node, manager, bus, sm, _ = _fake_node(
        [
            {"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
            {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500},
        ]
    )
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)
    await hook.on_llm_end(SimpleNamespace(usage=object()), None, None)
    await hook.on_llm_end(SimpleNamespace(usage=object()), None, None)

    assert len(manager.added) == 2
    second = manager.added[1]
    assert second.output["cumulative"]["total_tokens"] == 1500
    assert second.output["delta"]["total_tokens"] == 900
    assert second.output["delta"]["input_tokens"] == 700
    assert second.output["delta"]["output_tokens"] == 200


@pytest.mark.asyncio
async def test_on_start_resets_baseline_between_turns():
    """A new ``on_start`` clears the prior cumulative, so the first call of
    the next turn re-reports the absolute SDK accumulator as delta even if
    the SDK keeps the counter monotonic across turns."""
    node, manager, _, _, _ = _fake_node(
        [
            {"input_tokens": 500, "total_tokens": 500},
            {"input_tokens": 800, "total_tokens": 800},
        ]
    )
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)
    await hook.on_llm_end(SimpleNamespace(usage=object()), None, None)

    # Pretend the user starts a fresh turn — baseline should reset.
    await hook.on_start(None, None)
    await hook.on_llm_end(SimpleNamespace(usage=object()), None, None)

    second_turn = manager.added[1]
    # Without the reset the delta would be (800-500)=300; with the reset it
    # is the full 800.
    assert second_turn.output["delta"]["total_tokens"] == 800


@pytest.mark.asyncio
async def test_emit_manual_uses_same_publish_pipeline():
    """``emit_manual`` is the entry point for models that don't drive
    ``on_llm_end``; it must publish via the same manager / bus / notify
    plumbing so consumers don't need a separate code path."""
    node, manager, bus, sm, notify = _fake_node([])  # unused — manual skips extractor
    hook = TokenUsageHook(node)
    await hook.emit_manual({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "last_call_input_tokens": 10})
    assert len(manager.added) == 1
    assert len(bus.published) == 1
    assert sm.upsert_running_turn_usage.called
    notify.assert_called_once()
    assert manager.added[0].output["cumulative"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_on_llm_end_swallows_extractor_failure():
    """The hook lives inside the SDK run loop; any exception it raises
    crashes the user turn. Test that a buggy ``_extract_usage_info`` is
    handled silently and no action is enqueued."""
    node, manager, bus, sm, notify = _fake_node([])
    node.model._extract_usage_info = MagicMock(side_effect=RuntimeError("boom"))
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)
    await hook.on_llm_end(SimpleNamespace(usage=object()), None, None)
    assert manager.added == []
    assert bus.published == []
    sm.upsert_running_turn_usage.assert_not_called()
    notify.assert_not_called()


@pytest.mark.asyncio
async def test_on_llm_end_skips_when_no_action_history_bound():
    """Outside an active stream the node clears ``_current_action_history``;
    the hook should still update the in-memory snapshot so other callers
    (e.g. ``get_last_turn_usage`` invoked from tests) observe the change,
    but must not blow up trying to enqueue an action."""
    node, manager, bus, sm, _ = _fake_node([{"input_tokens": 7, "total_tokens": 7}])
    node._current_action_history = None
    hook = TokenUsageHook(node)
    await hook.on_start(None, None)
    await hook.on_llm_end(SimpleNamespace(usage=object()), None, None)
    assert manager.added == []
    # Snapshot still updated for status-bar fan-out.
    snapshot = node.running_turn_usage
    assert snapshot.total_tokens == 7
    assert snapshot.input_tokens == 7
    # Persistence still happens — resume should see the data.
    sm.upsert_running_turn_usage.assert_called_once()
