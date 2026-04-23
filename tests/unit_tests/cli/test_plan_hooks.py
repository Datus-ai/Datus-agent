# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/plan_hooks.py — PlanModeHooks.

All external dependencies (InteractionBroker, SQLiteSession, SessionTodoStorage)
are mocked. Tests cover state transitions, _is_pending_update, and hook methods.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.cli.plan_hooks import PlanModeHooks, UserCancelledException

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broker():
    b = MagicMock()
    b.request = AsyncMock()
    return b


@pytest.fixture
def session():
    return MagicMock()


def _make_hooks(broker, session, auto_mode=False):
    with patch("datus.tools.func_tool.plan_tools.SessionTodoStorage") as MockStorage:
        mock_storage = MagicMock()
        MockStorage.return_value = mock_storage
        hooks = PlanModeHooks(broker=broker, session=session, auto_mode=auto_mode)
        hooks.todo_storage = mock_storage
        return hooks


def _make_todo_list(items=None):
    todo_list = MagicMock()
    if items is None:
        items = []
    todo_list.items = items
    return todo_list


def _make_item(content="Do something", status="pending", item_id="id1"):
    item = MagicMock()
    item.content = content
    item.status = status
    item.id = item_id
    return item


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


class TestPlanModeHooksInit:
    def test_default_state(self, broker, session):
        hooks = _make_hooks(broker, session)
        assert hooks.plan_phase == "generating"
        assert hooks.execution_mode == "manual"
        assert hooks.auto_mode is False
        assert hooks.replan_feedback == ""
        assert hooks._state_transitions == []

    def test_auto_mode_sets_execution_mode(self, broker, session):
        hooks = _make_hooks(broker, session, auto_mode=True)
        assert hooks.execution_mode == "auto"
        assert hooks.auto_mode is True


# ---------------------------------------------------------------------------
# Tests: _transition_state
# ---------------------------------------------------------------------------


class TestTransitionState:
    def test_transition_changes_phase(self, broker, session):
        hooks = _make_hooks(broker, session)
        hooks._transition_state("confirming")
        assert hooks.plan_phase == "confirming"

    def test_transition_records_history(self, broker, session):
        hooks = _make_hooks(broker, session)
        hooks._transition_state("executing", {"mode": "auto"})
        assert len(hooks._state_transitions) == 1
        t = hooks._state_transitions[0]
        assert t["from_state"] == "generating"
        assert t["to_state"] == "executing"
        assert t["context"] == {"mode": "auto"}


# ---------------------------------------------------------------------------
# Tests: on_start / on_end / on_handoff (trivial async methods)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncHookStubs:
    async def test_on_start(self, broker, session):
        hooks = _make_hooks(broker, session)
        await hooks.on_start(MagicMock(), MagicMock())  # should not raise

    async def test_on_end(self, broker, session):
        hooks = _make_hooks(broker, session)
        await hooks.on_end(MagicMock(), MagicMock(), MagicMock())  # should not raise


# ---------------------------------------------------------------------------
# Tests: on_tool_end — todo_write sets flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnToolEnd:
    async def test_todo_write_sets_pending_flag(self, broker, session):
        hooks = _make_hooks(broker, session)
        tool = MagicMock()
        tool.name = "todo_write"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, MagicMock())
        assert hooks._plan_generated_pending is True

    async def test_other_tool_no_flag(self, broker, session):
        hooks = _make_hooks(broker, session)
        tool = MagicMock()
        tool.name = "other_tool"
        await hooks.on_tool_end(MagicMock(), MagicMock(), tool, MagicMock())
        assert hooks._plan_generated_pending is False


# ---------------------------------------------------------------------------
# Tests: on_llm_end — triggers _on_plan_generated when pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnLlmEnd:
    async def test_triggers_on_plan_generated_when_pending(self, broker, session):
        hooks = _make_hooks(broker, session)
        hooks._plan_generated_pending = True
        hooks._on_plan_generated = AsyncMock()

        await hooks.on_llm_end(MagicMock(), MagicMock(), MagicMock())

        hooks._on_plan_generated.assert_called_once()
        assert hooks._plan_generated_pending is False

    async def test_no_trigger_when_not_pending(self, broker, session):
        hooks = _make_hooks(broker, session)
        hooks._plan_generated_pending = False
        hooks._on_plan_generated = AsyncMock()

        await hooks.on_llm_end(MagicMock(), MagicMock(), MagicMock())

        hooks._on_plan_generated.assert_not_called()

    async def test_no_trigger_when_not_generating_phase(self, broker, session):
        hooks = _make_hooks(broker, session)
        hooks._plan_generated_pending = True
        hooks.plan_phase = "executing"
        hooks._on_plan_generated = AsyncMock()

        await hooks.on_llm_end(MagicMock(), MagicMock(), MagicMock())

        hooks._on_plan_generated.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _is_pending_update
# ---------------------------------------------------------------------------


class TestIsPendingUpdate:
    def test_returns_true_for_pending_status(self, broker, session):
        hooks = _make_hooks(broker, session)
        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"status": "pending", "id": "1"})
        assert hooks._is_pending_update(ctx) is True

    def test_returns_false_for_non_pending_status(self, broker, session):
        hooks = _make_hooks(broker, session)
        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"status": "completed", "id": "1"})
        assert hooks._is_pending_update(ctx) is False

    def test_returns_false_when_no_tool_arguments(self, broker, session):
        hooks = _make_hooks(broker, session)
        ctx = MagicMock(spec=[])  # no tool_arguments attribute
        assert hooks._is_pending_update(ctx) is False

    def test_returns_false_when_tool_arguments_none(self, broker, session):
        hooks = _make_hooks(broker, session)
        ctx = MagicMock()
        ctx.tool_arguments = None
        assert hooks._is_pending_update(ctx) is False

    def test_returns_false_on_invalid_json(self, broker, session):
        hooks = _make_hooks(broker, session)
        ctx = MagicMock()
        ctx.tool_arguments = "not valid json"
        assert hooks._is_pending_update(ctx) is False


# ---------------------------------------------------------------------------
# Tests: _on_plan_generated — no todo list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnPlanGenerated:
    async def test_no_todo_list_sends_error_message(self, broker, session):
        hooks = _make_hooks(broker, session)
        hooks.todo_storage.get_todo_list.return_value = None
        broker.request.return_value = [["1"]]

        await hooks._on_plan_generated()

        broker.request.assert_awaited_once()

    async def test_auto_mode_transitions_to_executing(self, broker, session):
        hooks = _make_hooks(broker, session, auto_mode=True)
        todo_list = _make_todo_list([_make_item()])
        hooks.todo_storage.get_todo_list.return_value = todo_list
        broker.request.return_value = [["1"]]

        await hooks._on_plan_generated()

        assert hooks.plan_phase == "executing"
        assert hooks.execution_mode == "auto"

    async def test_manual_mode_choice_1_executes(self, broker, session):
        hooks = _make_hooks(broker, session)
        todo_list = _make_todo_list([_make_item()])
        hooks.todo_storage.get_todo_list.return_value = todo_list
        broker.request.return_value = [["1"]]

        await hooks._on_plan_generated()

        assert hooks.plan_phase == "executing"
        assert hooks.execution_mode == "manual"

    async def test_manual_mode_choice_2_auto_execute(self, broker, session):
        hooks = _make_hooks(broker, session)
        todo_list = _make_todo_list([_make_item()])
        hooks.todo_storage.get_todo_list.return_value = todo_list
        broker.request.return_value = [["2"]]

        await hooks._on_plan_generated()

        assert hooks.execution_mode == "auto"
        assert hooks.plan_phase == "executing"

    async def test_manual_mode_choice_4_cancel_raises(self, broker, session):
        hooks = _make_hooks(broker, session)
        todo_list = _make_todo_list([_make_item()])
        hooks.todo_storage.get_todo_list.return_value = todo_list
        broker.request.return_value = [["4"]]

        with pytest.raises(UserCancelledException):
            await hooks._on_plan_generated()

        assert hooks.plan_phase == "cancelled"


# ---------------------------------------------------------------------------
# Tests: get_plan_tools
# ---------------------------------------------------------------------------


class TestGetPlanTools:
    def test_get_plan_tools_returns_tools(self, broker, session):
        hooks = _make_hooks(broker, session)
        mock_tools = [MagicMock(), MagicMock()]
        with patch("datus.tools.func_tool.plan_tools.PlanTool") as MockPlanTool:
            mock_plan_tool = MagicMock()
            mock_plan_tool.available_tools.return_value = mock_tools
            MockPlanTool.return_value = mock_plan_tool
            result = hooks.get_plan_tools()
        assert result is mock_tools
