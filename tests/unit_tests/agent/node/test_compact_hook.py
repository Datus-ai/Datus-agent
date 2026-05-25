# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for CompactHook: RunHooks integration that drives compact
passes from inside the Agents SDK Runner loop.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.agent.node.compact_hook import CompactHook


def _fake_node(*, mode_choice: str) -> MagicMock:
    """Build a stand-in for AgenticNode exposing only the surface CompactHook touches.

    ``_decide_compact_mode`` is async (it reads session items), so the mock
    uses ``AsyncMock``; tests that need ``mode_choice`` to vary just rebuild
    the stub.
    """
    node = MagicMock()
    node._decide_compact_mode = AsyncMock(return_value=mode_choice)
    node.compact = AsyncMock(return_value={"mode": mode_choice, "success": True})
    node._pending_compact_tasks = set()
    return node


@pytest.mark.asyncio
async def test_on_tool_end_noop_does_not_call_compact():
    """When the dispatcher returns ``noop`` the hook must NOT call
    ``node.compact`` — that would burn an extra await on every tool call
    even when there is nothing to do.
    """
    node = _fake_node(mode_choice="noop")
    hook = CompactHook(node)
    await hook.on_tool_end(context=None, agent=None, tool=None, result="ok")
    node.compact.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_tool_end_runs_major_synchronously():
    """Major must complete before the SDK loop yields; otherwise the next
    turn would re-read the unchanged session and overflow again.
    """
    node = _fake_node(mode_choice="major")
    hook = CompactHook(node)
    await hook.on_tool_end(context=None, agent=None, tool=None, result="ok")
    node.compact.assert_awaited_once_with(mode="major", reason="hook_major")


@pytest.mark.asyncio
async def test_on_tool_end_schedules_minor_asynchronously():
    """Minor must not block the SDK loop — disk I/O is fire-and-forget so
    the next tool call can proceed while archives are written.
    """
    node = _fake_node(mode_choice="minor")
    hook = CompactHook(node)
    await hook.on_tool_end(context=None, agent=None, tool=None, result="ok")
    # ``compact`` was called via ``create_task``; pump the loop once so the
    # task can run.
    await asyncio.sleep(0)
    node.compact.assert_awaited_once_with(mode="minor", reason="hook_minor")


@pytest.mark.asyncio
async def test_on_tool_end_minor_task_kept_strongly_referenced():
    """The background minor task must be added to ``_pending_compact_tasks``
    so asyncio's weak-ref scheduler does not GC it before it runs. A done
    callback then removes it so the set never grows unbounded.

    Regression: relying on ``asyncio.create_task`` without a strong ref means
    the task can be collected mid-flight in low-load scenarios — silently
    dropping the minor compact.
    """
    node = _fake_node(mode_choice="minor")
    # Slow compact so the task is still pending when we inspect the set.
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_compact(*, mode, reason):
        started.set()
        await proceed.wait()
        return {"mode": mode, "success": True}

    node.compact = AsyncMock(side_effect=slow_compact)
    hook = CompactHook(node)

    await hook.on_tool_end(context=None, agent=None, tool=None, result="ok")
    await started.wait()
    # Task must be registered while still running.
    assert len(node._pending_compact_tasks) == 1
    proceed.set()
    # Let the task complete and the done-callback fire.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert node._pending_compact_tasks == set()


@pytest.mark.asyncio
async def test_decide_failure_does_not_break_run_loop():
    """A buggy dispatcher must NEVER crash the SDK run loop; the worst case
    is we miss one compact opportunity and recover next round.
    """
    node = MagicMock()
    node._decide_compact_mode = AsyncMock(side_effect=RuntimeError("decide blew up"))
    node.compact = AsyncMock()
    node._pending_compact_tasks = set()
    hook = CompactHook(node)
    # Should NOT raise.
    await hook.on_tool_end(context=None, agent=None, tool=None, result="ok")
    # compact never invoked because dispatch failed.
    node.compact.assert_not_awaited()


@pytest.mark.asyncio
async def test_major_compact_failure_is_swallowed():
    """A failing major pass must not crash the SDK loop either — the next
    overflow guard or user turn will retry.
    """
    node = _fake_node(mode_choice="major")
    node.compact = AsyncMock(side_effect=RuntimeError("compact broke"))
    hook = CompactHook(node)
    # No exception expected — the hook logs and continues.
    await hook.on_tool_end(context=None, agent=None, tool=None, result="ok")
    node.compact.assert_awaited_once()


def test_compose_hooks_wires_compact_hook():
    """``CompactHook`` must be wired in by ``AgenticNode._compose_hooks``,
    otherwise ``on_tool_end`` never fires from inside the SDK Runner and the
    rolling-window minor + in-loop major compact triggers are dead code.

    Regression: the first cut of this refactor defined ``CompactHook`` but
    forgot to instantiate it in ``_compose_hooks`` — the hook was only
    reachable via the not-yet-existing ``compact_hook.py``  import. This
    test pins the wiring contract so a future refactor can't drop it.
    """
    from typing import AsyncGenerator, Optional
    from unittest.mock import patch

    from datus.agent.node.agentic_node import AgenticNode
    from datus.schemas.action_history import ActionHistory, ActionHistoryManager

    class _N(AgenticNode):
        async def execute_stream(
            self, action_history_manager: Optional[ActionHistoryManager] = None
        ) -> AsyncGenerator[ActionHistory, None]:
            yield  # pragma: no cover

        def get_node_name(self) -> str:
            return "chat"

    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _N.__new__(_N)
    node.agent_config = None
    node.execution_mode = "interactive"
    node._ensure_permission_hooks = lambda: None
    node.permission_hooks = None
    # With no extras and no permission hook, the single returned object must
    # BE the CompactHook — not None and not some other wrapper.
    assert isinstance(node._compose_hooks(), CompactHook)


def test_compact_hook_disabled_when_compact_off():
    """When both ``major`` and ``minor`` are disabled the hook is pointless;
    ``_compose_hooks`` must skip instantiation entirely so the SDK doesn't
    pay the per-tool-call dispatcher cost.
    """
    from typing import AsyncGenerator, Optional
    from unittest.mock import patch

    from datus.agent.node.agentic_node import AgenticNode
    from datus.configuration.agent_config import CompactConfig
    from datus.schemas.action_history import ActionHistory, ActionHistoryManager

    class _N(AgenticNode):
        async def execute_stream(
            self, action_history_manager: Optional[ActionHistoryManager] = None
        ) -> AsyncGenerator[ActionHistory, None]:
            yield  # pragma: no cover

        def get_node_name(self) -> str:
            return "chat"

    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _N.__new__(_N)
    node.agent_config = None
    node.execution_mode = "interactive"
    node._ensure_permission_hooks = lambda: None
    node.permission_hooks = None
    cfg = CompactConfig()
    cfg.major.enabled = False
    cfg.minor.enabled = False
    node._compact_cfg = cfg
    # No extras, no permission hook, compact disabled → no hooks at all.
    assert node._compose_hooks() is None
