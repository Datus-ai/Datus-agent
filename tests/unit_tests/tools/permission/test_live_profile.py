"""Running turns must re-evaluate pending gates without replaying tool effects."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from datus.cli.execution_state import InteractionBroker
from datus.schemas.interaction_event import InteractionEvent
from datus.tools.permission.live_profile import LivePermissionProfile
from datus.tools.permission.permission_hooks import PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.profiles import get_profile
from datus.tools.registry.tool_registry import ToolRegistry


def setup_gate(profile="normal", raw=None):
    config = SimpleNamespace(_raw_permissions=raw or {}, active_profile_name="normal")
    live = LivePermissionProfile(config, profile)
    manager = PermissionManager(global_config=get_profile(profile), active_profile=profile)
    live.register(manager)
    broker = InteractionBroker()
    registry = ToolRegistry({"upsert_osi_metrics": "semantic_tools"})
    hooks = PermissionHooks(broker, manager, "chat", registry)
    ctx = SimpleNamespace(tool_arguments=json.dumps({"path": "subject/semantic_models/fitness.yml"}))
    return live, manager, broker, hooks, ctx


@pytest.mark.asyncio
async def test_switch_releases_pending_write_and_does_not_answer_unrelated_questions():
    live, manager, broker, hooks, ctx = setup_gate()
    question = asyncio.create_task(
        broker.request([InteractionEvent(title="Question", content="Which metric?", allow_free_text=True)])
    )
    ordinary = await broker._output_queue.get()
    gate = asyncio.create_task(hooks.on_tool_start(ctx, None, SimpleNamespace(name="upsert_osi_metrics")))
    permission = await asyncio.wait_for(broker._output_queue.get(), 1)
    assert broker.is_pending(permission.action_id)

    live.switch("dangerous")
    await asyncio.wait_for(gate, 1)

    assert not broker.is_pending(permission.action_id)
    assert broker.is_pending(ordinary.action_id)
    assert not question.done()
    rechecked = await broker._output_queue.get()
    assert rechecked.output == {"resolution": "permission_mode_changed"}
    assert manager._session_approvals == {}
    await broker.submit(ordinary.action_id, [["recovery"]])
    assert await question == [["recovery"]]


@pytest.mark.asyncio
async def test_running_and_future_children_inherit_switch_but_other_runs_do_not():
    live, parent, _, _, _ = setup_gate()
    child = PermissionManager(global_config=get_profile("normal"))
    live.register(child)
    other, other_manager, _, _, _ = setup_gate()
    parent.approve_for_session("semantic_tools", "upsert_osi_metrics")
    live.switch("dangerous")
    assert parent.active_profile == child.active_profile == "dangerous"
    future = PermissionManager(global_config=get_profile("normal"))
    live.register(future)
    assert future.active_profile == "dangerous"
    assert other.profile == other_manager.active_profile == "normal"
    assert live.agent_config.active_profile_name == "normal"
    live.switch("normal")
    assert parent.active_profile == child.active_profile == future.active_profile == "normal"
    assert not parent._session_approvals


@pytest.mark.asyncio
async def test_downgrade_restores_write_confirmation_and_cancel_cleans_gate():
    live, _, broker, hooks, ctx = setup_gate("dangerous")
    tool = SimpleNamespace(name="upsert_osi_metrics")
    await hooks.on_tool_start(ctx, None, tool)
    live.switch("normal")
    gate = asyncio.create_task(hooks.on_tool_start(ctx, None, tool))
    permission = await asyncio.wait_for(broker._output_queue.get(), 1)
    assert not gate.done()
    gate.cancel()
    with pytest.raises(asyncio.CancelledError):
        await gate
    assert not broker.is_pending(permission.action_id)


@pytest.mark.asyncio
async def test_explicit_ask_survives_switch_instead_of_blanket_approval():
    raw = {"rules": [{"tool": "semantic_tools", "pattern": "upsert_osi_metrics", "permission": "ask"}]}
    live, _, broker, hooks, ctx = setup_gate(raw=raw)
    gate = asyncio.create_task(hooks.on_tool_start(ctx, None, SimpleNamespace(name="upsert_osi_metrics")))
    first = await asyncio.wait_for(broker._output_queue.get(), 1)
    live.switch("dangerous")
    retired = await asyncio.wait_for(broker._output_queue.get(), 1)
    second = await asyncio.wait_for(broker._output_queue.get(), 1)
    assert retired.action_id == first.action_id
    assert broker.is_pending(second.action_id)
    await broker.submit(second.action_id, [["y"]])
    await asyncio.wait_for(gate, 1)


def test_failed_switch_restores_every_manager_and_existing_session_grants():
    from unittest.mock import patch

    live, parent, _, _, _ = setup_gate()
    child = PermissionManager(global_config=get_profile("normal"))
    live.register(child)
    for manager in (parent, child):
        manager.approve_for_session("semantic_tools", "upsert_osi_metrics")
    configs = {manager: manager.global_config for manager in (parent, child)}
    changed = live.changed
    original = child.switch_profile

    def fail_after_rebuild(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("broken plugin")

    with patch.object(child, "switch_profile", side_effect=fail_after_rebuild):
        with pytest.raises(RuntimeError, match="broken plugin"):
            live.switch("dangerous")

    assert live.profile == "normal"
    assert live.changed is changed and not changed.is_set()
    for manager in (parent, child):
        assert manager.active_profile == "normal"
        assert manager.global_config is configs[manager]
        assert manager._session_approvals
