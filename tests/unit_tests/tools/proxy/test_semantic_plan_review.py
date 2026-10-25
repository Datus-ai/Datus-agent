"""The web review freezes content, and only acceptance can commit it."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from datus.api.routes.chat_routes import get_tool_preview
from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools
from datus.tools.permission.permission_hooks import PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.profiles import get_profile
from datus.tools.proxy.semantic_plan_review import apply_semantic_plan_review
from datus.tools.proxy.tool_result_channel import ToolResultChannel
from datus.tools.registry.tool_registry import ToolRegistry


class ReviewChannel(ToolResultChannel):
    def __init__(self):
        super().__init__()
        self.waiting = asyncio.Event()

    async def wait_for(self, call_id, timeout=None):
        self.waiting.set()
        return await super().wait_for(call_id, timeout)


def node_for(root):
    writer = SemanticPlanFileTools(root)
    return SimpleNamespace(
        semantic_plan_file_tool=writer,
        tools=writer.available_tools(),
        tool_channel=ReviewChannel(),
        proxied_tool_names=set(),
        scope="owner",
    )


async def start_review(node, summary="first", call_id="call-1"):
    node.tool_channel.waiting.clear()
    task = asyncio.create_task(
        node.tools[0].on_invoke_tool(
            SimpleNamespace(tool_call_id=call_id),
            json.dumps({"model_name": "sales", "plan_json": {"summary": summary}}),
        )
    )
    await asyncio.wait_for(node.tool_channel.waiting.wait(), 1)
    return task, node.tool_review_previews[call_id]


@pytest.mark.asyncio
async def test_accept_commits_the_server_preview_not_client_content(tmp_path):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    registry = ToolRegistry()
    registry.register_tools("semantic_tools", node.tools)
    hooks = PermissionHooks(
        broker=None,
        permission_manager=PermissionManager(get_profile("normal"), active_profile="normal"),
        node_name="semantic_modeling",
        tool_registry=registry,
        non_interactive=True,
        proxied_tool_names=node.proxied_tool_names,
    )
    await hooks.on_tool_start(SimpleNamespace(tool_call_id="call-1"), None, node.tools[0])
    running, preview = await start_review(node)
    target = tmp_path / preview["filePath"]
    assert preview["oldText"] == ""
    assert not target.exists()

    await node.tool_channel.publish("call-1", {"success": 1, "result": {"newText": "malicious browser content"}})
    result = await running

    assert target.read_text() == preview["newText"]
    assert result["result"]["review_status"] == "accepted"
    assert not node.tool_review_previews


@pytest.mark.asyncio
async def test_cancel_then_feedback_reuses_the_accepted_baseline(tmp_path):
    node = node_for(tmp_path)
    saved = node.semantic_plan_file_tool.write_semantic_model_plan_file("sales", {"summary": "accepted"})
    target = tmp_path / saved.result["path"]
    baseline = target.read_text()
    apply_semantic_plan_review(node)
    running, first = await start_review(node, "rejected")
    await node.tool_channel.publish("call-1", {"success": 0})
    assert (await running)["success"] == 0
    assert target.read_text() == baseline

    running, revised = await start_review(node, "revised", "call-2")
    assert revised["filePath"] == first["filePath"]
    assert revised["oldText"] == baseline
    assert json.loads(revised["newText"])["plan"]["summary"] == "revised"
    await node.tool_channel.publish("call-2", {"success": 1})
    assert (await running)["success"] == 1
    assert target.read_text() == revised["newText"]


@pytest.mark.asyncio
async def test_accept_rejects_a_stale_preview(tmp_path):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, preview = await start_review(node)
    saved = node.semantic_plan_file_tool.write_semantic_model_plan_file("sales", {"summary": "concurrent write"})
    target = tmp_path / saved.result["path"]
    concurrent = target.read_text()
    await node.tool_channel.publish("call-1", {"success": 1})
    result = await running
    assert result["success"] == 0
    assert "changed during review" in result["error"]
    assert target.read_text() == concurrent
    assert preview["newText"] != concurrent


@pytest.mark.asyncio
async def test_stop_discards_pending_proposal_without_a_write(tmp_path):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, preview = await start_review(node)
    node.tool_channel.cancel_all("stopped")
    assert (await running)["success"] == 0
    assert not (tmp_path / preview["filePath"]).exists()
    assert not node.tool_review_previews


@pytest.mark.asyncio
async def test_preview_is_scoped_to_the_session_owner(tmp_path):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, preview = await start_review(node)
    task = SimpleNamespace(node=node, status="running")
    svc = SimpleNamespace(
        task_manager=SimpleNamespace(get_task=lambda session_id: task if session_id == "session" else None)
    )
    try:
        result = await get_tool_preview(svc, SimpleNamespace(user_id="owner"), "session", "call-1")
        assert result.data == preview
        for user, session in [("stranger", "session"), ("owner", "unknown")]:
            with pytest.raises(HTTPException) as error:
                await get_tool_preview(svc, SimpleNamespace(user_id=user), session, "call-1")
            assert error.value.status_code == 404
    finally:
        node.tool_channel.cancel_all()
        await running


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        "not json",
        "{}",
        '{"model_name":"../outside","plan_json":{}}',
        '{"model_name":"sales","plan_json":{"metrics":[{"name":"a","depends_on":["missing"]}]}}',
    ],
)
async def test_bad_proposals_fail_before_waiting_or_writing(tmp_path, arguments):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    result = await node.tools[0].on_invoke_tool(SimpleNamespace(tool_call_id="bad"), arguments)
    assert result["success"] == 0
    assert not node.tool_channel.waiting.is_set()
    assert not list(tmp_path.rglob("semantic-model-plan.json"))


@pytest.mark.asyncio
async def test_expired_review_and_late_accept_never_write(tmp_path, monkeypatch):
    monkeypatch.setattr("datus.tools.proxy.semantic_plan_review.DEFAULT_RESULT_TIMEOUT", 0.01)
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, preview = await start_review(node)
    assert (await running)["success"] == 0
    assert await node.tool_channel.publish("call-1", {"success": 1}) is False
    assert not (tmp_path / preview["filePath"]).exists()
    assert not node.tool_review_previews


@pytest.mark.asyncio
async def test_finished_task_has_no_reviewable_call(tmp_path):
    node = node_for(tmp_path)
    task = SimpleNamespace(node=node, status="completed")
    svc = SimpleNamespace(task_manager=SimpleNamespace(get_task=lambda _: task))
    with pytest.raises(HTTPException) as error:
        await get_tool_preview(svc, SimpleNamespace(user_id="owner"), "session", "missing")
    assert error.value.status_code == 404
