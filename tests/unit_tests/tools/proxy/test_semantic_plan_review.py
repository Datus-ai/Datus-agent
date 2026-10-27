"""The web review freezes content, and only acceptance can commit it."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from datus.api.models.cli_models import SSEDataType
from datus.api.services.action_sse_converter import action_to_sse_event
from datus.schemas.action_bus import ActionBus
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
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
        action_bus=ActionBus(),
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
@pytest.mark.parametrize("permission", [None, PermissionLevel.ALLOW, PermissionLevel.ASK])
@pytest.mark.parametrize("accept", [False, True])
async def test_web_review_still_required_after_permission_check(tmp_path, permission, accept):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    registry = ToolRegistry()
    registry.register_tools("semantic_tools", node.tools)
    config = get_profile("normal")
    if permission is not None:
        config = config.merge_with(
            PermissionConfig(
                rules=[PermissionRule(tool="semantic_tools", pattern=node.tools[0].name, permission=permission)]
            )
        )
    hooks = PermissionHooks(
        broker=None,
        permission_manager=PermissionManager(config, active_profile="normal"),
        node_name="semantic_modeling",
        tool_registry=registry,
        non_interactive=permission != PermissionLevel.ASK,
        proxied_tool_names=node.proxied_tool_names,
    )
    hooks._request_user_confirmation = AsyncMock(return_value=True)
    await hooks.on_tool_start(SimpleNamespace(tool_call_id="call-1"), None, node.tools[0])
    if permission == PermissionLevel.ASK:
        hooks._request_user_confirmation.assert_awaited_once()
    else:
        hooks._request_user_confirmation.assert_not_awaited()
    running, preview = await start_review(node)
    target = tmp_path / preview["filePath"]
    assert preview["oldText"] == ""
    assert not target.exists()

    await node.tool_channel.publish(
        "call-1", {"success": int(accept), "result": {"newText": "malicious browser content"}}
    )
    result = await running

    if accept:
        assert target.read_text() == preview["newText"]
        assert result["result"]["review_status"] == "accepted"
    else:
        assert result["success"] == 0
        assert not target.exists()
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
@pytest.mark.parametrize("accept", [False, True])
async def test_prepared_snapshot_streams_on_the_same_card_before_a_decision(tmp_path, accept):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    args = {"model_name": "sales", "plan_json": {"summary": "review through SSE"}}

    async def primary():
        yield ActionHistory(
            action_id="stream-call",
            action_type="tool_call",
            role=ActionRole.TOOL,
            status=ActionStatus.PROCESSING,
            input={"function_name": node.tools[0].name, "arguments": args},
        )
        result = await node.tools[0].on_invoke_tool(SimpleNamespace(tool_call_id="stream-call"), json.dumps(args))
        yield ActionHistory(
            action_id="complete_stream-call",
            action_type="tool_result",
            role=ActionRole.TOOL,
            status=ActionStatus.SUCCESS if accept else ActionStatus.FAILED,
            input={"function_name": node.tools[0].name, "arguments": args},
            output=result,
        )

    events = []
    existed_before_decision = []
    async for action in node.action_bus.merge(primary()):
        event = action_to_sse_event(
            action,
            len(events),
            action.action_id,
            proxied_tool_names=node.proxied_tool_names,
            tool_previews=node.tool_review_previews,
        )
        events.append(event)
        if event.data.type == SSEDataType.UPDATE_MESSAGE:
            preview = event.data.payload.content[0].payload["preview"]
            existed_before_decision.append((tmp_path / preview["filePath"]).exists())
            await node.tool_channel.publish("stream-call", {"success": int(accept)})

    assert [event.data.type for event in events] == [
        SSEDataType.CREATE_MESSAGE,
        SSEDataType.UPDATE_MESSAGE,
        SSEDataType.CREATE_MESSAGE,
    ]
    update = events[1]
    part = update.data.payload.content[0]
    preview = part.payload["preview"]
    assert update.data.payload.message_id == "stream-call"
    assert part.payload["callToolId"] == "stream-call"
    assert part.payload["proxied"] is True
    assert preview["oldText"] == ""
    assert json.loads(preview["newText"])["plan"]["summary"] == "review through SSE"
    assert existed_before_decision == [False]
    written = {path.name: path.read_text() for path in tmp_path.rglob("semantic-model-plan.json")}
    assert written == ({"semantic-model-plan.json": preview["newText"]} if accept else {})
    assert not node.tool_review_previews
