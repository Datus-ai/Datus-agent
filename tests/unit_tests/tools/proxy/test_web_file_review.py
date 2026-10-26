"""One web confirmation preserves server-owned file tools across every child type."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agents import FunctionTool
from fastapi import HTTPException

from datus.api.models.chat_models import ToolReviewInput
from datus.api.routes.chat_routes import execute_tool_review
from datus.api.services.action_sse_converter import _build_tool_call_content
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_hooks import PermissionDeniedException, PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.profiles import get_profile
from datus.tools.proxy.proxy_tool import _FS_DEPENDENT_NODES, apply_proxy_tools
from datus.tools.proxy.tool_result_channel import ToolResultChannel
from datus.tools.proxy.web_file_review import apply_web_file_review, stamp_tool_execution
from datus.tools.registry.tool_registry import ToolRegistry


def make_node(name, tools):
    return SimpleNamespace(
        tools=tools,
        tool_channel=ToolResultChannel(),
        tool_registry=ToolRegistry({t.name: "filesystem_tools" for t in tools}),
        proxied_tool_names=set(),
        file_review_tool_names=set(),
        get_node_name=lambda: name,
        execution_mode="interactive",
        scope="owner",
    )


def tool_action(name="write_file"):
    return ActionHistory(
        action_id="call-1",
        role=ActionRole.TOOL,
        action_type="tool_call",
        input={"function_name": name, "arguments": {}},
        status=ActionStatus.PROCESSING,
    )


async def start_call(node, name, args):
    tool = next(t for t in node.tools if t.name == name)
    task = asyncio.create_task(tool.on_invoke_tool(SimpleNamespace(tool_call_id="call-1"), json.dumps(args)))
    for _ in range(10):
        if "call-1" in node.web_file_review_context.pending:
            break
        await asyncio.sleep(0)
    return task, node.web_file_review_context.pending["call-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("node_name", sorted(_FS_DEPENDENT_NODES))
@pytest.mark.parametrize("operation", ["write_file", "edit_file", "delete_file"])
async def test_all_server_file_children_use_one_card_and_original_executor(tmp_path, node_name, operation):
    path = tmp_path / "note.txt"
    path.write_text("before")
    guard, callback = Mock(), Mock()
    fs = FilesystemFuncTool(root_path=str(tmp_path), strict=True, mutation_guard=guard, mutation_callback=callback)
    node = make_node(node_name, fs.available_tools())
    parent = make_node("chat", fs.available_tools())
    apply_proxy_tools(parent, ["write_file", "edit_file", "delete_file"])
    apply_web_file_review(parent)
    apply_proxy_tools(node, parent.proxy_tool_patterns, channel=parent.tool_channel)
    apply_web_file_review(node, context=parent.web_file_review_context)
    broker = SimpleNamespace(request=AsyncMock())
    hooks = PermissionHooks(
        broker=broker,
        permission_manager=PermissionManager(get_profile("normal")),
        node_name=node_name,
        tool_registry=node.tool_registry,
        proxied_tool_names=node.proxied_tool_names,
        file_review_tool_names=node.file_review_tool_names,
    )
    args = {"path": "note.txt", "content": "after"} if operation == "write_file" else {"path": "note.txt"}
    if operation == "edit_file":
        args.update(old_string="before", new_string="after")
    await hooks.on_tool_start(
        SimpleNamespace(tool_arguments=json.dumps(args)), None, next(t for t in node.tools if t.name == operation)
    )
    broker.request.assert_not_called()
    action = tool_action(operation)
    stamp_tool_execution(action, node)
    payload = _build_tool_call_content(action, parent.proxied_tool_names)[0].payload
    assert payload["proxied"] is False and payload["serverReview"] is True
    # Forwarding through another ancestor must retain the child's ownership.
    stamp_tool_execution(action, parent)
    assert action.tool_execution == "server_review"

    running, review = await start_call(node, operation, args)
    assert path.read_text() == "before"
    result, duplicate = await asyncio.gather(review.execute(), review.execute())
    assert result == duplicate and result["success"] == 1
    guard.assert_called_once()
    callback.assert_called_once()
    assert not running.done()  # Accept & Edit has not finished yet.
    if operation == "delete_file":
        assert not path.exists()
    else:
        assert path.read_text() == "after"
        path.write_text("user edit")
    await parent.tool_channel.publish("call-1", {"success": 1, "result": "User finished editing"})
    assert (await running)["success"] == 1
    assert not parent.web_file_review_context.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [False, True])
async def test_guard_failure_and_cancel_never_write(tmp_path, accept):
    guard = Mock(side_effect=ValueError("outside authoring target"))
    fs = FilesystemFuncTool(root_path=str(tmp_path), strict=True, mutation_guard=guard)
    node = make_node("semantic_modeling", fs.available_tools())
    apply_web_file_review(node)
    running, review = await start_call(node, "write_file", {"path": "blocked.txt", "content": "bad"})
    if accept:
        result = await review.execute()
        assert result["success"] == 0 and "outside authoring target" in result["error"]
    else:
        await node.tool_channel.publish("call-1", {"success": 0, "error": "cancel"})
    assert (await running)["success"] == 0
    assert not (tmp_path / "blocked.txt").exists()


@pytest.mark.asyncio
async def test_review_keeps_explicit_deny_and_workflow_restrictions(tmp_path):
    fs = FilesystemFuncTool(root_path=str(tmp_path))
    node = make_node("gen_sql_summary", fs.available_tools())
    apply_web_file_review(node)
    config = get_profile("normal").merge_with(
        PermissionConfig(
            rules=[PermissionRule(tool="filesystem_tools", pattern="write_file", permission=PermissionLevel.DENY)]
        )
    )
    hooks = PermissionHooks(
        broker=None,
        permission_manager=PermissionManager(config),
        node_name="gen_sql_summary",
        tool_registry=node.tool_registry,
        file_review_tool_names=node.file_review_tool_names,
    )
    with pytest.raises(PermissionDeniedException):
        await hooks.on_tool_start(
            SimpleNamespace(tool_arguments="{}"), None, next(t for t in node.tools if t.name == "write_file")
        )
    workflow = make_node("gen_sql_summary", fs.available_tools())
    workflow.execution_mode = "workflow"
    apply_web_file_review(workflow)
    assert not workflow.file_review_tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["waiting", "editing"])
async def test_stopping_review_releases_waiters_without_reexecution(tmp_path, phase):
    fs = FilesystemFuncTool(root_path=str(tmp_path))
    node = make_node("gen_sql_summary", fs.available_tools())
    apply_web_file_review(node)
    running, review = await start_call(node, "write_file", {"path": "note.txt", "content": "after"})
    execution = None
    if phase == "editing":
        execution = await review.execute()
    assert (execution or {}).get("success") == (1 if phase == "editing" else None)
    node.tool_channel.cancel_all("stop")
    assert (await running)["success"] == 0
    assert not node.web_file_review_context.pending
    assert (tmp_path / "note.txt").exists() == (phase == "editing")


@pytest.mark.asyncio
async def test_http_approval_scoped_and_idempotent_and_ignores_browser_arguments(tmp_path):
    fs = FilesystemFuncTool(root_path=str(tmp_path), strict=True)
    node = make_node("semantic_modeling", fs.available_tools())
    apply_web_file_review(node)
    running, _ = await start_call(node, "write_file", {"path": "note.txt", "content": "frozen"})
    svc = SimpleNamespace(task_manager=SimpleNamespace(get_task=lambda _: SimpleNamespace(node=node, status="running")))
    request = ToolReviewInput(session_id="s", call_tool_id="call-1", content="forged")
    with pytest.raises(HTTPException) as error:
        await execute_tool_review(request, svc, SimpleNamespace(user_id="other"))
    assert error.value.status_code == 404
    first, second = await asyncio.gather(
        *[execute_tool_review(request, svc, SimpleNamespace(user_id="owner")) for _ in range(2)]
    )
    assert first.data == second.data and first.data["success"] == 1
    assert (tmp_path / "note.txt").read_text() == "frozen"
    await node.tool_channel.publish("call-1", {"success": 1})
    await running


def test_client_children_and_nested_server_calls_keep_their_own_execution():
    tool = FunctionTool(
        name="write_file", description="write", params_json_schema={"type": "object"}, on_invoke_tool=AsyncMock()
    )
    client = make_node("custom_child", [tool])
    apply_proxy_tools(client, ["write_file"])
    apply_web_file_review(client)
    action = tool_action()
    stamp_tool_execution(action, client)
    payload = _build_tool_call_content(action, set())[0].payload
    assert payload["proxied"] is True and not payload.get("serverReview")
    server = make_node("gen_visual_report", [tool])
    action = tool_action()
    stamp_tool_execution(action, server)
    stamp_tool_execution(action, client)
    assert _build_tool_call_content(action, {"write_file"})[0].payload["proxied"] is False


@pytest.mark.asyncio
async def test_unanswered_review_expires_and_late_approval_cannot_execute(tmp_path, monkeypatch):
    monkeypatch.setattr("datus.tools.proxy.web_file_review.DEFAULT_RESULT_TIMEOUT", 0.001)
    fs = FilesystemFuncTool(root_path=str(tmp_path))
    node = make_node("gen_sql_summary", fs.available_tools())
    apply_web_file_review(node)
    running, review = await start_call(node, "write_file", {"path": "note.txt", "content": "after"})
    assert (await running)["success"] == 0
    assert (await review.execute())["success"] == 0
    assert not (tmp_path / "note.txt").exists()


@pytest.mark.asyncio
async def test_route_rejects_stale_or_unknown_reviews_without_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("datus.api.routes.chat_routes.asyncio.sleep", AsyncMock())
    node = make_node("gen_sql_summary", [])
    apply_web_file_review(node)
    task = SimpleNamespace(node=node, status="running")
    svc = SimpleNamespace(task_manager=SimpleNamespace(get_task=lambda _: task))
    with pytest.raises(HTTPException):
        await execute_tool_review(
            ToolReviewInput(session_id="s", call_tool_id="missing"), svc, SimpleNamespace(user_id="owner")
        )
    task.status = "completed"
    with pytest.raises(HTTPException):
        await execute_tool_review(
            ToolReviewInput(session_id="s", call_tool_id="missing"), svc, SimpleNamespace(user_id="owner")
        )


@pytest.mark.asyncio
async def test_failed_executor_releases_http_waiter_and_never_trusts_a_success_report():
    original = AsyncMock(side_effect=ValueError("validation failed"))
    tool = FunctionTool(
        name="edit_file", description="edit", params_json_schema={"type": "object"}, on_invoke_tool=original
    )
    node = make_node("semantic_modeling", [tool])
    apply_web_file_review(node)
    running, review = await start_call(node, "edit_file", {})
    approval = asyncio.create_task(review.execute())
    with pytest.raises(ValueError, match="validation failed"):
        await running
    assert (await approval)["success"] == 0
    original.assert_awaited_once()
