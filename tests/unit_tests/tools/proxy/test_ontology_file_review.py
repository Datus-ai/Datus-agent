"""Web ontology decisions use immutable bytes and never change CLI writers."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from datus.api.services.action_sse_converter import _build_tool_call_content
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_hooks import PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.profiles import get_profile
from datus.tools.proxy.proxy_tool import apply_proxy_tools
from datus.tools.proxy.semantic_plan_review import apply_semantic_plan_review
from datus.tools.proxy.tool_result_channel import ToolResultChannel
from datus.tools.registry.tool_registry import ToolRegistry

PATH = "subject/semantic_models/ontology/flights/ontology.yml"
BEFORE = "name: Flights\nontology:\n- concept: Flight\n  description: before\n"
AFTER = BEFORE.replace("before", "reviewed")


class Channel(ToolResultChannel):
    def __init__(self):
        super().__init__()
        self.waiting = asyncio.Event()

    async def wait_for(self, call_id, timeout=None):
        self.waiting.set()
        return await super().wait_for(call_id, timeout)


def node_for(root, *, name="chat", channel=None, guard=None):
    fs = FilesystemFuncTool(root_path=str(root), mutation_guard=guard)
    node = SimpleNamespace(
        filesystem_func_tool=fs,
        tools=fs.available_tools(),
        tool_channel=channel or Channel(),
        proxied_tool_names=set(),
        tool_registry=ToolRegistry(),
        get_node_name=lambda: name,
        permission_manager=PermissionManager(get_profile("normal"), active_profile="normal"),
    )
    node.tool_registry.register_tools("filesystem_tools", node.tools)
    apply_proxy_tools(node, ["write_file", "edit_file", "delete_file"])
    return node


def tool(node, name):
    return next(tool for tool in node.tools if tool.name == name)


async def start(node, name="write_file", args=None, call_id="call"):
    node.tool_channel.waiting.clear()
    args = args or {"path": PATH, "content": AFTER}
    running = asyncio.create_task(
        tool(node, name).on_invoke_tool(SimpleNamespace(tool_call_id=call_id), json.dumps(args))
    )
    await asyncio.wait_for(node.tool_channel.waiting.wait(), 1)
    return running, node.tool_review_previews[call_id]


@pytest.fixture
def target(tmp_path):
    target = tmp_path / PATH
    target.parent.mkdir(parents=True)
    target.write_text(BEFORE)
    return target


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["chat", "semantic_modeling"])
@pytest.mark.parametrize("method", ["write_file", "edit_file"])
@pytest.mark.parametrize("accept", [False, True])
async def test_web_review_preserves_bytes_until_decided(tmp_path, target, name, method, accept):
    node = node_for(tmp_path, name=name)
    previous_proxy_names = set(node.proxied_tool_names)
    apply_semantic_plan_review(node)
    args = (
        {"path": PATH, "content": AFTER}
        if method == "write_file"
        else {"path": PATH, "old_string": "before", "new_string": "reviewed"}
    )
    running, preview = await start(node, method, args)
    assert target.read_text() == BEFORE
    assert preview == {"filePath": PATH, "oldText": BEFORE, "newText": AFTER, "reviewType": "ontology"}
    # A per-path adapter cannot advertise ordinary semantic-model writes as
    # browser-executed merely because this node supports ontology review.
    assert node.proxied_tool_names == previous_proxy_names
    await node.tool_channel.publish("call", {"success": int(accept), "result": {"newText": "untrusted"}})
    result = await running
    assert result["success"] == int(accept)
    assert target.read_text() == (AFTER if accept else BEFORE)
    assert not node.tool_review_previews


@pytest.mark.asyncio
async def test_first_write_and_revised_proposal_after_cancel(tmp_path):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, preview = await start(node)
    assert preview["oldText"] == ""
    await node.tool_channel.publish("call", {"success": 0})
    assert (await running)["success"] == 0
    assert not (tmp_path / PATH).exists()
    revised = AFTER + "description: revised after feedback\n"
    running, second = await start(node, args={"path": PATH, "content": revised}, call_id="revised")
    assert second["oldText"] == ""
    await node.tool_channel.publish("revised", {"success": 1})
    assert (await running)["success"] == 1
    assert (tmp_path / PATH).read_text() == revised


@pytest.mark.asyncio
async def test_native_writer_invalidates_pending_preview(tmp_path, target):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, _ = await start(node)
    concurrent = FilesystemFuncTool(root_path=str(tmp_path))
    assert concurrent.write_file(PATH, "concurrent").success == 1
    await node.tool_channel.publish("call", {"success": 1})
    result = await running
    assert result["success"] == 0
    assert "changed during review" in result["error"]
    assert target.read_text() == "concurrent"


@pytest.mark.asyncio
async def test_delegated_review_shares_parent_preview_and_decision_channel(tmp_path, target):
    parent = node_for(tmp_path)
    apply_semantic_plan_review(parent)
    child = node_for(tmp_path, name="semantic_modeling", channel=parent.tool_channel)
    apply_semantic_plan_review(child, context=parent.semantic_plan_review_context)
    running, preview = await start(child)
    assert parent.tool_review_previews["call"] == preview
    await parent.tool_channel.publish("call", {"success": 1})
    assert (await running)["success"] == 1
    assert target.read_text() == AFTER


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "timeout"])
async def test_stopped_or_expired_review_never_writes(tmp_path, target, reason, monkeypatch):
    monkeypatch.setattr("datus.tools.proxy.ontology_file_review.DEFAULT_RESULT_TIMEOUT", 0.01)
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, _ = await start(node)
    if reason == "stop":
        node.tool_channel.cancel_all("stopped")
    assert (await running)["success"] == 0
    assert target.read_text() == BEFORE
    assert not node.tool_review_previews
    await node.tool_channel.publish("call", {"success": 1})
    assert target.read_text() == BEFORE


@pytest.mark.asyncio
async def test_changed_symlink_target_cannot_redirect_acceptance(tmp_path, target):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    running, _ = await start(node)
    destination = target.with_name("other.yml")
    destination.write_text("not approved")
    target.unlink()
    target.symlink_to(destination)
    await node.tool_channel.publish("call", {"success": 1})
    result = await running
    assert result["success"] == 0
    assert "path changed" in result["error"]
    assert destination.read_text() == "not approved"


@pytest.mark.asyncio
@pytest.mark.parametrize("old", ["", "missing", "n"])
async def test_invalid_replacement_is_rejected_before_review(tmp_path, target, old):
    node = node_for(tmp_path)
    apply_semantic_plan_review(node)
    result = await tool(node, "edit_file").on_invoke_tool(
        SimpleNamespace(tool_call_id="bad"), json.dumps({"path": PATH, "old_string": old, "new_string": "bad"})
    )
    assert result["success"] == 0
    assert target.read_text() == BEFORE
    assert not node.tool_channel.waiting.is_set()


@pytest.mark.asyncio
async def test_native_domain_guard_is_preserved(tmp_path, target):
    def guard(path):
        raise ValueError("selected model only")

    node = node_for(tmp_path, guard=guard)
    apply_semantic_plan_review(node)
    result = await tool(node, "write_file").on_invoke_tool(
        SimpleNamespace(tool_call_id="bad"), json.dumps({"path": PATH, "content": AFTER})
    )
    assert result["success"] == 0
    assert "selected model only" in result["error"]
    assert target.read_text() == BEFORE
    assert not node.tool_channel.waiting.is_set()


@pytest.mark.asyncio
async def test_other_files_keep_the_original_server_executor(tmp_path):
    node = node_for(tmp_path, name="semantic_modeling")
    apply_semantic_plan_review(node)
    result = await tool(node, "write_file").on_invoke_tool(
        SimpleNamespace(tool_call_id="other"), json.dumps({"path": "notes.txt", "content": "ordinary"})
    )
    assert result["success"] == 1
    assert (tmp_path / "notes.txt").read_text() == "ordinary"
    assert not node.tool_review_previews


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [PermissionLevel.ALLOW, PermissionLevel.ASK, PermissionLevel.DENY])
async def test_permission_rules_are_enforced_by_review(tmp_path, target, level):
    node = node_for(tmp_path)
    node.permission_manager = PermissionManager(
        PermissionConfig(rules=[PermissionRule(tool="filesystem_tools", pattern="write_file", permission=level)])
    )
    apply_semantic_plan_review(node)
    hooks = PermissionHooks(
        broker=None,
        permission_manager=node.permission_manager,
        node_name="chat",
        tool_registry=node.tool_registry,
        proxied_tool_names=node.proxied_tool_names,
    )
    hooks._request_user_confirmation = AsyncMock(return_value=True)
    ctx = SimpleNamespace(tool_call_id="call", tool_arguments=json.dumps({"path": PATH, "content": AFTER}))
    if level == PermissionLevel.DENY:
        from datus.tools.permission.permission_hooks import PermissionDeniedException

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, None, tool(node, "write_file"))
        result = await tool(node, "write_file").on_invoke_tool(ctx, ctx.tool_arguments)
        assert result["success"] == 0
        assert target.read_text() == BEFORE
    else:
        await hooks.on_tool_start(ctx, None, tool(node, "write_file"))
        hooks._request_user_confirmation.assert_not_awaited()
        running, _ = await start(node)
        await node.tool_channel.publish("call", {"success": 0})
        assert (await running)["success"] == 0
        assert target.read_text() == BEFORE


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("path", [PATH, "/project/" + PATH, "subject/semantic_models/flights/flights.yml"])
def test_sse_review_metadata_is_per_path_and_web_mode(enabled, path):
    action = SimpleNamespace(input={"function_name": "edit_file", "arguments": {"path": path}}, action_id="call")
    payload = _build_tool_call_content(action, set(), enabled)[0].model_dump()["payload"]
    reviewed = enabled and "/ontology/" in path
    assert payload.get("reviewType") == ("ontology" if reviewed else None)
    assert payload["proxied"] is reviewed


def test_cli_keeps_direct_filesystem_execution(tmp_path, target):
    fs = FilesystemFuncTool(root_path=str(tmp_path))
    result = fs.edit_file(PATH, "before", "CLI")
    assert result.success == 1
    assert target.read_text() == BEFORE.replace("before", "CLI")
