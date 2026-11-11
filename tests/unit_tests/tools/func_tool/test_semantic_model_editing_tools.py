"""Semantic-domain registration and file boundaries for OSI object mutations."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.metric_filesystem_tools import MetricFilesystemFuncTool
from datus.tools.permission.permission_config import PermissionConfig, PermissionRule
from datus.tools.permission.permission_hooks import FilesystemPolicy, PermissionDeniedException, PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.profiles import get_profile
from datus.tools.registry.contracts import ToolDefinitionError, clone_registered_tool, tool_definition
from datus.tools.registry.tool_registry import ToolRegistry


@pytest.fixture
def file_context(tmp_path):
    """Give both providers the same isolated working root."""
    root = tmp_path / "project"
    root.mkdir()
    return MetricFilesystemFuncTool(root_path=str(root), strict=True)


@pytest.mark.asyncio
async def test_each_semantic_edit_keeps_strict_file_boundaries(file_context, tmp_path):
    """All four real SDK tools reject external targets before parsing or writing."""
    target = tmp_path / "outside.yml"
    target.write_text("name: untouched\n")
    editor = file_context.semantic_editor
    for tool in editor.available_tools():
        definition = tool_definition(tool)
        assert definition.tool_id == f"semantic_tools.{tool.name}"
        assert definition.access == "write"
        assert definition.direct_call is False
        args = dict(path=str(target), datasets_json=[], metrics_json=[], metric_names=[], dataset_names=[])
        args = {key: value for key, value in args.items() if key in tool.params_json_schema["properties"]}
        result = await tool.on_invoke_tool(None, json.dumps(args))
        assert result["success"] == 0
        assert target.read_text() == "name: untouched\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["normal", "auto", "dangerous"])
async def test_semantic_edits_cannot_escape_workflow_root(file_context, tmp_path, profile):
    """Moving families must retain the EXTERNAL non-interactive gate."""
    registry = ToolRegistry()
    tools = file_context.semantic_editor.available_tools()
    registry.register_tools("semantic_tools", tools)
    broker = SimpleNamespace(request=AsyncMock())
    manager = PermissionManager(global_config=get_profile(profile), active_profile=profile)
    hooks = PermissionHooks(
        broker,
        manager,
        "semantic_modeling",
        registry,
        fs_policy=FilesystemPolicy(root_path=file_context._root_resolved, current_node="semantic_modeling"),
        non_interactive=True,
    )
    for tool in tools:
        ctx = SimpleNamespace(tool_arguments=json.dumps({"path": str(tmp_path / "outside.yml")}))
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, None, tool)
    broker.request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["normal", "auto"])
async def test_object_deletes_still_require_confirmation(file_context, profile):
    """A semantic domain does not convert object deletion into an internal-path bypass."""
    tools = file_context.semantic_editor.available_tools()
    registry = ToolRegistry()
    registry.register_tools("semantic_tools", tools)
    broker = SimpleNamespace(request=AsyncMock(return_value=[["y"]]))
    manager = PermissionManager(global_config=get_profile(profile), active_profile=profile)
    hooks = PermissionHooks(
        broker,
        manager,
        "semantic_modeling",
        registry,
        fs_policy=FilesystemPolicy(root_path=file_context._root_resolved, current_node="semantic_modeling"),
    )
    for tool in tools:
        before = broker.request.await_count
        await hooks.on_tool_start(SimpleNamespace(tool_arguments='{"path":"model.yml"}'), None, tool)
        expected = tool.name.startswith("delete_") or profile == "normal"
        assert broker.request.await_count - before == int(expected)


@pytest.mark.asyncio
async def test_restricted_host_denial_precedes_semantic_file_bypass(file_context):
    """Even AUTO cannot restore an explicitly denied semantic mutation."""
    tools = file_context.semantic_editor.available_tools()
    registry = ToolRegistry()
    registry.register_tools("semantic_tools", tools)
    config = PermissionConfig(rules=[PermissionRule(tool="semantic_tools", pattern="*", permission="deny")])
    manager = PermissionManager(global_config=config, active_profile="auto")
    broker = SimpleNamespace(request=AsyncMock())
    hooks = PermissionHooks(
        broker,
        manager,
        "semantic_modeling",
        registry,
        fs_policy=FilesystemPolicy(root_path=file_context._root_resolved, current_node="semantic_modeling"),
        enforce_method_denials=True,
    )
    for tool in tools:
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(SimpleNamespace(tool_arguments='{"path":"model.yml"}'), None, tool)
    broker.request.assert_not_called()


@pytest.mark.asyncio
async def test_real_file_operations_keep_their_names_without_legacy_alias(tmp_path):
    """Write/edit/delete remain callable; renaming write_file to create_file is rejected."""
    filesystem = FilesystemFuncTool(root_path=str(tmp_path))
    tools = {tool.name: tool for tool in filesystem.available_tools()}
    for name in ("write_file", "edit_file", "delete_file"):
        definition = tool_definition(tools[name])
        assert definition.tool_id == f"filesystem_tools.{name}"
        assert definition.direct_call is False
    with pytest.raises(ToolDefinitionError):
        clone_registered_tool(tools["write_file"], name="create_file")
    assert (await tools["write_file"].on_invoke_tool(None, '{"path":"probe.txt","content":"before"}'))["success"] == 1
    assert (
        await tools["edit_file"].on_invoke_tool(None, '{"path":"probe.txt","old_string":"before","new_string":"after"}')
    )["success"] == 1
    assert (tmp_path / "probe.txt").read_text() == "after"
    assert (await tools["delete_file"].on_invoke_tool(None, '{"path":"probe.txt"}'))["success"] == 1
    assert not (tmp_path / "probe.txt").exists()
