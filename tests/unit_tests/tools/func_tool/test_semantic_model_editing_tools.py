"""Semantic-domain registration and file boundaries for OSI object mutations."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.semantic_model_editing_tools import SemanticModelEditingTools
from datus.tools.func_tool.semantic_model_filesystem_tools import SemanticModelFilesystemTools
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
    return SemanticModelFilesystemTools(root_path=str(root), strict=True)


@pytest.mark.asyncio
async def test_each_semantic_edit_keeps_strict_file_boundaries(file_context, tmp_path):
    """All four real SDK tools reject external targets before parsing or writing."""
    target = tmp_path / "outside.yml"
    target.write_text("name: untouched\n")
    editor = SemanticModelEditingTools(file_context.workspace)
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
    tools = SemanticModelEditingTools(file_context.workspace).available_tools()
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
    tools = SemanticModelEditingTools(file_context.workspace).available_tools()
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
    tools = SemanticModelEditingTools(file_context.workspace).available_tools()
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


@pytest.fixture
def bound_editor(tmp_path):
    """Bind real file revision tracking before testing input and persistence failures."""
    import hashlib

    import yaml

    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState

    document = {
        "version": "0.2.0.dev0",
        "name": "sales",
        "datasets": [{"name": "orders", "source": "orders"}],
        "metrics": [
            {
                "name": "revenue",
                "description": "Revenue",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.amount)"}]},
            }
        ],
    }
    target = tmp_path / "model.yml"
    target.write_text(yaml.safe_dump(document))
    state = OsiSemanticModelTargetState()

    def bind():
        """Bind the current bytes as the selected authoring revision."""
        state.select(
            {
                "semantic_model_name": "sales",
                "semantic_model_file": "model.yml",
                "absolute_path": str(target),
                "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
            mode="bound",
        )

    bind()
    filesystem = SemanticModelFilesystemTools(root_path=str(tmp_path), osi_target_state=state)
    return SemanticModelEditingTools(filesystem.workspace), target, bind


@pytest.mark.parametrize(
    "method,parameter,payload,message",
    [
        ("upsert_osi_datasets", "datasets_json", [], "non-empty"),
        ("upsert_osi_datasets", "datasets_json", [42], "JSON object"),
        ("upsert_osi_datasets", "datasets_json", [{}], "name is required"),
        ("upsert_osi_datasets", "datasets_json", [{"name": "a"}, {"name": "a"}], "duplicate"),
        ("upsert_osi_metrics", "metrics_json", [], "non-empty"),
        ("upsert_osi_metrics", "metrics_json", [42], "JSON object"),
        ("upsert_osi_metrics", "metrics_json", [{}], "name is required"),
        ("upsert_osi_metrics", "metrics_json", [{"name": "a"}, {"name": "a"}], "duplicate"),
        ("delete_osi_datasets", "dataset_names", "orders", "JSON array"),
        ("delete_osi_datasets", "dataset_names", [" "], "non-empty"),
        ("delete_osi_metrics", "metric_names", "revenue", "list"),
        ("delete_osi_metrics", "metric_names", [" "], "non-empty"),
    ],
)
def test_malformed_objects_never_mutate_the_bound_model(bound_editor, method, parameter, payload, message):
    """Bad object batches are rejected before revision or file state can change."""
    editor, target, _ = bound_editor
    original = target.read_bytes()
    result = getattr(editor, method)(str(target), **{parameter: payload})
    assert result.success == 0 and message.lower() in result.error.lower()
    assert target.read_bytes() == original
    editor.workspace.target_state.require_current_revision(target)


@pytest.mark.parametrize(
    "method,parameter,payload",
    [
        ("upsert_osi_datasets", "datasets_json", [{"name": "orders", "source": "new_orders"}]),
        (
            "upsert_osi_metrics",
            "metrics_json",
            [
                {
                    "name": "revenue",
                    "description": "Revenue",
                    "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]},
                }
            ],
        ),
        ("delete_osi_datasets", "dataset_names", ["orders"]),
        ("delete_osi_metrics", "metric_names", ["revenue"]),
    ],
)
@pytest.mark.parametrize(
    "content", [b"[]\n", b"name: sales\n", b"datasets: wrong\nmetrics: wrong\n", b"[unclosed", b"\xff"]
)
def test_corrupt_models_are_preserved_for_repair(bound_editor, method, parameter, payload, content):
    """Invalid existing documents must not be silently replaced by a valid batch."""
    editor, target, bind = bound_editor
    target.write_bytes(content)
    bind()
    result = getattr(editor, method)(str(target), **{parameter: payload})
    assert result.success == 0
    assert any(word in result.error for word in ("load", "root", "datasets", "metrics", "model"))
    assert target.read_bytes() == content


@pytest.mark.parametrize(
    "method,parameter,payload",
    [
        ("upsert_osi_datasets", "datasets_json", [{"name": "orders", "source": "new_orders"}]),
        (
            "upsert_osi_metrics",
            "metrics_json",
            [
                {
                    "name": "revenue",
                    "description": "Revenue",
                    "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]},
                }
            ],
        ),
        ("delete_osi_datasets", "dataset_names", ["orders"]),
        ("delete_osi_metrics", "metric_names", ["revenue"]),
    ],
)
def test_disk_errors_preserve_the_original_revision(bound_editor, monkeypatch, method, parameter, payload):
    """A failed atomic write reports failure without publishing a new revision."""
    editor, target, _ = bound_editor
    original = target.read_bytes()
    monkeypatch.setattr(editor.workspace, "validate_document", lambda _: None)

    def fail_write(*_args):
        """Simulate an unavailable disk at the atomic commit boundary."""
        raise OSError("disk full")

    monkeypatch.setattr("datus.tools.func_tool.semantic_model_editing_tools.atomic_write_text", fail_write)
    result = getattr(editor, method)(str(target), **{parameter: payload})
    assert result.success == 0 and "disk full" in result.error
    assert target.read_bytes() == original
    editor.workspace.target_state.require_current_revision(target)
