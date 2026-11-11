"""Cross-provider authoring state and generic path-policy integration."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState
from datus.tools.func_tool.semantic_model_editing_tools import SemanticModelEditingTools
from datus.tools.func_tool.semantic_model_filesystem_tools import SemanticModelFilesystemTools
from datus.tools.func_tool.semantic_model_workspace import SemanticModelWorkspace
from datus.tools.registry.contracts import tool_definition


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["bound", "planned"])
async def test_file_and_object_sdk_edits_share_revision_and_whole_run_rollback(tmp_path, mode):
    """Alternating real tool calls retain one snapshot, context and publish evidence."""
    target = tmp_path / "sales.yml"
    dataset = {
        "name": "orders",
        "source": "orders",
        "fields": [{"name": "amount", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "amount"}]}}],
    }
    original = yaml.safe_dump(
        {
            "version": "0.2.0.dev0",
            "name": "sales",
            "description": "Original model",
            "datasets": [dataset],
            "metrics": [],
        },
        sort_keys=False,
    ).encode()
    if mode == "bound":
        target.write_bytes(original)
    state = OsiSemanticModelTargetState()
    state.select(
        {
            "semantic_model_name": "sales",
            "semantic_model_file": target.name,
            "absolute_path": str(target),
            "artifact_sha256": hashlib.sha256(original).hexdigest() if mode == "bound" else "",
        },
        mode=mode,
    )
    evidence = GenerationEvidence()

    def record_mutation(path):
        """Apply the same revision and evidence callback as the real authoring node."""
        evidence.record_artifact_mutation(path)
        state.record_selected_write()

    filesystem = SemanticModelFilesystemTools(
        root_path=str(tmp_path),
        osi_target_state=state,
        generation_evidence=evidence,
        mutation_guard=state.require_selected_path,
        mutation_callback=record_mutation,
    )
    workspace = filesystem.workspace
    editor = SemanticModelEditingTools(workspace)
    tools = {tool.name: tool for tool in [*filesystem.available_tools(), *editor.available_tools()]}
    context = SimpleNamespace(context={})
    created = await tools["upsert_osi_datasets"].on_invoke_tool(
        context, json.dumps({"path": target.name, "datasets_json": [dataset]})
    )
    assert created["success"] == 1, created
    state.require_selected_revision(target)
    assert filesystem.tool_ctx.get() is context

    evidence.validation_passed = True
    content = target.read_text()
    changed = (
        content.replace("name: sales", "name: sales\ndescription: Edited model")
        if mode == "planned"
        else content.replace("Original model", "Edited model")
    )
    edited = await tools["edit_file"].on_invoke_tool(
        context, json.dumps({"path": target.name, "old_string": content, "new_string": changed})
    )
    assert edited["success"] == 1, edited
    state.require_selected_revision(target)
    assert evidence.validation_passed is False
    assert tool_definition(tools["edit_file"]).tool_id == "filesystem_tools.edit_file"

    metric = {
        "name": "revenue",
        "description": "Revenue",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.amount)"}]},
    }
    updated = await tools["upsert_osi_metrics"].on_invoke_tool(
        context, json.dumps({"path": target.name, "metrics_json": [metric]})
    )
    assert updated["success"] == 1, updated
    assert tool_definition(tools["upsert_osi_metrics"]).tool_id == "semantic_tools.upsert_osi_metrics"
    state.require_selected_revision(target)
    assert workspace.ensure_metric_uids(str(target))["revenue"]
    assert state.touched_metric_names == ["revenue"]
    assert workspace.rollback_failed_authoring()
    if mode == "bound":
        assert target.read_bytes() == original
        state.require_selected_revision(target)
    else:
        assert not target.exists()
        assert state.artifact_sha256 == ""
    assert not state.target_mutated
    assert state.touched_metric_names == []
    assert workspace.rollback_failed_authoring() is False


def test_workspace_can_use_generic_file_access_without_semantic_filesystem(tmp_path):
    """Object mutations require access policy and state, not a semantic file subclass."""
    filesystem = FilesystemFuncTool(root_path=str(tmp_path), strict=True)
    workspace = SemanticModelWorkspace(
        resolve_path=filesystem._classify,
        reject_external_path=filesystem._strict_reject,
        strict=filesystem.strict,
        mutation_guard=filesystem._mutation_guard_error,
        notify_mutation=filesystem._notify_mutation,
        set_tool_context=filesystem.set_tool_context,
    )
    editor = SemanticModelEditingTools(workspace)
    outside = tmp_path.parent / "outside.yml"
    result = editor.upsert_osi_datasets(str(outside), [{"name": "orders", "source": "orders"}])
    assert result.success == 0
    assert "outside" in result.error.lower()
    assert workspace.rollback_failed_authoring() is False


def test_semantic_raw_edit_validates_without_changing_generic_file_operations(tmp_path):
    """A semantic provider rejects non-OSI text while the generic provider edits it."""
    target = tmp_path / "notes.txt"
    target.write_text("before")
    semantic = SemanticModelFilesystemTools(root_path=str(tmp_path))
    rejected = semantic.edit_file(target.name, "before", "after")
    assert rejected.success == 0
    assert "OSI" in rejected.error
    assert target.read_text() == "before"
    generic = FilesystemFuncTool(root_path=str(tmp_path))
    assert generic.edit_file(target.name, "before", "after").success == 1
    assert target.read_text() == "after"
