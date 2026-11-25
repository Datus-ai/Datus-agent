"""Behavior of slug-addressed authoring tools and Studio's pinned edit target."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from datus.tools.func_tool import (
    DashboardArtifactTools,
    DashboardFilesystemFuncTool,
    ReportArtifactTools,
    ReportFilesystemFuncTool,
)


@pytest.fixture(params=["report", "dashboard"])
def existing(request, tmp_path):
    kind = request.param
    tools_cls, fs_cls = (
        (ReportArtifactTools, ReportFilesystemFuncTool)
        if kind == "report"
        else (DashboardArtifactTools, DashboardFilesystemFuncTool)
    )
    config = SimpleNamespace(project_root=str(tmp_path))

    def tools(**kwargs):
        return tools_cls(agent_config=config, db_func_tool=None, user_message="Show net revenue.", **kwargs)

    for slug in ("current", "other"):
        creator = tools_cls(agent_config=config, db_func_tool=None, user_message="Create revenue analysis.")
        result = getattr(creator, f"start_new_{kind}")(slug, "Revenue analysis", "Revenue by month")
        assert result.success, result.error
        (creator.render_dir / "app.jsx").write_text("export default function App() { return null; }")
    return kind, tmp_path / f"{kind}s/current", tools, fs_cls(root_path=str(tmp_path))


@pytest.mark.asyncio
async def test_sdk_can_validate_an_existing_slug_before_any_write(existing):
    kind, directory, make_tools, _ = existing
    slug_field = f"{kind}_slug"
    notes = directory / "analysis/intent.md"
    before = notes.read_bytes()
    for pinned in (False, True):
        tools = make_tools(requested_slug="current") if pinned else make_tools()
        validation = next(tool for tool in tools.available_tools() if tool.name == "validate_render")
        args = {} if pinned else {slug_field: "current"}
        if not pinned:
            assert slug_field in validation.params_json_schema["required"]
        result = await validation.on_invoke_tool(None, json.dumps(args))
        assert result["success"], result["error"]
        assert result["result"]["artifact_slug"] == "current"
        assert result["result"]["app_jsx_path"] == f"{kind}s/current/render/app.jsx"
        assert notes.read_bytes() == before


def test_successful_file_edits_record_one_intent_even_when_retried(existing):
    kind, directory, make_tools, fs = existing
    notes = directory / "analysis/intent.md"
    before = notes.read_bytes()
    target = f"{kind}s/current/render/app.jsx"
    tools = make_tools(requested_slug="current", intent_timestamp="2026-02-03T10:00:00.123456+00:00")
    fs.artifact_target = tools.target
    assert not fs.edit_file(target, "unknown", "replacement").success
    assert not fs.write_file(f"{kind}s/other/render/app.jsx", "wrong target").success
    assert notes.read_bytes() == before
    assert fs.edit_file(target, "return null", "return <main/>").success
    assert notes.read_bytes().startswith(before)
    assert "return <main/>" in (directory / "render/app.jsx").read_text()

    def write_component(index):
        return fs.write_file(f"{kind}s/current/render/cards/parts/item{index}.jsx", "export default 1")

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result.success for result in pool.map(write_component, range(8)))
    fs.artifact_target = make_tools(requested_slug="current", intent_timestamp=tools.target.intent_timestamp).target
    assert fs.delete_file(f"{kind}s/current/render/cards/parts/item0.jsx").success
    assert notes.read_text().count("Show net revenue.") == 1
    fs.artifact_target = make_tools(requested_slug="current").target
    assert fs.delete_file(f"{kind}s/current/render/cards/parts/item1.jsx").success
    assert notes.read_text().count("Show net revenue.") == 2


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_studio_targets_are_forwarded_to_visual_tools(kind):
    from unittest.mock import MagicMock

    from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode
    from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode
    from datus.agent.node.node_factory import create_node_input
    from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeInput
    from datus.schemas.gen_visual_report_models import GenVisualReportNodeInput

    cls, input_cls = (
        (GenVisualReportAgenticNode, GenVisualReportNodeInput)
        if kind == "report"
        else (GenVisualDashboardAgenticNode, GenVisualDashboardNodeInput)
    )
    node = MagicMock(spec=cls)
    node.agent_config = SimpleNamespace(project_root="/tmp")
    node.db_func_tool = None
    node.semantic_tools = None
    for message in (
        f'Update <artifact-chart-card {kind}-slug="current" chart-id="sales" />',
        f"Repair\n{kind} slug: current\nError: missing label",
    ):
        tools = cls._make_artifact_tools(node, create_node_input(message, node))
        assert tools.target.requested_slug == "current"
    typed = create_node_input("Please fix the heading", node, artifact_slug="current")
    restored = input_cls.model_validate(json.loads(json.dumps(typed.model_dump())))
    assert restored.intent_timestamp == typed.intent_timestamp
    assert cls._make_artifact_tools(node, restored).target.requested_slug == "current"
