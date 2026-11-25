"""Behavior of slug-addressed authoring tools and Studio's pinned edit target."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from datus.tools.func_tool import (
    DashboardArtifactTools,
    DashboardFilesystemFuncTool,
    DBFuncTool,
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

    def tools(db_func_tool=None, **kwargs):
        return tools_cls(agent_config=config, db_func_tool=db_func_tool, user_message="Show net revenue.", **kwargs)

    for slug in ("current", "other"):
        creator = tools_cls(agent_config=config, db_func_tool=None, user_message="Create revenue analysis.")
        result = getattr(creator, f"start_new_{kind}")(slug, "Revenue analysis", "Revenue by month")
        assert result.success == 1, result.error
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
        assert (slug_field in validation.params_json_schema["properties"]) == (not pinned)
        result = await validation.on_invoke_tool(None, json.dumps(args))
        assert result["success"] == 1, result["error"]
        assert result["result"]["artifact_slug"] == "current"
        assert result["result"]["app_jsx_path"] == f"{kind}s/current/render/app.jsx"
        assert notes.read_bytes() == before


@pytest.mark.asyncio
async def test_freeform_tools_explain_how_to_recover_the_slug(existing):
    kind, directory, make_tools, _ = existing
    field = f"{kind}_slug"
    tools = make_tools()
    args = {"name": "sales", "goal": "Count sales", "hypothesis": "Sales exist"}
    is_dashboard = kind == "dashboard"
    sql_args = {"sql_template": "SELECT 1", "sample_params": {}} if is_dashboard else {"sql": "SELECT 1"}
    metric_args = {"params": [], "sample_params": {}} if is_dashboard else {}
    inputs = {
        "validate_render": {},
        "save_query_template" if is_dashboard else "save_query": {**args, **sql_args},
        "save_metric_query_template" if is_dashboard else "save_metric_query": {**args, "query": {}, **metric_args},
    }
    history = (directory / "analysis/intent.md").read_bytes()
    for tool in tools.available_tools():
        if tool.name not in inputs:
            continue
        assert field in tool.params_json_schema["required"]
        for target_args in ({}, {field: "Sales summary"}, {field: "unknown"}):
            result = await tool.on_invoke_tool(None, json.dumps({**inputs[tool.name], **target_args}))
            assert result["success"] == 0
            assert f"{kind}s/*/manifest.json" in result["error"]
            assert field in result["error"]
            assert "name" in result["error"] and "retry" in result["error"]
    assert tools.target.slug is None
    assert (directory / "analysis/intent.md").read_bytes() == history


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
    assert fs.edit_file(target, "return null", "return <main/>").success == 1
    assert notes.read_bytes().startswith(before)
    assert "return <main/>" in (directory / "render/app.jsx").read_text()

    def write_component(index):
        return fs.write_file(f"{kind}s/current/render/cards/parts/item{index}.jsx", "export default 1")

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result.success for result in pool.map(write_component, range(8)))
    fs.artifact_target = make_tools(requested_slug="current", intent_timestamp=tools.target.intent_timestamp).target
    assert fs.delete_file(f"{kind}s/current/render/cards/parts/item0.jsx").success == 1
    assert notes.read_text().count("Show net revenue.") == 1
    fs.artifact_target = make_tools(requested_slug="current").target
    assert fs.delete_file(f"{kind}s/current/render/cards/parts/item1.jsx").success == 1
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


def test_create_can_retry_after_manifest_write_failure(existing, monkeypatch):
    kind, directory, make_tools, _ = existing
    create = getattr(make_tools(), f"start_new_{kind}")
    original = (directory / "manifest.json").read_bytes()

    def fail_write(*args):
        raise OSError("manifest write failed")

    with monkeypatch.context() as patch:
        patch.setattr("datus.tools.func_tool._visual_artifact_target._atomic_write_text", fail_write)
        assert not create("current", "Duplicate", "Existing target").success
        assert not create("retry", "Retry", "New target").success
    assert (directory / "manifest.json").read_bytes() == original
    assert create("retry", "Retry", "New target").success == 1


def test_resume_existing_manifest_before_first_render(existing):
    kind, directory, make_tools, fs = existing
    from datus.tools.db_tools.config import SQLiteConfig
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector

    db = directory.parent.parent / "sales.sqlite"
    db.touch()
    first = make_tools(db_func_tool=DBFuncTool(connector_or_manager=SQLiteConnector(SQLiteConfig(db_path=str(db)))))
    assert getattr(first, f"start_new_{kind}")("resume", "Resume me", "Interrupted creation").success == 1
    args = dict(name="total", goal="Show the baseline", hypothesis="The baseline is one")
    if kind == "report":
        saved = first.save_query(sql="SELECT 1 AS total", **args)
    else:
        saved = first.save_query_template(sql_template="-- @datus-params\nSELECT 1 AS total", sample_params={}, **args)
    assert saved.success == 1, saved.error
    queries = first.queries_dir
    before = {p.name: p.read_bytes() for p in queries.iterdir()}
    resumed = make_tools()
    fs.artifact_target = resumed.target
    incomplete = resumed.validate_render("resume")
    assert incomplete.success == 0 and "app.jsx" in incomplete.error
    path = f"{kind}s/resume/render/app.jsx"
    write = fs.write_file(path, "export default function App() { return null; }")
    assert write.success == 1, write.error
    assert resumed.validate_render("resume").result["app_jsx_path"] == path
    assert {p.name: p.read_bytes() for p in queries.iterdir()} == before


@pytest.mark.parametrize("invalid", ["missing", "malformed", "wrong_slug", "wrong_kind"])
def test_incomplete_artifact_requires_matching_manifest(existing, invalid):
    kind, directory, make_tools, fs = existing
    target = directory.parent / "incomplete"
    target.mkdir()
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["slug"] = "other" if invalid == "wrong_slug" else "incomplete"
    if invalid == "wrong_kind":
        manifest["kind"] = "dashboard" if kind == "report" else "report"
    if invalid != "missing":
        (target / "manifest.json").write_text("{" if invalid == "malformed" else json.dumps(manifest))
    fs.artifact_target = make_tools().target
    write = fs.write_file(f"{kind}s/incomplete/render/app.jsx", "export default 1")
    assert write.success == 0
    assert not (target / "render/app.jsx").exists()
