# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``GenVisualReportAgenticNode``.

Design principle: NO mocks except LLM.

Covers:
* Node initialization wires the expected tools.
* ``ReportFilesystemFuncTool`` replaces the default filesystem tool.
* ``_prepare_artifacts`` registers the artifact tools but leaves the
  report slug unbound — the LLM owns the new/edit decision at runtime.
* End-to-end streaming run: LLM calls bind_existing_report (against a
  pre-seeded dir) and validate_render.
* CLI mode compiles ``index.html`` after a successful validate_render.
* Binding-required failure when the LLM never calls start/bind_report.
* Incomplete-artifact failure when validate_render is never called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_visual_report_models import GenVisualReportNodeInput
from datus.tools.func_tool import (
    DBFuncTool,
    ReportArtifactTools,
    ReportFilesystemFuncTool,
    SemanticTools,
)
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_tool_then_response,
)


def _make_node(real_agent_config, **overrides):
    from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode

    kwargs = dict(
        node_id="vr_node_test",
        description="Visual report node",
        node_type=NodeType.TYPE_GEN_VISUAL_REPORT,
        agent_config=real_agent_config,
        node_name="gen_visual_report",
    )
    kwargs.update(overrides)
    return GenVisualReportAgenticNode(**kwargs)


_APP_JSX_TEMPLATE = """\
/** @datus-title {title} */
import React from 'react';
import {{ useDatusArtifact }} from '@datus/web-artifact';

export default function App() {{
  const {{ useQuerySql }} = useDatusArtifact();
  const {{ data }} = useQuerySql('{data_ref}');
  return React.createElement('pre', null, JSON.stringify(data?.rows ?? []));
}}
"""


def _seed_render_on_disk(project_root: Path, report_slug: str, *, data_ref: str = "queries/q") -> None:
    """Seed a minimal validated render tree + matching query so renderer-side tests run."""
    report_dir = project_root / "reports" / report_slug
    (report_dir / "queries").mkdir(parents=True, exist_ok=True)
    (report_dir / "render").mkdir(exist_ok=True)
    (report_dir / "render" / "app.jsx").write_text(
        _APP_JSX_TEMPLATE.format(title="stub", data_ref=data_ref),
        encoding="utf-8",
    )
    slug = data_ref.split("/", 1)[-1]
    (report_dir / "queries" / f"{slug}.sql").write_text("SELECT 1", encoding="utf-8")
    (report_dir / "queries" / f"{slug}.json").write_text(
        '{"executed_at":"2026-05-13T00:00:00Z","datasource":"x","row_count":0,'
        '"columns":[{"name":"a","type":"integer"}],"rows":[]}',
        encoding="utf-8",
    )
    # manifest.json is part of the report contract — validate_render rejects
    # the artifact if it's missing.
    (report_dir / "manifest.json").write_text(
        f'{{"slug":"{report_slug}","name":"seeded report","description":"Unit-test seeded report.",'
        '"kind":"report","created_at":"2026-05-13T00:00:00Z"}\n',
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Initialization                                                              #
# --------------------------------------------------------------------------- #


class TestGenVisualReportInit:
    def test_basic_init(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        assert node.get_node_name() == "gen_visual_report"
        assert isinstance(node.db_func_tool, DBFuncTool)
        assert isinstance(node.semantic_tools, SemanticTools)
        assert isinstance(node.filesystem_func_tool, ReportFilesystemFuncTool)
        assert node.report_artifact_tools is None
        assert node._active_report_slug is None

    def test_tools_include_filesystem_and_db(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        tool_names = {t.name for t in node.tools}
        # DB tool surface
        assert "list_tables" in tool_names
        # Filesystem tool surface — write_file is how the LLM authors render/
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "delete_file" in tool_names
        # Pre-execution: artifact tools are not registered yet
        assert "save_query" not in tool_names
        assert "validate_render" not in tool_names

    def test_tool_category_map_registers_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Filesystem tools must be declared in the ``filesystem_tools``
        category so ``PermissionHooks._handle_filesystem_zone`` engages and
        ``apply_proxy_tools`` recognises them as excluded via
        ``_FS_DEPENDENT_NODES``.
        """
        node = _make_node(real_agent_config)
        mapping = node._tool_category_map()
        assert "filesystem_tools" in mapping
        fs_tool_names = {t.name for t in mapping["filesystem_tools"]}
        assert {"read_file", "write_file", "edit_file", "delete_file"}.issubset(fs_tool_names)
        # db_tools and semantic_tools also surface under their own categories.
        assert "db_tools" in mapping
        assert "semantic_tools" in mapping

    def test_metric_discovery_tools_exposed_when_metrics_present(self, real_agent_config, mock_llm_create, monkeypatch):
        """When the project has indexed metrics, the visual report node must
        expose ``search_metrics`` and ``get_metrics`` to the LLM —
        otherwise the prompt advertises metric-first discovery while the
        tools are silently missing, and the model falls back to deriving
        SQL from raw table DDL. ``ContextSearchTools`` gates these tools
        on ``has_metrics``; patching ``_show_metrics`` to ``True`` is the
        minimal way to simulate a metric-rich project in a unit test and
        also exercises the ``context_search_tools.*`` wildcard expansion
        end-to-end.
        """
        from datus.tools.func_tool.context_search import ContextSearchTools

        monkeypatch.setattr(ContextSearchTools, "_show_metrics", lambda self: True)

        node = _make_node(real_agent_config)
        assert isinstance(node.context_search_tools, ContextSearchTools)
        tool_names = {t.name for t in node.tools}
        # search_metrics + get_metrics prove the metric discovery branch fired.
        # list_subject_tree is force-included whenever any context_search
        # branch fires, so it's a canary for the wildcard wiring itself.
        assert {"search_metrics", "get_metrics", "list_subject_tree"}.issubset(tool_names)

    def test_apply_proxy_tools_keeps_filesystem_tools_unwrapped(self, real_agent_config, mock_llm_create):
        """End-to-end check on a real node: ``apply_proxy_tools`` invoked
        with the web-source pattern ``["write_file", "edit_file"]`` must
        leave both filesystem tools un-proxied because
        ``gen_visual_report`` is in ``_FS_DEPENDENT_NODES``.

        Guards against a time-of-check regression: the exclusion only
        fires when ``tool_registry`` is already populated, so this test
        does NOT pre-fill the registry — it relies on
        ``apply_proxy_tools`` triggering ``_populate_tool_registry``
        eagerly.
        """
        from datus.tools.proxy.proxy_tool import apply_proxy_tools

        node = _make_node(real_agent_config)
        # Snapshot the original ``on_invoke_tool`` callable for each fs
        # tool before applying the proxy wrapper.
        before = {t.name: t.on_invoke_tool for t in node.tools if t.name in {"write_file", "edit_file"}}
        assert before, "test setup: expected write_file/edit_file in node.tools"

        apply_proxy_tools(node, ["write_file", "edit_file"])

        after = {t.name: t.on_invoke_tool for t in node.tools if t.name in {"write_file", "edit_file"}}
        for name, original in before.items():
            assert after[name] is original, f"{name} was proxied despite gen_visual_report fs-dependent exclusion"


# --------------------------------------------------------------------------- #
# Pre-execution artifact wiring                                               #
# --------------------------------------------------------------------------- #


class TestPrepareReportArtifacts:
    def test_registers_intent_tools_without_binding(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        user_input = GenVisualReportNodeInput(user_message="北美一季度门店销售分析")
        node.input = user_input

        node._prepare_artifacts(user_input)

        assert isinstance(node.report_artifact_tools, ReportArtifactTools)
        assert node._active_artifact_slug is None
        assert node.report_artifact_tools.report_slug is None
        assert node.report_artifact_tools.mode is None

        tool_names = {t.name for t in node.tools}
        assert "start_new_report" in tool_names
        assert "bind_existing_report" in tool_names
        assert "save_query" in tool_names
        assert "validate_render" in tool_names

        reports_root = Path(real_agent_config.project_root) / "reports"
        # No directories are created until the LLM commits to a slug.
        assert not reports_root.exists() or sorted(p.name for p in reports_root.iterdir()) == []


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_stream_end_to_end(real_agent_config, mock_llm_create):
    """LLM binds an existing report (pre-seeded on disk) and validates the render tree.

    This covers the agentic node's result-extraction + html-compile path
    without needing the mock LLM to author a fresh render tree end-to-end
    (that path is exercised in the artifact-tools tests). Pre-seeding
    ``render/`` + ``queries/`` + ``manifest.json`` and using
    ``bind_existing_report`` keeps the test purely additive over the unit
    coverage.
    """
    project_root = Path(real_agent_config.project_root)
    existing_slug = "e2e_demo"
    _seed_render_on_disk(project_root, existing_slug, data_ref="queries/avg_sat_reading")

    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(
                        name="bind_existing_report",
                        arguments=json.dumps({"report_slug": existing_slug}),
                    ),
                    MockToolCall(name="validate_render", arguments="{}"),
                ],
                content="Report validated.",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualReportNodeInput(
        user_message=f"check {existing_slug}",
        database="california_schools",
    )

    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)

    final = actions[-1]
    assert final.role == ActionRole.ASSISTANT
    assert final.status == ActionStatus.SUCCESS

    result = final.output
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["report_slug"] == existing_slug
    assert result["app_jsx_path"] == f"reports/{existing_slug}/render/app.jsx"
    assert result["render_file_count"] == 1
    # No save_query in this run — the seed wrote the query file directly.
    assert result["query_count"] == 0

    report_dir = project_root / "reports" / existing_slug
    expected_html_rel = f"reports/{existing_slug}/index.html"
    assert result["html_path"] == expected_html_rel
    assert (report_dir / "index.html").is_file()

    # CLI mode streams a WORKFLOW status message carrying the compiled HTML's
    # absolute path so the user can reopen it after closing the browser tab.
    path_actions = [a for a in actions if a.action_type == "report_html_path"]
    assert len(path_actions) == 1, "expected exactly one report_html_path action in CLI mode"
    path_action = path_actions[0]
    assert path_action.role == ActionRole.WORKFLOW
    abs_html = str((report_dir / "index.html").resolve())
    assert abs_html in path_action.messages
    assert path_action.output["html_path"] == abs_html
    assert path_action.output["url"].startswith("file://")

    # Artifact card fields surface through the result so the SSE artifact
    # event can be built without touching disk a second time.
    assert result["artifact_kind"] == "report"
    assert result["artifact_mode"] == "edit"  # bind_existing_report path
    assert result["name"] == "seeded report"
    assert result["description"] == "Unit-test seeded report."
    assert result["created_at"] == "2026-05-13T00:00:00Z"


class TestReportDistResolution:
    """Verify the CLI flag → node_config priority for offline asset overrides."""

    def _make_dist(self, base: Path, name: str) -> Path:
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.css").write_text(f"/* {name} css */", encoding="utf-8")
        (d / "index.umd.js").write_text(f"/* {name} js */", encoding="utf-8")
        return d

    def test_cli_override_wins_over_node_config(self, real_agent_config, mock_llm_create, tmp_path):
        node_dist = self._make_dist(tmp_path / "vendors", "from-node-config")
        cli_dist = self._make_dist(tmp_path / "vendors", "from-cli-flag")

        node = _make_node(real_agent_config)
        node.node_config["report_dist"] = str(node_dist)
        real_agent_config.report_dist_cli_override = str(cli_dist)

        report_slug = "priority_check_001"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_slug)
        node._active_artifact_slug = report_slug

        html_rel = node._maybe_compile_html(report_slug)
        assert html_rel == f"reports/{report_slug}/index.html"

        copied_css = Path(real_agent_config.project_root) / "reports" / report_slug / "_assets" / "index.css"
        assert copied_css.read_text(encoding="utf-8") == "/* from-cli-flag css */"

    def test_node_config_used_when_cli_flag_absent(self, real_agent_config, mock_llm_create, tmp_path):
        node_dist = self._make_dist(tmp_path / "vendors", "node-only")

        node = _make_node(real_agent_config)
        node.node_config["report_dist"] = str(node_dist)
        if hasattr(real_agent_config, "report_dist_cli_override"):
            delattr(real_agent_config, "report_dist_cli_override")

        report_slug = "priority_check_002"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_slug)
        node._active_artifact_slug = report_slug

        node._maybe_compile_html(report_slug)
        copied_css = Path(real_agent_config.project_root) / "reports" / report_slug / "_assets" / "index.css"
        assert copied_css.read_text(encoding="utf-8") == "/* node-only css */"


class TestHtmlPathStreamMessage:
    """Verify the compiled-HTML absolute path is surfaced into the action stream."""

    def test_post_validate_hook_emits_path_action_in_cli_mode(self, real_agent_config, mock_llm_create):
        from datus.schemas.gen_visual_report_models import GenVisualReportNodeResult

        node = _make_node(real_agent_config)
        report_slug = "path_msg_cli"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_slug)
        node._active_artifact_slug = report_slug

        result = GenVisualReportNodeResult(success=True)
        action = node._post_validate_hook(report_slug, result)

        assert isinstance(action, ActionHistory)
        assert action.action_type == "report_html_path"
        assert action.role == ActionRole.WORKFLOW
        abs_html = str((Path(real_agent_config.project_root) / "reports" / report_slug / "index.html").resolve())
        assert action.output["html_path"] == abs_html
        assert abs_html in action.messages
        # The relative path is still recorded on the result for SaaS/task consumers.
        assert result.html_path == f"reports/{report_slug}/index.html"

    def test_post_validate_hook_returns_none_in_non_cli_mode(self, real_agent_config, mock_llm_create):
        from datus.schemas.gen_visual_report_models import GenVisualReportNodeResult

        # filesystem_strict flips the node into SaaS/API mode, which renders
        # reports dynamically server-side and never compiles a standalone HTML.
        real_agent_config.filesystem_strict = True
        node = _make_node(real_agent_config)
        report_slug = "path_msg_saas"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_slug)
        node._active_artifact_slug = report_slug

        result = GenVisualReportNodeResult(success=True)
        action = node._post_validate_hook(report_slug, result)

        assert action is None
        assert result.html_path is None
        assert not (Path(real_agent_config.project_root) / "reports" / report_slug / "index.html").exists()


class _InlineThread:
    """Synchronous stand-in for ``threading.Thread`` so tests don't need sleeps."""

    def __init__(self, target=None, daemon=False, **kwargs):
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        if self._target is not None:
            self._target()


class TestAutoOpenInBrowser:
    """Verify ``_maybe_open_in_browser`` gates on ``agent_config.report_auto_open``."""

    def test_opens_browser_when_flag_enabled(self, real_agent_config, mock_llm_create, monkeypatch):
        node = _make_node(real_agent_config)
        real_agent_config.report_auto_open = True
        report_slug = "auto_open_yes"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_slug)
        node._active_artifact_slug = report_slug

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_slug)

        assert len(opened) == 1, f"expected one webbrowser.open call, got {opened}"
        assert opened[0].startswith("file://")
        assert opened[0].endswith(f"reports/{report_slug}/index.html")

    def test_does_not_open_when_flag_disabled(self, real_agent_config, mock_llm_create, monkeypatch):
        node = _make_node(real_agent_config)
        real_agent_config.report_auto_open = False
        report_slug = "auto_open_no"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_slug)
        node._active_artifact_slug = report_slug

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_slug)

        assert opened == [], f"webbrowser.open must not be called; got {opened}"

    def test_does_not_open_when_attribute_missing(self, real_agent_config, mock_llm_create, monkeypatch):
        node = _make_node(real_agent_config)
        if hasattr(real_agent_config, "report_auto_open"):
            delattr(real_agent_config, "report_auto_open")
        report_slug = "auto_open_default"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_slug)
        node._active_artifact_slug = report_slug

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_slug)

        assert opened == []


@pytest.mark.asyncio
async def test_execute_stream_without_binding_and_no_answer_marks_failure(real_agent_config, mock_llm_create):
    """No binding AND no prose answer (the run did nothing) → binding-required failure."""
    from tests.unit_tests.mock_llm_model import build_simple_response

    mock_llm_create.reset(
        responses=[
            build_simple_response(""),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualReportNodeInput(user_message="forgetful run")

    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)

    final = actions[-1]
    result = final.output
    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["app_jsx_path"] is None
    assert result["report_slug"] is None
    assert result["query_count"] == 0
    error = result.get("error") or ""
    assert "start_new_report" in error
    assert "bind_existing_report" in error
    # When no binding ever happened, the artifact-card fields stay None —
    # the SSE dispatcher then skips the artifact event and falls through
    # to the regular error path.
    assert result["artifact_kind"] == "report"
    assert result["artifact_mode"] is None
    assert result["name"] is None
    assert result["description"] is None
    assert result["created_at"] is None


@pytest.mark.asyncio
async def test_execute_stream_informational_answer_ends_normally(real_agent_config, mock_llm_create):
    """An informational question answered in prose without binding a report
    ends normally instead of surfacing the internal missing-binding error."""
    from tests.unit_tests.mock_llm_model import build_simple_response

    answer = "Earlier you switched the revenue chart to monthly and added a YoY column."
    mock_llm_create.reset(responses=[build_simple_response(answer)])

    node = _make_node(real_agent_config)
    node.input = GenVisualReportNodeInput(user_message="what did I change?")

    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)

    final = actions[-1]
    assert final.status == ActionStatus.SUCCESS
    result = final.output
    assert isinstance(result, dict)
    assert result["success"] is True
    assert not result.get("error")
    assert result["report_slug"] is None
    assert result["app_jsx_path"] is None
    assert result["response"] == answer


@pytest.mark.asyncio
async def test_execute_stream_bound_but_no_validate_marks_failure(real_agent_config, mock_llm_create):
    """LLM binds but never calls validate_render → distinct incomplete-artifact failure."""
    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(
                        name="start_new_report",
                        arguments=json.dumps(
                            {
                                "slug": "halfway",
                                "name": "halfway",
                                "description": "Bound but never validated — unit-test fixture.",
                            }
                        ),
                    ),
                ],
                content="I bound a report but forgot to finalize.",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualReportNodeInput(user_message="bound-then-quit run")

    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)

    final = actions[-1]
    result = final.output
    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["app_jsx_path"] is None
    assert result["report_slug"] == "halfway"
    assert result["query_count"] == 0
    assert "validate_render never returned success" in (result.get("error") or "")
    # start_new_report ran so mode=new and the manifest fields are populated;
    # the slug exists but the run still counts as failed because
    # validate_render never succeeded.
    assert result["artifact_mode"] == "new"
    assert result["name"] == "halfway"
    assert result["description"] == "Bound but never validated — unit-test fixture."
    assert result["created_at"]  # ISO timestamp written by start_new_report
