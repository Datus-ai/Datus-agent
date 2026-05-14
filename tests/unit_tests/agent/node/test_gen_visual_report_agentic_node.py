# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``GenVisualReportAgenticNode``.

Design principle: NO mocks except LLM.

Covers:
* Node initialization wires the expected tools.
* ``ReportFilesystemFuncTool`` replaces the default filesystem tool.
* ``_prepare_report_artifacts`` registers the artifact tools but leaves the
  report id unbound — the LLM owns the new/edit decision at runtime.
* End-to-end streaming run: LLM calls start_new_report, save_query,
  write_file (for render/*.jsx) and finally validate_render.
* The LLM-facing hint surfaces when the user references existing reports on disk.
* CLI mode compiles ``index.html`` after a successful validate_render.
* Binding-required failure when the LLM never calls start/bind_report.
* Incomplete-artifact failure when validate_render is never called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
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


def _seed_render_on_disk(project_root: Path, report_id: str, *, data_ref: str = "queries/q") -> None:
    """Seed a minimal validated render tree + matching query so renderer-side tests run."""
    report_dir = project_root / "reports" / report_id
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
        '{"name":"seeded report","description":"Unit-test seeded report.",'
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
        assert node._active_report_id is None

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


# --------------------------------------------------------------------------- #
# Pre-execution artifact wiring                                               #
# --------------------------------------------------------------------------- #


class TestPrepareReportArtifacts:
    def test_registers_intent_tools_without_binding(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        user_input = GenVisualReportNodeInput(user_message="北美一季度门店销售分析")
        node.input = user_input

        node._prepare_report_artifacts(user_input)

        assert isinstance(node.report_artifact_tools, ReportArtifactTools)
        assert node._active_report_id is None
        assert node.report_artifact_tools.report_id is None
        assert node.report_artifact_tools.mode is None

        tool_names = {t.name for t in node.tools}
        assert "start_new_report" in tool_names
        assert "bind_existing_report" in tool_names
        assert "save_query" in tool_names
        assert "validate_render" in tool_names

        reports_root = Path(real_agent_config.project_root) / "reports"
        assert sorted(p.name for p in reports_root.glob("rpt_*")) == []


class TestEnhancedMessageHint:
    def test_hint_added_when_user_references_existing_report(self, real_agent_config, mock_llm_create):
        project_root = Path(real_agent_config.project_root)
        existing_id = "rpt_existing_demo_260513_aaaaaa"
        _seed_render_on_disk(project_root, existing_id)

        node = _make_node(real_agent_config)
        user_input = GenVisualReportNodeInput(user_message=f"修改 {existing_id} 报告，补充一个 YoY 分析章节")

        message = node._build_enhanced_message(user_input)
        assert existing_id in message
        assert "bind_existing_report" in message
        assert "start_new_report" in message

    def test_no_hint_when_no_reports_directory(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        user_input = GenVisualReportNodeInput(user_message="generate a new sales overview")
        message = node._build_enhanced_message(user_input)
        assert "bind_existing_report" not in message
        assert "start_new_report" not in message


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_stream_end_to_end(real_agent_config, mock_llm_create):
    """LLM binds an existing report (pre-seeded on disk) and validates the render tree.

    This covers the agentic node's result-extraction + html-compile path
    without needing the mock LLM to reference a dynamically-allocated id
    (the write_file authoring path is exercised end-to-end in the artifact
    tools tests). Pre-seeding ``render/`` + ``queries/`` and using
    ``bind_existing_report`` keeps the test purely additive over the unit
    coverage and avoids brittle id placeholder gymnastics.
    """
    project_root = Path(real_agent_config.project_root)
    existing_id = "rpt_e2e_demo_260514_aabbcc"
    _seed_render_on_disk(project_root, existing_id, data_ref="queries/avg_sat_reading")

    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(
                        name="bind_existing_report",
                        arguments=json.dumps({"report_id": existing_id}),
                    ),
                    MockToolCall(name="validate_render", arguments="{}"),
                ],
                content="Report validated.",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualReportNodeInput(
        user_message=f"check {existing_id}",
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
    assert result["report_id"] == existing_id
    assert result["app_jsx_path"] == f"reports/{existing_id}/render/app.jsx"
    assert result["render_file_count"] == 1
    # No save_query in this run — the seed wrote the query file directly.
    assert result["query_count"] == 0

    report_dir = project_root / "reports" / existing_id
    expected_html_rel = f"reports/{existing_id}/index.html"
    assert result["html_path"] == expected_html_rel
    assert (report_dir / "index.html").is_file()


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

        report_id = "rpt_priority_check_001"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        html_rel = node._maybe_compile_html(report_id)
        assert html_rel == f"reports/{report_id}/index.html"

        copied_css = Path(real_agent_config.project_root) / "reports" / report_id / "_assets" / "index.css"
        assert copied_css.read_text(encoding="utf-8") == "/* from-cli-flag css */"

    def test_node_config_used_when_cli_flag_absent(self, real_agent_config, mock_llm_create, tmp_path):
        node_dist = self._make_dist(tmp_path / "vendors", "node-only")

        node = _make_node(real_agent_config)
        node.node_config["report_dist"] = str(node_dist)
        if hasattr(real_agent_config, "report_dist_cli_override"):
            delattr(real_agent_config, "report_dist_cli_override")

        report_id = "rpt_priority_check_002"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        node._maybe_compile_html(report_id)
        copied_css = Path(real_agent_config.project_root) / "reports" / report_id / "_assets" / "index.css"
        assert copied_css.read_text(encoding="utf-8") == "/* node-only css */"


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
        report_id = "rpt_auto_open_yes"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_id)

        assert len(opened) == 1, f"expected one webbrowser.open call, got {opened}"
        assert opened[0].startswith("file://")
        assert opened[0].endswith(f"reports/{report_id}/index.html")

    def test_does_not_open_when_flag_disabled(self, real_agent_config, mock_llm_create, monkeypatch):
        node = _make_node(real_agent_config)
        real_agent_config.report_auto_open = False
        report_id = "rpt_auto_open_no"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_id)

        assert opened == [], f"webbrowser.open must not be called; got {opened}"

    def test_does_not_open_when_attribute_missing(self, real_agent_config, mock_llm_create, monkeypatch):
        node = _make_node(real_agent_config)
        if hasattr(real_agent_config, "report_auto_open"):
            delattr(real_agent_config, "report_auto_open")
        report_id = "rpt_auto_open_default"
        _seed_render_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_id)

        assert opened == []


@pytest.mark.asyncio
async def test_execute_stream_without_binding_marks_failure(real_agent_config, mock_llm_create):
    """LLM never binds a report → run reports a binding-required failure."""
    from tests.unit_tests.mock_llm_model import build_simple_response

    mock_llm_create.reset(
        responses=[
            build_simple_response("I gathered context but never bound a report."),
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
    assert result["report_id"] is None
    assert result["query_count"] == 0
    error = result.get("error") or ""
    assert "start_new_report" in error
    assert "bind_existing_report" in error


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
    assert result["report_id"] is not None and result["report_id"].startswith("rpt_halfway_")
    assert result["query_count"] == 0
    assert "validate_render never returned success" in (result.get("error") or "")
