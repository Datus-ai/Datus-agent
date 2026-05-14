# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``GenVisualReportAgenticNode``.

Design principle: NO mocks except LLM (same as test_gen_report_agentic_node).

Covers:
* Node initialization wires the expected tools.
* ``ReportFilesystemFuncTool`` replaces the default filesystem tool.
* ``_prepare_report_artifacts`` registers the artifact tools but leaves the
  report id unbound — the LLM owns the new/edit decision at runtime.
* End-to-end streaming run where the LLM calls ``start_new_report``,
  ``save_query``, and ``save_main_jsx`` against a real SQLite database,
  persisting the artifact under ``project_root/reports/<id>/``.
* The LLM-facing hint surfaces when the user references existing reports on disk.
* CLI mode compiles ``index.html`` after a successful run.
* When the LLM never binds a report, the run reports a binding-error.
* When the LLM binds but never calls ``save_main_jsx``, the run reports an
  incomplete-artifact error.
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
    build_simple_response,
    build_tool_then_response,
)

# --------------------------------------------------------------------------- #
# Initialization                                                              #
# --------------------------------------------------------------------------- #


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


_MAIN_JSX_SOURCE_TEMPLATE = """\
/** @datus-title {title} */
import React from 'react';
import {{ useDatusArtifact }} from '@datus/web-artifact';

export default function Demo() {{
  const {{ useQuerySql }} = useDatusArtifact();
  const {{ data }} = useQuerySql('{data_ref}');
  return React.createElement('pre', null, JSON.stringify(data?.rows ?? []));
}}
"""


def _seed_main_jsx_on_disk(project_root: Path, report_id: str) -> None:
    """Seed a minimal main.jsx + queries pair so renderer-side tests find them."""
    report_dir = project_root / "reports" / report_id
    (report_dir / "queries").mkdir(parents=True, exist_ok=True)
    (report_dir / "main.jsx").write_text(
        _MAIN_JSX_SOURCE_TEMPLATE.format(title="stub", data_ref="queries/q"),
        encoding="utf-8",
    )
    (report_dir / "queries" / "q.sql").write_text("SELECT 1", encoding="utf-8")
    (report_dir / "queries" / "q.json").write_text(
        '{"executed_at":"2026-05-13T00:00:00Z","datasource":"x","row_count":0,'
        '"columns":[{"name":"a","type":"integer"}],"rows":[]}',
        encoding="utf-8",
    )


class TestGenVisualReportInit:
    def test_basic_init(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        assert node.get_node_name() == "gen_visual_report"
        assert isinstance(node.db_func_tool, DBFuncTool)
        assert isinstance(node.semantic_tools, SemanticTools)
        assert isinstance(node.filesystem_func_tool, ReportFilesystemFuncTool)
        # Artifact tools are bound at execute_stream time, not init.
        assert node.report_artifact_tools is None
        assert node._active_report_id is None

    def test_tools_include_filesystem_and_db(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        tool_names = {t.name for t in node.tools}
        # DB tool surface
        assert "list_tables" in tool_names
        # Filesystem tool surface
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        # Pre-execution: artifact tools are not registered yet
        assert "save_query" not in tool_names
        assert "save_main_jsx" not in tool_names


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
        assert "save_main_jsx" in tool_names

        # No rpt_<...> subdirectory created prematurely.
        reports_root = Path(real_agent_config.project_root) / "reports"
        report_subdirs = sorted(p.name for p in reports_root.glob("rpt_*"))
        assert report_subdirs == []


class TestEnhancedMessageHint:
    def test_hint_added_when_user_references_existing_report(self, real_agent_config, mock_llm_create):
        project_root = Path(real_agent_config.project_root)
        existing_id = "rpt_existing_demo_260513_aaaaaa"
        _seed_main_jsx_on_disk(project_root, existing_id)

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
    """LLM binds a fresh report, saves one query, then a valid main.jsx."""

    start_new_args = json.dumps({"title": "California SAT"})
    save_query_args = json.dumps(
        {
            "name": "avg_sat_reading",
            "sql": "SELECT 'state' AS scope, AVG(AvgScrRead) AS avg_read FROM satscores GROUP BY 'state'",
            "description": "Average SAT reading score statewide",
        }
    )

    jsx_source = _MAIN_JSX_SOURCE_TEMPLATE.format(title="California SAT report", data_ref="queries/avg_sat_reading")
    save_main_jsx_args = json.dumps({"jsx_code": jsx_source})

    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(name="start_new_report", arguments=start_new_args),
                    MockToolCall(name="save_query", arguments=save_query_args),
                    MockToolCall(name="save_main_jsx", arguments=save_main_jsx_args),
                ],
                content="Report generated.",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualReportNodeInput(
        user_message="Average SAT reading score statewide",
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
    assert result["report_id"].startswith("rpt_")
    assert result["main_jsx_path"].endswith("main.jsx")
    assert result["query_count"] == 1

    report_dir = Path(real_agent_config.project_root) / "reports" / result["report_id"]
    assert (report_dir / "main.jsx").is_file()
    assert (report_dir / "queries" / "avg_sat_reading.sql").is_file()
    assert (report_dir / "queries" / "avg_sat_reading.json").is_file()

    expected_html_rel = f"reports/{result['report_id']}/index.html"
    assert result["html_path"] == expected_html_rel
    assert (report_dir / "index.html").is_file()


class TestReportDistResolution:
    """Verify the CLI flag → node_config priority for offline asset overrides."""

    def _make_dist(self, base: Path, name: str) -> Path:
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "datus-report.css").write_text(f"/* {name} css */", encoding="utf-8")
        (d / "datus-report.umd.js").write_text(f"/* {name} js */", encoding="utf-8")
        return d

    def test_cli_override_wins_over_node_config(self, real_agent_config, mock_llm_create, tmp_path):
        node_dist = self._make_dist(tmp_path / "vendors", "from-node-config")
        cli_dist = self._make_dist(tmp_path / "vendors", "from-cli-flag")

        node = _make_node(real_agent_config)
        node.node_config["report_dist"] = str(node_dist)
        real_agent_config.report_dist_cli_override = str(cli_dist)

        report_id = "rpt_priority_check_001"
        _seed_main_jsx_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        html_rel = node._maybe_compile_html(report_id)
        assert html_rel == f"reports/{report_id}/index.html"

        copied_css = Path(real_agent_config.project_root) / "reports" / report_id / "_assets" / "datus-report.css"
        assert copied_css.read_text(encoding="utf-8") == "/* from-cli-flag css */"

    def test_node_config_used_when_cli_flag_absent(self, real_agent_config, mock_llm_create, tmp_path):
        node_dist = self._make_dist(tmp_path / "vendors", "node-only")

        node = _make_node(real_agent_config)
        node.node_config["report_dist"] = str(node_dist)
        if hasattr(real_agent_config, "report_dist_cli_override"):
            delattr(real_agent_config, "report_dist_cli_override")

        report_id = "rpt_priority_check_002"
        _seed_main_jsx_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        node._maybe_compile_html(report_id)
        copied_css = Path(real_agent_config.project_root) / "reports" / report_id / "_assets" / "datus-report.css"
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
        _seed_main_jsx_on_disk(Path(real_agent_config.project_root), report_id)
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
        _seed_main_jsx_on_disk(Path(real_agent_config.project_root), report_id)
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
        _seed_main_jsx_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_id)

        assert opened == []


@pytest.mark.asyncio
async def test_execute_stream_without_binding_marks_failure(real_agent_config, mock_llm_create):
    """LLM never binds a report → run reports a binding-required failure."""
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
    assert result["main_jsx_path"] is None
    assert result["report_id"] is None
    assert result["query_count"] == 0
    error = result.get("error") or ""
    assert "start_new_report" in error
    assert "bind_existing_report" in error


@pytest.mark.asyncio
async def test_execute_stream_bound_but_no_main_jsx_marks_failure(real_agent_config, mock_llm_create):
    """LLM binds but never calls save_main_jsx → distinct incomplete-artifact failure."""
    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(name="start_new_report", arguments=json.dumps({"title": "halfway"})),
                ],
                content="I bound a report but forgot to finalize the JSX.",
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
    assert result["main_jsx_path"] is None
    assert result["report_id"] is not None and result["report_id"].startswith("rpt_halfway_")
    assert result["query_count"] == 0
    assert "save_main_jsx was never called" in (result.get("error") or "")
