# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for ``GenVisualReportAgenticNode``.

Design principle: NO mocks except LLM (same as test_gen_report_agentic_node).

Covers:
* Node initialization wires the expected tools
* ``ReportFilesystemFuncTool`` replaces the default filesystem tool
* Allocation of a fresh ``report_id`` per ``execute_stream`` call
* End-to-end streaming run where the LLM calls ``save_query`` then
  ``save_manifest`` against a real SQLite database, persisting the
  artifact under ``project_root/reports/<id>/``
* CLI mode compiles ``index.html`` after a successful run
* When ``save_manifest`` is never called, the run is reported as failed
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
        assert "save_manifest" not in tool_names


# --------------------------------------------------------------------------- #
# Pre-execution artifact wiring                                               #
# --------------------------------------------------------------------------- #


class TestPrepareReportArtifacts:
    def test_prepare_allocates_report_id_and_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        user_input = GenVisualReportNodeInput(user_message="北美一季度门店销售分析")
        node.input = user_input

        node._prepare_report_artifacts(user_input)

        # report_id format is `rpt_<slug>_<yymmdd>_<rand6>` per the contract.
        active_id = node._active_report_id or ""
        assert active_id.startswith("rpt_"), f"unexpected report id: {active_id!r}"
        assert len(active_id) >= len("rpt_") + len("000000") + 1, f"report id too short: {active_id!r}"
        assert isinstance(node.report_artifact_tools, ReportArtifactTools)
        tool_names = {t.name for t in node.tools}
        assert "save_query" in tool_names
        assert "save_manifest" in tool_names
        report_dir = Path(real_agent_config.project_root) / "reports" / node._active_report_id
        assert (report_dir / "queries").is_dir()


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_stream_end_to_end(real_agent_config, mock_llm_create):
    """LLM saves one query, then a valid manifest. Artifact should be on disk."""

    save_query_args = json.dumps(
        {
            "name": "avg_sat_reading",
            "sql": "SELECT 'state' AS scope, AVG(AvgScrRead) AS avg_read FROM satscores GROUP BY 'state'",
            "description": "Average SAT reading score statewide",
        }
    )

    manifest_obj = {
        "version": "1.0",
        "title": "California SAT report",
        "created_at": "2026-05-13T10:00:00Z",
        "sections": [
            {"id": "blk_001", "type": "markdown", "content": "# California SAT report"},
            {
                "id": "blk_002",
                "type": "chart",
                "data_ref": "queries/avg_sat_reading",
                "spec": {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "scope", "type": "nominal"},
                        "y": {"field": "avg_read", "type": "quantitative"},
                    },
                },
            },
        ],
    }
    save_manifest_args = json.dumps({"manifest_json": json.dumps(manifest_obj)})

    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(name="save_query", arguments=save_query_args),
                    MockToolCall(name="save_manifest", arguments=save_manifest_args),
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
    assert result["manifest_path"].endswith("manifest.json")
    assert result["query_count"] == 1

    report_dir = Path(real_agent_config.project_root) / "reports" / result["report_id"]
    assert (report_dir / "manifest.json").is_file()
    assert (report_dir / "queries" / "avg_sat_reading.sql").is_file()
    assert (report_dir / "queries" / "avg_sat_reading.json").is_file()

    # CLI mode produces an index.html alongside the manifest, at the path the
    # result advertises (project-root-relative).
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

    def _seed_manifest_on_disk(self, project_root: Path, report_id: str) -> None:
        report_dir = project_root / "reports" / report_id
        (report_dir / "queries").mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": "1.0",
            "id": report_id,
            "title": "stub",
            "created_at": "2026-05-13T10:00:00Z",
            "sections": [{"id": "blk_001", "type": "markdown", "content": "# hi"}],
        }
        (report_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_cli_override_wins_over_node_config(self, real_agent_config, mock_llm_create, tmp_path):
        node_dist = self._make_dist(tmp_path / "vendors", "from-node-config")
        cli_dist = self._make_dist(tmp_path / "vendors", "from-cli-flag")

        node = _make_node(real_agent_config)
        node.node_config["report_dist"] = str(node_dist)
        real_agent_config.report_dist_cli_override = str(cli_dist)

        report_id = "rpt_priority_check_001"
        self._seed_manifest_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        html_rel = node._maybe_compile_html(report_id)
        assert html_rel == f"reports/{report_id}/index.html"

        copied_css = Path(real_agent_config.project_root) / "reports" / report_id / "_assets" / "datus-report.css"
        # CLI override beat node_config; the CLI copy ended up on disk.
        assert copied_css.read_text(encoding="utf-8") == "/* from-cli-flag css */"

    def test_node_config_used_when_cli_flag_absent(self, real_agent_config, mock_llm_create, tmp_path):
        node_dist = self._make_dist(tmp_path / "vendors", "node-only")

        node = _make_node(real_agent_config)
        node.node_config["report_dist"] = str(node_dist)
        # Ensure no leftover CLI override is hanging on the shared fixture.
        if hasattr(real_agent_config, "report_dist_cli_override"):
            delattr(real_agent_config, "report_dist_cli_override")

        report_id = "rpt_priority_check_002"
        self._seed_manifest_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        node._maybe_compile_html(report_id)
        copied_css = Path(real_agent_config.project_root) / "reports" / report_id / "_assets" / "datus-report.css"
        assert copied_css.read_text(encoding="utf-8") == "/* node-only css */"


class _InlineThread:
    """Synchronous stand-in for ``threading.Thread``.

    The node's ``_maybe_open_in_browser`` schedules ``webbrowser.open`` on a
    daemon thread so a slow platform launcher does not block the CLI. Tests
    need that call to happen before assertions run, so we replace
    ``threading.Thread`` with this drop-in that invokes ``target`` inline
    on ``start()``. Eliminates the need for sleep-based waits (P0 violation).
    """

    def __init__(self, target=None, daemon=False, **kwargs):
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        if self._target is not None:
            self._target()


class TestAutoOpenInBrowser:
    """Verify ``_maybe_open_in_browser`` gates on ``agent_config.report_auto_open``."""

    def _seed_manifest_on_disk(self, project_root: Path, report_id: str) -> None:
        report_dir = project_root / "reports" / report_id
        (report_dir / "queries").mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": "1.0",
            "id": report_id,
            "title": "stub",
            "created_at": "2026-05-13T10:00:00Z",
            "sections": [{"id": "blk_001", "type": "markdown", "content": "# hi"}],
        }
        (report_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_opens_browser_when_flag_enabled(self, real_agent_config, mock_llm_create, monkeypatch):
        node = _make_node(real_agent_config)
        real_agent_config.report_auto_open = True
        report_id = "rpt_auto_open_yes"
        self._seed_manifest_on_disk(Path(real_agent_config.project_root), report_id)
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
        self._seed_manifest_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_id)

        assert opened == [], f"webbrowser.open must not be called; got {opened}"

    def test_does_not_open_when_attribute_missing(self, real_agent_config, mock_llm_create, monkeypatch):
        """No ``report_auto_open`` attribute (e.g. SaaS path) must default to no-open."""
        node = _make_node(real_agent_config)
        if hasattr(real_agent_config, "report_auto_open"):
            delattr(real_agent_config, "report_auto_open")
        report_id = "rpt_auto_open_default"
        self._seed_manifest_on_disk(Path(real_agent_config.project_root), report_id)
        node._active_report_id = report_id

        opened = []
        monkeypatch.setattr("threading.Thread", _InlineThread)
        monkeypatch.setattr("webbrowser.open", lambda url, *a, **kw: opened.append(url) or True)

        node._maybe_compile_html(report_id)

        assert opened == []


@pytest.mark.asyncio
async def test_execute_stream_without_manifest_marks_failure(real_agent_config, mock_llm_create):
    """If LLM never calls save_manifest, the run reports failure."""
    mock_llm_create.reset(
        responses=[
            build_simple_response("I gathered context but did not finalize the manifest."),
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
    assert result["manifest_path"] is None
    assert result["query_count"] == 0
    assert "save_manifest was never called" in (result.get("error") or "")
