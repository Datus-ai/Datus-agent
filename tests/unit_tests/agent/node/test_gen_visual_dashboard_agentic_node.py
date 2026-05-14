# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``GenVisualDashboardAgenticNode``.

Design principle: NO mocks except LLM.

Covers:
* Node initialization wires the expected tools (db, filesystem, semantic).
* ``DashboardFilesystemFuncTool`` replaces the default filesystem tool.
* ``_prepare_dashboard_artifacts`` registers the artifact tools without
  binding a dashboard id.
* The bugfix: repeated ``_prepare_dashboard_artifacts`` calls on the same
  node instance must NOT stack duplicate tool wrappers.
* ``_detect_referenced_dashboard_ids`` surfaces on-disk ids mentioned in
  the user message; the LLM-facing hint includes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeInput
from datus.tools.func_tool import (
    DashboardArtifactTools,
    DashboardFilesystemFuncTool,
    DBFuncTool,
    SemanticTools,
)
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_tool_then_response,
)


def _make_node(real_agent_config, **overrides):
    from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode

    kwargs = dict(
        node_id="vd_node_test",
        description="Visual dashboard node",
        node_type=NodeType.TYPE_GEN_VISUAL_DASHBOARD,
        agent_config=real_agent_config,
        node_name="gen_visual_dashboard",
    )
    kwargs.update(overrides)
    return GenVisualDashboardAgenticNode(**kwargs)


def _seed_dashboard_on_disk(project_root: Path, dashboard_id: str) -> None:
    """Seed a minimal dashboard layout so referenced-id detection finds it."""
    dash_dir = project_root / "dashboards" / dashboard_id
    (dash_dir / "queries").mkdir(parents=True, exist_ok=True)
    (dash_dir / "render").mkdir(exist_ok=True)
    (dash_dir / "render" / "app.jsx").write_text(
        "export default function App() { return null; }\n",
        encoding="utf-8",
    )
    # manifest.json is part of the dashboard contract — validate_render
    # rejects the artifact when it's missing.
    (dash_dir / "manifest.json").write_text(
        '{"name":"seeded dashboard","description":"Unit-test seeded dashboard.",'
        '"kind":"dashboard","created_at":"2026-05-13T00:00:00Z"}\n',
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Initialization                                                              #
# --------------------------------------------------------------------------- #


class TestGenVisualDashboardInit:
    def test_basic_init(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        assert node.get_node_name() == "gen_visual_dashboard"
        assert isinstance(node.db_func_tool, DBFuncTool)
        assert isinstance(node.semantic_tools, SemanticTools)
        assert isinstance(node.filesystem_func_tool, DashboardFilesystemFuncTool)
        assert node.dashboard_artifact_tools is None
        assert node._active_dashboard_id is None

    def test_tools_include_filesystem_and_db(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        tool_names = {t.name for t in node.tools}
        assert "list_tables" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "delete_file" in tool_names
        # Pre-execution: artifact tools are not registered yet.
        assert "save_query_template" not in tool_names
        assert "validate_render" not in tool_names
        assert "start_new_dashboard" not in tool_names


# --------------------------------------------------------------------------- #
# Pre-execution artifact wiring                                               #
# --------------------------------------------------------------------------- #


class TestPrepareDashboardArtifacts:
    def test_registers_intent_tools_without_binding(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message="一个销售概览 dashboard")
        node.input = user_input

        node._prepare_dashboard_artifacts(user_input)

        assert isinstance(node.dashboard_artifact_tools, DashboardArtifactTools)
        assert node._active_dashboard_id is None
        assert node.dashboard_artifact_tools.dashboard_id is None

        tool_names = {t.name for t in node.tools}
        assert "start_new_dashboard" in tool_names
        assert "bind_existing_dashboard" in tool_names
        assert "save_query_template" in tool_names
        assert "validate_render" in tool_names

        dashboards_root = Path(real_agent_config.project_root) / "dashboards"
        assert sorted(p.name for p in dashboards_root.glob("dash_*")) == []

    def test_repeated_calls_do_not_duplicate_tools(self, real_agent_config, mock_llm_create):
        """Bugfix regression: ``execute_stream`` may run twice per instance.

        Before the fix, each call ``self.tools.extend(...)``ed the artifact
        tools, leaving stale wrappers bound to the previous instance. The
        replace-by-name logic must keep the tool list stable across runs.
        """
        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message="dashboard please")
        node.input = user_input

        node._prepare_dashboard_artifacts(user_input)
        first_tool_count = len(node.tools)
        first_dashboard_artifact_tools = node.dashboard_artifact_tools

        # Run preparation again — should swap the instance, not stack tools.
        node._prepare_dashboard_artifacts(user_input)
        assert len(node.tools) == first_tool_count
        assert node.dashboard_artifact_tools is not first_dashboard_artifact_tools

        # No name should appear twice.
        names = [t.name for t in node.tools]
        assert len(names) == len(set(names))

        # All four artifact-tool names are still present after the swap.
        artifact_names = {"start_new_dashboard", "bind_existing_dashboard", "save_query_template", "validate_render"}
        assert artifact_names.issubset(set(names))


# --------------------------------------------------------------------------- #
# Enhanced-message hint                                                       #
# --------------------------------------------------------------------------- #


class TestEnhancedMessageHint:
    def test_hint_added_when_user_references_existing_dashboard(self, real_agent_config, mock_llm_create):
        project_root = Path(real_agent_config.project_root)
        existing_id = "dash_existing_demo_260514_aabbcc"
        _seed_dashboard_on_disk(project_root, existing_id)

        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message=f"修改 {existing_id}，加一个 region 过滤器")

        message = node._build_enhanced_message(user_input)
        assert existing_id in message
        assert "bind_existing_dashboard" in message
        assert "start_new_dashboard" in message

    def test_no_hint_when_no_dashboards_directory(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message="generate a new revenue dashboard")
        message = node._build_enhanced_message(user_input)
        assert "bind_existing_dashboard" not in message
        assert "start_new_dashboard" not in message

    def test_referenced_id_must_exist_on_disk_to_surface(self, real_agent_config, mock_llm_create):
        # Reference an id whose folder does not exist; no hint should appear.
        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message="参考 dash_nonexistent_260514_aaaaaa 生成新的")
        message = node._build_enhanced_message(user_input)
        assert "bind_existing_dashboard" not in message

    def test_extra_message_parts_added_when_db_context_present(self, real_agent_config, mock_llm_create):
        """Catalog / database / schema all surface on the enhanced message header."""
        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message="ok", catalog="cat", database="db", db_schema="s")
        message = node._build_enhanced_message(user_input)
        assert "Catalog: cat" in message
        assert "Database context: db" in message
        assert "Schema: s" in message


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_stream_end_to_end(real_agent_config, mock_llm_create):
    """LLM binds an existing dashboard (pre-seeded on disk) and validates the render tree.

    Mirrors the ``test_execute_stream_end_to_end`` pattern in the report
    suite: pre-seed disk so the LLM can use ``bind_existing_dashboard``
    without having to know a dynamically-allocated id, then drive a
    minimal ``bind → validate_render`` script via the mock LLM.
    """
    project_root = Path(real_agent_config.project_root)
    existing_id = "dash_e2e_demo_260514_abcdef"
    dash_dir = project_root / "dashboards" / existing_id
    (dash_dir / "queries").mkdir(parents=True, exist_ok=True)
    (dash_dir / "render").mkdir(exist_ok=True)
    (dash_dir / "render" / "app.jsx").write_text(
        "import React from 'react';\n"
        "import { useDatusArtifact } from '@datus/web-artifact';\n"
        "export default function App() {\n"
        "  const { useQuerySql } = useDatusArtifact();\n"
        "  useQuerySql('queries/dummy', {});\n"
        "  return null;\n"
        "}\n",
        encoding="utf-8",
    )
    # A matching template so the param-key contract resolves.
    (dash_dir / "queries" / "dummy.sql.j2").write_text(
        "-- @datus-params x:string:optional\nSELECT 1 AS a",
        encoding="utf-8",
    )
    (dash_dir / "queries" / "dummy.params.json").write_text(
        json.dumps(
            {
                "slug": "dummy",
                "description": "",
                "datasource": "default",
                "params": [{"name": "x", "type": "string", "required": False}],
                "columns": [{"name": "a", "type": "integer"}],
                "sample_params": {},
                "sample_row_count": 1,
                "saved_at": "2026-05-14T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    # Seed manifest.json — validate_render requires it on disk.
    (dash_dir / "manifest.json").write_text(
        '{"name":"e2e demo dashboard","description":"End-to-end seeded dashboard.",'
        '"kind":"dashboard","created_at":"2026-05-14T00:00:00Z"}\n',
        encoding="utf-8",
    )

    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(
                        name="bind_existing_dashboard",
                        arguments=json.dumps({"dashboard_id": existing_id}),
                    ),
                    MockToolCall(name="validate_render", arguments="{}"),
                ],
                content="Dashboard validated.",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualDashboardNodeInput(
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
    assert result["dashboard_id"] == existing_id
    assert result["app_jsx_path"] == f"dashboards/{existing_id}/render/app.jsx"
    assert result["render_file_count"] == 1
    # No save_query_template in this run — the seed wrote the template directly.
    assert result["template_count"] == 0


@pytest.mark.asyncio
async def test_execute_stream_fails_when_no_binding(real_agent_config, mock_llm_create):
    """If the LLM never calls start_/bind_existing_dashboard, the node fails clearly."""
    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[],
                content="I changed my mind.",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualDashboardNodeInput(user_message="hi", database="california_schools")
    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)

    final = actions[-1]
    assert final.status == ActionStatus.FAILED
    result = final.output
    assert isinstance(result, dict)
    assert result["success"] is False
    assert "start_new_dashboard" in (result.get("error") or "")


class TestNodeFactoryDashboardBranch:
    """Exercises the ``gen_visual_*`` cases in ``node_factory``.

    Kept here (rather than in ``test_node.py``) so the visual-artifact
    factory coverage doesn't ride on the pre-existing brittle
    ``TestNodeFactory`` fixtures. Covers both ``gen_visual_dashboard`` and
    ``gen_visual_report`` branches since diff-cover bundles both as part
    of this feature branch's introduction.
    """

    def test_factory_returns_dashboard_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode
        from datus.agent.node.node_factory import create_interactive_node

        node = create_interactive_node(
            subagent_name="gen_visual_dashboard",
            agent_config=real_agent_config,
            execution_mode="interactive",
        )
        assert isinstance(node, GenVisualDashboardAgenticNode)
        assert node.get_node_name() == "gen_visual_dashboard"

    def test_factory_returns_report_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode
        from datus.agent.node.node_factory import create_interactive_node

        node = create_interactive_node(
            subagent_name="gen_visual_report",
            agent_config=real_agent_config,
            execution_mode="interactive",
        )
        assert isinstance(node, GenVisualReportAgenticNode)
        assert node.get_node_name() == "gen_visual_report"

    def test_create_node_input_returns_dashboard_input(self, real_agent_config, mock_llm_create):
        from datus.agent.node.node_factory import create_node_input

        node = _make_node(real_agent_config)
        node_input = create_node_input(
            user_message="dashboard please",
            node=node,
            catalog="cat",
            database="db",
            db_schema="s",
        )
        assert isinstance(node_input, GenVisualDashboardNodeInput)
        assert node_input.user_message == "dashboard please"
        assert node_input.catalog == "cat"
        assert node_input.database == "db"
        assert node_input.db_schema == "s"

    def test_create_node_input_returns_report_input(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode
        from datus.agent.node.node_factory import create_interactive_node, create_node_input
        from datus.schemas.gen_visual_report_models import GenVisualReportNodeInput

        report_node = create_interactive_node(
            subagent_name="gen_visual_report",
            agent_config=real_agent_config,
            execution_mode="interactive",
        )
        assert isinstance(report_node, GenVisualReportAgenticNode)
        node_input = create_node_input(
            user_message="report please",
            node=report_node,
            catalog="cat",
            database="db",
            db_schema="s",
        )
        assert isinstance(node_input, GenVisualReportNodeInput)
        assert node_input.user_message == "report please"


@pytest.mark.asyncio
async def test_execute_stream_fails_when_validate_render_not_called(real_agent_config, mock_llm_create):
    """The subagent terminates with a clear error when the LLM never finalises."""
    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(
                        name="start_new_dashboard",
                        arguments=json.dumps(
                            {
                                "name": "incomplete",
                                "description": "Bound but never validated — unit-test fixture.",
                            }
                        ),
                    ),
                ],
                content="started but never validated",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualDashboardNodeInput(user_message="start", database="california_schools")
    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)

    final = actions[-1]
    assert final.status == ActionStatus.FAILED
    result = final.output
    assert result["success"] is False
    # ``_active_dashboard_id`` got captured even though validate_render didn't run.
    assert result["dashboard_id"] is not None
    assert "validate_render" in (result.get("error") or "")
