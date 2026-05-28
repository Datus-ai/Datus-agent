# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``GenVisualDashboardAgenticNode``.

Design principle: NO mocks except LLM.

Covers:
* Node initialization wires the expected tools (db, filesystem, semantic).
* ``DashboardFilesystemFuncTool`` replaces the default filesystem tool.
* ``_prepare_artifacts`` registers the artifact tools without binding a
  dashboard slug.
* The bugfix: repeated ``_prepare_artifacts`` calls on the same node
  instance must NOT stack duplicate tool wrappers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
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


def _seed_dashboard_on_disk(project_root: Path, dashboard_slug: str) -> None:
    """Seed a minimal dashboard layout for end-to-end tests."""
    dash_dir = project_root / "dashboards" / dashboard_slug
    (dash_dir / "queries").mkdir(parents=True, exist_ok=True)
    (dash_dir / "render").mkdir(exist_ok=True)
    (dash_dir / "render" / "app.jsx").write_text(
        "export default function App() { return null; }\n",
        encoding="utf-8",
    )
    # manifest.json is part of the dashboard contract — validate_render
    # rejects the artifact when it's missing.
    (dash_dir / "manifest.json").write_text(
        f'{{"slug":"{dashboard_slug}","name":"seeded dashboard","description":"Unit-test seeded dashboard.",'
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
        assert node._active_dashboard_slug is None

    def test_session_id_is_honored(self, real_agent_config, mock_llm_create):
        # Regression: BaseVisualArtifactAgenticNode must accept session_id and
        # forward it to the base node. When dropped, the node mints a fresh
        # random session id each turn, so the SQLite history never matches the
        # chat session and multi-turn context is silently lost.
        node = _make_node(real_agent_config, session_id="vd_session_fixed")
        assert node.session_id == "vd_session_fixed"

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

    def test_tool_category_map_registers_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Same contract as the visual report node: filesystem tools must
        be declared under ``filesystem_tools`` so permission gating and
        ``_FS_DEPENDENT_NODES`` exclusion in ``apply_proxy_tools`` apply.
        """
        node = _make_node(real_agent_config)
        mapping = node._tool_category_map()
        assert "filesystem_tools" in mapping
        fs_tool_names = {t.name for t in mapping["filesystem_tools"]}
        assert {"read_file", "write_file", "edit_file", "delete_file"}.issubset(fs_tool_names)
        assert "db_tools" in mapping
        assert "semantic_tools" in mapping

    def test_metric_discovery_tools_exposed_when_metrics_present(self, real_agent_config, mock_llm_create, monkeypatch):
        """Mirrors the report-node contract: with metrics indexed, the
        dashboard node must expose ``search_metrics`` and ``get_metrics``
        so the LLM can discover the metric registry instead of
        re-deriving SQL from raw schema. Also exercises the
        ``context_search_tools.*`` wildcard end-to-end.
        """
        from datus.tools.func_tool.context_search import ContextSearchTools

        monkeypatch.setattr(ContextSearchTools, "_show_metrics", lambda self: True)

        node = _make_node(real_agent_config)
        assert isinstance(node.context_search_tools, ContextSearchTools)
        tool_names = {t.name for t in node.tools}
        assert {"search_metrics", "get_metrics", "list_subject_tree"}.issubset(tool_names)

    def test_apply_proxy_tools_keeps_filesystem_tools_unwrapped(self, real_agent_config, mock_llm_create):
        """End-to-end check mirroring the visual report case: web-source
        ``["write_file", "edit_file"]`` patterns must leave the dashboard
        node's filesystem tools un-proxied because
        ``gen_visual_dashboard`` is in ``_FS_DEPENDENT_NODES``.

        Does not pre-fill ``tool_registry``; verifies that
        ``apply_proxy_tools`` populates it eagerly so the exclusion
        fires.
        """
        from datus.tools.proxy.proxy_tool import apply_proxy_tools

        node = _make_node(real_agent_config)
        before = {t.name: t.on_invoke_tool for t in node.tools if t.name in {"write_file", "edit_file"}}
        assert before, "test setup: expected write_file/edit_file in node.tools"

        apply_proxy_tools(node, ["write_file", "edit_file"])

        after = {t.name: t.on_invoke_tool for t in node.tools if t.name in {"write_file", "edit_file"}}
        for name, original in before.items():
            assert after[name] is original, f"{name} was proxied despite gen_visual_dashboard fs-dependent exclusion"


# --------------------------------------------------------------------------- #
# Pre-execution artifact wiring                                               #
# --------------------------------------------------------------------------- #


class TestPrepareDashboardArtifacts:
    def test_registers_intent_tools_without_binding(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message="一个销售概览 dashboard")
        node.input = user_input

        node._prepare_artifacts(user_input)

        assert isinstance(node.dashboard_artifact_tools, DashboardArtifactTools)
        assert node._active_artifact_slug is None
        assert node.dashboard_artifact_tools.dashboard_slug is None

        tool_names = {t.name for t in node.tools}
        assert "start_new_dashboard" in tool_names
        assert "bind_existing_dashboard" in tool_names
        assert "save_query_template" in tool_names
        assert "validate_render" in tool_names

        dashboards_root = Path(real_agent_config.project_root) / "dashboards"
        assert not dashboards_root.exists() or sorted(p.name for p in dashboards_root.iterdir()) == []

    def test_repeated_calls_do_not_duplicate_tools(self, real_agent_config, mock_llm_create):
        """Bugfix regression: ``execute_stream`` may run twice per instance.

        Before the fix, each call ``self.tools.extend(...)``ed the artifact
        tools, leaving stale wrappers bound to the previous instance. The
        replace-by-name logic must keep the tool list stable across runs.
        """
        node = _make_node(real_agent_config)
        user_input = GenVisualDashboardNodeInput(user_message="dashboard please")
        node.input = user_input

        node._prepare_artifacts(user_input)
        first_tool_count = len(node.tools)
        first_dashboard_artifact_tools = node.dashboard_artifact_tools

        # Run preparation again — should swap the instance, not stack tools.
        node._prepare_artifacts(user_input)
        assert len(node.tools) == first_tool_count
        assert node.dashboard_artifact_tools is not first_dashboard_artifact_tools

        # No name should appear twice.
        names = [t.name for t in node.tools]
        assert len(names) == len(set(names))

        # All four artifact-tool names are still present after the swap.
        artifact_names = {"start_new_dashboard", "bind_existing_dashboard", "save_query_template", "validate_render"}
        assert artifact_names.issubset(set(names))


# --------------------------------------------------------------------------- #
# Enhanced-message wiring                                                     #
# --------------------------------------------------------------------------- #


class TestEnhancedMessage:
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
    """LLM binds an existing dashboard (pre-seeded on disk) and validates the render tree."""
    project_root = Path(real_agent_config.project_root)
    existing_slug = "e2e_demo"
    dash_dir = project_root / "dashboards" / existing_slug
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
        f'{{"slug":"{existing_slug}","name":"e2e demo dashboard","description":"End-to-end seeded dashboard.",'
        '"kind":"dashboard","created_at":"2026-05-14T00:00:00Z"}\n',
        encoding="utf-8",
    )

    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[
                    MockToolCall(
                        name="bind_existing_dashboard",
                        arguments=json.dumps({"dashboard_slug": existing_slug}),
                    ),
                    MockToolCall(name="validate_render", arguments="{}"),
                ],
                content="Dashboard validated.",
            ),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualDashboardNodeInput(
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
    assert result["dashboard_slug"] == existing_slug
    assert result["app_jsx_path"] == f"dashboards/{existing_slug}/render/app.jsx"
    assert result["render_file_count"] == 1
    # No save_query_template in this run — the seed wrote the template directly.
    assert result["template_count"] == 0

    # Artifact card fields are populated from manifest.json so the SSE
    # artifact event ships everything the frontend needs in one hop.
    assert result["artifact_kind"] == "dashboard"
    assert result["artifact_mode"] == "edit"
    assert result["name"] == "e2e demo dashboard"
    assert result["description"] == "End-to-end seeded dashboard."
    assert result["created_at"] == "2026-05-14T00:00:00Z"

    # CLI mode compiles a standalone HTML next to the artifact and emits a
    # WORKFLOW status action carrying both the path AND the ``datus --web``
    # launch hint so the user can copy-paste the command for the live-query
    # backend.
    expected_html_rel = f"dashboards/{existing_slug}/index.html"
    assert result["html_path"] == expected_html_rel
    assert (dash_dir / "index.html").is_file()

    path_actions = [a for a in actions if a.action_type == "dashboard_html_path"]
    assert len(path_actions) == 1, "expected exactly one dashboard_html_path action in CLI mode"
    path_action = path_actions[0]
    assert path_action.role == ActionRole.WORKFLOW
    abs_html = str((dash_dir / "index.html").resolve())
    assert abs_html in path_action.messages
    assert path_action.output["html_path"] == abs_html
    assert path_action.output["url"].startswith("file://")
    assert "/api/v1/dashboard/query" in path_action.output["query_endpoint"]
    # The launch command names the active datasource (so it's copy-pasteable)
    # AND keeps the executable + flag stable for docs-link references.
    assert path_action.output["datus_web_command"].startswith("datus --web --datasource ")


@pytest.mark.asyncio
async def test_execute_stream_fails_when_no_binding_and_no_answer(real_agent_config, mock_llm_create):
    """No binding AND no prose answer (the run did nothing) → fails clearly."""
    mock_llm_create.reset(
        responses=[
            build_tool_then_response(
                tool_calls=[],
                content="",
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


@pytest.mark.asyncio
async def test_execute_stream_informational_answer_ends_normally(real_agent_config, mock_llm_create):
    """An informational question (e.g. "what changes did I make?") answered in
    prose without binding an artifact must end normally — not surface the
    internal "Run finished without binding a dashboard" error."""
    answer = "In this session you changed 2 charts: revenue trend → bar, product mix → donut."
    mock_llm_create.reset(
        responses=[
            build_tool_then_response(tool_calls=[], content=answer),
        ]
    )

    node = _make_node(real_agent_config)
    node.input = GenVisualDashboardNodeInput(user_message="what changes did I make?", database="california_schools")
    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)

    final = actions[-1]
    assert final.status == ActionStatus.SUCCESS
    result = final.output
    assert isinstance(result, dict)
    assert result["success"] is True
    assert not result.get("error")
    assert result["dashboard_slug"] is None
    assert result["app_jsx_path"] is None
    assert result["response"] == answer


class TestDashboardHtmlPathStreamMessage:
    """Verify the compiled-HTML status action surfaces the path + launch hint."""

    def _seed_full_dashboard(self, project_root: Path, slug: str) -> Path:
        """Seed render/queries/manifest fixtures so ``_maybe_compile_html`` succeeds."""
        dash_dir = project_root / "dashboards" / slug
        (dash_dir / "render").mkdir(parents=True, exist_ok=True)
        (dash_dir / "queries").mkdir(parents=True, exist_ok=True)
        (dash_dir / "render" / "app.jsx").write_text(
            "import React from 'react';\nexport default function App() { return null; }\n",
            encoding="utf-8",
        )
        (dash_dir / "queries" / "q.sql.j2").write_text("-- @datus-params\nSELECT 1\n", encoding="utf-8")
        (dash_dir / "queries" / "q.params.json").write_text(
            json.dumps(
                {
                    "slug": "q",
                    "description": "",
                    "datasource": "warehouse",
                    "params": [],
                    "columns": [{"name": "a", "type": "integer"}],
                    "sample_params": {},
                    "sample_row_count": 1,
                    "saved_at": "2026-05-14T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        (dash_dir / "manifest.json").write_text(
            f'{{"slug":"{slug}","name":"d","description":"d","kind":"dashboard","created_at":"2026-05-14T00:00:00Z"}}\n',
            encoding="utf-8",
        )
        return dash_dir

    def test_post_validate_hook_emits_action_in_cli_mode(self, real_agent_config, mock_llm_create):
        from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeResult

        node = _make_node(real_agent_config)
        dashboard_slug = "path_msg_cli"
        dash_dir = self._seed_full_dashboard(Path(real_agent_config.project_root), dashboard_slug)
        node._active_artifact_slug = dashboard_slug

        result = GenVisualDashboardNodeResult(success=True)
        action = node._post_validate_hook(dashboard_slug, result)

        assert isinstance(action, ActionHistory)
        assert action.action_type == "dashboard_html_path"
        assert action.role == ActionRole.WORKFLOW

        abs_html = str((dash_dir / "index.html").resolve())
        assert action.output["html_path"] == abs_html
        assert abs_html in action.messages

        # The query endpoint defaults to the agent --web URL so the iframe
        # knows where to POST live-query requests.
        assert action.output["query_endpoint"].endswith("/api/v1/dashboard/query")
        # The launch hint surfaces the exact command the user should run.
        assert "datus --web --datasource" in action.output["datus_web_command"]
        assert "datus --web" in action.messages
        # Relative path is still recorded on the result for SaaS/task consumers.
        assert result.html_path == f"dashboards/{dashboard_slug}/index.html"

    def test_post_validate_hook_returns_none_in_non_cli_mode(self, real_agent_config, mock_llm_create):
        from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeResult

        # filesystem_strict flips the node into SaaS/API mode — dashboards
        # then render dynamically through the backend's /dashboard/detail
        # endpoint and never compile a standalone HTML.
        real_agent_config.filesystem_strict = True
        node = _make_node(real_agent_config)
        dashboard_slug = "path_msg_saas"
        self._seed_full_dashboard(Path(real_agent_config.project_root), dashboard_slug)
        node._active_artifact_slug = dashboard_slug

        result = GenVisualDashboardNodeResult(success=True)
        action = node._post_validate_hook(dashboard_slug, result)

        assert action is None
        assert result.html_path is None
        assert not (Path(real_agent_config.project_root) / "dashboards" / dashboard_slug / "index.html").exists()

    def test_post_validate_hook_uses_node_config_overrides(self, real_agent_config, mock_llm_create):
        """``agentic_nodes.gen_visual_dashboard.{web_host,web_port,query_endpoint}``
        overrides surface in both the rendered HTML and the action payload."""
        from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeResult

        node = _make_node(real_agent_config)
        node.node_config["web_host"] = "192.168.1.10"
        node.node_config["web_port"] = 9000
        dashboard_slug = "with_overrides"
        dash_dir = self._seed_full_dashboard(Path(real_agent_config.project_root), dashboard_slug)
        node._active_artifact_slug = dashboard_slug

        result = GenVisualDashboardNodeResult(success=True)
        action = node._post_validate_hook(dashboard_slug, result)
        assert action is not None

        expected_endpoint = "http://192.168.1.10:9000/api/v1/dashboard/query"
        assert action.output["query_endpoint"] == expected_endpoint
        body = (dash_dir / "index.html").read_text(encoding="utf-8")
        assert f"queryEndpoint: '{expected_endpoint}'" in body
        # Non-default port + non-localhost host both surface on the command.
        cmd = action.output["datus_web_command"]
        assert "--port 9000" in cmd
        assert "--host 192.168.1.10" in cmd


class TestConfiguredDatasource:
    """``_configured_datasource`` powers the copy-paste ``datus --web --datasource``
    hint — it must reflect what the active session is actually pointing at
    instead of always falling back to the YAML default.

    The helper is a pure getattr cascade over ``agent_config`` — these tests
    swap ``node.agent_config`` with a ``SimpleNamespace`` stub so we can
    exercise the three branches (live binding / YAML default / neither)
    without going through the real ``current_datasource`` setter, which
    refuses unknown names by design.
    """

    def test_prefers_current_datasource_over_services_default(self, real_agent_config, mock_llm_create):
        from types import SimpleNamespace

        node = _make_node(real_agent_config)
        node.agent_config = SimpleNamespace(
            current_datasource="live_ds",
            services=SimpleNamespace(default_datasource="ignored_ds"),
        )

        assert node._configured_datasource() == "live_ds"

    def test_falls_back_to_services_default_when_current_unset(self, real_agent_config, mock_llm_create):
        from types import SimpleNamespace

        node = _make_node(real_agent_config)
        node.agent_config = SimpleNamespace(
            current_datasource="",
            services=SimpleNamespace(default_datasource="fallback_ds"),
        )

        assert node._configured_datasource() == "fallback_ds"

    def test_returns_none_when_nothing_is_configured(self, real_agent_config, mock_llm_create):
        from types import SimpleNamespace

        node = _make_node(real_agent_config)
        node.agent_config = SimpleNamespace(
            current_datasource=None,
            services=SimpleNamespace(default_datasource=None),
        )

        assert node._configured_datasource() is None

    def test_returns_none_when_services_attribute_is_missing(self, real_agent_config, mock_llm_create):
        """Defence-in-depth: the helper must still return None (not raise)
        when ``agent_config`` lacks a ``services`` attribute — the
        downstream message just renders ``<your_datasource>``."""
        from types import SimpleNamespace

        node = _make_node(real_agent_config)
        node.agent_config = SimpleNamespace(current_datasource=None)

        assert node._configured_datasource() is None


class TestNodeFactoryDashboardBranch:
    """Exercises the ``gen_visual_*`` cases in ``node_factory``."""

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

    def test_factory_forwards_session_id_to_dashboard(self, real_agent_config, mock_llm_create):
        # Regression: the API path passes the HTTP session id as both node_id
        # and session_id so the node's SQLite history matches the chat session.
        # The gen_visual_* branches previously dropped session_id, breaking
        # multi-turn history for dashboards/reports.
        from datus.agent.node.node_factory import create_interactive_node

        node = create_interactive_node(
            subagent_name="gen_visual_dashboard",
            agent_config=real_agent_config,
            node_id="api_session_42",
            session_id="api_session_42",
        )
        assert node.session_id == "api_session_42"

    def test_factory_forwards_session_id_to_report(self, real_agent_config, mock_llm_create):
        from datus.agent.node.node_factory import create_interactive_node

        node = create_interactive_node(
            subagent_name="gen_visual_report",
            agent_config=real_agent_config,
            node_id="api_session_99",
            session_id="api_session_99",
        )
        assert node.session_id == "api_session_99"

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
                                "slug": "incomplete",
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
    # ``_active_dashboard_slug`` got captured even though validate_render didn't run.
    assert result["dashboard_slug"] == "incomplete"
    assert "validate_render" in (result.get("error") or "")
