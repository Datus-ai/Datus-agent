# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenVisualDashboardAgenticNode — parameterized dashboard generation.

Companion to ``GenVisualReportAgenticNode``. Instead of pre-baked JSON
result files, this node produces a parameterized React-JSX dashboard
artifact under ``<project_root>/dashboards/<slug>/``:

* ``render/app.jsx`` — the React entry module the LLM authors (default
  export); it owns the filter state and imports the chart components.
* ``queries/<slug>.sql.j2`` + ``queries/<slug>.params.json`` — per-query
  Jinja2 SQL template plus its declared parameter metadata.
* ``manifest.json`` — ``{slug, name, description, kind, created_at}``.

At view time the backend renders the template with user-selected filter
values and executes it live against the bound datasource — see
``Datus-backend/datus_backend/services/dashboard_service.py``. Common
machinery lives in :class:`BaseVisualArtifactAgenticNode`; this file
owns the dashboard-specific artifact wiring and result model.
"""

from __future__ import annotations

from typing import List, Optional

from datus.agent.node.base_visual_artifact_agentic_node import BaseVisualArtifactAgenticNode
from datus.schemas.action_history import ActionHistory
from datus.schemas.gen_visual_dashboard_models import (
    GenVisualDashboardNodeInput,
    GenVisualDashboardNodeResult,
)
from datus.tools.func_tool.dashboard_artifact_tools import (
    DashboardArtifactTools,
    DashboardFilesystemFuncTool,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenVisualDashboardAgenticNode(
    BaseVisualArtifactAgenticNode[GenVisualDashboardNodeInput, GenVisualDashboardNodeResult]
):
    """
    Visual dashboard subagent.

    Sets up semantic / db / context-search tools plus the dashboard-specific
    ``DashboardArtifactTools`` (save_query_template / validate_render) and a
    hardened ``DashboardFilesystemFuncTool`` that denies direct writes to
    dashboard artifact paths.

    The LLM chooses the ``dashboard_slug`` on every fresh
    ``start_new_dashboard`` call; the system prompt directs it to
    ``glob('dashboards/*')`` first so the chosen slug doesn't collide with
    an existing one.
    """

    NODE_NAME = "gen_visual_dashboard"
    ARTIFACT_KIND = "dashboard"
    ARTIFACT_ROOT_DIR_NAME = "dashboards"
    FILESYSTEM_TOOL_CLS = DashboardFilesystemFuncTool
    QUERY_SAVE_ACTION_TYPE = "save_query_template"
    FALLBACK_TEMPLATE_NAME = "gen_visual_dashboard_system"

    def get_node_name(self) -> str:
        return self.configured_node_name or self.NODE_NAME

    # ────────── Convenience accessors ──────────

    @property
    def _active_dashboard_slug(self) -> Optional[str]:
        return self._active_artifact_slug

    @_active_dashboard_slug.setter
    def _active_dashboard_slug(self, value: Optional[str]) -> None:
        self._active_artifact_slug = value

    @property
    def dashboard_artifact_tools(self) -> Optional[DashboardArtifactTools]:
        return self.artifact_tools  # type: ignore[return-value]

    @dashboard_artifact_tools.setter
    def dashboard_artifact_tools(self, value: Optional[DashboardArtifactTools]) -> None:
        self.artifact_tools = value

    # ────────── Hooks the base class calls ──────────

    def _make_artifact_tools(self) -> DashboardArtifactTools:
        return DashboardArtifactTools(
            agent_config=self.agent_config,
            db_func_tool=self.db_func_tool,
        )

    def _read_artifact_slug_from_tools(self) -> Optional[str]:
        tools = self.artifact_tools
        if tools is None:
            return None
        return getattr(tools, "dashboard_slug", None)

    def _build_success_result(
        self,
        *,
        user_input: GenVisualDashboardNodeInput,
        response_content: str,
        artifact_slug: Optional[str],
        app_jsx_rel_path: Optional[str],
        render_file_count: int,
        query_actions: List[ActionHistory],
        tokens_used: int,
        all_actions: List[ActionHistory],
        tool_calls: List[ActionHistory],
    ) -> GenVisualDashboardNodeResult:
        return GenVisualDashboardNodeResult(
            success=app_jsx_rel_path is not None,
            response=response_content,
            dashboard_slug=artifact_slug,
            app_jsx_path=app_jsx_rel_path,
            render_file_count=render_file_count,
            template_count=len(query_actions),
            tokens_used=tokens_used,
            action_history=[a.model_dump() for a in all_actions],
            execution_stats={
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": sorted({a.action_type for a in tool_calls}),
                "total_tokens": tokens_used,
            },
        )

    def _build_error_result(self, exc: BaseException) -> GenVisualDashboardNodeResult:
        return GenVisualDashboardNodeResult(
            success=False,
            error=str(exc),
            response="Sorry, I encountered an error while generating the visual dashboard.",
            dashboard_slug=self._active_artifact_slug,
            tokens_used=0,
        )

    # No CLI-mode HTML compile for dashboards — they need a live
    # datasource to execute the parameterized SQL templates, so the
    # standalone HTML route doesn't apply. The default no-op
    # ``_post_validate_hook`` on the base class is fine.
