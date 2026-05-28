"""API routes for the visual-dashboard artifact.

* ``GET /api/v1/dashboard/detail`` — returns render/* + template metadata
  for a dashboard slug.
* ``POST /api/v1/dashboard/query`` — renders a saved Jinja2 SQL template
  against the supplied filter values and executes it live through the
  project's connector.

Published-version snapshotting (``visual_dashboards`` /
``visual_dashboard_versions`` in Postgres) and the companion
``ask_dashboard`` subagent live in the Datus-backend SaaS wrapper —
not here.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Query

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.dashboard_models import (
    DashboardDetail,
    DashboardQueryRequest,
    SqlQueryResultEnvelope,
)

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


def _project_files_root(svc: ServiceDep) -> Path:
    """Project files root the dashboard artifacts live under.

    Mirrors the convention in ``kb_routes`` — ``{agent.home}/files`` is
    where the per-project tree is anchored in both the agent-only path
    (``datus --web``) and the SaaS path (which injects the same shape
    through ``get_project_files_root``).
    """
    return Path(os.path.join(svc.agent_config.home, "files"))


@router.get(
    "/dashboard/detail",
    response_model=Result[DashboardDetail],
    summary="Get Dashboard Artifact Detail",
    description=(
        "Return the render/ tree (app.jsx + sibling modules) plus the parameter "
        "metadata for every saved query template under a dashboard produced by the "
        "gen_visual_dashboard subagent."
    ),
)
async def get_dashboard_detail(
    svc: ServiceDep,
    slug: str = Query(..., description="Dashboard slug, e.g. 'revenue_overview'"),
) -> Result[DashboardDetail]:
    return await svc.dashboard.get_detail(
        project_files_root=_project_files_root(svc),
        dashboard_slug=slug,
    )


@router.post(
    "/dashboard/query",
    response_model=Result[SqlQueryResultEnvelope],
    summary="Run Dashboard Query",
    description=(
        "Render a saved Jinja2 SQL template with the supplied filter values and "
        "execute it live against the project's bound datasource. Returns the result "
        "envelope expected by RemoteQueryArtifactProvider in @datus/web-artifact."
    ),
)
async def run_dashboard_query(
    body: DashboardQueryRequest,
    svc: ServiceDep,
) -> Result[SqlQueryResultEnvelope]:
    return await svc.dashboard.run_query(
        project_files_root=_project_files_root(svc),
        dashboard_slug=body.dashboard_slug,
        query_slug=body.query_slug,
        params=body.params,
        published_version=body.published_version,
        # Agent-only deployment: no Postgres-backed version snapshots, so no loader.
        published_template_loader=None,
    )
