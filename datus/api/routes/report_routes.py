"""API route for the visual-report artifact.

``GET /api/v1/report/detail`` — returns the render/ tree (app.jsx + sibling
modules) plus the full set of queries/*.sql and queries/*.json files for
a report produced by the ``gen_visual_report`` subagent.

Publish (``visual_reports`` / ``visual_report_versions`` snapshots) and
the companion ``ask_report`` subagent live in the Datus-backend SaaS
wrapper — not here.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Query

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.report_models import ReportDetail

router = APIRouter(prefix="/api/v1", tags=["report"])


def _project_files_root(svc: ServiceDep) -> Path:
    """Project files root the report artifacts live under.

    Mirrors the convention in ``kb_routes`` / ``dashboard_routes`` —
    ``{agent.home}/files`` is where the per-project tree is anchored in
    both the agent-only path (``datus --web``) and the SaaS path (which
    injects the same shape through ``get_project_files_root``).
    """
    return Path(os.path.join(svc.agent_config.home, "files"))


@router.get(
    "/report/detail",
    response_model=Result[ReportDetail],
    summary="Get Report Artifact Detail",
    description=(
        "Return the render/ tree (app.jsx + sibling modules) plus the full set of "
        "queries/*.sql and queries/*.json files for a report produced by the "
        "gen_visual_report subagent."
    ),
)
async def get_report_detail(
    svc: ServiceDep,
    slug: str = Query(..., description="Report slug, e.g. 'account_activity_q1'"),
) -> Result[ReportDetail]:
    return await svc.report.get_detail(
        project_files_root=_project_files_root(svc),
        report_slug=slug,
    )
