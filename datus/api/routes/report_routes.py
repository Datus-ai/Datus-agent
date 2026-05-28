"""API route for the visual-report artifact.

``GET /api/v1/report/detail`` — returns the render/ tree (app.jsx + sibling
modules) plus the full set of queries/*.sql and queries/*.json files for
a report produced by the ``gen_visual_report`` subagent.

Publish (``visual_reports`` / ``visual_report_versions`` snapshots) and
the companion ``ask_report`` subagent live in the Datus-backend SaaS
wrapper — not here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.report_models import ReportDetail

router = APIRouter(prefix="/api/v1", tags=["report"])


def _project_files_root(svc: ServiceDep) -> Path:
    """Project files root the report artifacts live under.

    ``agent_config.project_root`` is the universal anchor:

    * Agent CLI (``datus --web``): ``AgentConfig`` defaults
      ``project_root`` to ``os.getcwd()``, so ``gen_visual_report``
      writes ``<CWD>/reports/<slug>/`` and this route reads from the
      same tree.
    * Datus-backend SaaS: :mod:`datus_backend.config_loader` sets
      ``project_root = project_files_dir`` (i.e.
      ``<tenant>/<ws>/<project_id>/files``), which is where the SaaS
      subagent writes its artifacts.

    Previous incarnations of this helper composed ``{agent.home}/files``
    by analogy with ``kb_routes`` — that only worked in the SaaS layout
    where ``home == project_dir`` and ``project_root == home/files``;
    in CLI it resolved to ``~/.datus/files`` and produced
    REPORT_NOT_FOUND on every detail lookup.
    """
    return Path(svc.agent_config.project_root)


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
