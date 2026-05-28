"""Pydantic models for the visual-report API.

Mirrors the wire shape consumed by the @datus/web-artifact-render report
viewer. The DB-bound fields on :class:`ReportDetail`
(``subagent`` / ``report_id`` / ``published_version`` /
``published_at``) are intentionally ``Optional`` / default ``0``: the
agent-only path (``datus --web``) never populates them, while the
Datus-backend wrapper enriches them from Postgres before responding.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from datus.api.models.dashboard_models import ArtifactFile
from datus.schemas.artifact_manifest import ArtifactManifest

__all__ = [
    "ArtifactFile",
    "ReportDetail",
    "ReportSubAgentInfo",
]


class ReportSubAgentInfo(BaseModel):
    """Compact pointer to the ``ask_report`` subagent bound to this report.

    Only populated when the report has been published on the SaaS
    backend (``visual_reports.subagent_id``). The agent-only path
    always leaves this as ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="SubAgent.id — opens the CommonTab agent editor")
    name: str = Field(..., description="SubAgent.name (mirrors the report's display name at create time)")
    description: str = Field(..., description="SubAgent.description")


class ReportDetail(BaseModel):
    """Wire shape of ``GET /api/v1/report/detail``.

    ``files`` is the slug-relative flat list covering every artifact file
    the report owns (render/ tree + queries/<slug>.sql / .json pre-baked
    result pairs + analysis/ sidecars). Unlike dashboards, reports inline
    their query results into the bundle — there's no live-query path at
    view time, so ``@datus/web-common/modules/report`` (and the standalone
    ``@datus/web-artifact-render`` UMD viewer it ships with) only needs
    this list to render the entire artifact.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(..., description="Report slug, e.g. 'account_activity_q1'")
    name: str = Field(..., description="Human-readable display name (read from manifest.json)")
    description: str = Field(..., description="One-paragraph description of what the report covers (manifest.json)")
    manifest: ArtifactManifest = Field(
        ..., description="Full manifest.json contents (slug + name + description + kind + created_at)"
    )
    created_at: Optional[str] = Field(None, description="ISO 8601 timestamp (render/app.jsx mtime)")
    files: List[ArtifactFile] = Field(
        ...,
        description=(
            "Flat list of every artifact file under reports/<slug>/ that passes the "
            "per-prefix allowlist (render/{.jsx,.js,.css,.json,.md}, queries/{.sql,.json}, "
            "analysis/{.md,.json}). manifest.json is intentionally NOT included — "
            "the parsed structured form is on ``manifest`` above. Sorted by path."
        ),
    )
    subagent: Optional[ReportSubAgentInfo] = Field(
        None,
        description=(
            "The ``ask_report`` SubAgent bound to this report by ``/report/publish``. "
            "``None`` before the first successful publish or when running without a SaaS DB."
        ),
    )
    report_id: Optional[str] = Field(
        None,
        description=(
            "``visual_reports.id`` for this (workspace, project, slug) once a "
            "publish has landed; ``None`` before the first publish or when running "
            "without a SaaS DB. Use this to build the saas viewer URL ``/report/<report_id>``."
        ),
    )
    published_version: int = Field(
        0,
        description=(
            "Latest ``visual_report_versions.version`` for this (workspace, project, slug). "
            "``0`` when nothing has been published yet or when running without a SaaS DB."
        ),
    )
    published_at: Optional[str] = Field(
        None,
        description=(
            "ISO 8601 UTC timestamp of the latest published version's ``created_at``. "
            "``None`` when nothing has been published yet or when running without a SaaS DB."
        ),
    )
