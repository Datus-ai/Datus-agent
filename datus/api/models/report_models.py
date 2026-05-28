"""Pydantic models for the visual-report API.

Mirrors the wire shape consumed by the @datus/web-artifact-render
report viewer. Models here only carry fields the agent actually
populates: the on-disk artifact bundle.

Publication-side fields (``subagent``, ``report_id``,
``published_version``, ``published_at``) live in a downstream SaaS
host's subclass — they're meaningful only when a Postgres deployment
sits behind the agent and have no analogue on the agent-only path.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from datus.api.models.dashboard_models import ArtifactFile
from datus.schemas.artifact_manifest import ArtifactManifest

__all__ = [
    "ArtifactFile",
    "ReportDetail",
]


class ReportDetail(BaseModel):
    """Wire shape of ``GET /api/v1/report/detail`` (agent-only path).

    ``files`` is the slug-relative flat list covering every artifact file
    the report owns (render/ tree + queries/<slug>.sql / .json pre-baked
    result pairs + analysis/ sidecars). Unlike dashboards, reports inline
    their query results into the bundle — there's no live-query path at
    view time, so ``@datus/web-common/modules/report`` (and the standalone
    ``@datus/web-artifact-render`` UMD viewer it ships with) only needs
    this list to render the entire artifact.

    A downstream SaaS host extends this with the publication-side
    fields (``subagent`` / ``report_id`` / ``published_version`` /
    ``published_at``) via its own subclass when a Postgres deployment
    is present.
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
