"""Pydantic models for the visual-dashboard API.

Mirrors the wire shape consumed by the @datus/web-artifact-render
dashboard viewer. Models here only carry fields the agent actually
populates: on-disk bundle assembly + Jinja2 template metadata.

Publication-side fields (``subagent``, ``dashboard_id``,
``published_version``, ``published_at``) live in a downstream SaaS
host's subclass — they're meaningful only when a Postgres deployment
sits behind the agent and have no analogue on the agent-only path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from datus.schemas.artifact_manifest import ArtifactManifest
from datus.schemas.gen_visual_dashboard_models import QueryTemplateMetaFile
from datus.schemas.gen_visual_report_models import QueryColumnMeta

__all__ = [
    "ArtifactFile",
    "DashboardDetail",
    "DashboardQueryRequest",
    "SqlQueryResultEnvelope",
]


class ArtifactFile(BaseModel):
    """Flat-wire entry for a single artifact file inside ``dashboards/<slug>/``
    (and, by symmetry, ``reports/<slug>/`` on the SaaS side).

    ``path`` is slug-relative, including the top-level directory
    (e.g. ``render/charts/trend.jsx``, ``queries/sales.sql.j2``,
    ``analysis/insights.json``). Matches ``IArtifactFileEntry`` on TS.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "Slug-relative path, including the top-level directory "
            "(e.g. 'render/app.jsx', 'queries/foo.sql.j2', 'analysis/intent.md')."
        ),
    )
    content: str = Field(..., description="Raw UTF-8 source text")


class DashboardDetail(BaseModel):
    """Wire shape of ``GET /api/v1/dashboard/detail`` (agent-only path).

    ``files`` is the slug-relative flat list covering every artifact file
    the dashboard owns (render/ tree + queries/*.sql.j2 / queries/*.params.json
    + analysis/). ``templates`` stays as a parsed sidecar for outer-panel
    UI that wants to drive filter affordances without re-parsing the
    .params.json bytes; the iframe itself only needs the render slice of
    ``files``.

    A downstream SaaS host extends this with the publication-side
    fields (``subagent`` / ``dashboard_id`` / ``published_version`` /
    ``published_at``) via its own subclass when a Postgres deployment
    is present.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(..., description="Dashboard slug, e.g. 'revenue_overview'")
    name: str = Field(..., description="Human-readable display name (read from manifest.json)")
    description: str = Field(..., description="One-paragraph description of what the dashboard tracks (manifest.json)")
    manifest: ArtifactManifest = Field(
        ..., description="Full manifest.json contents (slug + name + description + kind + created_at)"
    )
    created_at: Optional[str] = Field(None, description="ISO 8601 timestamp (render/app.jsx mtime)")
    files: List[ArtifactFile] = Field(
        ...,
        description=(
            "Flat list of every artifact file under dashboards/<slug>/ that passes the "
            "per-prefix allowlist (render/{.jsx,.js,.css}, queries/{.sql.j2,.params.json}, "
            "analysis/{.md,.json}). manifest.json is intentionally NOT included — "
            "the parsed structured form is on ``manifest`` above. Sorted by path."
        ),
    )
    templates: List[QueryTemplateMetaFile] = Field(
        default_factory=list,
        description="Per-slug Jinja2 template metadata (params declaration, columns, sample_params)",
    )


class SqlQueryResultEnvelope(BaseModel):
    """Matches the ISqlQueryResult shape consumed by ``useQuerySql`` on the frontend.

    The frontend hook destructures ``data?.rows ?? []`` / ``data?.columns ?? []``
    and reads ``data?.sql`` for the optional "View SQL" affordance — keep the
    field names in lockstep with ``packages/web-artifact/src/types.ts``.
    """

    model_config = ConfigDict(extra="forbid")

    executed_at: str = Field(..., description="ISO 8601 UTC timestamp of the executing query")
    datasource: str = Field(..., description="Logical datasource the query ran against")
    row_count: int = Field(..., ge=0)
    columns: List[QueryColumnMeta] = Field(..., description="Column name + inferred semantic type")
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Result rows; each is a {column → scalar} map")
    sql: Optional[str] = Field(
        None,
        description=(
            "The fully rendered SQL the connector executed (after Jinja2 + bind-value substitution). "
            "Surfaced so the in-iframe <ChartEntryMore> menu can show 'View SQL'."
        ),
    )


class DashboardQueryRequest(BaseModel):
    """Request body for ``POST /api/v1/dashboard/query``."""

    model_config = ConfigDict(extra="forbid")

    dashboard_slug: str = Field(..., description="Dashboard slug, e.g. 'revenue_overview'")
    query_slug: str = Field(
        ...,
        description=(
            "Query template slug — the filename stem of the "
            "``dashboards/<dashboard_slug>/queries/<query_slug>.sql.j2`` + "
            "``.params.json`` sibling pair."
        ),
    )
    params: Dict[str, Any] = Field(default_factory=dict, description="User-selected filter values keyed by param name")
    published_version: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "When set, render the template from an immutable snapshot at this version "
            "instead of the on-disk buffer. Only supported by a SaaS host with a "
            "version-snapshot store; the agent-only ``datus --web`` path rejects "
            "non-null values with INVALID_PUBLISHED_VERSION."
        ),
    )
