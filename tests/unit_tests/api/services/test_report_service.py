# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.api.services.report_service`` — CI level, zero external deps.

Covers the on-disk artifact bundle walk:

* Happy path: ``files`` is a flat slug-relative list including the
  render/ tree, queries/ pre-baked SQL + JSON result pairs, and any
  analysis/ sidecars.
* The per-prefix walker allowlist drops files that don't match.
* Missing ``render/app.jsx`` / ``manifest.json`` surfaces as
  ``REPORT_NOT_FOUND``; malformed slugs surface as
  ``INVALID_REPORT_SLUG``.

The PostgreSQL-bound enrichment (``subagent`` / ``report_id`` /
``published_version`` / ``published_at``) is layered on top by the
Datus-backend SaaS wrapper and exercised by its own
``tests/unit/test_report_service_subagent.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.api.services.report_service import ReportService

_SAMPLE_APP_JSX = """\
/** @datus-title Demo Report */
import React from 'react';
import { useDatusArtifact } from '@datus/web-artifact';
import KpiBanner from './kpi-banner';

export default function Demo() {
  const { useQuerySql } = useDatusArtifact();
  const { data } = useQuerySql('queries/q');
  return React.createElement(KpiBanner, { rows: data?.rows ?? [] });
}
"""

_SAMPLE_KPI_BANNER_JSX = """\
import React from 'react';
export default function KpiBanner({ rows }) {
  return React.createElement('div', null, rows.length, ' rows');
}
"""


def _write_report(
    project_files_root: Path,
    *,
    report_slug: str = "demo_001",
    name: str = "Demo Report",
    description: str = "Smoke-test report used by the report-service unit tests.",
    render_files: dict | None = None,
    queries: dict | None = None,
    write_manifest: bool = True,
) -> Path:
    """Lay out a minimal on-disk report fixture under ``<root>/reports/<slug>/``."""
    report_dir = project_files_root / "reports" / report_slug
    (report_dir / "queries").mkdir(parents=True, exist_ok=True)
    (report_dir / "render").mkdir(parents=True, exist_ok=True)
    render_files = render_files or {
        "app.jsx": _SAMPLE_APP_JSX,
        "kpi-banner.jsx": _SAMPLE_KPI_BANNER_JSX,
    }
    for rel, content in render_files.items():
        target = report_dir / "render" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    queries = queries or {"q.sql": "SELECT 1", "q.json": '{"row_count":0,"rows":[]}'}
    for cur_name, content in queries.items():
        (report_dir / "queries" / cur_name).write_text(content, encoding="utf-8")
    if write_manifest:
        (report_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "slug": report_slug,
                    "name": name,
                    "description": description,
                    "kind": "report",
                    "created_at": "2026-05-14T10:00:00Z",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return report_dir


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_detail_returns_flat_files(tmp_path: Path):
    _write_report(tmp_path)
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="demo_001",
    )
    assert result.success is True
    payload = result.data
    assert payload.slug == "demo_001"
    # name + description are read from manifest.json — not parsed from app.jsx.
    assert payload.name == "Demo Report"
    assert payload.description.startswith("Smoke-test report")
    # manifest carries the full on-disk JSON for clients that prefer the nested shape.
    assert payload.manifest.slug == "demo_001"
    assert payload.manifest.name == "Demo Report"
    assert payload.manifest.kind == "report"

    # The flat ``files`` list is the canonical bundle — every entry's
    # ``path`` is slug-relative including its top-level directory prefix.
    paths = {f.path for f in payload.files}
    assert paths == {
        "render/app.jsx",
        "render/kpi-banner.jsx",
        "queries/q.sql",
        "queries/q.json",
    }
    app_entry = next(f for f in payload.files if f.path == "render/app.jsx")
    assert "useDatusArtifact" in app_entry.content

    # created_at comes from app.jsx mtime — must be a usable ISO 8601 UTC string.
    assert payload.created_at.endswith("Z") and "T" in payload.created_at

    # Publication-side fields (subagent / report_id / published_version /
    # published_at) are not part of the agent-side ``ReportDetail`` schema
    # — they live on Datus-backend's ``PublishedReportDetail`` subclass.
    # The presence of any such attribute here would mean the subclass
    # leaked into agent code.
    assert not hasattr(payload, "subagent")
    assert not hasattr(payload, "report_id")
    assert not hasattr(payload, "published_version")
    assert not hasattr(payload, "published_at")


@pytest.mark.asyncio
async def test_get_detail_walks_nested_render_subdirs(tmp_path: Path):
    _write_report(
        tmp_path,
        report_slug="nested_001",
        render_files={
            "app.jsx": _SAMPLE_APP_JSX,
            "kpi-banner.jsx": _SAMPLE_KPI_BANNER_JSX,
            "charts/trend.jsx": "import React from 'react';\nexport default () => null;\n",
            "shared/colors.js": "export default ['#fff'];\n",
        },
    )
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="nested_001",
    )
    assert result.success is True
    render_paths = [f.path for f in result.data.files if f.path.startswith("render/")]
    assert render_paths == [
        "render/app.jsx",
        "render/charts/trend.jsx",
        "render/kpi-banner.jsx",
        "render/shared/colors.js",
    ]


@pytest.mark.asyncio
async def test_get_detail_filters_render_files_by_allowlist(tmp_path: Path):
    """``render/`` walker honours the per-prefix allowlist:

    * ``.jsx`` / ``.js`` / ``.css`` — primary module sources, always allowed.
    * ``.json`` / ``.md`` — sidecar fixtures / notes the LLM may park next
      to its modules; allowed but the renderer just ignores them.
    * anything else (e.g. ``.bin``) — silently dropped so a stray scratch
      file doesn't bloat the bundle.
    """
    _write_report(
        tmp_path,
        report_slug="filter_suffix",
        render_files={
            "app.jsx": _SAMPLE_APP_JSX,
            "kpi-banner.jsx": _SAMPLE_KPI_BANNER_JSX,
        },
    )
    render_root = tmp_path / "reports" / "filter_suffix" / "render"
    (render_root / "data.json").write_text('{"x":1}', encoding="utf-8")
    (render_root / "notes.md").write_text("# notes", encoding="utf-8")
    (render_root / "stray.bin").write_text("binary blob", encoding="utf-8")

    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="filter_suffix",
    )
    assert result.success is True
    render_paths = {f.path for f in result.data.files if f.path.startswith("render/")}
    assert render_paths == {
        "render/app.jsx",
        "render/kpi-banner.jsx",
        "render/data.json",
        "render/notes.md",
    }
    assert "render/stray.bin" not in render_paths


@pytest.mark.asyncio
async def test_get_detail_returns_name_and_description_from_manifest(tmp_path: Path):
    _write_report(
        tmp_path,
        report_slug="manifest_only_001",
        name="账号活跃报告",
        description="季度账号活跃情况分析。",
        render_files={
            "app.jsx": "import React from 'react';\nexport default function R() { return null; }\n",
        },
    )
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="manifest_only_001",
    )
    assert result.success is True
    assert result.data.name == "账号活跃报告"
    assert result.data.description == "季度账号活跃情况分析。"


@pytest.mark.asyncio
async def test_get_detail_returns_queries_in_sorted_order(tmp_path: Path):
    _write_report(
        tmp_path,
        queries={
            "z_last.sql": "SELECT 1",
            "z_last.json": '{"row_count":0,"rows":[]}',
            "a_first.sql": "SELECT 2",
            "a_first.json": '{"row_count":0,"rows":[]}',
        },
    )
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="demo_001",
    )
    assert result.success is True
    query_paths = [f.path for f in result.data.files if f.path.startswith("queries/")]
    assert query_paths == [
        "queries/a_first.json",
        "queries/a_first.sql",
        "queries/z_last.json",
        "queries/z_last.sql",
    ]


@pytest.mark.asyncio
async def test_only_sql_and_json_queries_included(tmp_path: Path):
    _write_report(
        tmp_path,
        queries={
            "q.sql": "SELECT 1",
            "q.json": '{"row_count":0,"rows":[]}',
            "ignored.txt": "should not be returned",
            "binary.bin": "should not be returned",
        },
    )
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="demo_001",
    )
    assert result.success is True
    query_paths = {f.path for f in result.data.files if f.path.startswith("queries/")}
    assert query_paths == {"queries/q.sql", "queries/q.json"}


@pytest.mark.asyncio
async def test_detail_picks_up_analysis_files_via_flat_walker(tmp_path: Path):
    """Files dropped under ``analysis/`` appear in the flat ``files`` list.

    Anchors the contract that the walker fans out across every directory
    in ``_REPORT_ARTIFACT_DIRS`` — adding new sidecar files (insights,
    suggested questions, …) requires zero changes to the detail endpoint
    or its Pydantic model; consumers parse the JSON payloads themselves.
    """
    _write_report(tmp_path, report_slug="with_analysis")
    analysis_dir = tmp_path / "reports" / "with_analysis" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "suggested_questions.json").write_text(
        json.dumps(
            [
                {
                    "question": "Which categories are oversaturated?",
                    "related_queries": ["q"],
                    "related_insight": None,
                    "priority": 0.91,
                }
            ]
        ),
        encoding="utf-8",
    )
    (analysis_dir / "intent.md").write_text("### prompt\n", encoding="utf-8")

    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="with_analysis",
    )
    assert result.success is True
    analysis_paths = {f.path for f in result.data.files if f.path.startswith("analysis/")}
    assert analysis_paths == {
        "analysis/suggested_questions.json",
        "analysis/intent.md",
    }


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_manifest_rejected(tmp_path: Path):
    """A bound report without manifest.json is treated as not-found."""
    _write_report(
        tmp_path,
        report_slug="no_manifest_001",
        write_manifest=False,
    )
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="no_manifest_001",
    )
    assert result.success is False
    assert result.errorCode == "REPORT_NOT_FOUND"
    assert "manifest.json" in (result.errorMessage or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_id",
    ["", "UPPER_BAD", "ok/../etc", "ok with spaces", "中文"],
)
async def test_invalid_report_slug_rejected(tmp_path: Path, bad_id: str):
    """Defence-in-depth: malformed / traversal slugs fail the regex guard
    before any I/O so a crafted slug can't reach the filesystem walker."""
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug=bad_id,
    )
    assert result.success is False
    assert result.errorCode == "INVALID_REPORT_SLUG"


@pytest.mark.asyncio
async def test_report_not_found(tmp_path: Path):
    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="missing_xxx",
    )
    assert result.success is False
    assert result.errorCode == "REPORT_NOT_FOUND"


@pytest.mark.asyncio
async def test_report_missing_app_jsx_reports_not_found(tmp_path: Path):
    """A render/ directory without app.jsx is still REPORT_NOT_FOUND."""
    report_dir = tmp_path / "reports" / "no_app"
    (report_dir / "render").mkdir(parents=True, exist_ok=True)
    (report_dir / "queries").mkdir(parents=True, exist_ok=True)
    (report_dir / "render" / "kpi-banner.jsx").write_text(_SAMPLE_KPI_BANNER_JSX, encoding="utf-8")

    result = await ReportService().get_detail(
        project_files_root=tmp_path,
        report_slug="no_app",
    )
    assert result.success is False
    assert result.errorCode == "REPORT_NOT_FOUND"
