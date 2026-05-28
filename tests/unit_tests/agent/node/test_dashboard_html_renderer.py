# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.agent.node.visual_artifact.dashboard_html_renderer``.

The dashboard renderer reuses the shared
:mod:`datus.agent.node.visual_artifact._artifact_html_renderer` machinery —
these tests cover the dashboard-specific surface (template path,
allowlist, placeholders) plus the offline / CDN switch and the
live-query endpoint the agent ``--web`` server hosts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.agent.node.visual_artifact.dashboard_html_renderer import (
    DEFAULT_QUERY_ENDPOINT,
    render_dashboard_html,
)

_APP_JSX_TEMPLATE = """\
/** @datus-title {title} */
import React from 'react';
import {{ useDatusArtifact }} from '@datus/web-artifact';

export default function App() {{
  const {{ useQuerySql }} = useDatusArtifact();
  const {{ data }} = useQuerySql('queries/{slug}', {{ region: 'APAC' }});
  return React.createElement('pre', null, JSON.stringify(data?.rows ?? []));
}}
"""


def _seed_dashboard(
    project_root: Path,
    *,
    dashboard_slug: str = "demo",
    query_slug: str = "by_region",
    title: str = "Demo Dashboard",
) -> Path:
    dash_dir = project_root / "dashboards" / dashboard_slug
    (dash_dir / "render").mkdir(parents=True, exist_ok=True)
    (dash_dir / "render" / "app.jsx").write_text(
        _APP_JSX_TEMPLATE.format(title=title, slug=query_slug),
        encoding="utf-8",
    )
    (dash_dir / "queries").mkdir(parents=True, exist_ok=True)
    (dash_dir / "queries" / f"{query_slug}.sql.j2").write_text(
        "-- @datus-params region:string\nSELECT 1\n",
        encoding="utf-8",
    )
    (dash_dir / "queries" / f"{query_slug}.params.json").write_text(
        json.dumps(
            {
                "slug": query_slug,
                "description": "",
                "datasource": "warehouse",
                "params": [{"name": "region", "type": "string", "required": True}],
                "columns": [{"name": "a", "type": "integer"}],
                "sample_params": {"region": "APAC"},
                "sample_row_count": 1,
                "saved_at": "2026-05-13T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (dash_dir / "manifest.json").write_text(
        f'{{"slug":"{dashboard_slug}","name":"{title}","description":"d",'
        '"kind":"dashboard","created_at":"2026-05-13T00:00:00Z"}\n',
        encoding="utf-8",
    )
    return dash_dir


def _read_payload(html_body: str) -> dict:
    """Extract the inlined dashboard payload from the rendered HTML body."""
    marker = '<script type="application/json" id="datus-dashboard-data">'
    start = html_body.index(marker) + len(marker)
    end = html_body.index("</script>", start)
    return json.loads(html_body[start:end].replace("<\\/", "</"))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_render_dashboard_html_writes_index_next_to_artifact(tmp_path: Path):
    _seed_dashboard(tmp_path, dashboard_slug="demo_001")
    out_path = render_dashboard_html(project_root=tmp_path, dashboard_slug="demo_001")
    assert out_path == tmp_path / "dashboards" / "demo_001" / "index.html"
    assert out_path.is_file()


def test_render_dashboard_html_bakes_default_query_endpoint(tmp_path: Path):
    """No ``query_endpoint`` override → the default ``--web`` URL is baked in
    so the iframe knows where to POST every filter change."""
    _seed_dashboard(tmp_path, dashboard_slug="defaults")
    out_path = render_dashboard_html(project_root=tmp_path, dashboard_slug="defaults")
    body = out_path.read_text(encoding="utf-8")

    assert DEFAULT_QUERY_ENDPOINT in body
    # initDashboard call is wired (template is in single quotes around the placeholder).
    assert "DatusArtifact.initDashboard" in body
    # Auth + project propagation was dropped — the iframe no longer
    # forwards anything beyond ``queryEndpoint``. Keep an explicit guard
    # so a future regression that re-introduces those props gets caught.
    assert "authToken" not in body
    assert "projectId" not in body


def test_render_dashboard_html_threads_custom_endpoint_through(tmp_path: Path):
    _seed_dashboard(tmp_path, dashboard_slug="custom_be")
    out_path = render_dashboard_html(
        project_root=tmp_path,
        dashboard_slug="custom_be",
        query_endpoint="https://api.example.com/api/v1/dashboard/query",
    )
    body = out_path.read_text(encoding="utf-8")

    assert "queryEndpoint: 'https://api.example.com/api/v1/dashboard/query'" in body
    # Default value must NOT leak in when an override is supplied.
    assert DEFAULT_QUERY_ENDPOINT not in body


def test_render_dashboard_html_escapes_query_endpoint_for_js_single_quoted(tmp_path: Path):
    """The endpoint is slotted into a single-quoted JS string literal — a
    crafted value must NOT be able to close the literal, inject JS, or
    close the surrounding ``<script>`` block."""
    _seed_dashboard(tmp_path, dashboard_slug="injection_check")
    # All four escape vectors: single quote, backslash, newline, and
    # the ``</`` sequence that would otherwise terminate ``<script>``.
    crafted = "http://evil'\\bad\n</script><script>alert(1)</script>x"
    out_path = render_dashboard_html(
        project_root=tmp_path,
        dashboard_slug="injection_check",
        query_endpoint=crafted,
    )
    body = out_path.read_text(encoding="utf-8")

    # The escaped form (what we want) is present.
    assert ("queryEndpoint: 'http://evil\\'\\\\bad\\n<\\/script><script>alert(1)<\\/script>x'") in body
    # The raw ``</script>`` payload must NOT appear inside the
    # ``initDashboard`` script block — if it did, the browser would
    # terminate ``<script>`` early and execute the injected snippet.
    # We allow the literal ``</script>`` ONLY as the natural closer of
    # the inline ``initDashboard`` block; nothing else.
    init_idx = body.index("DatusArtifact.initDashboard")
    init_close = body.index("</script>", init_idx)
    init_block = body[init_idx:init_close]
    assert "</script>" not in init_block
    assert "alert(1)" in init_block  # text survives, but neutralised
    assert "<\\/script>" in init_block


def test_render_dashboard_html_includes_flat_files_with_template_pair(tmp_path: Path):
    """Payload is a single ``files`` list including both .sql.j2 + .params.json
    sibling files (the dashboard walker allowlist accepts both)."""
    _seed_dashboard(tmp_path, dashboard_slug="demo_003")
    out_path = render_dashboard_html(project_root=tmp_path, dashboard_slug="demo_003")
    payload = _read_payload(out_path.read_text(encoding="utf-8"))

    file_paths = {entry["path"] for entry in payload["files"]}
    assert "render/app.jsx" in file_paths
    assert "queries/by_region.sql.j2" in file_paths
    assert "queries/by_region.params.json" in file_paths


def test_render_dashboard_html_extracts_title_annotation(tmp_path: Path):
    _seed_dashboard(tmp_path, dashboard_slug="with_title", title="2026 Q1 Sales")
    out_path = render_dashboard_html(project_root=tmp_path, dashboard_slug="with_title")
    body = out_path.read_text(encoding="utf-8")
    assert "<title>Datus Dashboard — 2026 Q1 Sales</title>" in body

    payload = _read_payload(body)
    assert payload["title"] == "2026 Q1 Sales"


def test_render_dashboard_html_falls_back_to_slug_for_title(tmp_path: Path):
    """When app.jsx omits the @datus-title annotation, the slug is used."""
    dash_dir = tmp_path / "dashboards" / "no_title"
    (dash_dir / "queries").mkdir(parents=True, exist_ok=True)
    (dash_dir / "render").mkdir(exist_ok=True)
    (dash_dir / "render" / "app.jsx").write_text(
        "import React from 'react';\nexport default function App() { return null; }\n",
        encoding="utf-8",
    )
    out_path = render_dashboard_html(project_root=tmp_path, dashboard_slug="no_title")
    body = out_path.read_text(encoding="utf-8")
    assert "<title>Datus Dashboard — no_title</title>" in body


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------


def test_render_dashboard_html_rejects_invalid_slug(tmp_path: Path):
    """Defence-in-depth: the slug regex blocks path traversal before any I/O."""
    with pytest.raises(ValueError, match="invalid dashboard_slug"):
        render_dashboard_html(project_root=tmp_path, dashboard_slug="../escape")


def test_render_dashboard_html_missing_render_raises(tmp_path: Path):
    (tmp_path / "dashboards" / "missing" / "queries").mkdir(parents=True)
    (tmp_path / "dashboards" / "missing" / "render").mkdir()
    with pytest.raises(FileNotFoundError):
        render_dashboard_html(project_root=tmp_path, dashboard_slug="missing")


# ---------------------------------------------------------------------------
# CDN / offline switch
# ---------------------------------------------------------------------------


def test_render_dashboard_html_defaults_to_cdn(tmp_path: Path):
    _seed_dashboard(tmp_path, dashboard_slug="cdn_default")
    out_path = render_dashboard_html(project_root=tmp_path, dashboard_slug="cdn_default")
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-artifact-render" in body
    assert "index.css" in body
    assert "index.umd.js" in body
    assert not (tmp_path / "dashboards" / "cdn_default" / "_assets").exists()


def _seed_dist(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.css").write_text("/* offline css */", encoding="utf-8")
    (dist_dir / "index.umd.js").write_text("/* offline js */", encoding="utf-8")


def test_render_dashboard_html_offline_kwarg_copies_assets(tmp_path: Path):
    _seed_dashboard(tmp_path, dashboard_slug="offline_001")
    dist_dir = tmp_path / "vendor" / "datus-artifact-dist"
    _seed_dist(dist_dir)

    out_path = render_dashboard_html(
        project_root=tmp_path,
        dashboard_slug="offline_001",
        dashboard_dist=dist_dir,
    )
    body = out_path.read_text(encoding="utf-8")
    assert "_assets/index.css" in body
    assert "_assets/index.umd.js" in body
    assert "https://unpkg.com/" not in body

    copied_assets = tmp_path / "dashboards" / "offline_001" / "_assets"
    assert (copied_assets / "index.css").read_text(encoding="utf-8") == "/* offline css */"
    assert (copied_assets / "index.umd.js").read_text(encoding="utf-8") == "/* offline js */"


def test_render_dashboard_html_invalid_dist_falls_back_to_cdn(tmp_path: Path):
    _seed_dashboard(tmp_path, dashboard_slug="invalid_dist")
    incomplete = tmp_path / "vendor" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "index.css").write_text("/* partial */", encoding="utf-8")
    # No index.umd.js — the resolver should reject and fall back to CDN.

    out_path = render_dashboard_html(
        project_root=tmp_path,
        dashboard_slug="invalid_dist",
        dashboard_dist=incomplete,
    )
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-artifact-render" in body
    assert not (tmp_path / "dashboards" / "invalid_dist" / "_assets").exists()
