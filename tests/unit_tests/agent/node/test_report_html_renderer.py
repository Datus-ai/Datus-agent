# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the CLI HTML renderer that compiles report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.agent.node.report_html_renderer import render_report_html

_APP_JSX = """\
/** @datus-title Demo report <update & fix> */
import React from 'react';
import KpiBanner from './kpi-banner';
import { useDatusArtifact } from '@datus/web-artifact';

// Sentinel </script> exercises the JSON-escape path independently of the
// title (the title regex strips '/' so the title alone cannot carry it).
const SENTINEL = '</script>';

export default function App() {
  const { useQuerySql } = useDatusArtifact();
  const { data } = useQuerySql('queries/q');
  return React.createElement(KpiBanner, { rows: data?.rows ?? [], sentinel: SENTINEL });
}
"""

_KPI_BANNER_JSX = """\
import React from 'react';
export default function KpiBanner({ rows }) {
  return React.createElement('div', null, rows.length, ' rows');
}
"""


def _seed_report(project_root: Path, *, report_slug: str = "demo_001") -> Path:
    report_dir = project_root / "reports" / report_slug
    (report_dir / "queries").mkdir(parents=True)
    (report_dir / "render").mkdir()
    (report_dir / "render" / "app.jsx").write_text(_APP_JSX, encoding="utf-8")
    (report_dir / "render" / "kpi-banner.jsx").write_text(_KPI_BANNER_JSX, encoding="utf-8")
    (report_dir / "queries" / "q.sql").write_text("SELECT 1", encoding="utf-8")
    (report_dir / "queries" / "q.json").write_text('{"row_count":0,"rows":[]}', encoding="utf-8")
    return report_dir


def test_render_report_html_substitutes_payload(tmp_path: Path):
    _seed_report(tmp_path)
    out_path = render_report_html(project_root=tmp_path, report_slug="demo_001")
    body = out_path.read_text(encoding="utf-8")
    assert "__DATUS_REPORT_DATA__" not in body
    assert "__DATUS_REPORT_TITLE__" not in body
    assert "Demo report" in body
    # The title is HTML-escaped before being injected into <title>, so the
    # raw '<', '&', '>' from the annotation must appear as entities.
    assert "<title>Datus Report — Demo report &lt;update &amp; fix&gt;</title>" in body
    # </script> from the source must be escaped in the JSON data block so it
    # doesn't close the embedded <script type="application/json"> prematurely.
    assert "</script></script>" not in body


def test_render_report_html_writes_index_file(tmp_path: Path):
    _seed_report(tmp_path, report_slug="demo_002")
    out_path = render_report_html(project_root=tmp_path, report_slug="demo_002")
    assert out_path == tmp_path / "reports" / "demo_002" / "index.html"
    assert out_path.is_file()


def test_render_report_html_includes_render_files_and_queries(tmp_path: Path):
    _seed_report(tmp_path, report_slug="demo_003")
    out_path = render_report_html(project_root=tmp_path, report_slug="demo_003")
    body = out_path.read_text(encoding="utf-8")

    start = body.index('id="datus-report-data">') + len('id="datus-report-data">')
    end = body.index("</script>", start)
    payload_raw = body[start:end]
    payload_unescaped = payload_raw.replace("<\\/", "</")
    data = json.loads(payload_unescaped)

    assert data["slug"] == "demo_003"
    render_names = {f["name"] for f in data["render_files"]}
    assert render_names == {"app.jsx", "kpi-banner.jsx"}
    app_entry = next(f for f in data["render_files"] if f["name"] == "app.jsx")
    assert "useDatusArtifact" in app_entry["content"]
    query_names = {q["name"] for q in data["queries"]}
    assert query_names == {"q.sql", "q.json"}
    assert "T" in data["created_at"] and data["created_at"].endswith("Z")


def test_render_report_html_walks_nested_render_dirs(tmp_path: Path):
    _seed_report(tmp_path, report_slug="nested_001")
    report_dir = tmp_path / "reports" / "nested_001"
    (report_dir / "render" / "charts").mkdir()
    (report_dir / "render" / "charts" / "trend.jsx").write_text(
        "import React from 'react';\nexport default () => React.createElement('div');\n",
        encoding="utf-8",
    )
    out_path = render_report_html(project_root=tmp_path, report_slug="nested_001")
    body = out_path.read_text(encoding="utf-8")
    start = body.index('id="datus-report-data">') + len('id="datus-report-data">')
    end = body.index("</script>", start)
    data = json.loads(body[start:end].replace("<\\/", "</"))
    render_names = {f["name"] for f in data["render_files"]}
    assert "charts/trend.jsx" in render_names


def test_render_report_html_missing_app_jsx_raises(tmp_path: Path):
    (tmp_path / "reports" / "missing" / "queries").mkdir(parents=True)
    (tmp_path / "reports" / "missing" / "render").mkdir()
    with pytest.raises(FileNotFoundError):
        render_report_html(project_root=tmp_path, report_slug="missing")


def test_render_report_html_defaults_to_cdn(tmp_path: Path):
    _seed_report(tmp_path, report_slug="cdn_default")
    out_path = render_report_html(project_root=tmp_path, report_slug="cdn_default")
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-report" in body
    assert "index.css" in body
    assert "index.umd.js" in body
    assert not (tmp_path / "reports" / "cdn_default" / "_assets").exists()


def _seed_dist(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.css").write_text("/* offline css */", encoding="utf-8")
    (dist_dir / "index.umd.js").write_text("/* offline js */", encoding="utf-8")


def test_render_report_html_offline_kwarg_copies_assets(tmp_path: Path):
    _seed_report(tmp_path, report_slug="offline_001")
    dist_dir = tmp_path / "vendor" / "datus-report-dist"
    _seed_dist(dist_dir)

    out_path = render_report_html(
        project_root=tmp_path,
        report_slug="offline_001",
        report_dist=dist_dir,
    )
    body = out_path.read_text(encoding="utf-8")
    assert "_assets/index.css" in body
    assert "_assets/index.umd.js" in body
    assert "https://unpkg.com/" not in body

    copied_assets = tmp_path / "reports" / "offline_001" / "_assets"
    assert (copied_assets / "index.css").read_text(encoding="utf-8") == "/* offline css */"
    assert (copied_assets / "index.umd.js").read_text(encoding="utf-8") == "/* offline js */"


def test_render_report_html_invalid_dist_falls_back_to_cdn(tmp_path: Path):
    _seed_report(tmp_path, report_slug="invalid_dist")
    incomplete = tmp_path / "vendor" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "index.css").write_text("/* partial */", encoding="utf-8")

    out_path = render_report_html(
        project_root=tmp_path,
        report_slug="invalid_dist",
        report_dist=incomplete,
    )
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-report" in body
    assert not (tmp_path / "reports" / "invalid_dist" / "_assets").exists()


def test_render_report_html_ignores_environment_variables(tmp_path: Path, monkeypatch):
    """``DATUS_REPORT_DIST`` was removed — the renderer must not read it."""
    _seed_report(tmp_path, report_slug="no_env_lookup")
    dist_dir = tmp_path / "vendor" / "would-be-env"
    _seed_dist(dist_dir)
    monkeypatch.setenv("DATUS_REPORT_DIST", str(dist_dir))

    out_path = render_report_html(project_root=tmp_path, report_slug="no_env_lookup")
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-report" in body
    assert not (tmp_path / "reports" / "no_env_lookup" / "_assets").exists()


def test_render_report_html_falls_back_to_report_slug_for_title(tmp_path: Path):
    """When app.jsx omits the @datus-title annotation, the report slug is used."""
    report_dir = tmp_path / "reports" / "no_title"
    (report_dir / "queries").mkdir(parents=True)
    (report_dir / "render").mkdir()
    (report_dir / "render" / "app.jsx").write_text(
        "import React from 'react';\nexport default function R() { return null; }\n",
        encoding="utf-8",
    )
    out_path = render_report_html(project_root=tmp_path, report_slug="no_title")
    body = out_path.read_text(encoding="utf-8")
    assert "<title>Datus Report — no_title</title>" in body
