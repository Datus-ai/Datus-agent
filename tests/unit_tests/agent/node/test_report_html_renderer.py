# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the CLI HTML renderer that compiles report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.agent.node.report_html_renderer import render_report_html


def _seed_report(project_root: Path, *, report_id: str = "rpt_demo_001") -> Path:
    report_dir = project_root / "reports" / report_id
    (report_dir / "queries").mkdir(parents=True)
    manifest = {
        "version": "1.0",
        "id": report_id,
        "title": "Demo report </script>",
        "created_at": "2026-05-13T10:00:00Z",
        "sections": [
            {"id": "blk_001", "type": "markdown", "content": "# hi"},
        ],
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (report_dir / "queries" / "q.sql").write_text("SELECT 1", encoding="utf-8")
    (report_dir / "queries" / "q.json").write_text('{"row_count":0,"rows":[]}', encoding="utf-8")
    return report_dir


def test_render_report_html_substitutes_payload(tmp_path: Path):
    _seed_report(tmp_path)
    out_path = render_report_html(project_root=tmp_path, report_id="rpt_demo_001")
    body = out_path.read_text(encoding="utf-8")
    assert "__DATUS_REPORT_DATA__" not in body
    assert "__DATUS_REPORT_TITLE__" not in body
    assert "Demo report" in body
    # </script> from the title must be escaped so it doesn't close the data block.
    assert "</script></script>" not in body


def test_render_report_html_writes_index_file(tmp_path: Path):
    _seed_report(tmp_path, report_id="rpt_demo_002")
    out_path = render_report_html(project_root=tmp_path, report_id="rpt_demo_002")
    assert out_path == tmp_path / "reports" / "rpt_demo_002" / "index.html"
    assert out_path.is_file()


def test_render_report_html_includes_queries_in_payload(tmp_path: Path):
    _seed_report(tmp_path, report_id="rpt_demo_003")
    out_path = render_report_html(project_root=tmp_path, report_id="rpt_demo_003")
    body = out_path.read_text(encoding="utf-8")
    # The data block should be a single line of valid JSON between the data tags.
    start = body.index('id="datus-report-data">') + len('id="datus-report-data">')
    end = body.index("</script>", start)
    payload = body[start:end]
    payload_unescaped = payload.replace("<\\/", "</")
    data = json.loads(payload_unescaped)
    assert data["manifest"]["id"] == "rpt_demo_003"
    query_names = {q["name"] for q in data["queries"]}
    assert query_names == {"q.sql", "q.json"}


def test_render_report_html_missing_manifest_raises(tmp_path: Path):
    (tmp_path / "reports" / "rpt_missing" / "queries").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        render_report_html(project_root=tmp_path, report_id="rpt_missing")


def test_render_report_html_defaults_to_cdn(tmp_path: Path):
    _seed_report(tmp_path, report_id="rpt_cdn_default")
    out_path = render_report_html(project_root=tmp_path, report_id="rpt_cdn_default")
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-report" in body
    assert "datus-report.css" in body
    assert "datus-report.umd.js" in body
    assert not (tmp_path / "reports" / "rpt_cdn_default" / "_assets").exists()


def _seed_dist(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "datus-report.css").write_text("/* offline css */", encoding="utf-8")
    (dist_dir / "datus-report.umd.js").write_text("/* offline js */", encoding="utf-8")


def test_render_report_html_offline_kwarg_copies_assets(tmp_path: Path):
    _seed_report(tmp_path, report_id="rpt_offline_001")
    dist_dir = tmp_path / "vendor" / "datus-report-dist"
    _seed_dist(dist_dir)

    out_path = render_report_html(
        project_root=tmp_path,
        report_id="rpt_offline_001",
        report_dist=dist_dir,
    )
    body = out_path.read_text(encoding="utf-8")
    # Local relative paths replace the CDN.
    assert "_assets/datus-report.css" in body
    assert "_assets/datus-report.umd.js" in body
    assert "https://unpkg.com/" not in body

    copied_assets = tmp_path / "reports" / "rpt_offline_001" / "_assets"
    assert (copied_assets / "datus-report.css").read_text(encoding="utf-8") == "/* offline css */"
    assert (copied_assets / "datus-report.umd.js").read_text(encoding="utf-8") == "/* offline js */"


def test_render_report_html_invalid_dist_falls_back_to_cdn(tmp_path: Path):
    _seed_report(tmp_path, report_id="rpt_invalid_dist")
    # Directory exists but only contains the css — js missing, should fall back.
    incomplete = tmp_path / "vendor" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "datus-report.css").write_text("/* partial */", encoding="utf-8")

    out_path = render_report_html(
        project_root=tmp_path,
        report_id="rpt_invalid_dist",
        report_dist=incomplete,
    )
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-report" in body
    assert not (tmp_path / "reports" / "rpt_invalid_dist" / "_assets").exists()


def test_render_report_html_ignores_environment_variables(tmp_path: Path, monkeypatch):
    """``DATUS_REPORT_DIST`` was removed — the renderer must not read it.

    Locks the contract so a future revert of the env-var fallback fails this
    test rather than silently re-enabling the legacy code path. The dist is
    valid; if env-var support is back, the assets would be copied and the
    assertion below would flip.
    """
    _seed_report(tmp_path, report_id="rpt_no_env_lookup")
    dist_dir = tmp_path / "vendor" / "would-be-env"
    _seed_dist(dist_dir)
    monkeypatch.setenv("DATUS_REPORT_DIST", str(dist_dir))

    out_path = render_report_html(project_root=tmp_path, report_id="rpt_no_env_lookup")
    body = out_path.read_text(encoding="utf-8")
    assert "https://unpkg.com/@datus/web-report" in body
    assert not (tmp_path / "reports" / "rpt_no_env_lookup" / "_assets").exists()
