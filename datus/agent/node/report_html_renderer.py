# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Compile a Datus report artifact into a single self-contained ``index.html``.

Used only by the Datus-CLI path. SaaS deployments render dynamically through
the backend ``/api/v1/report/detail`` endpoint and do not call this function.

Two asset-loading modes, mirroring ``datus.cli.web.chatbot``:

* **CDN mode (default)** — the rendered HTML loads ``@datus/web-report`` from
  ``unpkg.com`` at a pinned version. Requires network at view time.
* **Offline mode** — caller passes ``report_dist`` (or a node-config /
  ``DATUS_REPORT_DIST`` env override). The two assets are copied next to
  the ``index.html`` under ``_assets/`` and the template is rewritten to
  reference them via relative paths. The result opens correctly through
  ``file://`` with no network access.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report_index.html"
_DATA_PLACEHOLDER = "__DATUS_REPORT_DATA__"
_TITLE_PLACEHOLDER = "__DATUS_REPORT_TITLE__"
_CSS_URL_PLACEHOLDER = "__DATUS_REPORT_CSS_URL__"
_JS_URL_PLACEHOLDER = "__DATUS_REPORT_JS_URL__"

# CDN URLs used when no offline dist is supplied. Keep the pinned version in
# lockstep with ``packages/web-report/package.json``.
_CDN_REPORT_VERSION = "0.1.0"
_CDN_REPORT_CSS = f"https://unpkg.com/@datus/web-report@{_CDN_REPORT_VERSION}/dist/datus-report.css"
_CDN_REPORT_JS = f"https://unpkg.com/@datus/web-report@{_CDN_REPORT_VERSION}/dist/datus-report.umd.js"

# Filename pair we expect inside ``report_dist``. Same names as those emitted
# by ``packages/web-report`` (``vite build``).
_DIST_CSS_NAME = "datus-report.css"
_DIST_JS_NAME = "datus-report.umd.js"

# Subdirectory under ``reports/<id>/`` where local assets are copied.
_ASSETS_SUBDIR = "_assets"


def _read_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_queries(queries_dir: Path) -> List[Dict[str, str]]:
    """Return file entries in deterministic order (alphabetical by name)."""
    if not queries_dir.is_dir():
        return []
    entries: List[Dict[str, str]] = []
    for path in sorted(queries_dir.iterdir(), key=lambda p: p.name):
        if path.suffix not in {".sql", ".json"} or not path.is_file():
            continue
        entries.append({"name": path.name, "content": path.read_text(encoding="utf-8")})
    return entries


def _escape_for_script_tag(payload: str) -> str:
    """Escape `</` sequences so the JSON survives being embedded in a <script> block."""
    return payload.replace("</", "<\\/")


def _resolve_dist(report_dist: Optional[Path]) -> Optional[Path]:
    """Validate ``report_dist`` and return the resolved directory or ``None``.

    Falls back to ``$DATUS_REPORT_DIST`` when the caller did not pass an
    explicit path, matching how ``datus.cli.web.chatbot`` honours its
    ``--chatbot-dist`` flag.
    """
    candidate: Optional[str]
    if report_dist is not None:
        candidate = str(report_dist)
    else:
        candidate = os.environ.get("DATUS_REPORT_DIST") or None
    if not candidate:
        return None

    resolved = Path(candidate).expanduser().resolve()
    if not resolved.is_dir():
        logger.warning("report_dist %s is not a directory; falling back to CDN.", resolved)
        return None

    missing = [name for name in (_DIST_CSS_NAME, _DIST_JS_NAME) if not (resolved / name).is_file()]
    if missing:
        logger.warning(
            "report_dist %s is missing required assets %s; falling back to CDN.",
            resolved,
            missing,
        )
        return None
    return resolved


def _copy_offline_assets(report_dir: Path, dist_dir: Path) -> tuple[str, str]:
    """Copy css + umd next to the report and return the relative URLs to use.

    The copy is idempotent — repeated renders against the same directory
    overwrite the previous payload, which is what we want when the user
    updates their local build.
    """
    assets_dir = report_dir / _ASSETS_SUBDIR
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dist_dir / _DIST_CSS_NAME, assets_dir / _DIST_CSS_NAME)
    shutil.copy2(dist_dir / _DIST_JS_NAME, assets_dir / _DIST_JS_NAME)
    return (
        f"{_ASSETS_SUBDIR}/{_DIST_CSS_NAME}",
        f"{_ASSETS_SUBDIR}/{_DIST_JS_NAME}",
    )


def render_report_html(
    *,
    project_root: Path,
    report_id: str,
    report_dist: Optional[Path] = None,
) -> Path:
    """
    Compile ``reports/<report_id>/index.html`` from manifest + queries.

    Args:
        project_root: ``AgentConfig.project_root``; resolved absolute path.
        report_id: target report id (matches the directory name).
        report_dist: optional path to a local ``@datus/web-report`` ``dist/``
            directory containing ``datus-report.css`` and
            ``datus-report.umd.js``. When provided and valid, the two files
            are copied next to the generated HTML and the template links to
            them via relative paths (so the page works offline through
            ``file://``). When ``None``, falls back to the ``DATUS_REPORT_DIST``
            environment variable, then to the pinned unpkg CDN.

    Returns:
        Absolute path to the generated ``index.html``.

    Raises:
        FileNotFoundError: if ``manifest.json`` is missing.
        OSError: on read/write failures.
    """
    project_root = project_root.resolve()
    report_dir = project_root / "reports" / report_id
    manifest_path = report_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found under {report_dir}")

    manifest = _read_manifest(manifest_path)
    queries = _read_queries(report_dir / "queries")

    dist_dir = _resolve_dist(report_dist)
    if dist_dir is not None:
        css_url, js_url = _copy_offline_assets(report_dir, dist_dir)
        logger.info("Offline mode: copied web-report assets from %s", dist_dir)
    else:
        css_url, js_url = _CDN_REPORT_CSS, _CDN_REPORT_JS

    template_html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = {"manifest": manifest, "queries": queries}
    payload_json = _escape_for_script_tag(json.dumps(payload, ensure_ascii=False))
    title = manifest.get("title", report_id)
    rendered = (
        template_html.replace(_DATA_PLACEHOLDER, payload_json)
        .replace(_TITLE_PLACEHOLDER, title)
        .replace(_CSS_URL_PLACEHOLDER, css_url)
        .replace(_JS_URL_PLACEHOLDER, js_url)
    )

    out_path = report_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    logger.info("report HTML written to %s", out_path)
    return out_path
