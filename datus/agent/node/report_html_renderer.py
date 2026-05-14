# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Compile a Datus report artifact into a single self-contained ``index.html``.

Used only by the Datus-CLI path. SaaS deployments render dynamically through
the backend ``/api/v1/report/detail`` endpoint and do not call this function.

The generated HTML inlines two payloads next to ``@datus/web-report``:

* ``main.jsx`` source as a ``<script type="application/json">`` tag (the
  agent's ``save_main_jsx`` output, untouched).
* ``queries/<slug>.sql`` + ``.json`` as ``[{name, content}, ...]`` entries.

``@datus/web-report`` boots the standalone viewer, which spins up the
sandboxed iframe runtime that Babel-compiles ``main.jsx`` and renders it.

Two asset-loading modes, mirroring ``datus.cli.web.chatbot``:

* **CDN mode (default)** — the rendered HTML loads ``@datus/web-report`` from
  ``unpkg.com`` at a pinned version. Requires network at view time.
* **Offline mode** — caller passes ``report_dist`` (resolved upstream from
  the ``--report-dist`` CLI flag or ``agentic_nodes.gen_visual_report.report_dist``).
  The two assets are copied next to the ``index.html`` under ``_assets/``
  and the template is rewritten to reference them via relative paths so
  the result opens through ``file://`` with no network access.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report_index.html"
_DATA_PLACEHOLDER = "__DATUS_REPORT_DATA__"
_TITLE_PLACEHOLDER = "__DATUS_REPORT_TITLE__"
_CSS_URL_PLACEHOLDER = "__DATUS_REPORT_CSS_URL__"
_JS_URL_PLACEHOLDER = "__DATUS_REPORT_JS_URL__"

# Best-effort title extraction from a JSDoc-style annotation at the top of
# main.jsx, e.g. ``/** @datus-title 2026 Q1 NA Sales Report */``. The runtime
# falls back to the report id when this isn't present.
_TITLE_ANNOTATION_RE = re.compile(r"@datus-title\s+([^\n*/]+?)(?:\s*\*/|\s*\n|$)")

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


def _extract_title(main_jsx: str, fallback: str) -> str:
    match = _TITLE_ANNOTATION_RE.search(main_jsx[:2048])
    if match:
        title = match.group(1).strip()
        if title:
            return title
    return fallback


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

    Returns ``None`` (caller falls back to CDN) when the path is unset,
    not a directory, or missing one of the required asset files.
    """
    if not report_dist:
        return None

    resolved = Path(report_dist).expanduser().resolve()
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
    Compile ``reports/<report_id>/index.html`` from main.jsx + queries.

    Args:
        project_root: ``AgentConfig.project_root``; resolved absolute path.
        report_id: target report id (matches the directory name).
        report_dist: optional path to a local ``@datus/web-report`` ``dist/``
            directory containing ``datus-report.css`` and
            ``datus-report.umd.js``. When provided and valid, the two files
            are copied next to the generated HTML and the template links to
            them via relative paths (so the page works offline through
            ``file://``). When ``None`` (or the directory is missing /
            incomplete), the template links to the pinned unpkg CDN instead.

    Returns:
        Absolute path to the generated ``index.html``.

    Raises:
        FileNotFoundError: if ``main.jsx`` is missing.
        OSError: on read/write failures.
    """
    project_root = project_root.resolve()
    report_dir = project_root / "reports" / report_id
    main_jsx_path = report_dir / "main.jsx"
    if not main_jsx_path.is_file():
        raise FileNotFoundError(f"main.jsx not found under {report_dir}")

    main_jsx = main_jsx_path.read_text(encoding="utf-8")
    queries = _read_queries(report_dir / "queries")

    dist_dir = _resolve_dist(report_dist)
    if dist_dir is not None:
        css_url, js_url = _copy_offline_assets(report_dir, dist_dir)
        logger.info("Offline mode: copied web-report assets from %s", dist_dir)
    else:
        css_url, js_url = _CDN_REPORT_CSS, _CDN_REPORT_JS

    template_html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    created_at = _dt.datetime.fromtimestamp(main_jsx_path.stat().st_mtime, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    title = _extract_title(main_jsx, report_id)
    payload = {
        "id": report_id,
        "title": title,
        "created_at": created_at,
        "main_jsx": main_jsx,
        "queries": queries,
    }
    payload_json = _escape_for_script_tag(json.dumps(payload, ensure_ascii=False))
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
