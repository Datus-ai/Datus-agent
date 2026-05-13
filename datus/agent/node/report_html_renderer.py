# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Compile a Datus report artifact into a single self-contained ``index.html``.

Used only by the Datus-CLI path. SaaS deployments render dynamically through
the backend ``/api/v1/report/detail`` endpoint and do not call this function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report_index.html"
_DATA_PLACEHOLDER = "__DATUS_REPORT_DATA__"
_TITLE_PLACEHOLDER = "__DATUS_REPORT_TITLE__"


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


def render_report_html(*, project_root: Path, report_id: str) -> Path:
    """
    Compile ``reports/<report_id>/index.html`` from manifest + queries.

    Args:
        project_root: ``AgentConfig.project_root``; resolved absolute path.
        report_id: target report id (matches the directory name).

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

    template_html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = {"manifest": manifest, "queries": queries}
    payload_json = _escape_for_script_tag(json.dumps(payload, ensure_ascii=False))
    title = manifest.get("title", report_id)
    rendered = template_html.replace(_DATA_PLACEHOLDER, payload_json).replace(_TITLE_PLACEHOLDER, title)

    out_path = report_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    logger.info("report HTML written to %s", out_path)
    return out_path
