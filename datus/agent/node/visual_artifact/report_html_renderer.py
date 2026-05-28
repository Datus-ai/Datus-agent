# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Compile a Datus report artifact into a single self-contained ``index.html``.

Used only by the Datus-CLI path. SaaS deployments render dynamically through
the backend ``/api/v1/report/detail`` endpoint and do not call this function.

The generated HTML inlines a single payload next to ``@datus/web-artifact-render``:

* ``files: [{path, content}, ...]`` inside one ``<script type="application/json">``
  block. ``path`` is **slug-relative** and includes the top-level directory
  prefix (e.g. ``render/app.jsx``, ``render/charts/trend.jsx``,
  ``queries/sales_by_zone.sql``). Allowed prefixes: ``render/`` (.jsx / .js /
  .css / .json, recursive — JSON allowed for sidecars an LLM may park next
  to its modules) and ``queries/`` (.sql / .json, one level). This matches
  the ``IPublishedReportArtifact`` / ``IReportDetail`` shape that
  ``@datus/web-artifact-render`` consumes via ``splitArtifactFiles(detail.files)``.

``@datus/web-artifact-render`` boots the standalone viewer, which spins up
the sandboxed iframe runtime; the runtime Babel-compiles each module on
demand and renders the default export of ``render/app.jsx``. The UMD
global ``DatusArtifact`` exposes ``initReport`` / ``initDashboard``;
this renderer drives ``initReport``.

Two asset-loading modes, mirroring ``datus.cli.web.chatbot``:

* **CDN mode (default)** — the rendered HTML loads
  ``@datus/web-artifact-render`` from ``unpkg.com`` at a pinned version.
  Requires network at view time.
* **Offline mode** — caller passes ``report_dist`` (resolved upstream from
  the ``--report-dist`` CLI flag or ``agentic_nodes.gen_visual_report.report_dist``).
  The two assets are copied next to the ``index.html`` under ``_assets/``
  and the template is rewritten to reference them via relative paths so
  the result opens through ``file://`` with no network access.

The shared walker / template-slotting / asset-resolution machinery
lives in :mod:`datus.agent.node.visual_artifact._artifact_html_renderer`;
this module only owns the report-kind config + the public entrypoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from datus.agent.node.visual_artifact._artifact_html_renderer import (
    ArtifactHtmlSpec,
    render_artifact_html,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report_index.html"

# Per-prefix allowlist driving the flat artifact walker. Kept in
# lockstep with the report API service's walker so the CLI-emitted
# HTML payload and the SaaS ``IPublishedReportArtifact`` share one
# shape.
_REPORT_ARTIFACT_DIRS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "render": ((".jsx", ".js", ".css", ".json"), True),
    "queries": ((".sql", ".json"), False),
}

# Accepted shape for ``report_slug`` — restricting this up front prevents
# path traversal (``..``) or absolute-path components from escaping
# ``reports/`` when the slug is joined into
# ``project_root / "reports" / report_slug``.
_REPORT_SLUG_RE = re.compile(r"^[a-z0-9_]{1,80}$")

_REPORT_SPEC = ArtifactHtmlSpec(
    kind="report",
    root_dir_name="reports",
    slug_regex=_REPORT_SLUG_RE,
    artifact_dirs=_REPORT_ARTIFACT_DIRS,
    template_path=_TEMPLATE_PATH,
    data_placeholder="__DATUS_REPORT_DATA__",
    title_placeholder="__DATUS_REPORT_TITLE__",
    css_url_placeholder="__DATUS_REPORT_CSS_URL__",
    js_url_placeholder="__DATUS_REPORT_JS_URL__",
    extra_placeholders={},
)


def render_report_html(
    *,
    project_root: Path,
    report_slug: str,
    report_dist: Optional[Path] = None,
) -> Path:
    """
    Compile ``reports/<report_slug>/index.html`` from render/ + queries.

    Args:
        project_root: ``AgentConfig.project_root``; resolved absolute path.
        report_slug: target report slug (matches the directory name).
        report_dist: optional path to a local ``@datus/web-artifact-render``
            ``dist/`` directory containing ``index.css`` and
            ``index.umd.js``. When provided and valid, the two files
            are copied next to the generated HTML and the template links to
            them via relative paths (so the page works offline through
            ``file://``). When ``None`` (or the directory is missing /
            incomplete), the template links to the pinned unpkg CDN instead.

    Returns:
        Absolute path to the generated ``index.html``.

    Raises:
        ValueError: if ``report_slug`` is malformed.
        FileNotFoundError: if ``render/app.jsx`` is missing.
        OSError: on read/write failures.
    """
    return render_artifact_html(
        spec=_REPORT_SPEC,
        project_root=project_root,
        slug=report_slug,
        dist=report_dist,
    )
