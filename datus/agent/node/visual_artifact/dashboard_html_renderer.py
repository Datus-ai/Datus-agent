# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Compile a Datus dashboard artifact into a single self-contained ``index.html``.

Used only by the Datus-CLI path. SaaS deployments render dynamically
through the backend ``/api/v1/dashboard/detail`` endpoint and do not
call this function.

Unlike the report HTML (which inlines pre-baked JSON results next to the
JSX), the dashboard HTML inlines a parameterized template payload and
bootstraps ``DatusArtifact.initDashboard`` against a **live query
endpoint** — the renderer iframe issues
``POST <queryEndpoint>?dashboard_slug=...&query_slug=...`` for every
filter change. That endpoint is the agent's own ``--web`` server: launch
it with ``datus --web --datasource <ds>`` and the compiled HTML at
``dashboards/<slug>/index.html`` becomes interactive.

Three asset / runtime modes:

* **CDN mode (default)** — the rendered HTML loads
  ``@datus/web-artifact-render`` from ``unpkg.com``. Needs network access
  for the bundle; the query backend is still the local ``datus --web``
  server unless overridden.
* **Offline mode** — caller passes ``dashboard_dist`` and the CSS/UMD
  pair is copied next to the HTML under ``_assets/`` so it opens through
  ``file://`` without hitting the CDN.
* **Custom backend** — ``query_endpoint`` is baked into the HTML at
  compile time. Default points at ``http://localhost:8501/api/v1/dashboard/query``
  (matches the ``--host`` / ``--port`` defaults of ``datus --web``);
  callers running against a different host or a SaaS-style deployment
  override it.

The shared walker / template-slotting / asset-resolution machinery
lives in :mod:`datus.agent.node.visual_artifact._artifact_html_renderer`;
this module only owns the dashboard-kind config + the public entrypoint.
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

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard_index.html"

# Per-prefix allowlist driving the flat artifact walker. ``queries/``
# carries Jinja2 templates + params metadata (vs. report's pre-executed
# SQL + JSON result pairs). Kept in lockstep with the dashboard API
# service's walker so the CLI-emitted HTML payload and the SaaS
# ``IDashboardDetail`` share one shape.
_DASHBOARD_ARTIFACT_DIRS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "render": ((".jsx", ".js", ".css", ".json"), True),
    "queries": ((".sql.j2", ".params.json"), False),
}

# Same shape as ``DASHBOARD_SLUG_RE`` in
# ``datus.schemas.gen_visual_dashboard_models`` — duplicated here so the
# CLI renderer has no schema-side import and stays usable in trimmed
# environments (e.g. notebooks loading the renderer directly).
_DASHBOARD_SLUG_RE = re.compile(r"^[a-z0-9_]{1,80}$")

#: Default query backend baked into the compiled HTML when the caller
#: doesn't override ``query_endpoint``. Matches the default ``--host`` /
#: ``--port`` of ``datus --web`` (see ``datus.cli.main``).
DEFAULT_QUERY_ENDPOINT = "http://localhost:8501/api/v1/dashboard/query"


def _escape_js_single_quoted(value: str) -> str:
    """Escape ``value`` so it can be slotted into a single-quoted JS string.

    The dashboard template wraps the query-endpoint placeholder in
    ``'...'``; without this escape, a value containing ``'`` could close
    the literal, and a ``</script>`` substring could close the surrounding
    ``<script>`` block. The endpoint is normally derived from
    ``agent.yml`` keys, but the agent is open-source and shipped to
    third-party deployments, so defence-in-depth is cheap and matches
    the ``_escape_for_script_tag`` treatment the JSON payload already
    gets.
    """
    return (
        value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r").replace("</", "<\\/")
    )


_DASHBOARD_SPEC = ArtifactHtmlSpec(
    kind="dashboard",
    root_dir_name="dashboards",
    slug_regex=_DASHBOARD_SLUG_RE,
    artifact_dirs=_DASHBOARD_ARTIFACT_DIRS,
    template_path=_TEMPLATE_PATH,
    data_placeholder="__DATUS_DASHBOARD_DATA__",
    title_placeholder="__DATUS_DASHBOARD_TITLE__",
    css_url_placeholder="__DATUS_DASHBOARD_CSS_URL__",
    js_url_placeholder="__DATUS_DASHBOARD_JS_URL__",
    extra_placeholders={},  # filled per call by render_dashboard_html
)


def render_dashboard_html(
    *,
    project_root: Path,
    dashboard_slug: str,
    dashboard_dist: Optional[Path] = None,
    query_endpoint: Optional[str] = None,
) -> Path:
    """
    Compile ``dashboards/<dashboard_slug>/index.html`` from render/ + queries.

    Args:
        project_root: ``AgentConfig.project_root``; resolved absolute path.
        dashboard_slug: target dashboard slug (matches the directory name).
        dashboard_dist: optional path to a local ``@datus/web-artifact-render``
            ``dist/`` directory containing ``index.css`` and
            ``index.umd.js``. When provided and valid, the two files
            are copied next to the generated HTML and the template links to
            them via relative paths (so the page works offline through
            ``file://``). When ``None`` (or the directory is missing /
            incomplete), the template links to the pinned unpkg CDN instead.
        query_endpoint: absolute URL the rendered HTML will POST to for
            every dashboard query. Defaults to
            :data:`DEFAULT_QUERY_ENDPOINT` (``http://localhost:8501/api/v1/dashboard/query``).

    Returns:
        Absolute path to the generated ``index.html``.

    Raises:
        ValueError: if ``dashboard_slug`` is malformed.
        FileNotFoundError: if ``render/app.jsx`` is missing.
        OSError: on read/write failures.
    """
    spec = ArtifactHtmlSpec(
        kind=_DASHBOARD_SPEC.kind,
        root_dir_name=_DASHBOARD_SPEC.root_dir_name,
        slug_regex=_DASHBOARD_SPEC.slug_regex,
        artifact_dirs=_DASHBOARD_SPEC.artifact_dirs,
        template_path=_DASHBOARD_SPEC.template_path,
        data_placeholder=_DASHBOARD_SPEC.data_placeholder,
        title_placeholder=_DASHBOARD_SPEC.title_placeholder,
        css_url_placeholder=_DASHBOARD_SPEC.css_url_placeholder,
        js_url_placeholder=_DASHBOARD_SPEC.js_url_placeholder,
        extra_placeholders={
            "__DATUS_QUERY_ENDPOINT__": _escape_js_single_quoted(query_endpoint or DEFAULT_QUERY_ENDPOINT),
        },
    )
    return render_artifact_html(
        spec=spec,
        project_root=project_root,
        slug=dashboard_slug,
        dist=dashboard_dist,
    )
