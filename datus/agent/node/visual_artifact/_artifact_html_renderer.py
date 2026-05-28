# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared CLI HTML compile machinery for visual-artifact subagents.

Both ``gen_visual_report`` and ``gen_visual_dashboard`` compile a single
self-contained ``index.html`` next to their on-disk artifact directory.
The pipeline is identical:

1. Validate the slug (defence-in-depth against path traversal).
2. Read the artifact's render/ + queries/ tree into a slug-relative
   ``files: [{path, content}, ...]`` payload — same shape that the SaaS
   ``IPublished{Report,Dashboard}Artifact`` carries.
3. Resolve asset URLs: a local ``--{report,report}-dist`` directory when
   the caller passed one (offline ``file://`` mode), the pinned unpkg
   CDN otherwise.
4. Slot the payload + asset URLs into the artifact-kind-specific HTML
   template and write it to ``<artifact_dir>/index.html``.

What differs across kinds is centralised in :class:`ArtifactHtmlSpec`:

* ``root_dir_name`` — ``"reports"`` or ``"dashboards"``;
* ``artifact_dirs`` — per-prefix walker allowlist
  (``render/`` uses .jsx/.js/.css/.json; queries are ``.sql/.json`` for
  reports and ``.sql.j2/.params.json`` for dashboards);
* ``template_path`` — the HTML template file to slot the payload into;
* ``extra_placeholders`` — kind-specific placeholders the caller can
  fill in (the dashboard template needs ``__DATUS_QUERY_ENDPOINT__``
  because ``DatusArtifact.initDashboard`` issues live queries against
  a running ``datus --web`` server).

The CDN URLs both kinds use point at ``@datus/web-artifact-render`` —
one UMD bundle now serves both viewers (Datus-saas#412).
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# ``index.css`` / ``index.umd.js`` are the package's vite outputs;
# unchanged across the ``web-report`` → ``web-artifact-render`` rename.
_DIST_CSS_NAME = "index.css"
_DIST_JS_NAME = "index.umd.js"

# Pinned package version on unpkg. Keep in lockstep with
# ``packages/web-artifact-render/package.json`` on the SaaS side.
_CDN_BUNDLE_VERSION = "~0.1.0"

#: CDN URL for the shared UMD CSS bundle.
CDN_BUNDLE_CSS = f"https://unpkg.com/@datus/web-artifact-render@{_CDN_BUNDLE_VERSION}/dist/{_DIST_CSS_NAME}"

#: CDN URL for the shared UMD JS bundle (exposes ``window.DatusArtifact``).
CDN_BUNDLE_JS = f"https://unpkg.com/@datus/web-artifact-render@{_CDN_BUNDLE_VERSION}/dist/{_DIST_JS_NAME}"

# Subdirectory under ``<artifact_dir>/`` where local dist assets are copied
# in offline mode.
_ASSETS_SUBDIR = "_assets"

# Best-effort title extraction from a JSDoc-style annotation at the top of
# render/app.jsx, e.g. ``/** @datus-title 2026 Q1 NA Sales Report */``. The
# runtime falls back to the artifact slug when this isn't present.
_TITLE_ANNOTATION_RE = re.compile(r"@datus-title\s+([^\n*/]+?)(?:\s*\*/|\s*\n|$)")


@dataclass(frozen=True)
class ArtifactHtmlSpec:
    """Per-artifact-kind config plugged into :func:`render_artifact_html`.

    Frozen + dataclass(eq) so callers can stash module-level singletons
    (``_REPORT_SPEC`` / ``_DASHBOARD_SPEC``) and pass them through
    without each call constructing a fresh object.
    """

    #: ``"report"`` / ``"dashboard"`` — purely for log messages.
    kind: str

    #: Top-level on-disk directory (``"reports"`` / ``"dashboards"``)
    #: that the slug is resolved under.
    root_dir_name: str

    #: Accepted shape for the slug. Restricting up front prevents path
    #: traversal (``..``) or absolute-path components from escaping
    #: ``<project_root>/<root_dir_name>/``.
    slug_regex: "re.Pattern[str]"

    #: Per-prefix walker allowlist. Each entry maps a slug-relative
    #: top-level directory to ``(allowed_suffixes, recursive)``.
    #:
    #: Report kind:
    #:   * ``render/`` — recursive (.jsx/.js/.css/.json)
    #:   * ``queries/`` — one level (.sql / .json result pairs)
    #: Dashboard kind:
    #:   * ``render/`` — recursive (.jsx/.js/.css/.json)
    #:   * ``queries/`` — one level (.sql.j2 + .params.json template pairs)
    artifact_dirs: Mapping[str, Tuple[Tuple[str, ...], bool]]

    #: HTML template file (under ``templates/``) the payload is slotted
    #: into. Must declare the kind-specific bootstrap call.
    template_path: Path

    #: Placeholder for the JSON-encoded ``{slug,title,created_at,files}``
    #: payload (e.g. ``__DATUS_REPORT_DATA__`` / ``__DATUS_DASHBOARD_DATA__``).
    data_placeholder: str

    #: Title placeholder substituted into ``<title>`` / fallback copy.
    title_placeholder: str

    #: Asset URL placeholders (CSS + JS) — same names across kinds today,
    #: but parameterised in case a future template renames one half.
    css_url_placeholder: str
    js_url_placeholder: str

    #: Optional extra ``placeholder -> value`` map filled in at the call
    #: site. Used by the dashboard kind to inject the live-query endpoint
    #: + auth + project id; report kind passes ``{}``.
    extra_placeholders: Dict[str, str] = field(default_factory=dict)


def _extract_title(app_jsx: str, fallback: str) -> str:
    """Pull the ``@datus-title`` annotation out of the first 2 KB of
    ``app.jsx``, or fall back to the artifact slug."""
    match = _TITLE_ANNOTATION_RE.search(app_jsx[:2048])
    if match:
        title = match.group(1).strip()
        if title:
            return title
    return fallback


def _read_artifact_files(
    artifact_dir: Path,
    artifact_dirs: Mapping[str, Tuple[Tuple[str, ...], bool]],
) -> List[Dict[str, str]]:
    """Return ``[{path, content}, ...]`` sorted deterministically.

    ``path`` is slug-relative and includes the top-level directory (e.g.
    ``render/app.jsx``, ``queries/q.sql``, ``queries/q.sql.j2``). Files
    outside the per-prefix allowlist (or with disallowed suffixes) are
    silently dropped so a stray scratch file doesn't bloat the inline
    payload.

    Each candidate is resolved before reading so that a symlink under
    ``render/`` / ``queries/`` cannot exfiltrate a file from outside the
    artifact directory into the inline HTML payload — the LLM controls
    these paths and a stray ``ln -s /etc/passwd render/foo.jsx`` would
    otherwise end up in the bundle.

    Allowlist matching is ``endswith``-based so compound suffixes like
    ``.sql.j2`` / ``.params.json`` (dashboard kind) work alongside plain
    suffixes like ``.sql`` / ``.json`` (report kind).
    """
    artifact_dir_resolved = artifact_dir.resolve()
    entries: List[Dict[str, str]] = []
    for sub, (allowed_suffixes, recursive) in artifact_dirs.items():
        root = artifact_dir / sub
        if not root.is_dir():
            continue
        iterator = root.rglob("*") if recursive else root.iterdir()
        for path in iterator:
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            if not any(name_lower.endswith(suffix) for suffix in allowed_suffixes):
                continue
            resolved = path.resolve()
            try:
                rel = resolved.relative_to(artifact_dir_resolved).as_posix()
            except ValueError:
                # Symlink (or other indirection) escapes the artifact root —
                # drop silently rather than leak content from outside.
                continue
            entries.append({"path": rel, "content": resolved.read_text(encoding="utf-8")})
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _escape_for_script_tag(payload: str) -> str:
    """Escape ``</`` so the JSON survives being embedded in a <script> block."""
    return payload.replace("</", "<\\/")


def _resolve_dist(dist: Optional[Path]) -> Optional[Path]:
    """Validate ``dist`` and return the resolved directory or ``None``.

    Returns ``None`` (caller falls back to CDN) when the path is unset,
    not a directory, or missing one of the required asset files.
    """
    if not dist:
        return None

    resolved = Path(dist).expanduser().resolve()
    if not resolved.is_dir():
        logger.warning("artifact dist %s is not a directory; falling back to CDN.", resolved)
        return None

    missing = [name for name in (_DIST_CSS_NAME, _DIST_JS_NAME) if not (resolved / name).is_file()]
    if missing:
        logger.warning(
            "artifact dist %s is missing required assets %s; falling back to CDN.",
            resolved,
            missing,
        )
        return None
    return resolved


def _copy_offline_assets(artifact_dir: Path, dist_dir: Path) -> Tuple[str, str]:
    """Copy css + umd next to the artifact and return the relative URLs to use.

    The copy is idempotent — repeated renders against the same directory
    overwrite the previous payload, which is what we want when the user
    updates their local build.
    """
    assets_dir = artifact_dir / _ASSETS_SUBDIR
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dist_dir / _DIST_CSS_NAME, assets_dir / _DIST_CSS_NAME)
    shutil.copy2(dist_dir / _DIST_JS_NAME, assets_dir / _DIST_JS_NAME)
    return (
        f"{_ASSETS_SUBDIR}/{_DIST_CSS_NAME}",
        f"{_ASSETS_SUBDIR}/{_DIST_JS_NAME}",
    )


def render_artifact_html(
    *,
    spec: ArtifactHtmlSpec,
    project_root: Path,
    slug: str,
    dist: Optional[Path] = None,
) -> Path:
    """Compile ``<artifact_dir>/index.html`` from render/ + queries.

    Args:
        spec: per-kind config (allowlist, template path, placeholders, …).
        project_root: ``AgentConfig.project_root``; resolved to an
            absolute path before the artifact dir is composed.
        slug: target artifact slug (matches the on-disk directory name).
        dist: optional path to a local ``@datus/web-artifact-render``
            ``dist/`` directory containing ``index.css`` / ``index.umd.js``.
            When provided and valid, the two files are copied next to the
            generated HTML and the template links to them via relative
            paths (so the page works offline through ``file://``). When
            ``None`` (or the directory is missing / incomplete), the
            template links to the pinned unpkg CDN instead.

    Returns:
        Absolute path to the generated ``index.html``.

    Raises:
        ValueError: if ``slug`` fails ``spec.slug_regex``.
        FileNotFoundError: if ``render/app.jsx`` is missing.
        OSError: on read/write failures.
    """
    if not spec.slug_regex.fullmatch(slug):
        raise ValueError(f"invalid {spec.kind}_slug {slug!r}; expected to match {spec.slug_regex.pattern}")
    project_root = project_root.resolve()
    artifact_dir = project_root / spec.root_dir_name / slug
    app_jsx_path = artifact_dir / "render" / "app.jsx"
    if not app_jsx_path.is_file():
        raise FileNotFoundError(f"render/app.jsx not found under {artifact_dir}")

    app_jsx = app_jsx_path.read_text(encoding="utf-8")
    files = _read_artifact_files(artifact_dir, spec.artifact_dirs)

    dist_dir = _resolve_dist(dist)
    if dist_dir is not None:
        css_url, js_url = _copy_offline_assets(artifact_dir, dist_dir)
        logger.info("Offline mode: copied web-artifact-render assets from %s", dist_dir)
    else:
        css_url, js_url = CDN_BUNDLE_CSS, CDN_BUNDLE_JS

    template_html = spec.template_path.read_text(encoding="utf-8")
    created_at = _dt.datetime.fromtimestamp(app_jsx_path.stat().st_mtime, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    title = _extract_title(app_jsx, slug)
    payload = {
        "slug": slug,
        "title": title,
        "created_at": created_at,
        "files": files,
    }
    payload_json = _escape_for_script_tag(json.dumps(payload, ensure_ascii=False))

    rendered = (
        template_html.replace(spec.data_placeholder, payload_json)
        .replace(spec.title_placeholder, html.escape(title))
        .replace(spec.css_url_placeholder, css_url)
        .replace(spec.js_url_placeholder, js_url)
    )
    for placeholder, value in spec.extra_placeholders.items():
        rendered = rendered.replace(placeholder, value)

    out_path = artifact_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    logger.info("%s HTML written to %s", spec.kind, out_path)
    return out_path
