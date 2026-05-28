"""Report artifact reader.

Walks ``reports/<slug>/`` produced by the ``gen_visual_report``
subagent and returns a flat ``files`` list plus the parsed
``manifest`` — the bundle the ``@datus/web-artifact-render`` viewer
consumes.

This service is the canonical implementation. An optional SaaS host
can layer publication-side enrichment on top and own publish, both of
which require a Postgres schema that has no agent-only counterpart.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datus.api.models.base_models import Result
from datus.api.models.report_models import ArtifactFile, ReportDetail
from datus.configuration.agent_config import AgentConfig
from datus.schemas.artifact_manifest import ArtifactManifest
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# 25 MB total bundle ceiling — guards the wire response from a runaway
# report that ships every probe SQL alongside its JSX.
_MAX_BUNDLE_BYTES: int = 25 * 1024 * 1024
_MAX_FILES: int = 200

# Per-prefix allowlist driving the flat artifact walker. Each entry maps
# a slug-relative top-level directory to ``(allowed_suffixes, recursive)``:
#
#   * ``render/``   — recursive (.jsx subdirs like charts/, shared/);
#                     also accepts .json / .md sidecars LLMs may write
#                     alongside their JSX modules
#   * ``queries/``  — one level (LLM writes flat <slug>.sql/.json pairs)
#   * ``analysis/`` — one level (intent.md, insights.json, …)
#
# Adding a new top-level dir under reports/<slug>/ is a one-line entry;
# everything else (walking, byte/file caps, sort, detail) adapts
# automatically.
_REPORT_ARTIFACT_DIRS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "render": ((".jsx", ".js", ".css", ".json", ".md"), True),
    "queries": ((".sql", ".json"), False),
    "analysis": ((".md", ".json"), False),
}

# Same shape as the SaaS-side ``visual_reports.slug`` column constraint.
REPORT_SLUG_RE = re.compile(r"^[a-z0-9_]{1,80}$")


def _resolve_report_dir(project_files_root: Path, report_slug: str) -> Optional[Path]:
    """Resolve ``<project_files_root>/reports/<slug>`` safely.

    Returns ``None`` when the slug fails the regex or when the resolved
    path escapes ``reports/`` (defence-in-depth against symlinks pointing
    outside the workspace). Callers map ``None`` to the
    ``INVALID_REPORT_SLUG`` error code.
    """
    if not REPORT_SLUG_RE.fullmatch(report_slug or ""):
        return None
    report_dir = (project_files_root / "reports" / report_slug).resolve()
    reports_root = (project_files_root / "reports").resolve()
    # ``Path.is_relative_to`` is OS-agnostic — the previous
    # ``str(...).startswith(... + "/")`` form only worked on POSIX.
    if report_dir != reports_root and not report_dir.is_relative_to(reports_root):
        return None
    return report_dir


def _iter_artifact_files(artifact_dir: Path) -> List[Path]:
    """Walk ``artifact_dir`` and return allowed files sorted by slug-relative path.

    Honours the per-prefix allowlist; files under any other directory or
    whose name doesn't end in one of the listed suffix patterns are
    silently dropped so a stray scratch file doesn't trip detail.

    Each candidate is resolved before being kept so a symlink under
    ``render/`` / ``queries/`` / ``analysis/`` cannot exfiltrate a file
    from outside the artifact directory into the inline bundle — the LLM
    controls these paths and a stray ``ln -s /etc/passwd render/foo.jsx``
    would otherwise survive the ``is_file()`` probe (which follows
    symlinks).
    """
    artifact_dir_resolved = artifact_dir.resolve()
    found: List[Path] = []
    for sub, (allowed_suffixes, recursive) in _REPORT_ARTIFACT_DIRS.items():
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
            if not resolved.is_file():
                continue
            try:
                resolved.relative_to(artifact_dir_resolved)
            except ValueError:
                # Symlink (or other indirection) escapes the artifact root —
                # drop silently rather than leak content from outside.
                continue
            found.append(path)
    found.sort(key=lambda p: p.relative_to(artifact_dir).as_posix())
    return found


class ReportService:
    """Service for ``GET /api/v1/report/detail``.

    Constructed per-project (cached by ``DatusService``); safe to share
    across requests for the same project.
    """

    def __init__(self, agent_config: Optional[AgentConfig] = None):
        # ``agent_config`` is accepted for parity with the other agent
        # services (and so the same constructor signature works whether
        # the caller has the config or not). The current contract doesn't
        # need it for ``get_detail`` — bundle assembly is purely on-disk.
        self.agent_config = agent_config

    # -- detail --------------------------------------------------------------

    async def get_detail(
        self,
        *,
        project_files_root: Path,
        report_slug: str,
    ) -> Result[ReportDetail]:
        """Load the on-disk artifact bundle for a report slug.

        Returns the slim agent-side :class:`ReportDetail`; a SaaS host
        that needs publication-side fields layers them on via its own
        subclass.
        """
        report_dir = _resolve_report_dir(project_files_root, report_slug)
        if report_dir is None:
            return Result(
                success=False,
                errorCode="INVALID_REPORT_SLUG",
                errorMessage=(
                    f"report_slug must match {REPORT_SLUG_RE.pattern} and resolve under the project's reports/"
                ),
            )

        app_jsx_path = report_dir / "render" / "app.jsx"
        manifest_path = report_dir / "manifest.json"

        exists = await asyncio.to_thread(app_jsx_path.is_file)
        if not exists:
            return Result(
                success=False,
                errorCode="REPORT_NOT_FOUND",
                errorMessage=f"report {report_slug!r} not found",
            )

        manifest_exists = await asyncio.to_thread(manifest_path.is_file)
        if not manifest_exists:
            return Result(
                success=False,
                errorCode="REPORT_NOT_FOUND",
                errorMessage=f"reports/{report_slug}/manifest.json is missing",
            )

        try:
            manifest_text = await asyncio.to_thread(manifest_path.read_text, "utf-8")
            manifest = ArtifactManifest.model_validate(json.loads(manifest_text))
        except Exception as exc:
            logger.exception("Corrupt manifest.json for %s: %s", report_slug, exc)
            return Result(
                success=False,
                errorCode="REPORT_NOT_FOUND",
                errorMessage=f"reports/{report_slug}/manifest.json is corrupt: {exc}",
            )

        try:
            file_paths: List[Path] = await asyncio.to_thread(_iter_artifact_files, report_dir)
        except OSError as exc:
            logger.exception("Failed walking reports/%s: %s", report_slug, exc)
            return Result(success=False, errorCode="REPORT_NOT_FOUND", errorMessage=str(exc))

        files: List[ArtifactFile] = []
        total_bytes = 0
        app_jsx_seen = False

        for path in file_paths:
            if len(files) >= _MAX_FILES:
                logger.warning("Truncated artifact files at %d entries for %s", _MAX_FILES, report_slug)
                break
            try:
                content = await asyncio.to_thread(path.read_text, "utf-8")
            except OSError as exc:
                logger.exception("Failed reading artifact file %s for %s: %s", path, report_slug, exc)
                return Result(success=False, errorCode="REPORT_NOT_FOUND", errorMessage=str(exc))
            total_bytes += len(content.encode("utf-8"))
            if total_bytes > _MAX_BUNDLE_BYTES:
                logger.warning(
                    "Report %s exceeded %d-byte bundle limit; truncating to %d files",
                    report_slug,
                    _MAX_BUNDLE_BYTES,
                    len(files),
                )
                break
            rel = path.relative_to(report_dir).as_posix()
            files.append(ArtifactFile(path=rel, content=content))
            if rel == "render/app.jsx":
                app_jsx_seen = True

        if not app_jsx_seen:
            # The existence probe above passed but the file didn't make it
            # into the returned bundle — almost always means a tight byte
            # cap dropped it. Fail loudly: the renderer can't bootstrap
            # without it and a partial bundle is worse than an explicit
            # error.
            return Result(
                success=False,
                errorCode="REPORT_NOT_FOUND",
                errorMessage="render/app.jsx missing from bundle",
            )

        try:
            mtime = await asyncio.to_thread(lambda: app_jsx_path.stat().st_mtime)
            created_at: Optional[str] = _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except OSError:
            created_at = None

        return Result(
            success=True,
            data=ReportDetail(
                slug=report_slug,
                name=manifest.name,
                description=manifest.description,
                manifest=manifest,
                created_at=created_at,
                files=files,
            ),
        )
