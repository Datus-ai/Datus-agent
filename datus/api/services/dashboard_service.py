"""Dashboard artifact reader + live query executor.

Two endpoints power the dashboard viewer:

* ``GET /api/v1/dashboard/detail`` — returns a flat ``files`` list
  covering the authored ``render/`` tree + ``queries/`` Jinja2 templates
  + ``analysis/`` sidecars under ``dashboards/<slug>/``, plus the parsed
  ``templates`` metadata sidecar that outer-panel UI uses to drive filter
  affordances without re-parsing ``.params.json`` bytes.
* ``POST /api/v1/dashboard/query`` — renders a saved Jinja2 SQL template
  against the supplied filter values and executes it live through the
  project's connector. The result envelope matches the
  ``ISqlQueryResult`` shape consumed by ``RemoteQueryArtifactProvider``.

This service is the canonical implementation. An optional SaaS host
can layer publication-side enrichment on top and supply a
``published_template_loader`` so the same wire contract serves both
the agent-only ``datus --web`` path and a multi-tenant deployment.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from datus.api.models.base_models import Result
from datus.api.models.dashboard_models import (
    ArtifactFile,
    DashboardDetail,
    SqlQueryResultEnvelope,
)
from datus.configuration.agent_config import AgentConfig
from datus.schemas.artifact_manifest import ArtifactManifest
from datus.schemas.gen_visual_dashboard_models import (
    DASHBOARD_SLUG_RE,
    QUERY_SLUG_RE,
    QueryTemplateMetaFile,
    TemplateParamDecl,
)
from datus.schemas.gen_visual_report_models import QueryColumnMeta
from datus.schemas.metric_artifact_query import MetricQueryFile
from datus.tools.func_tool.dashboard_artifact_tools import render_dashboard_template
from datus.tools.func_tool.report_artifact_tools import _normalize_value
from datus.utils.artifact_files import iter_artifact_files
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_MAX_QUERY_ROWS = 100_000
# 25 MB total bundle ceiling — guards the wire response from a runaway
# dashboard that ships every probe SQL alongside its JSX.
_MAX_BUNDLE_BYTES: int = 25 * 1024 * 1024
_MAX_FILES: int = 200

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


PublishedTemplateLoader = Callable[[int], Awaitable[Result[Tuple[str | MetricQueryFile, str]]]]


def _resolve_dashboard_dir(project_files_root: Path, dashboard_slug: str) -> Optional[Path]:
    """Resolve ``<project_files_root>/dashboards/<slug>`` safely.

    Returns ``None`` when the slug fails the regex or when the resolved
    path escapes ``dashboards/`` (defence-in-depth against symlinks
    pointing outside the workspace). Callers map ``None`` to the
    ``INVALID_DASHBOARD_SLUG`` error code.
    """
    if not DASHBOARD_SLUG_RE.fullmatch(dashboard_slug or ""):
        return None
    dashboard_dir = (project_files_root / "dashboards" / dashboard_slug).resolve()
    dashboards_root = (project_files_root / "dashboards").resolve()
    # ``Path.is_relative_to`` is OS-agnostic — the previous
    # ``str(...).startswith(... + "/")`` form only worked on POSIX.
    if dashboard_dir != dashboards_root and not dashboard_dir.is_relative_to(dashboards_root):
        return None
    return dashboard_dir


def _iter_artifact_files(artifact_dir: Path) -> List[Path]:
    return iter_artifact_files(artifact_dir, "dashboard")


def _coerce_param_value(decl: TemplateParamDecl, raw: Any) -> Any:
    """Coerce a request-supplied param to its declared type.

    Raises ``ValueError`` with a user-readable message on mismatch. Arrays
    enforce per-element coercion against the base type. ``None`` is allowed
    for optional params (caller filters those out before reaching here).
    """

    def _coerce_scalar(value: Any, base: str) -> Any:
        if base == "string":
            if not isinstance(value, str):
                raise ValueError(f"expected string, got {type(value).__name__}")
            return value
        if base == "integer":
            if isinstance(value, bool):
                raise ValueError("expected integer, got boolean")
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.lstrip("-").isdigit():
                return int(value)
            raise ValueError(f"expected integer, got {type(value).__name__}")
        if base == "number":
            if isinstance(value, bool):
                raise ValueError("expected number, got boolean")
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError as exc:
                    raise ValueError(f"expected number, got {value!r}") from exc
            raise ValueError(f"expected number, got {type(value).__name__}")
        if base == "boolean":
            if isinstance(value, bool):
                return value
            raise ValueError(f"expected boolean, got {type(value).__name__}")
        if base == "date":
            if isinstance(value, str) and _ISO_DATE_RE.match(value):
                return value
            raise ValueError(f"expected ISO date string YYYY-MM-DD, got {value!r}")
        raise ValueError(f"unsupported base type {base!r}")

    base_type = decl.base_type
    if decl.is_array:
        if not isinstance(raw, list):
            raise ValueError(f"param {decl.name!r}: expected array, got {type(raw).__name__}")
        return [_coerce_scalar(item, base_type) for item in raw]
    return _coerce_scalar(raw, base_type)


def _validate_params(decls: List[TemplateParamDecl], supplied: Dict[str, Any]) -> Dict[str, Any]:
    """Return a coerced copy of ``supplied`` or raise ``ValueError`` on failure."""
    decl_by_name = {p.name: p for p in decls}
    declared_names = set(decl_by_name)
    unknown = set(supplied) - declared_names
    if unknown:
        raise ValueError(
            f"unknown params not declared in template: {sorted(unknown)}; template declares {sorted(declared_names)}"
        )
    missing_required = [p.name for p in decls if p.required and (p.name not in supplied or supplied[p.name] is None)]
    if missing_required:
        raise ValueError(f"missing required params: {missing_required}")

    coerced: Dict[str, Any] = {}
    for name, raw in supplied.items():
        decl = decl_by_name[name]
        if raw is None:
            if decl.required:
                raise ValueError(f"required param {name!r} cannot be null")
            continue
        try:
            coerced[name] = _coerce_param_value(decl, raw)
        except ValueError as exc:
            raise ValueError(f"param {name!r}: {exc}") from exc
    return coerced


async def _load_local_template_pair(
    project_files_root: Path, dashboard_slug: str, query_slug: str
) -> Result[Tuple[str | MetricQueryFile, str]]:
    """Read the on-disk ``queries/<query_slug>.{sql.j2,params.json}`` pair
    under ``dashboards/<dashboard_slug>/``.

    Returns ``(sql_template, meta_text)``.
    """
    dashboard_dir = _resolve_dashboard_dir(project_files_root, dashboard_slug)
    if dashboard_dir is None:
        return Result(
            success=False,
            errorCode="INVALID_DASHBOARD_SLUG",
            errorMessage=(
                f"dashboard_slug must match {DASHBOARD_SLUG_RE.pattern} and resolve under the project's dashboards/"
            ),
        )

    sql_path = dashboard_dir / "queries" / f"{query_slug}.sql.j2"
    meta_path = dashboard_dir / "queries" / f"{query_slug}.params.json"
    metric_path = dashboard_dir / "queries" / f"{query_slug}.metric.json"

    def _stat() -> bool:
        return (sql_path.is_file() or metric_path.is_file()) and meta_path.is_file()

    exists = await asyncio.to_thread(_stat)
    if not exists:
        return Result(
            success=False,
            errorCode="TEMPLATE_NOT_FOUND",
            errorMessage=f"queries/{query_slug}.sql.j2 + .params.json not found",
        )
    try:
        if metric_path.is_file():
            if sql_path.is_file():
                raise ValueError("query has both SQL and metric sources")
            for path in (metric_path, meta_path):
                if not path.resolve().is_relative_to(dashboard_dir):
                    raise ValueError("metric query sidecar escapes the dashboard directory")
            sql_template = MetricQueryFile.model_validate_json(await asyncio.to_thread(metric_path.read_text, "utf-8"))
        else:
            sql_template = await asyncio.to_thread(sql_path.read_text, "utf-8")
        meta_text = await asyncio.to_thread(meta_path.read_text, "utf-8")
    except (OSError, ValueError) as exc:
        logger.exception("Failed reading template files for %s/%s: %s", dashboard_slug, query_slug, exc)
        return Result(success=False, errorCode="TEMPLATE_NOT_FOUND", errorMessage=str(exc))
    return Result(success=True, data=(sql_template, meta_text))


class DashboardService:
    """Service for ``GET /api/v1/dashboard/detail`` + ``POST /api/v1/dashboard/query``.

    Constructed per-project (cached by ``DatusService``); safe to share
    across requests for the same project.
    """

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    # -- detail --------------------------------------------------------------

    async def get_detail(
        self,
        *,
        project_files_root: Path,
        dashboard_slug: str,
    ) -> Result[DashboardDetail]:
        """Load the on-disk artifact bundle for a dashboard slug.

        Returns the slim agent-side :class:`DashboardDetail`; a SaaS
        host that needs publication-side fields layers them on via its
        own subclass.
        """
        dashboard_dir = _resolve_dashboard_dir(project_files_root, dashboard_slug)
        if dashboard_dir is None:
            return Result(
                success=False,
                errorCode="INVALID_DASHBOARD_SLUG",
                errorMessage=(
                    f"dashboard_slug must match {DASHBOARD_SLUG_RE.pattern} and resolve under the project's dashboards/"
                ),
            )

        render_dir = dashboard_dir / "render"
        app_jsx_path = render_dir / "app.jsx"
        manifest_path = dashboard_dir / "manifest.json"

        exists = await asyncio.to_thread(app_jsx_path.is_file)
        if not exists:
            return Result(
                success=False,
                errorCode="DASHBOARD_NOT_FOUND",
                errorMessage=f"dashboard {dashboard_slug!r} not found",
            )

        manifest_exists = await asyncio.to_thread(manifest_path.is_file)
        if not manifest_exists:
            return Result(
                success=False,
                errorCode="DASHBOARD_NOT_FOUND",
                errorMessage=f"dashboards/{dashboard_slug}/manifest.json is missing",
            )

        try:
            manifest_text = await asyncio.to_thread(manifest_path.read_text, "utf-8")
            manifest = ArtifactManifest.model_validate(json.loads(manifest_text))
        except Exception as exc:
            logger.exception("Corrupt manifest.json for %s: %s", dashboard_slug, exc)
            return Result(
                success=False,
                errorCode="DASHBOARD_NOT_FOUND",
                errorMessage=f"dashboards/{dashboard_slug}/manifest.json is corrupt: {exc}",
            )

        try:
            file_paths: List[Path] = await asyncio.to_thread(_iter_artifact_files, dashboard_dir)
        except OSError as exc:
            logger.exception("Failed walking dashboards/%s: %s", dashboard_slug, exc)
            return Result(success=False, errorCode="DASHBOARD_NOT_FOUND", errorMessage=str(exc))

        files: List[ArtifactFile] = []
        total_bytes = 0
        app_jsx_seen = False

        for path in file_paths:
            if len(files) >= _MAX_FILES:
                logger.warning("Truncated artifact files at %d entries for %s", _MAX_FILES, dashboard_slug)
                break
            try:
                content = await asyncio.to_thread(path.read_text, "utf-8")
            except OSError as exc:
                logger.exception("Failed reading artifact file %s for %s: %s", path, dashboard_slug, exc)
                return Result(success=False, errorCode="DASHBOARD_NOT_FOUND", errorMessage=str(exc))
            total_bytes += len(content.encode("utf-8"))
            if total_bytes > _MAX_BUNDLE_BYTES:
                logger.warning(
                    "Dashboard %s exceeded %d-byte bundle limit; truncating to %d files",
                    dashboard_slug,
                    _MAX_BUNDLE_BYTES,
                    len(files),
                )
                break
            rel = path.relative_to(dashboard_dir).as_posix()
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
                errorCode="DASHBOARD_NOT_FOUND",
                errorMessage="render/app.jsx missing from bundle",
            )

        try:
            mtime = await asyncio.to_thread(lambda: app_jsx_path.stat().st_mtime)
            created_at: Optional[str] = _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except OSError:
            created_at = None

        templates: List[QueryTemplateMetaFile] = []
        queries_dir = dashboard_dir / "queries"
        if await asyncio.to_thread(queries_dir.is_dir):
            entries = await asyncio.to_thread(
                lambda: sorted(p for p in queries_dir.iterdir() if p.is_file() and p.name.endswith(".params.json")),
            )
            for path in entries:
                try:
                    content = await asyncio.to_thread(path.read_text, "utf-8")
                    meta = QueryTemplateMetaFile.model_validate(json.loads(content))
                except Exception as exc:
                    logger.warning("Skipping malformed template meta %s: %s", path, exc)
                    continue
                templates.append(meta)

        return Result(
            success=True,
            data=DashboardDetail(
                slug=dashboard_slug,
                name=manifest.name,
                description=manifest.description,
                manifest=manifest,
                created_at=created_at,
                files=files,
                templates=templates,
            ),
        )

    # -- run_query -----------------------------------------------------------

    async def _run_metric_query(self, query, meta, params, project_root, policy_context, published=False):
        from datus.tools import func_tool as func_tool_mod
        from datus.tools.func_tool.metric_artifact_tools import (
            execute_metric_artifact_query,
            execute_published_metric_query,
        )
        from datus.tools.func_tool.semantic_tools import SemanticTools

        if (
            query.name != meta.slug
            or query.datasource != meta.datasource
            or query.parameter_names() != {p.name for p in meta.params}
        ):
            return Result(
                success=False,
                errorCode="TEMPLATE_CORRUPT",
                errorMessage="metric recipe does not match its parameter metadata",
            )
        try:

            def execute():
                db_tool = func_tool_mod.DBFuncTool(
                    agent_config=self.agent_config, sub_agent_name="gen_visual_dashboard"
                )
                semantic_tools = SemanticTools(
                    agent_config=self.agent_config,
                    sub_agent_name="gen_visual_dashboard",
                    runtime_db_context_provider=lambda: {"datasource": query.datasource},
                )
                if published:
                    return execute_published_metric_query(
                        query,
                        runtime_config=semantic_tools.runtime.config,
                        db_tool=db_tool,
                        params=params,
                        policy_context=policy_context,
                    )
                return execute_metric_artifact_query(
                    query,
                    semantic_tools=semantic_tools,
                    db_tool=db_tool,
                    project_root=Path(self.agent_config.project_root),
                    params=params,
                    policy_context=policy_context,
                    saved=query,
                )

            payload, _ = await asyncio.to_thread(execute)
            return Result(success=True, data=SqlQueryResultEnvelope(**payload.model_dump()))
        except Exception as exc:
            if getattr(exc, "code", None) in {ErrorCode.POLICY_DENIED, ErrorCode.POLICY_DENIED.code}:
                code = "POLICY_DENIED"
            elif "METRIC_MODEL_SNAPSHOT_MISSING" in str(exc):
                code = "METRIC_MODEL_SNAPSHOT_MISSING"
            elif "METRIC_MODEL_CHANGED" in str(exc):
                code = "METRIC_MODEL_CHANGED"
            else:
                code = "METRIC_QUERY_FAILED"
            return Result(success=False, errorCode=code, errorMessage=str(exc))

    async def run_query(
        self,
        *,
        project_files_root: Path,
        dashboard_slug: str,
        query_slug: str,
        params: Dict[str, Any],
        policy_context: Optional[Dict[str, Any]] = None,
        published_version: Optional[int] = None,
        published_template_loader: Optional[PublishedTemplateLoader] = None,
    ) -> Result[SqlQueryResultEnvelope]:
        """Render + execute a dashboard query template.

        Two source paths for the template pair:

        * ``published_version is None`` (IDE live-edit preview, default):
          read the on-disk
          ``dashboards/<dashboard_slug>/queries/<query_slug>.{sql.j2,params.json}``
          pair under the project's files root.
        * ``published_version`` is an int (SaaS published-dashboard
          viewer): defer to ``published_template_loader`` to pull the
          template pair out of the SaaS-backend version snapshot. The
          agent-only path has no DB, so it returns
          ``INVALID_PUBLISHED_VERSION`` when no loader is wired.
        """
        if not QUERY_SLUG_RE.fullmatch(query_slug or ""):
            return Result(
                success=False,
                errorCode="INVALID_QUERY_SLUG",
                errorMessage=f"query_slug must match {QUERY_SLUG_RE.pattern}",
            )
        if not isinstance(params, dict):
            return Result(
                success=False,
                errorCode="INVALID_PARAMS",
                errorMessage="params must be a JSON object",
            )

        if published_version is not None:
            if published_template_loader is None:
                return Result(
                    success=False,
                    errorCode="INVALID_PUBLISHED_VERSION",
                    errorMessage=(
                        "published_version is not supported in this deployment; omit the field for live-edit preview."
                    ),
                )
            if published_version < 1:
                return Result(
                    success=False,
                    errorCode="INVALID_PUBLISHED_VERSION",
                    errorMessage=f"published_version must be a positive integer, got {published_version!r}",
                )
            template_result = await published_template_loader(published_version)
        else:
            template_result = await _load_local_template_pair(project_files_root, dashboard_slug, query_slug)

        if not template_result.success or template_result.data is None:
            return Result(
                success=False,
                errorCode=template_result.errorCode,
                errorMessage=template_result.errorMessage,
            )
        sql_template, meta_text = template_result.data

        try:
            meta = QueryTemplateMetaFile.model_validate(json.loads(meta_text))
        except Exception as exc:
            logger.exception("Corrupt params.json for %s/%s: %s", dashboard_slug, query_slug, exc)
            return Result(
                success=False,
                errorCode="TEMPLATE_CORRUPT",
                errorMessage=f"queries/{query_slug}.params.json is corrupt: {exc}",
            )

        try:
            coerced = _validate_params(meta.params, params)
        except ValueError as exc:
            return Result(
                success=False,
                errorCode="INVALID_PARAMS",
                errorMessage=str(exc),
            )

        if isinstance(sql_template, MetricQueryFile):
            return await self._run_metric_query(
                sql_template,
                meta,
                coerced,
                project_files_root,
                policy_context,
                published=published_version is not None,
            )

        try:
            rendered_sql = render_dashboard_template(sql_template, meta.params, coerced)
        except ValueError as exc:
            logger.exception("Render error for %s/%s: %s", dashboard_slug, query_slug, exc)
            return Result(
                success=False,
                errorCode="TEMPLATE_RENDER_ERROR",
                errorMessage=str(exc),
            )

        # Late import + module attribute lookup so unit tests can monkeypatch
        # ``DBFuncTool`` without ripping out the bound symbol.
        try:
            from datus.tools import func_tool as func_tool_mod
        except Exception as exc:  # pragma: no cover - import path is stable
            logger.exception("DBFuncTool import failed: %s", exc)
            return Result(success=False, errorCode="QUERY_EXECUTION_FAILED", errorMessage=str(exc))

        try:
            db_tool = func_tool_mod.DBFuncTool(agent_config=self.agent_config, sub_agent_name="gen_visual_dashboard")
            connector = db_tool._get_connector(meta.datasource or None)
        except Exception as exc:
            logger.exception("Failed to resolve datasource for %s/%s: %s", dashboard_slug, query_slug, exc)
            return Result(
                success=False,
                errorCode="DATASOURCE_UNAVAILABLE",
                errorMessage=f"failed to resolve datasource {meta.datasource!r}: {exc}",
            )

        try:
            # Enforced-read path (SQL policy + multi-statement rejection). View-time
            # re-execution runs with live user params on every dashboard open, so an
            # unbounded rendered statement here is the highest-blast-radius surface.
            exec_result = await asyncio.to_thread(
                db_tool.execute_read_enforced,
                rendered_sql,
                connector,
                datasource=meta.datasource or "",
                result_format="list",
                policy_context=policy_context,
            )
        except Exception as exc:
            logger.exception("Query execution crashed for %s/%s: %s", dashboard_slug, query_slug, exc)
            return Result(
                success=False,
                errorCode="QUERY_EXECUTION_FAILED",
                errorMessage=str(exc),
            )

        if not getattr(exec_result, "success", False):
            error_text = getattr(exec_result, "error", None) or "unknown error"
            # A policy refusal is not a failed query: the caller may not see
            # these rows, which is a different thing to say and a different
            # thing for a viewer to render. Passed through unwrapped — it is
            # already a sentence addressed to whoever is reading it, and
            # "query failed:" in front of it helps nobody.
            if getattr(exec_result, "error_code", None) == ErrorCode.POLICY_DENIED.code:
                return Result(success=False, errorCode="POLICY_DENIED", errorMessage=error_text)
            return Result(
                success=False,
                errorCode="QUERY_EXECUTION_FAILED",
                errorMessage=f"query failed: {error_text}",
            )

        rows_raw = getattr(exec_result, "sql_return", None) or []
        if not isinstance(rows_raw, list):
            return Result(
                success=False,
                errorCode="QUERY_EXECUTION_FAILED",
                errorMessage="unexpected result format from connector; expected list of dicts",
            )
        if len(rows_raw) > _MAX_QUERY_ROWS:
            logger.warning(
                "Dashboard query %s/%s returned %d rows; truncating to %d",
                dashboard_slug,
                query_slug,
                len(rows_raw),
                _MAX_QUERY_ROWS,
            )
            rows_raw = rows_raw[:_MAX_QUERY_ROWS]

        rows: List[Dict[str, Any]] = []
        column_names: List[str] = []
        for row in rows_raw:
            if not isinstance(row, dict):
                return Result(
                    success=False,
                    errorCode="QUERY_EXECUTION_FAILED",
                    errorMessage="unexpected row format from connector; expected dict per row",
                )
            rows.append({k: _normalize_value(v) for k, v in row.items()})
            for key in row.keys():
                if key not in column_names:
                    column_names.append(key)

        # Prefer the column types persisted at save-time (LLM trusted them);
        # for any new columns the query produced that the save-time render
        # didn't see, fall back to ``string`` rather than guess.
        saved_types = {c.name: c.type for c in meta.columns}
        columns_meta = [QueryColumnMeta(name=c, type=saved_types.get(c, "string")) for c in column_names]

        return Result(
            success=True,
            data=SqlQueryResultEnvelope(
                executed_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                datasource=meta.datasource,
                row_count=len(rows),
                columns=columns_meta,
                rows=rows,
                sql=rendered_sql,
            ),
        )
