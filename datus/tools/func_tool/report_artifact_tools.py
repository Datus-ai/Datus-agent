# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tools for producing report artifacts (render/*.jsx + queries/*).

Three complementary tools live here:

* ``ReportArtifactTools.save_query`` — runs a read-only SQL through the
  existing ``DBFuncTool`` connector, infers column semantic types, and
  atomically persists ``<slug>.sql`` and ``<slug>.json`` under the
  report's ``queries/`` directory.
* ``ReportArtifactTools.validate_render`` — the terminal action of this
  subagent. Walks ``reports/<id>/render/`` and verifies the entry point,
  import graph, and ``useQuerySql`` references resolve cleanly.
* ``ReportFilesystemFuncTool`` wraps the standard ``FilesystemFuncTool`` to
  reject writes/edits targeting ``queries/*`` (use ``save_query``) and
  keep ``render/*`` writes limited to JSX/JS/CSS files.

The author writes the actual report by calling ``write_file`` (and
``edit_file`` / ``delete_file``) against ``reports/<id>/render/*.jsx`` —
plain tool calls the LLM already knows. This avoids the per-tool-call
output-token truncation that bit a single-shot ``save_main_jsx`` for
real-world (~25 KB) reports.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agents import Tool

from datus.schemas.artifact_manifest import ArtifactManifest
from datus.schemas.gen_visual_report_models import (
    QUERY_SLUG_RE,
    REPORT_SLUG_RE,
    ColumnSemanticType,
    QueryResultFile,
    extract_query_slug,
)
from datus.tools.func_tool._artifact_filesystem_base import ArtifactFilesystemFuncTool
from datus.tools.func_tool._visual_artifact_helpers import (
    append_intent_section,
    coerce_uses_arg,
    upsert_manifest_after_save,
    utc_now_iso,
    write_query_brief,
)
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+\-]\d{2}:?\d{2})?)?$")
_MAX_QUERY_BYTES = 5 * 1024 * 1024  # 5 MB hard cap per query result file

# Catches every literal-string argument to useQuerySql(...). Template strings
# and dynamic expressions are intentionally skipped (the system prompt allows
# them for enumerable filter selectors that resolve at runtime).
_USE_QUERY_SQL_LITERAL_RE = re.compile(r"useQuerySql\s*\(\s*['\"]([^'\"\\\n]+)['\"]\s*\)")

# Catches both `import ... from '<path>'` and `export ... from '<path>'`
# variants. Side-effect imports `import './foo'` are matched via the
# alternate branch.
_IMPORT_PATH_RE = re.compile(
    r"""
    (?:^|\W)                                   # not inside a word
    (?:import|export)\s+                       # the keyword
    (?:                                        # binding clause is optional
        (?:[^'"\n;]+?\s+from\s+)               # ... from
        |                                      # OR
        (?=['"])                               # side-effect form (just a path)
    )?
    ['"]([^'"]+)['"]                           # the captured path
    """,
    re.VERBOSE | re.MULTILINE,
)

# `export default` at the top of a line — used to confirm the entry module
# exposes a renderable component. Matches `export default function ...`,
# `export default class ...`, `export default foo`, `export default ({...})
# => ...`, etc.
_DEFAULT_EXPORT_RE = re.compile(r"(?m)^\s*export\s+default\b")

# Bare specifiers an authored render module is allowed to import. Keep in
# lockstep with the module map inside the iframe runtime
ALLOWED_BARE_MODULES: frozenset[str] = frozenset(
    {
        "react",
        "recharts",
        "lucide-react",
        "d3-format",
        "dayjs",
        "@datus/web-artifact",
    }
)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via tempfile + rename, on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def _infer_column_type(values: List[Any]) -> ColumnSemanticType:
    """Infer a semantic column type from a sample of values."""
    saw_bool = False
    saw_int = False
    saw_float = False
    saw_other = False
    saw_str_iso_date = 0
    saw_str_total = 0

    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            saw_bool = True
        elif isinstance(value, int):
            saw_int = True
        elif isinstance(value, float):
            saw_float = True
        elif isinstance(value, str):
            saw_str_total += 1
            if _ISO_DATE_RE.match(value):
                saw_str_iso_date += 1
        else:
            type_name = type(value).__name__
            if type_name in {"datetime", "date"}:
                saw_str_total += 1
                saw_str_iso_date += 1
            else:
                saw_other = True

    if saw_bool and not (saw_int or saw_float or saw_other or saw_str_total):
        return "boolean"
    if saw_int and not (saw_bool or saw_float or saw_other or saw_str_total):
        return "integer"
    if (saw_int or saw_float) and not (saw_bool or saw_other or saw_str_total):
        return "number"
    if saw_str_total and saw_str_iso_date == saw_str_total and saw_str_total > 0:
        return "date"
    return "string"


def _normalize_value(value: Any) -> Any:
    """Coerce DB-driver scalar values into JSON-safe types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    type_name = type(value).__name__
    if type_name in {"datetime", "date"}:
        return value.isoformat()
    if type_name == "Decimal":
        try:
            as_int = int(value)
            if as_int == value:
                return as_int
        except (TypeError, ValueError):
            pass
        return float(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return str(value)


def _looks_like_select(sql: str) -> bool:
    head = sql.lstrip()
    while head.startswith("--") or head.startswith("/*"):
        if head.startswith("--"):
            nl = head.find("\n")
            head = head[nl + 1 :] if nl >= 0 else ""
        else:
            end = head.find("*/")
            head = head[end + 2 :] if end >= 0 else ""
        head = head.lstrip()
    return head[:6].upper().startswith(("SELECT", "WITH", "SHOW", "DESCRI", "EXPLAI", "PRAGMA"))


# --------------------------------------------------------------------------- #
# Render module resolution                                                    #
# --------------------------------------------------------------------------- #


def _module_key(rel_path: str) -> str:
    """Strip a .jsx/.js extension to get the canonical module key.

    `rel_path` is relative to the `render/` directory, e.g.
    ``"app.jsx"`` → ``"app"`` or ``"charts/trend.jsx"`` → ``"charts/trend"``.
    """
    return re.sub(r"\.(jsx|js)$", "", rel_path)


def _resolve_relative_import(caller_key: str, spec: str, module_keys: Set[str]) -> Optional[str]:
    """Resolve a relative import spec to a module key (or None when it doesn't).

    Mirrors the iframe runtime's resolution so static validation can refuse
    references the renderer would also reject. ``caller_key`` is the
    importing module (e.g. ``"app"`` or ``"charts/trend"``); ``spec`` is the
    relative path (``"./kpi-banner"``, ``"../shared/util"``). Returns the
    resolved module key on success.
    """
    parts: List[str] = caller_key.split("/")[:-1] if "/" in caller_key else []
    spec_segments = spec.split("/")
    # The renderer accepts both extension-less and extension-full imports.
    if spec_segments:
        spec_segments[-1] = re.sub(r"\.(jsx|js)$", "", spec_segments[-1])

    for seg in spec_segments:
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                return None  # escape attempt outside render/
            parts.pop()
        else:
            parts.append(seg)

    candidate = "/".join(parts)
    if not candidate:
        return None  # someone wrote `import './'` — meaningless

    if candidate in module_keys:
        return candidate
    indexed = f"{candidate}/index"
    if indexed in module_keys:
        return indexed
    return None


# --------------------------------------------------------------------------- #
# Filesystem wrapper                                                          #
# --------------------------------------------------------------------------- #


class ReportFilesystemFuncTool(ArtifactFilesystemFuncTool):
    """Filesystem tool that protects the report artifact tree.

    * ``reports/<id>/queries/*`` — read-only via the filesystem layer.
      Writes must go through ``save_query`` so the SQL is actually executed
      and the result JSON is well-formed.
    * ``reports/<id>/render/*`` — writable, but only ``.jsx`` / ``.js`` /
      ``.css`` files. JSON or other data formats are denied here so the
      LLM can't smuggle query payloads into the rendered tree.
    * Anything else under the project root inherits the parent's policy.
    """

    ARTIFACT_ROOT_DIR_NAME = "reports"
    SAVE_QUERY_TOOL_NAME = "save_query"
    ARTIFACT_KIND = "report"


# --------------------------------------------------------------------------- #
# Artifact tools                                                              #
# --------------------------------------------------------------------------- #


class ReportArtifactTools:
    """LLM-facing tools that produce the report artifact tree.

    Lifecycle:

    1. The owning node constructs one instance per execution with no
       active report slug.
    2. The LLM declares intent and binds the active report by calling
       **exactly one** of ``start_new_report`` (create) or
       ``bind_existing_report`` (edit). The system prompt enumerates the
       decision criteria.
    3. ``save_query`` writes query artifacts. ``write_file`` /
       ``edit_file`` / ``delete_file`` (from the filesystem tool) put
       JSX/JS/CSS under ``reports/<slug>/render/``.
    4. ``validate_render`` is the terminal action: it walks the render
       tree, checks the entry point, verifies every ``useQuerySql``
       slug exists, and confirms every relative import resolves. The
       subagent stops on its first success.
    """

    def __init__(
        self,
        *,
        agent_config,
        db_func_tool,
        user_message: str = "",
    ) -> None:
        self.agent_config = agent_config
        self._db_func_tool = db_func_tool
        # Raw user prompt that drove this node invocation — appended to
        # analysis/intent.md verbatim when the artifact is created / bound,
        # so the file becomes the authoritative log of what the user asked
        # for. Empty string is a tolerated edge case (programmatic
        # invocations / test harnesses); ``append_intent_section``
        # silently drops whitespace-only messages, renderer error
        # reports, and "continue / proceed" placeholders so the file
        # stays focused on real intent.
        self._user_message = user_message or ""

        project_root = Path(getattr(agent_config, "project_root", "")).resolve()
        if not project_root or str(project_root) == ".":
            raise ValueError("agent_config.project_root must be a non-empty directory")
        self._project_root = project_root

        # Lazy state — populated by start_new_report / bind_existing_report.
        self.report_slug: Optional[str] = None
        self.report_dir: Optional[Path] = None
        self.queries_dir: Optional[Path] = None
        self.render_dir: Optional[Path] = None
        self.analysis_dir: Optional[Path] = None
        # "new" | "edit" — surfaced to callers that want to differentiate
        # "fresh artifact" from "edit in-place" in their final response.
        self.mode: Optional[str] = None

    # -- public --------------------------------------------------------------

    def available_tools(self) -> List[Tool]:
        """Return tools registered with the agent framework."""
        return [
            trans_to_function_tool(self.start_new_report),
            trans_to_function_tool(self.bind_existing_report),
            # ``save_query.uses`` is a free-form ``Dict[str, List[str]]`` —
            # strict-mode JSON schema rejects ``additionalProperties: true``
            # which ``Dict[str, Any]`` emits. We validate the shape
            # ourselves via :func:`coerce_uses_arg` once the call lands.
            trans_to_function_tool(self.save_query, strict_mode=False),
            trans_to_function_tool(self.validate_render),
        ]

    # -- intent declaration --------------------------------------------------

    def start_new_report(self, slug: str, name: str, description: str) -> FuncToolResult:
        """
        Create a fresh report directory at ``reports/<slug>/``, write its manifest, and bind it.

        The LLM picks the ``slug`` — it doubles as the on-disk directory
        name and as the stable identifier surfaced everywhere downstream
        (SaaS list pages, IDE explorer, backend routes). **Before calling
        this tool the LLM must ``glob('reports/*')`` and confirm the
        chosen slug doesn't collide** — this tool refuses to overwrite
        an existing directory.

        Args:
            slug: Lowercase ASCII identifier matching ``^[a-z0-9_]{1,80}$``.
                Becomes the directory name (``reports/<slug>/``). Pick
                something semantic and stable (e.g.
                ``account_activity_q1_2026``); do NOT include personal
                information or timestamps unless they're load-bearing
                for disambiguation.
            name: Human-readable display name (any language is fine —
                Chinese / mixed scripts welcome). Required, max 200 chars.
            description: One-paragraph description of what the report
                argues / covers. Surfaced in list pages and IDE
                explorers next to the name. Required, max 1000 chars.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "report_slug": "<slug>",
                    "report_dir": "reports/<slug>",
                    "render_dir": "reports/<slug>/render",
                    "queries_dir": "reports/<slug>/queries",
                    "analysis_dir": "reports/<slug>/analysis",
                    "manifest_path": "reports/<slug>/manifest.json",
                    "mode": "new",
                }

            The activation also seeds ``analysis/intent.md`` with the
            user's original prompt (append-only fenced-code-block section)
            so the follow-up subagent has the raw question to anchor on.
        """
        if not slug or not REPORT_SLUG_RE.fullmatch(slug):
            return FuncToolResult(
                success=0,
                error=(
                    f"slug must match {REPORT_SLUG_RE.pattern} (lowercase letters / digits / underscores, "
                    f"1–80 chars); got {slug!r}. Pick a semantic identifier; the LLM is responsible for "
                    "uniqueness within reports/."
                ),
            )
        if not name or not name.strip():
            return FuncToolResult(success=0, error="name must be a non-empty display name (any language).")
        if not description or not description.strip():
            return FuncToolResult(
                success=0,
                error="description must be a non-empty one-paragraph description of what the report covers.",
            )
        candidate = self._project_root / "reports" / slug
        if candidate.exists():
            return FuncToolResult(
                success=0,
                error=(
                    f"reports/{slug}/ already exists. Pick a different slug — first `glob('reports/*')` "
                    "to see what's taken, or call `bind_existing_report` if you meant to edit it."
                ),
            )
        try:
            manifest = ArtifactManifest(
                slug=slug,
                name=name.strip(),
                description=description.strip(),
                kind="report",
                created_at=utc_now_iso(),
            )
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Manifest validation failed: {exc}")
        return self._activate(slug, mode="new", create_dirs=True, manifest=manifest)

    def bind_existing_report(self, report_slug: str) -> FuncToolResult:
        """
        Switch the active report to an existing one and bind subsequent saves there.

        Call this when the user asks to **modify / update / edit /
        append to** a specific named report. ``save_query`` overwrites
        same-named queries; ``write_file`` / ``edit_file`` /
        ``delete_file`` mutate ``render/`` in-place. Use ``read_file``
        + ``glob`` to inspect the existing tree before mutating it.

        When the user references the report by its display name rather
        than its slug (``"update the account activity report"``), the LLM
        should first ``glob('reports/*/manifest.json')`` and read each
        manifest's ``name`` to find the matching slug, then call this
        tool with that slug.

        Args:
            report_slug: target report slug, e.g. ``"account_activity_q1"``.
                Must match ``^[a-z0-9_]{1,80}$`` and the directory
                (including ``render/app.jsx``) must already exist under
                ``<project_root>/reports/``.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "report_slug": "<slug>",
                    "report_dir": "reports/<slug>",
                    "render_dir": "reports/<slug>/render",
                    "queries_dir": "reports/<slug>/queries",
                    "analysis_dir": "reports/<slug>/analysis",
                    "mode": "edit",
                }

            ``analysis/intent.md`` gets a new ``### [timestamp] mode: edit``
            section appended with the user's prompt so the running log of
            "what the user has asked over time" stays complete.
        """
        if not report_slug or not REPORT_SLUG_RE.fullmatch(report_slug):
            return FuncToolResult(
                success=0,
                error=f"report_slug must match {REPORT_SLUG_RE.pattern}; got {report_slug!r}",
            )
        candidate = self._project_root / "reports" / report_slug
        if not candidate.is_dir():
            return FuncToolResult(
                success=0,
                error=(
                    f"Report directory not found: reports/{report_slug}. "
                    "Use start_new_report() if you intended to create a new report."
                ),
            )
        if not (candidate / "render" / "app.jsx").is_file():
            return FuncToolResult(
                success=0,
                error=(
                    f"reports/{report_slug}/render/app.jsx is missing — the report is incomplete. "
                    "Cannot bind for editing."
                ),
            )
        return self._activate(report_slug, mode="edit", create_dirs=False)

    def _activate(
        self,
        report_slug: str,
        *,
        mode: str,
        create_dirs: bool,
        manifest: Optional[ArtifactManifest] = None,
    ) -> FuncToolResult:
        report_dir = self._project_root / "reports" / report_slug
        queries_dir = report_dir / "queries"
        render_dir = report_dir / "render"
        analysis_dir = report_dir / "analysis"
        manifest_path = report_dir / "manifest.json"
        if create_dirs:
            queries_dir.mkdir(parents=True, exist_ok=True)
            render_dir.mkdir(parents=True, exist_ok=True)
            analysis_dir.mkdir(parents=True, exist_ok=True)
        manifest_rel: Optional[str] = None
        if manifest is not None:
            try:
                _atomic_write_text(
                    manifest_path,
                    json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
                )
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Failed to write manifest.json: {exc}")
            manifest_rel = manifest_path.relative_to(self._project_root).as_posix()
        self.report_slug = report_slug
        self.report_dir = report_dir
        self.queries_dir = queries_dir
        self.render_dir = render_dir
        self.analysis_dir = analysis_dir
        self.mode = mode

        # Append the raw user prompt to analysis/intent.md before returning.
        # Failure here is non-fatal — the SQL/render pipeline is the load-
        # bearing artifact; the intent log is bonus context for the
        # follow-up subagent. We surface the warning in the tool result so
        # the LLM (and integration tests) can notice the regression.
        intent_warning = append_intent_section(
            analysis_dir,
            user_message=self._user_message,
            mode=mode,
            timestamp=utc_now_iso(),
        )

        result: Dict[str, Any] = {
            "report_slug": report_slug,
            "report_dir": f"reports/{report_slug}",
            "render_dir": f"reports/{report_slug}/render",
            "queries_dir": f"reports/{report_slug}/queries",
            "analysis_dir": f"reports/{report_slug}/analysis",
            "mode": mode,
        }
        if manifest_rel:
            result["manifest_path"] = manifest_rel
        if intent_warning:
            result["intent_warning"] = intent_warning
        return FuncToolResult(result=result)

    def _require_active(self, tool_name: str) -> Optional[FuncToolResult]:
        """Return a failure result when no report is bound, else ``None``."""
        if self.report_slug is None or self.report_dir is None or self.queries_dir is None:
            return FuncToolResult(
                success=0,
                error=(
                    f"No active report bound. Call start_new_report(slug=..., name=..., description=...) "
                    f"to create one, or bind_existing_report(report_slug=...) to edit an existing one, "
                    f"before calling {tool_name}()."
                ),
            )
        return None

    def save_query(
        self,
        name: str,
        sql: str,
        goal: str,
        hypothesis: str,
        uses: Optional[Dict[str, Any]] = None,
        caveats: str = "",
        datasource: str = "",
    ) -> FuncToolResult:
        """
        Run a read-only SQL, persist the SQL text, the result, AND the per-query brief sidecar.

        Args:
            name: Semantic slug for the query (e.g. "sales_by_store"). Matches
                ``^[a-z0-9_]{1,64}$``. Reused names overwrite the previous files
                (all three sidecars: ``.sql`` / ``.json`` / ``.brief.json``).
            sql: SELECT / WITH / SHOW / DESCRIBE / EXPLAIN. Multi-statement
                input is rejected. Comments inside the SQL are kept.
            goal: One-line research question this query answers, e.g.
                "distribution of high-risk signups across months". Becomes the
                first SQL comment line so a human reading ``.sql`` can recover
                intent. Required. Not persisted separately — the SQL header
                comment is the canonical store; the brief file holds only
                the fields a follow-up consultant would otherwise have to
                infer (hypothesis / uses / caveats).
            hypothesis: One-sentence concrete prediction the LLM expects this
                query to validate or refute (e.g. "high-risk signups cluster
                around promotional campaigns"). Required and non-empty. If
                you don't have a hypothesis, skip the query — placeholder
                hypotheses pollute the analysis layer.
            uses: Optional ``{"metrics": [{"path": [...], "name": "..."}],
                "reference_sql": [...], "ext_knowledge": [...]}``. Each
                bucket lists subject-library assets this query draws on,
                identified by their ``path`` + ``name`` pair (the same two
                fields ``list_metrics`` / ``search_metrics`` /
                ``list_subject_tree`` return). Empty / omitted is fine
                for pure ad-hoc queries. Surfaced verbatim in
                ``analysis/subject_refs.json`` for the follow-up
                subagent, deduped on ``(path, name)``. Malformed entries
                (missing ``path`` or ``name``, legacy string-id form)
                are rejected immediately.
            caveats: Before deciding this field is empty, check the SQL
                against these five gotchas a follow-up reader would otherwise
                miss: (1) JOIN type changing the denominator (LEFT vs INNER);
                (2) hardcoded value lists that won't auto-include new source
                values; (3) implicit time-window / scope filters not obvious
                from the query name; (4) NULL handling (COUNT(col) vs *, SUM
                over nullables, dimensions dropping NULL groups); (5)
                sampling, dedup, or non-standard aggregation (DISTINCT,
                weighted vs simple avg, top-K cutoffs). Write one concise
                sentence per applicable point. Truly routine aggregates with
                no implicit assumptions get an empty string — filler like
                "no caveats" is NOT acceptable.
            datasource: Logical datasource name. Empty string uses the default.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "name": "sales_by_store",
                    "sql_path": "reports/<id>/queries/sales_by_store.sql",
                    "json_path": "reports/<id>/queries/sales_by_store.json",
                    "brief_path": "reports/<id>/queries/sales_by_store.brief.json",
                    "data_ref": "queries/sales_by_store",
                    "row_count": 42,
                    "columns": [{"name": "...", "type": "..."}, ...],
                }

            The ``columns`` block is the authoritative source for axis-type
            decisions in subsequent render/*.jsx files.
        """
        not_bound = self._require_active("save_query")
        if not_bound is not None:
            return not_bound
        if not name or not QUERY_SLUG_RE.fullmatch(name):
            return FuncToolResult(
                success=0,
                error=f"name must match {QUERY_SLUG_RE.pattern}; got {name!r}",
            )
        if not sql or not sql.strip():
            return FuncToolResult(success=0, error="sql must not be empty")
        if not _looks_like_select(sql):
            return FuncToolResult(
                success=0,
                error="save_query only accepts read-only SQL (SELECT / WITH / SHOW / DESCRIBE / EXPLAIN).",
            )
        if not goal or not goal.strip():
            return FuncToolResult(
                success=0,
                error="goal must be a non-empty one-line research question (the question this query answers).",
            )
        if not hypothesis or not hypothesis.strip():
            return FuncToolResult(
                success=0,
                error=(
                    "hypothesis must be a non-empty one-sentence concrete prediction this query validates. "
                    "If you don't have a hypothesis, skip the query."
                ),
            )
        try:
            uses_obj = coerce_uses_arg(uses)
        except ValueError as exc:
            return FuncToolResult(success=0, error=f"uses argument invalid: {exc}")

        connector = None
        try:
            connector = self._db_func_tool._get_connector(datasource or None)
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to resolve datasource {datasource!r}: {exc}")

        ds_label = datasource or getattr(self._db_func_tool, "_default_datasource", "") or "default"

        try:
            execute_result = connector.execute_query(sql, result_format="list")
        except Exception as exc:
            logger.exception("save_query execute_query crashed", extra={"name": name})
            return FuncToolResult(success=0, error=f"Query execution failed: {exc}")

        if not execute_result.success:
            return FuncToolResult(
                success=0,
                error=f"Query failed: {execute_result.error}",
            )

        rows_raw = execute_result.sql_return or []
        if not isinstance(rows_raw, list):
            return FuncToolResult(
                success=0,
                error="Unexpected result format from connector; expected list of dicts",
            )
        rows: List[Dict[str, Any]] = []
        column_names: List[str] = []
        for row in rows_raw:
            if not isinstance(row, dict):
                return FuncToolResult(
                    success=0,
                    error="Unexpected row format from connector; expected dict per row",
                )
            normalized = {k: _normalize_value(v) for k, v in row.items()}
            rows.append(normalized)
            for key in row.keys():
                if key not in column_names:
                    column_names.append(key)

        columns_meta: List[Dict[str, str]] = []
        for col_name in column_names:
            sample = [row.get(col_name) for row in rows[:200]]
            columns_meta.append({"name": col_name, "type": _infer_column_type(sample)})

        if not columns_meta:
            return FuncToolResult(
                success=0,
                error="Query returned no columns. Refine the SQL so at least one column is selected.",
            )

        payload = {
            "executed_at": utc_now_iso(),
            "datasource": ds_label,
            "row_count": len(rows),
            "columns": columns_meta,
            "rows": rows,
        }

        try:
            QueryResultFile.model_validate(payload)
        except Exception as exc:
            return FuncToolResult(
                success=0,
                error=f"Query result failed schema validation: {exc}",
            )

        json_blob = json.dumps(payload, ensure_ascii=False, indent=2, default=_normalize_value)
        if len(json_blob.encode("utf-8")) > _MAX_QUERY_BYTES:
            return FuncToolResult(
                success=0,
                error=(
                    f"Query result exceeds the {_MAX_QUERY_BYTES // (1024 * 1024)} MB limit. "
                    "Aggregate or LIMIT the SQL before saving."
                ),
            )

        sql_path = self.queries_dir / f"{name}.sql"
        json_path = self.queries_dir / f"{name}.json"
        brief_path = self.queries_dir / f"{name}.brief.json"

        header_parts: List[str] = [f"-- {goal.strip()}"]
        header_parts.append(f"-- generated at {payload['executed_at']} for report {self.report_slug}")
        sql_text = "\n".join(header_parts) + "\n" + sql.rstrip() + "\n"

        try:
            _atomic_write_text(sql_path, sql_text)
            _atomic_write_text(json_path, json_blob)
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Failed to persist query files: {exc}")

        # Brief sidecar — write next, so the three-file bundle stays
        # in sync. If this fails, the SQL+JSON pair already exists on
        # disk; we surface the error so the LLM can retry without losing
        # the query result.
        brief_err = write_query_brief(
            self.queries_dir,
            name=name,
            hypothesis=hypothesis.strip(),
            uses=uses_obj,
            caveats=caveats.strip() if caveats else "",
        )
        if brief_err:
            return FuncToolResult(success=0, error=brief_err)

        # Manifest upsert (datasources union-add + updated_at bump). Soft
        # failure — log warning, expose in result, but don't fail the
        # whole tool call: the artifact's primary contract (SQL+data+
        # brief) is already on disk.
        manifest_warning = upsert_manifest_after_save(
            self.report_dir / "manifest.json",
            datasource=ds_label,
            timestamp=payload["executed_at"],
        )

        rel_sql = sql_path.relative_to(self._project_root).as_posix()
        rel_json = json_path.relative_to(self._project_root).as_posix()
        rel_brief = brief_path.relative_to(self._project_root).as_posix()

        result: Dict[str, Any] = {
            "name": name,
            "sql_path": rel_sql,
            "json_path": rel_json,
            "brief_path": rel_brief,
            "data_ref": f"queries/{name}",
            "row_count": len(rows),
            "columns": columns_meta,
            "preview_rows": rows[:3],
        }
        if manifest_warning:
            result["manifest_warning"] = manifest_warning
        return FuncToolResult(result=result)

    def validate_render(self) -> FuncToolResult:
        """
        Validate the assembled render/ tree. Terminal action of this subagent.

        Walks ``reports/<id>/render/`` and verifies:

        * ``render/app.jsx`` exists and contains an ``export default``.
        * Every ``useQuerySql('queries/<slug>')`` literal across all files
          resolves to a query whose ``.json`` is on disk.
        * Every ``import`` / ``export ... from`` path is either:
          - a bare specifier in the allowed list (``react``, ``recharts``,
            ``lucide-react``, ``d3-format``, ``dayjs``,
            ``@datus/web-artifact``), OR
          - a relative path that resolves to a file under ``render/``.
        * No file escapes ``render/`` via ``../`` import.

        Returns:
            FuncToolResult.result on success::

                {
                    "artifact_kind": "report",
                    "artifact_slug": "<id>",
                    "app_jsx_path": "reports/<id>/render/app.jsx",
                    "render_files": ["render/app.jsx", "render/kpi-banner.jsx", ...],
                    "query_refs": ["queries/foo", "queries/bar"],
                    "warnings": ["render/legacy.jsx is unreachable from app.jsx"],
                }

            ``artifact_kind`` / ``artifact_slug`` let the frontend refresh the
            live preview as soon as the validator passes, without waiting for
            the (multi-second) post-validate finalize LLM calls.

            On failure, ``success=0`` and ``error`` lists every issue found.
            Warnings (e.g. unreferenced files) are non-fatal — fix or ignore.
        """
        not_bound = self._require_active("validate_render")
        if not_bound is not None:
            return not_bound

        # Manifest must exist before render-tree validation — it's part of
        # the artifact contract that the list pages / IDE rely on. For
        # ``mode="new"`` runs ``start_new_report`` already wrote it; for
        # ``mode="edit"`` we expect it on disk from a previous create run.
        manifest_path = self.report_dir / "manifest.json"
        if not manifest_path.is_file():
            return FuncToolResult(
                success=0,
                error=(
                    f"reports/{self.report_slug}/manifest.json is missing. A report must always "
                    "have a manifest with name + description. Re-run start_new_report or "
                    "restore the manifest from a previous version."
                ),
            )
        try:
            ArtifactManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception as exc:
            return FuncToolResult(
                success=0,
                error=f"reports/{self.report_slug}/manifest.json is corrupt or off-spec: {exc}",
            )

        if not self.render_dir.is_dir():
            return FuncToolResult(
                success=0,
                error=(
                    f"render/ directory missing under reports/{self.report_slug}. "
                    "Write at least an app.jsx with write_file before calling validate_render."
                ),
            )

        # Walk the render tree.
        all_files: List[Path] = sorted(
            [p for p in self.render_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jsx", ".js"}]
        )
        if not all_files:
            return FuncToolResult(
                success=0,
                error=(
                    "render/ has no .jsx / .js files. Write at least an app.jsx with "
                    "write_file before calling validate_render."
                ),
            )

        app_jsx_path = self.render_dir / "app.jsx"
        if not app_jsx_path.is_file():
            return FuncToolResult(
                success=0,
                error="render/app.jsx is required as the entry module but is missing.",
            )

        modules: Dict[str, Dict[str, Any]] = {}
        for path in all_files:
            rel = path.relative_to(self.render_dir).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                return FuncToolResult(
                    success=0,
                    error=f"Failed to read render file {rel}: {exc}",
                )
            modules[_module_key(rel)] = {
                "rel": rel,
                "source": source,
                "imports": [],
                "query_refs": [],
            }

        module_keys: Set[str] = set(modules.keys())
        issues: List[str] = []
        query_refs: Set[str] = set()

        for key, mod in modules.items():
            source = mod["source"]

            # ---- useQuerySql literal slugs
            for match in _USE_QUERY_SQL_LITERAL_RE.finditer(source):
                literal = match.group(1)
                slug = extract_query_slug(literal)
                if slug is None:
                    issues.append(
                        f"render/{mod['rel']}: useQuerySql received an invalid literal sqlId "
                        f"{literal!r}. Use 'queries/<slug>' where <slug> matches ^[a-z0-9_]+$."
                    )
                    continue
                query_refs.add(f"queries/{slug}")
                json_path = self.queries_dir / f"{slug}.json"
                if not json_path.is_file():
                    issues.append(
                        f"render/{mod['rel']}: useQuerySql('queries/{slug}') points to a query "
                        "not produced via save_query."
                    )

            # ---- import / export … from paths
            for match in _IMPORT_PATH_RE.finditer(source):
                spec = match.group(1)
                if spec in ALLOWED_BARE_MODULES:
                    continue
                if spec.startswith("./") or spec.startswith("../"):
                    resolved = _resolve_relative_import(key, spec, module_keys)
                    if resolved is None:
                        issues.append(
                            f"render/{mod['rel']}: relative import {spec!r} does not resolve to a file under render/."
                        )
                    else:
                        mod["imports"].append(resolved)
                    continue
                # Bare specifier outside the allowed list.
                issues.append(
                    f"render/{mod['rel']}: import {spec!r} is not allowed. Only bare specifiers "
                    f"{sorted(ALLOWED_BARE_MODULES)} or relative paths under render/ are allowed."
                )

        if not _DEFAULT_EXPORT_RE.search(modules["app"]["source"]):
            issues.append(
                "render/app.jsx must include an `export default` (the renderer mounts the "
                "default export as the report's root component)."
            )

        if issues:
            return FuncToolResult(
                success=0,
                error="validate_render found "
                + ("1 issue:\n  - " if len(issues) == 1 else f"{len(issues)} issues:\n  - ")
                + "\n  - ".join(issues),
            )

        # Reachability from app.jsx via static imports — anything not visited
        # is a warning (the LLM can choose to delete_file or leave it).
        reachable: Set[str] = set()
        stack: List[str] = ["app"]
        while stack:
            k = stack.pop()
            if k in reachable:
                continue
            reachable.add(k)
            stack.extend(modules[k]["imports"])
        unreferenced = sorted(modules.keys() - reachable)
        warnings = [
            f"render/{modules[k]['rel']} is not imported by render/app.jsx (directly or transitively)"
            for k in unreferenced
        ]

        return FuncToolResult(
            result={
                "artifact_kind": "report",
                "artifact_slug": self.report_slug,
                "app_jsx_path": app_jsx_path.relative_to(self._project_root).as_posix(),
                "manifest_path": manifest_path.relative_to(self._project_root).as_posix(),
                "render_files": [f"render/{modules[k]['rel']}" for k in sorted(modules.keys())],
                "query_refs": sorted(query_refs),
                "warnings": warnings,
            }
        )
