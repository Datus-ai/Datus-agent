# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tools for producing report artifacts (main.jsx + queries/*).

Two complementary tools live here:

* ``ReportArtifactTools.save_query`` — runs a read-only SQL through the
  existing ``DBFuncTool`` connector, infers column semantic types, and
  atomically persists ``<slug>.sql`` and ``<slug>.json`` under the
  report's ``queries/`` directory.
* ``ReportArtifactTools.save_main_jsx`` — validates a candidate React
  component source (cross-checking ``useQuerySql('queries/...')``
  references against on-disk query files) and atomically writes
  ``main.jsx``.

``ReportFilesystemFuncTool`` wraps the standard ``FilesystemFuncTool`` to
reject writes/edits targeting paths that should only be touched via
``save_query`` / ``save_main_jsx``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Tool

from datus.schemas.gen_visual_report_models import (
    QUERY_SLUG_RE,
    REPORT_ID_RE,
    ColumnSemanticType,
    QueryResultFile,
    extract_query_slug,
)
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+\-]\d{2}:?\d{2})?)?$")
_MAX_QUERY_BYTES = 5 * 1024 * 1024  # 5 MB hard cap per query result file
_MAX_MAIN_JSX_BYTES = 512 * 1024  # 512 KB ceiling for the React source file

_SLUG_NON_ASCII_RE = re.compile(r"[^a-z0-9]+")

# Captures the literal-string argument of useQuerySql(...). Template strings
# and dynamic expressions are intentionally skipped (the system prompt allows
# them for enumerable filter selectors that resolve at runtime).
_USE_QUERY_SQL_LITERAL_RE = re.compile(r"useQuerySql\s*\(\s*['\"]([^'\"\\\n]+)['\"]\s*\)")


def _slugify_title(title: str, max_len: int = 32) -> str:
    """Best-effort ASCII slug from an LLM-supplied report title."""
    ascii_only = title.encode("ascii", errors="ignore").decode("ascii")
    slug = _SLUG_NON_ASCII_RE.sub("_", ascii_only.lower()).strip("_")
    return slug[:max_len]


def _allocate_report_id(title: str, project_root: Path) -> str:
    """Generate ``rpt_<title-slug>_<yymmdd>_<rand6>`` not colliding on disk."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%y%m%d")
    base_slug = _slugify_title(title) or "report"
    reports_root = project_root / "reports"
    for _ in range(8):
        suffix = uuid.uuid4().hex[:6]
        candidate = f"rpt_{base_slug}_{stamp}_{suffix}"
        candidate = candidate[:83]
        if not (reports_root / candidate).exists():
            return candidate
    raise RuntimeError("Failed to allocate a unique report id after 8 attempts")


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _looks_like_react_component(jsx: str) -> bool:
    """Tiny syntactic smoke test — catch obvious "wrong file" submissions."""
    if "export default" not in jsx:
        return False
    return ("function " in jsx) or ("=> {" in jsx) or ("=>" in jsx and "(" in jsx)


# --------------------------------------------------------------------------- #
# Filesystem wrapper                                                          #
# --------------------------------------------------------------------------- #


class ReportFilesystemFuncTool(FilesystemFuncTool):
    """Filesystem tool with deny rules for report artifact paths.

    Two patterns are write-protected (read/glob/grep still work):

    * ``reports/<id>/main.jsx``
    * ``reports/<id>/queries/<anything>``

    Any attempt to ``write_file`` or ``edit_file`` against these paths returns
    a clear error pointing the LLM at ``save_main_jsx`` / ``save_query``.
    """

    _DENY_RE = re.compile(r"^reports/[^/]+/(main\.jsx|queries/.+)$")

    def _is_report_artifact_path(self, path: str) -> Optional[str]:
        """Return the matching pattern label, or None when the path is free."""
        try:
            resolved = self._classify(path)
        except Exception:  # pragma: no cover - defensive
            return None
        try:
            rel = resolved.resolved.relative_to(self._root_resolved)
        except ValueError:
            return None
        rel_str = rel.as_posix()
        match = self._DENY_RE.match(rel_str)
        if not match:
            return None
        return "main.jsx" if match.group(1) == "main.jsx" else "queries/*"

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:  # type: ignore[override]
        match = self._is_report_artifact_path(path)
        if match == "main.jsx":
            return FuncToolResult(
                success=0,
                error=(
                    "main.jsx must not be written directly. Use the `save_main_jsx` tool, "
                    "which validates the component before writing."
                ),
            )
        if match == "queries/*":
            return FuncToolResult(
                success=0,
                error=(
                    "Files under reports/<id>/queries/ must not be written directly. "
                    "Use the `save_query` tool, which runs the SQL and writes both .sql and .json."
                ),
            )
        return super().write_file(path, content, file_type)

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:  # type: ignore[override]
        match = self._is_report_artifact_path(path)
        if match == "main.jsx":
            return FuncToolResult(
                success=0,
                error=(
                    "main.jsx cannot be edited in place. Use `read_file` to load it, "
                    "produce the full replacement source in your reasoning, then call "
                    "`save_main_jsx` with the new content."
                ),
            )
        if match == "queries/*":
            return FuncToolResult(
                success=0,
                error=(
                    "Query artifact files cannot be edited in place. Re-run `save_query` with "
                    "the same name to regenerate them."
                ),
            )
        return super().edit_file(path, old_string, new_string)


# --------------------------------------------------------------------------- #
# Artifact tools                                                              #
# --------------------------------------------------------------------------- #


class ReportArtifactTools:
    """LLM-facing tools that produce the report artifact tree.

    Lifecycle:

    1. The owning node constructs one instance per execution with no
       active report id.
    2. The LLM declares intent and binds the active report by calling
       **exactly one** of ``start_new_report`` (create) or
       ``bind_existing_report`` (edit). The system prompt enumerates the
       decision criteria.
    3. ``save_query`` and ``save_main_jsx`` operate against the active
       report. They fail-fast with a clear pointer when no report is
       bound, forcing the LLM to declare intent before writing.
    """

    def __init__(
        self,
        *,
        agent_config,
        db_func_tool,
    ) -> None:
        self.agent_config = agent_config
        self._db_func_tool = db_func_tool

        project_root = Path(getattr(agent_config, "project_root", "")).resolve()
        if not project_root or str(project_root) == ".":
            raise ValueError("agent_config.project_root must be a non-empty directory")
        self._project_root = project_root

        # Lazy state — populated by start_new_report / bind_existing_report.
        self.report_id: Optional[str] = None
        self.report_dir: Optional[Path] = None
        self.queries_dir: Optional[Path] = None
        # "new" | "edit" — surfaced to callers that want to differentiate
        # "fresh artifact" from "edit in-place" in their final response.
        self.mode: Optional[str] = None

    # -- public --------------------------------------------------------------

    def available_tools(self) -> List[Tool]:
        """Return tools registered with the agent framework."""
        return [
            trans_to_function_tool(self.start_new_report),
            trans_to_function_tool(self.bind_existing_report),
            trans_to_function_tool(self.save_query),
            trans_to_function_tool(self.save_main_jsx),
        ]

    # -- intent declaration --------------------------------------------------

    def start_new_report(self, title: str = "") -> FuncToolResult:
        """
        Allocate a fresh report directory and bind subsequent saves to it.

        Call this when the user's request is to **produce a new report
        artifact**, even if they reference one or more existing reports
        for context. To learn from a reference report, read its
        ``main.jsx`` / ``queries/*`` via the filesystem read tool and then
        build the new artifact here.

        Args:
            title: short ASCII title used to seed the report id's slug
                segment. Non-ASCII characters are stripped. Optional —
                pass an empty string to fall back to ``"report"``.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "report_id": "rpt_<slug>_<yymmdd>_<rand>",
                    "report_dir": "reports/<report_id>",
                    "mode": "new",
                }
        """
        try:
            new_id = _allocate_report_id(title, self._project_root)
        except RuntimeError as exc:
            return FuncToolResult(success=0, error=str(exc))
        return self._activate(new_id, mode="new", create_dirs=True)

    def bind_existing_report(self, report_id: str) -> FuncToolResult:
        """
        Switch the active report to an existing one and bind subsequent saves there.

        Call this when the user asks to **modify / update / edit /
        append to** a specific named report. ``save_query`` will overwrite
        same-named queries; ``save_main_jsx`` will replace ``main.jsx``.
        Use the filesystem read tool to load the current ``main.jsx``
        before mutating it.

        Args:
            report_id: target report id, e.g. ``"rpt_2026q1_na_sales"``.
                Must match ``^rpt_[a-z0-9_-]{1,80}$`` and the directory
                (including ``main.jsx``) must already exist under
                ``<project_root>/reports/``.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "report_id": "<report_id>",
                    "report_dir": "reports/<report_id>",
                    "mode": "edit",
                }
        """
        if not report_id or not REPORT_ID_RE.fullmatch(report_id):
            return FuncToolResult(
                success=0,
                error=f"report_id must match {REPORT_ID_RE.pattern}; got {report_id!r}",
            )
        candidate = self._project_root / "reports" / report_id
        if not candidate.is_dir():
            return FuncToolResult(
                success=0,
                error=(
                    f"Report directory not found: reports/{report_id}. "
                    "Use start_new_report() if you intended to create a new report."
                ),
            )
        if not (candidate / "main.jsx").is_file():
            return FuncToolResult(
                success=0,
                error=(f"reports/{report_id}/main.jsx is missing — the report is incomplete. Cannot bind for editing."),
            )
        return self._activate(report_id, mode="edit", create_dirs=False)

    def _activate(self, report_id: str, *, mode: str, create_dirs: bool) -> FuncToolResult:
        report_dir = self._project_root / "reports" / report_id
        queries_dir = report_dir / "queries"
        if create_dirs:
            queries_dir.mkdir(parents=True, exist_ok=True)
        self.report_id = report_id
        self.report_dir = report_dir
        self.queries_dir = queries_dir
        self.mode = mode
        return FuncToolResult(
            result={
                "report_id": report_id,
                "report_dir": f"reports/{report_id}",
                "mode": mode,
            }
        )

    def _require_active(self, tool_name: str) -> Optional[FuncToolResult]:
        """Return a failure result when no report is bound, else ``None``."""
        if self.report_id is None or self.report_dir is None or self.queries_dir is None:
            return FuncToolResult(
                success=0,
                error=(
                    f"No active report bound. Call start_new_report(title=...) to create one, "
                    f"or bind_existing_report(report_id=...) to edit an existing one, "
                    f"before calling {tool_name}()."
                ),
            )
        return None

    def save_query(
        self,
        name: str,
        sql: str,
        description: str = "",
        datasource: str = "",
    ) -> FuncToolResult:
        """
        Run a read-only SQL, persist the SQL text and the result, return column meta.

        Args:
            name: Semantic slug for the query (e.g. "sales_by_store"). Matches
                ``^[a-z0-9_]{1,64}$``. Reused names overwrite the previous files.
            sql: SELECT / WITH / SHOW / DESCRIBE / EXPLAIN. Multi-statement
                input is rejected. Comments inside the SQL are kept.
            description: Optional one-line semantic note. Becomes the first SQL
                comment line so future LLM turns can recover the intent.
            datasource: Logical datasource name. Empty string uses the default.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "name": "sales_by_store",
                    "sql_path": "reports/<id>/queries/sales_by_store.sql",
                    "json_path": "reports/<id>/queries/sales_by_store.json",
                    "data_ref": "queries/sales_by_store",
                    "row_count": 42,
                    "columns": [{"name": "...", "type": "..."}, ...],
                }

            The ``columns`` block is the authoritative source for axis-type
            decisions in subsequent ``save_main_jsx`` calls.
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
            "executed_at": _utc_now_iso(),
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

        header_parts: List[str] = []
        if description:
            header_parts.append(f"-- {description.strip()}")
        header_parts.append(f"-- generated at {payload['executed_at']} for report {self.report_id}")
        sql_text = "\n".join(header_parts) + "\n" + sql.rstrip() + "\n"

        try:
            _atomic_write_text(sql_path, sql_text)
            _atomic_write_text(json_path, json_blob)
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Failed to persist query files: {exc}")

        rel_sql = sql_path.relative_to(self._project_root).as_posix()
        rel_json = json_path.relative_to(self._project_root).as_posix()

        return FuncToolResult(
            result={
                "name": name,
                "sql_path": rel_sql,
                "json_path": rel_json,
                "data_ref": f"queries/{name}",
                "row_count": len(rows),
                "columns": columns_meta,
                "preview_rows": rows[:3],
            }
        )

    def save_main_jsx(self, jsx_code: str) -> FuncToolResult:
        """
        Validate a candidate React component source and atomically write it to disk.

        Args:
            jsx_code: The full ``main.jsx`` source. Must:
                * include ``export default function ...`` (or arrow form),
                * stay under 512 KB,
                * reference only ``queries/<slug>`` sqlIds that already
                  exist on disk via prior ``save_query`` calls.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "main_jsx_path": "reports/<id>/main.jsx",
                    "bytes": <int>,
                    "query_refs": ["queries/foo", "queries/bar"],
                }

        Template-string sqlIds (``\\`queries/by_month_${month}\\```) are
        intentionally NOT statically checked — they resolve from a literal
        enumeration at runtime. The system prompt warns the LLM that these
        will surface as ``errorMessage`` if the enumerated slug isn't
        actually published.
        """
        not_bound = self._require_active("save_main_jsx")
        if not_bound is not None:
            return not_bound

        if not isinstance(jsx_code, str):
            return FuncToolResult(success=0, error="jsx_code must be a string")
        if not jsx_code.strip():
            return FuncToolResult(success=0, error="jsx_code must not be empty")

        size = len(jsx_code.encode("utf-8"))
        if size > _MAX_MAIN_JSX_BYTES:
            return FuncToolResult(
                success=0,
                error=(
                    f"main.jsx exceeds the {_MAX_MAIN_JSX_BYTES // 1024} KB limit "
                    f"({size // 1024} KB). Split helpers/data into smaller pieces or simplify the layout."
                ),
            )

        if not _looks_like_react_component(jsx_code):
            return FuncToolResult(
                success=0,
                error=(
                    "jsx_code does not look like a React component module. "
                    "It must include `export default function ...` (or an arrow-form default export)."
                ),
            )

        # Static check: every literal useQuerySql('queries/<slug>') must
        # resolve to a saved query file. Template strings (backticks with
        # ${...}) are tolerated because they encode enumerable filter values.
        referenced_slugs: List[str] = []
        missing: List[str] = []
        seen: set[str] = set()
        for match in _USE_QUERY_SQL_LITERAL_RE.finditer(jsx_code):
            literal = match.group(1)
            slug = extract_query_slug(literal)
            if slug is None:
                # Loud signal — the LLM passed a malformed literal. Better
                # to flag than to silently skip.
                return FuncToolResult(
                    success=0,
                    error=(
                        f"useQuerySql received an invalid literal sqlId: {literal!r}. "
                        "Use 'queries/<slug>' where <slug> matches ^[a-z0-9_]+$."
                    ),
                )
            if slug in seen:
                continue
            seen.add(slug)
            referenced_slugs.append(f"queries/{slug}")
            json_path = self.queries_dir / f"{slug}.json"
            if not json_path.is_file():
                missing.append(f"queries/{slug}")

        if missing:
            return FuncToolResult(
                success=0,
                error=(
                    "These useQuerySql references point to queries that were not produced via save_query: "
                    + ", ".join(missing)
                    + ". Run save_query for each missing query before save_main_jsx."
                ),
            )

        main_jsx_path = self.report_dir / "main.jsx"
        try:
            _atomic_write_text(main_jsx_path, jsx_code)
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Failed to write main.jsx: {exc}")

        return FuncToolResult(
            result={
                "main_jsx_path": main_jsx_path.relative_to(self._project_root).as_posix(),
                "bytes": size,
                "query_refs": referenced_slugs,
            }
        )
