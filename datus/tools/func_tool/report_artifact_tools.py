# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tools for producing report artifacts (manifest.json + queries/*).

Two complementary tools live here:

* ``ReportArtifactTools.save_query`` — runs a read-only SQL through the
  existing ``DBFuncTool`` connector, infers column semantic types, and
  atomically persists ``<slug>.sql`` and ``<slug>.json`` under the
  report's ``queries/`` directory.
* ``ReportArtifactTools.save_manifest`` — validates a candidate manifest
  against :class:`ReportManifest` (cross-checking referenced ``data_ref``
  values against on-disk query files) and atomically writes
  ``manifest.json``.

``ReportFilesystemFuncTool`` wraps the standard ``FilesystemFuncTool`` to
reject writes/edits targeting paths that should only be touched via
``save_query`` / ``save_manifest``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Tool

from datus.schemas.gen_visual_report_models import (
    QUERY_SLUG_RE,
    REPORT_ID_RE,
    ColumnSemanticType,
    QueryResultFile,
    ReportManifest,
)
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+\-]\d{2}:?\d{2})?)?$")
_MAX_QUERY_BYTES = 5 * 1024 * 1024  # 5 MB hard cap per query result file


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
    """Infer a semantic column type from a sample of values.

    Heuristic:
        * all-None -> string
        * any bool -> boolean (handled before int because bool is int subclass)
        * all int -> integer
        * any float (or mixed numeric) -> number
        * all str matching ISO date/datetime -> date
        * otherwise -> string
    """
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
            # datetime / date objects from DB drivers, etc.
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
    # strip leading SQL comments
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
# Filesystem wrapper                                                          #
# --------------------------------------------------------------------------- #


class ReportFilesystemFuncTool(FilesystemFuncTool):
    """Filesystem tool with deny rules for report artifact paths.

    Two patterns are write-protected (read/glob/grep still work):

    * ``reports/<id>/manifest.json``
    * ``reports/<id>/queries/<anything>``

    Any attempt to ``write_file`` or ``edit_file`` against these paths returns
    a clear error pointing the LLM at ``save_manifest`` / ``save_query``.
    """

    _DENY_RE = re.compile(r"^reports/[^/]+/(manifest\.json|queries/.+)$")

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
        return "manifest.json" if match.group(1) == "manifest.json" else "queries/*"

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:  # type: ignore[override]
        match = self._is_report_artifact_path(path)
        if match == "manifest.json":
            return FuncToolResult(
                success=0,
                error=(
                    "manifest.json must not be written directly. Use the `save_manifest` tool, "
                    "which validates the schema before writing."
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
        if match == "manifest.json":
            return FuncToolResult(
                success=0,
                error=(
                    "manifest.json cannot be edited in place. Use `read_file` to load it, "
                    "modify the structure in your reasoning, then call `save_manifest` to overwrite."
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

    The owning node constructs one instance per execution with a fresh
    ``report_id`` and exposes both ``save_query`` and ``save_manifest`` via
    :meth:`available_tools`.

    The tools intentionally hide the disk layout: callers reference queries
    by *slug* (e.g. ``"sales_by_store"``) and never see the absolute path.
    """

    def __init__(
        self,
        *,
        agent_config,
        report_id: str,
        db_func_tool,
    ) -> None:
        if not REPORT_ID_RE.fullmatch(report_id):
            raise ValueError(f"report_id must match {REPORT_ID_RE.pattern}, got {report_id!r}")

        self.agent_config = agent_config
        self.report_id = report_id
        self._db_func_tool = db_func_tool

        project_root = Path(getattr(agent_config, "project_root", "")).resolve()
        if not project_root or str(project_root) == ".":
            raise ValueError("agent_config.project_root must be a non-empty directory")
        self.report_dir: Path = project_root / "reports" / report_id
        self.queries_dir: Path = self.report_dir / "queries"
        self.queries_dir.mkdir(parents=True, exist_ok=True)

        self._project_root = project_root

    # -- public --------------------------------------------------------------

    def available_tools(self) -> List[Tool]:
        """Return tools registered with the agent framework."""
        return [
            trans_to_function_tool(self.save_query),
            trans_to_function_tool(self.save_manifest),
        ]

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

            The ``columns`` block is the authoritative source for Vega-Lite
            encoding ``type`` decisions in subsequent ``save_manifest`` calls.
        """
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

    def save_manifest(self, manifest_json: str) -> FuncToolResult:
        """
        Validate a candidate manifest and atomically write it to disk.

        Args:
            manifest_json: The full manifest object encoded as a JSON string.
                Schema follows ``ReportManifest``. ``id`` must equal this
                run's ``report_id`` (omit it and we set it for you). Every
                chart/table ``data_ref`` must point to a query file already
                produced by ``save_query``.

        Returns:
            FuncToolResult.result is a dict like::

                {
                    "manifest_path": "reports/<id>/manifest.json",
                    "section_count": <int>,
                    "data_refs": [...],
                }
        """
        if isinstance(manifest_json, dict):
            # Allow direct dict input from internal callers and tests.
            manifest: Dict[str, Any] = manifest_json
        elif isinstance(manifest_json, str):
            try:
                manifest = json.loads(manifest_json)
            except json.JSONDecodeError as exc:
                return FuncToolResult(success=0, error=f"manifest_json is not valid JSON: {exc}")
            if not isinstance(manifest, dict):
                return FuncToolResult(success=0, error="manifest_json must decode to a JSON object")
        else:
            return FuncToolResult(success=0, error="manifest_json must be a JSON object string")

        manifest.setdefault("id", self.report_id)
        manifest.setdefault("version", "1.0")
        manifest.setdefault("created_at", _utc_now_iso())

        if manifest.get("id") != self.report_id:
            return FuncToolResult(
                success=0,
                error=(f"manifest.id must equal the current report id {self.report_id!r}; got {manifest.get('id')!r}"),
            )

        try:
            parsed = ReportManifest.model_validate(manifest)
        except Exception as exc:
            return FuncToolResult(success=0, error=f"manifest schema validation failed: {exc}")

        missing_refs: List[str] = []
        for ref in parsed.collect_data_refs():
            slug = ref.split("/", 1)[1]
            if not (self.queries_dir / f"{slug}.sql").exists() or not (self.queries_dir / f"{slug}.json").exists():
                missing_refs.append(ref)
        if missing_refs:
            return FuncToolResult(
                success=0,
                error=(
                    "These data_ref values point to queries that were not produced via save_query: "
                    + ", ".join(missing_refs)
                    + ". Run save_query for each missing query before save_manifest."
                ),
            )

        manifest_path = self.report_dir / "manifest.json"
        try:
            blob = parsed.model_dump_json(indent=2)
            _atomic_write_text(manifest_path, blob)
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Failed to write manifest: {exc}")

        return FuncToolResult(
            result={
                "manifest_path": manifest_path.relative_to(self._project_root).as_posix(),
                "section_count": len(parsed.sections),
                "data_refs": parsed.collect_data_refs(),
            }
        )
