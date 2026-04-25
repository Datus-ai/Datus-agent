# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unified per-tool one-line summary registry.

The same string is consumed by:

* SSE / API streams via ``ActionHistory.output["summary"]`` →
  ``datus.schemas.action_content_builder.build_tool_result_content`` →
  frontend ``shortDesc``.
* CLI compact rendering via ``ToolCallContent.compact_result`` in
  ``datus.cli.action_display.tool_content``.

Both call sites must produce identical wording, so the per-tool formatters
live in one place. Only the ``success`` path is per-tool; failure
summaries are produced uniformly by :func:`format_failure`.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Limits matching the previous behaviour in ``openai_compatible.py``.
SUMMARY_ERROR_MAX_CHARS = 100
SUMMARY_TEXT_MAX_CHARS = 80


# ── Generic helpers (public API) ────────────────────────────────────────


def pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def truncate_text(text: str, limit: int = SUMMARY_TEXT_MAX_CHARS) -> str:
    first_line = next((line for line in text.splitlines() if line.strip()), "").strip()
    if not first_line:
        return "Empty result"
    if len(first_line) <= limit:
        return first_line
    return first_line[:limit].rstrip() + "…"


def looks_like_failure(data: dict) -> bool:
    success = data.get("success")
    if success is False or success == 0:
        return True
    error = data.get("error")
    if isinstance(error, str) and error.strip():
        return True
    return False


def format_failure(data: dict) -> str:
    error = data.get("error")
    if not isinstance(error, str) or not error.strip():
        return "Failed"
    return f"Failed: {truncate_text(error, SUMMARY_ERROR_MAX_CHARS)}"


def is_empty_result(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
        return True
    return False


def format_list_envelope(value: dict) -> str:
    """Default rendering for a ``FuncToolListResult`` payload."""
    items_n = len(value.get("items") or [])
    total = value.get("total")
    has_more = value.get("has_more")
    base = pluralize(items_n, "item")
    if isinstance(total, int) and total != items_n:
        base = f"{base} of {total}"
    if has_more:
        base = f"{base} (+more)"
    return base


def format_generic_result(value: Any) -> str:
    """Tool-agnostic fallback when a per-tool formatter is missing or returns ``""``."""
    if isinstance(value, dict):
        if "items" in value and isinstance(value["items"], list):
            return format_list_envelope(value)
        for key in ("row_count", "affected_rows", "rows_affected"):
            if isinstance(value.get(key), int):
                return pluralize(value[key], "row")
        if isinstance(value.get("count"), int):
            return pluralize(value["count"], "item")
        if isinstance(value.get("rows"), int):
            return pluralize(value["rows"], "row")
        return "OK"
    if isinstance(value, list):
        return pluralize(len(value), "item")
    if isinstance(value, bool):
        return "OK" if value else "Failed"
    if isinstance(value, int):
        return pluralize(value, "row")
    if isinstance(value, str):
        return truncate_text(value)
    return "OK"


# ── Per-tool helpers (small utilities used by formatters below) ─────────


def _envelope_with_label(value: Any, singular: str, plural: str, *, with_preview: bool = True) -> str:
    """Render a ``FuncToolListResult`` payload with a tool-specific noun.

    When ``with_preview`` is True (default), the first few item names are
    appended (``"3 metrics: revenue, gmv, dau"``) when extractable.
    """
    if not isinstance(value, dict) or "items" not in value:
        return ""
    items = value.get("items") or []
    items_n = len(items)
    noun = singular if items_n == 1 else plural
    base = f"{items_n} {noun}"
    total = value.get("total")
    has_more = value.get("has_more")
    if isinstance(total, int) and total != items_n:
        base = f"{base} of {total}"
    if has_more:
        base = f"{base} (+more)"
    if with_preview and items_n:
        preview = _items_preview(items)
        if preview:
            base = f"{base}: {preview}"
    return base


def _list_count(value: Any, singular: str, plural: str) -> str:
    """Render a plain ``list`` payload (no envelope)."""
    if not isinstance(value, list):
        return ""
    n = len(value)
    return f"{n} {singular}" if n == 1 else f"{n} {plural}"


_ITEM_NAME_KEYS = (
    "name",
    "table_name",
    "database",
    "schema",
    "identifier",
    "title",
    "id",
    "job_name",
    "job_id",
    "dataset_name",
    "chart_name",
    "metric_name",
    "skill_name",
)


def _item_name(item: Any) -> str:
    """Best-effort extraction of a short identifier from a list item."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in _ITEM_NAME_KEYS:
            val = item.get(key)
            if val:
                return str(val)
    return ""


def _items_preview(items: list, max_show: int = 3) -> str:
    """Return ``"a, b, c"`` from the first ``max_show`` named items; ``""`` if none."""
    names: list = []
    for item in items[:max_show]:
        name = _item_name(item)
        if name:
            names.append(name)
        if len(names) >= max_show:
            break
    if not names:
        return ""
    preview = ", ".join(names)
    if len(items) > len(names):
        preview += ", ..."
    return preview


def _count_with_preview(items: list, singular: str, plural: str) -> str:
    """``N nouns: a, b, c`` when items have extractable names; otherwise just ``N nouns``."""
    n = len(items)
    header = f"{n} {singular}" if n == 1 else f"{n} {plural}"
    preview = _items_preview(items)
    return f"{header}: {preview}" if preview else header


# ── Tool-specific formatters ────────────────────────────────────────────
#
# Each formatter takes the unwrapped ``result`` field of a FuncToolResult
# (success path only) and returns a one-line summary, or ``""`` to fall
# back to the generic formatter.


# === Database tools (datus/tools/func_tool/database.py) ===


def _fmt_read_query(result: Any) -> str:
    if isinstance(result, dict):
        rows = result.get("original_rows")
        cols = result.get("column_count")
        if cols is None:
            compressed = result.get("compressed_data")
            if isinstance(compressed, str) and compressed:
                first_line = compressed.split("\n", 1)[0]
                if first_line:
                    cols = len(first_line.split(","))
        if isinstance(rows, int) and isinstance(cols, int):
            return f"{rows} × {cols} result"
        if isinstance(rows, int):
            return pluralize(rows, "row")
    if isinstance(result, list):
        return pluralize(len(result), "row")
    return ""


def _fmt_execute_write(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("row_count", "affected_rows", "rows_affected"):
            if isinstance(result.get(key), int):
                return f"wrote {pluralize(result[key], 'row')}"
    return ""


def _fmt_execute_ddl(result: Any) -> str:
    if isinstance(result, dict) and result.get("message"):
        return "DDL OK"
    return ""


def _fmt_describe_table(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    columns = result.get("columns") or result.get("schema")
    if isinstance(columns, list):
        return pluralize(len(columns), "column")
    return ""


def _fmt_get_table_ddl(result: Any) -> str:
    if isinstance(result, dict) and result.get("definition"):
        identifier = result.get("identifier") or result.get("table_name") or "table"
        return f"DDL of {identifier}"
    return ""


def _fmt_list_tables(result: Any) -> str:
    if isinstance(result, list):
        return _count_with_preview(result, "table", "tables")
    return ""


def _fmt_list_databases(result: Any) -> str:
    if isinstance(result, list):
        return _count_with_preview(result, "database", "databases")
    return ""


def _fmt_list_schemas(result: Any) -> str:
    if isinstance(result, list):
        return _count_with_preview(result, "schema", "schemas")
    return ""


def _fmt_search_table(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    metadata = result.get("metadata") or []
    if not isinstance(metadata, list):
        return ""
    n = len(metadata)
    sample = result.get("sample_data")
    if isinstance(sample, dict):
        sample_rows = sample.get("original_rows", 0) or 0
    elif isinstance(sample, list):
        sample_rows = len(sample)
    else:
        sample_rows = 0
    table_label = "table" if n == 1 else "tables"
    sample_label = "sample row" if sample_rows == 1 else "sample rows"
    if n == 0 and sample_rows == 0:
        return "no tables matched"
    if sample_rows:
        return f"{n} {table_label} and {sample_rows} {sample_label}"
    return f"{n} {table_label}"


def _fmt_transfer_query_result(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("row_count", "rows_transferred", "rows", "affected_rows"):
            if isinstance(result.get(key), int):
                target = result.get("target_table")
                base = f"transferred {pluralize(result[key], 'row')}"
                return f"{base} → {target}" if target else base
    return ""


# === BI tools (datus/tools/func_tool/bi_tools.py) ===


def _fmt_list_dashboards(result: Any) -> str:
    return _envelope_with_label(result, "dashboard", "dashboards")


def _fmt_get_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title") or result.get("name")
        charts = result.get("charts")
        if title and isinstance(charts, list):
            return f'dashboard "{title}" ({pluralize(len(charts), "chart")})'
        if title:
            return f'dashboard "{title}"'
        dash_id = result.get("dashboard_id") or result.get("id")
        if dash_id:
            return f"dashboard {dash_id}"
    return ""


def _fmt_list_charts(result: Any) -> str:
    return _envelope_with_label(result, "chart", "charts")


def _fmt_get_chart(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        chart_type = result.get("chart_type") or result.get("type")
        if name and chart_type:
            return f'chart "{name}" ({chart_type})'
        if name:
            return f'chart "{name}"'
        chart_id = result.get("chart_id") or result.get("id")
        if chart_id:
            return f"chart {chart_id}"
    return ""


def _fmt_get_chart_data(result: Any) -> str:
    if isinstance(result, dict):
        rows = result.get("row_count")
        if rows is None and isinstance(result.get("rows"), list):
            rows = len(result["rows"])
        cols = result.get("column_names")
        if isinstance(rows, int) and isinstance(cols, list):
            return f"{rows} rows × {len(cols)} cols"
        if isinstance(rows, int):
            return pluralize(rows, "row")
    return ""


def _fmt_list_datasets(result: Any) -> str:
    return _envelope_with_label(result, "dataset", "datasets")


def _fmt_create_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title") or result.get("name")
        dash_id = result.get("dashboard_id") or result.get("id")
        if title and dash_id:
            return f'created dashboard "{title}" ({dash_id})'
        if title:
            return f'created dashboard "{title}"'
        if dash_id:
            return f"created dashboard {dash_id}"
    return ""


def _fmt_update_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title") or result.get("name")
        dash_id = result.get("dashboard_id") or result.get("id")
        if title:
            return f'updated dashboard "{title}"'
        if dash_id:
            return f"updated dashboard {dash_id}"
    return ""


def _fmt_delete_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        deleted = result.get("deleted")
        dash_id = result.get("dashboard_id")
        if deleted is False and dash_id:
            return f"dashboard {dash_id} (not deleted)"
        if dash_id:
            return f"deleted dashboard {dash_id}"
        if deleted:
            return "deleted dashboard"
    return ""


def _fmt_create_chart(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        dash_id = result.get("dashboard_id")
        chart_id = result.get("chart_id") or result.get("id")
        if name and dash_id:
            return f'created chart "{name}" → dashboard {dash_id}'
        if name:
            return f'created chart "{name}"'
        if chart_id:
            return f"created chart {chart_id}"
    return ""


def _fmt_update_chart(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        chart_id = result.get("chart_id") or result.get("id")
        if name:
            return f'updated chart "{name}"'
        if chart_id:
            return f"updated chart {chart_id}"
    return ""


def _fmt_add_chart_to_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        chart_id = result.get("chart_id")
        dash_id = result.get("dashboard_id")
        if chart_id and dash_id:
            return f"chart {chart_id} → dashboard {dash_id}"
    return ""


def _fmt_delete_chart(result: Any) -> str:
    if isinstance(result, dict):
        chart_id = result.get("chart_id")
        if chart_id:
            return f"deleted chart {chart_id}"
        if result.get("deleted"):
            return "deleted chart"
    return ""


def _fmt_create_dataset(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name")
        dataset_id = result.get("dataset_id") or result.get("id")
        if name:
            return f'created dataset "{name}"'
        if dataset_id:
            return f"created dataset {dataset_id}"
    return ""


def _fmt_list_bi_databases(result: Any) -> str:
    return _list_count(result, "BI database", "BI databases")


def _fmt_delete_dataset(result: Any) -> str:
    if isinstance(result, dict):
        dataset_id = result.get("dataset_id")
        if dataset_id:
            return f"deleted dataset {dataset_id}"
        if result.get("deleted"):
            return "deleted dataset"
    return ""


def _fmt_write_query(result: Any) -> str:
    if isinstance(result, dict):
        rows = result.get("rows_written")
        table = result.get("table_name")
        if isinstance(rows, int) and table:
            return f"wrote {pluralize(rows, 'row')} → {table}"
        if isinstance(rows, int):
            return f"wrote {pluralize(rows, 'row')}"
    return ""


# === Semantic tools (datus/tools/func_tool/semantic_tools.py) ===


def _fmt_list_metrics(result: Any) -> str:
    return _envelope_with_label(result, "metric", "metrics")


def _fmt_get_dimensions(result: Any) -> str:
    return _envelope_with_label(result, "dimension", "dimensions")


def _fmt_query_metrics(result: Any) -> str:
    if isinstance(result, dict):
        cols = result.get("columns")
        data = result.get("data")
        rows: Optional[int] = None
        if isinstance(data, dict):
            rows = data.get("original_rows")
        if isinstance(cols, list) and isinstance(rows, int):
            return f"{rows} rows × {len(cols)} cols"
        if isinstance(rows, int):
            return pluralize(rows, "row")
        if isinstance(cols, list):
            return f"{len(cols)} columns"
    return ""


def _fmt_validate_semantic(result: Any) -> str:
    if isinstance(result, dict):
        valid = result.get("valid")
        issues = result.get("issues") or []
        if valid is True:
            return "valid"
        if valid is False:
            n = len(issues) if isinstance(issues, list) else 0
            return f"{pluralize(n, 'issue')}" if n else "invalid"
    return ""


def _fmt_attribution_analyze(result: Any) -> str:
    if isinstance(result, dict):
        ranking = result.get("dimension_ranking") or []
        selected = result.get("selected_dimensions") or []
        n_sel = len(selected) if isinstance(selected, list) else 0
        n_rank = len(ranking) if isinstance(ranking, list) else 0
        if n_sel and n_rank:
            return f"selected {n_sel} of {n_rank} dimensions"
        if n_sel:
            return f"selected {pluralize(n_sel, 'dimension')}"
    return ""


def _fmt_search_metrics(result: Any) -> str:
    return _list_count(result, "metric matched", "metrics matched")


def _fmt_search_reference_sql(result: Any) -> str:
    return _list_count(result, "reference SQL", "reference SQLs")


def _fmt_search_semantic_objects(result: Any) -> str:
    return _list_count(result, "semantic object", "semantic objects")


def _fmt_search_knowledge(result: Any) -> str:
    return _list_count(result, "knowledge entry", "knowledge entries")


# === Generation / semantic-model-gen tools ===


def _fmt_check_semantic_object_exists(result: Any) -> str:
    if isinstance(result, dict):
        kind = result.get("kind") or "object"
        if result.get("exists") is True:
            return f"{kind} exists"
        if result.get("exists") is False:
            return f"{kind} not found"
    return ""


def _fmt_check_semantic_model_exists(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("exists") is True:
            return "table exists"
        if result.get("exists") is False:
            return "table not found"
    return ""


def _fmt_end_semantic_model_generation(result: Any) -> str:
    if isinstance(result, dict):
        files = result.get("semantic_model_files")
        if isinstance(files, list):
            return f"validated {pluralize(len(files), 'semantic file')}"
    return ""


def _fmt_end_metric_generation(result: Any) -> str:
    if isinstance(result, dict):
        sync = result.get("sync") or {}
        if isinstance(sync, dict) and sync.get("success"):
            return "metric generated and synced"
        if result.get("metric_file"):
            return "metric generated"
    return ""


def _fmt_generate_sql_summary_id(result: Any) -> str:
    if isinstance(result, str) and result:
        return f"id: {truncate_text(result, 40)}"
    return ""


def _fmt_analyze_table_relationships(result: Any) -> str:
    """Reuse the inline ``summary`` field already produced by the tool."""
    if isinstance(result, dict) and isinstance(result.get("summary"), str):
        return result["summary"]
    if isinstance(result, dict) and isinstance(result.get("relationships"), list):
        return f"{pluralize(len(result['relationships']), 'relationship')}"
    return ""


def _fmt_analyze_column_usage_patterns(result: Any) -> str:
    if isinstance(result, dict) and isinstance(result.get("summary"), str):
        return result["summary"]
    if isinstance(result, dict) and isinstance(result.get("column_patterns"), dict):
        return f"{pluralize(len(result['column_patterns']), 'column')} analyzed"
    return ""


def _fmt_get_multiple_tables_ddl(result: Any) -> str:
    if isinstance(result, list):
        n = len(result)
        return f"DDL of {pluralize(n, 'table')}"
    return ""


# === Scheduler tools (datus/tools/func_tool/scheduler_tools.py) ===


def _fmt_submit_sql_job(result: Any) -> str:
    if isinstance(result, dict):
        job_name = result.get("job_name")
        job_id = result.get("job_id")
        if job_name and job_id:
            return f'submitted "{job_name}" ({job_id})'
        if job_id:
            return f"submitted {job_id}"
    return ""


def _fmt_submit_sparksql_job(result: Any) -> str:
    if isinstance(result, dict):
        job_name = result.get("job_name")
        job_id = result.get("job_id")
        if job_name and job_id:
            return f'submitted spark "{job_name}" ({job_id})'
        if job_id:
            return f"submitted spark {job_id}"
    return ""


def _fmt_trigger_scheduler_job(result: Any) -> str:
    if isinstance(result, dict):
        run_id = result.get("run_id")
        job_id = result.get("job_id")
        if job_id and run_id:
            return f"triggered {job_id} → run {run_id}"
        if job_id:
            return f"triggered {job_id}"
    return ""


def _fmt_get_scheduler_job(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("found") is False:
            return f"job {result.get('job_id', '?')} not found"
        job_name = result.get("job_name")
        status = result.get("status")
        if job_name and status:
            return f'job "{job_name}" ({status})'
        if result.get("job_id"):
            return f"job {result['job_id']}"
    return ""


def _fmt_list_scheduler_jobs(result: Any) -> str:
    return _envelope_with_label(result, "job", "jobs")


def _fmt_pause_job(result: Any) -> str:
    if isinstance(result, dict) and result.get("job_id"):
        return f"paused {result['job_id']}"
    return ""


def _fmt_resume_job(result: Any) -> str:
    if isinstance(result, dict) and result.get("job_id"):
        return f"resumed {result['job_id']}"
    return ""


def _fmt_delete_job(result: Any) -> str:
    if isinstance(result, dict) and result.get("job_id"):
        return f"deleted {result['job_id']}"
    return ""


def _fmt_update_job(result: Any) -> str:
    if isinstance(result, dict) and result.get("job_id"):
        return f"updated {result['job_id']}"
    return ""


def _fmt_list_job_runs(result: Any) -> str:
    return _envelope_with_label(result, "run", "runs")


def _fmt_get_run_log(result: Any) -> str:
    if isinstance(result, dict):
        run_id = result.get("run_id")
        log = result.get("log")
        if run_id and isinstance(log, str):
            lines = len(log.splitlines())
            return f"log of run {run_id} ({pluralize(lines, 'line')})"
        if run_id:
            return f"log of run {run_id}"
    return ""


def _fmt_list_scheduler_connections(result: Any) -> str:
    if isinstance(result, dict) and isinstance(result.get("total"), int):
        return f"{pluralize(result['total'], 'scheduler connection')}"
    return ""


# === Context search tools (datus/tools/func_tool/context_search.py) ===


def _fmt_list_subject_tree(result: Any) -> str:
    """Walk the nested taxonomy and aggregate per-leaf-kind counts."""
    if not isinstance(result, dict):
        return ""

    counts = {"metrics": 0, "reference_sql": 0, "knowledge": 0, "reference_template": 0}

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in counts:
                if isinstance(value, list):
                    counts[key] += len(value)
            elif isinstance(value, dict):
                walk(value)

    walk(result)
    parts = []
    if counts["metrics"]:
        parts.append(pluralize(counts["metrics"], "metric"))
    if counts["reference_sql"]:
        parts.append(f"{counts['reference_sql']} SQLs")
    if counts["knowledge"]:
        parts.append(f"{counts['knowledge']} knowledge")
    if counts["reference_template"]:
        parts.append(f"{counts['reference_template']} templates")
    if not parts:
        return "subject tree (empty)"
    return ", ".join(parts)


def _fmt_get_metrics(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name")
        if name:
            return f'metric "{name}"'
    if isinstance(result, list):
        return _list_count(result, "metric", "metrics")
    return ""


def _fmt_get_reference_sql(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name")
        if name:
            return f'reference SQL "{name}"'
    if isinstance(result, list):
        return _list_count(result, "reference SQL", "reference SQLs")
    return ""


def _fmt_get_knowledge(result: Any) -> str:
    return _list_count(result, "knowledge entry", "knowledge entries")


# === Reference template tools ===


def _fmt_search_reference_template(result: Any) -> str:
    return _list_count(result, "template", "templates")


def _fmt_get_reference_template(result: Any) -> str:
    if isinstance(result, dict) and result.get("name"):
        return f'template "{result["name"]}"'
    return ""


def _fmt_render_reference_template(result: Any) -> str:
    if isinstance(result, dict) and result.get("template_name"):
        return f'rendered "{result["template_name"]}"'
    return ""


def _fmt_execute_reference_template(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("template_name")
        query_result = result.get("query_result")
        rows: Optional[int] = None
        if isinstance(query_result, dict):
            rows = query_result.get("original_rows")
        if name and isinstance(rows, int):
            return f'{rows} rows from "{name}"'
        if name:
            return f'executed "{name}"'
    return ""


# === Filesystem tools ===


def _fmt_read_file(result: Any) -> str:
    """``read_file`` returns a plain content string; we only know the byte size."""
    if isinstance(result, str):
        line_count = result.count("\n") + (1 if result and not result.endswith("\n") else 0)
        return f"read {pluralize(line_count, 'line')}"
    return ""


def _fmt_write_file(result: Any) -> str:
    if isinstance(result, str):
        # The tool returns ``"File written successfully: {path}"``.
        marker = "File written successfully: "
        if result.startswith(marker):
            return f"wrote {result[len(marker) :]}"
        return truncate_text(result)
    return ""


def _fmt_edit_file(result: Any) -> str:
    if isinstance(result, str):
        marker = "File edited successfully: "
        if result.startswith(marker):
            return f"edited {result[len(marker) :]}"
        return truncate_text(result)
    return ""


def _fmt_glob(result: Any) -> str:
    if isinstance(result, dict):
        files = result.get("files")
        if isinstance(files, list):
            base = pluralize(len(files), "file")
            if result.get("truncated"):
                base = f"{base} (truncated)"
            return base
    return ""


def _fmt_grep(result: Any) -> str:
    if isinstance(result, dict):
        matches = result.get("matches")
        if isinstance(matches, list):
            base = pluralize(len(matches), "match") if len(matches) == 1 else f"{len(matches)} matches"
            if result.get("truncated"):
                base = f"{base} (truncated)"
            return base
    return ""


# === Plan / todo tools ===


def _fmt_todo_read(result: Any) -> str:
    if isinstance(result, dict):
        lists = result.get("lists") or []
        if not lists:
            return "no todos"
        first = lists[0] if isinstance(lists, list) else {}
        items = first.get("items", []) if isinstance(first, dict) else []
        total = len(items) if isinstance(items, list) else 0
        completed = sum(1 for it in items if isinstance(it, dict) and it.get("status") == "completed")
        return f"{completed}/{total} todos"
    return ""


def _fmt_todo_write(result: Any) -> str:
    if isinstance(result, dict):
        todo_list = result.get("todo_list") or {}
        items = todo_list.get("items", []) if isinstance(todo_list, dict) else []
        if isinstance(items, list):
            return f"{pluralize(len(items), 'todo')}"
    return ""


def _fmt_todo_update(result: Any) -> str:
    if isinstance(result, dict):
        item = result.get("updated_item") or {}
        if isinstance(item, dict):
            status = item.get("status")
            content = item.get("content")
            if status and content:
                return f'"{truncate_text(content, 40)}" → {status}'
            if status:
                return f"todo → {status}"
    return ""


# === Date / session tools ===


def _fmt_parse_temporal_expressions(result: Any) -> str:
    if isinstance(result, dict):
        dates = result.get("extracted_dates")
        if isinstance(dates, list):
            return f"parsed {pluralize(len(dates), 'expression')}"
    return ""


def _fmt_get_current_date(result: Any) -> str:
    if isinstance(result, dict) and result.get("current_date"):
        return str(result["current_date"])
    return ""


def _fmt_search_skill_usage(result: Any) -> str:
    if isinstance(result, dict):
        matches = result.get("matches")
        if isinstance(matches, list):
            n = len(matches)
            return f"{n} session match" if n == 1 else f"{n} session matches"
    return ""


# === Skill tools ===


def _fmt_load_skill(result: Any) -> str:
    if isinstance(result, dict):
        metadata = result.get("metadata") or {}
        name = metadata.get("name") or result.get("name")
        if name:
            return f'loaded "{name}"'
    return ""


def _fmt_validate_skill(result: Any) -> str:
    if isinstance(result, dict):
        skill_name = result.get("skill_name") or "skill"
        warnings = result.get("warnings", 0)
        if warnings:
            return f"{skill_name} valid ({pluralize(warnings, 'warning')})"
        return f"{skill_name} valid"
    return ""


# === Ask user / interaction ===


def _fmt_ask_user(result: Any) -> str:
    """``ask_user`` stores answers as a JSON-encoded string list."""
    text: Optional[str] = None
    if isinstance(result, str) and result.strip():
        text = result
    elif isinstance(result, dict):
        text = result.get("content") or result.get("answer")
    if not text:
        return ""
    try:
        decoded = json.loads(text) if isinstance(text, str) and text.lstrip().startswith("[") else None
    except (TypeError, ValueError):
        decoded = None
    if isinstance(decoded, list) and decoded:
        first = decoded[0]
        if isinstance(first, dict):
            ans = first.get("answer")
            if ans is not None:
                preview = ans if isinstance(ans, str) else str(ans)
                if len(decoded) > 1:
                    return f'"{truncate_text(preview, 40)}" (+{len(decoded) - 1} more)'
                return f'"{truncate_text(preview, 40)}"'
        return f"{len(decoded)} answers"
    if isinstance(text, str):
        return f'"{truncate_text(text, 40)}"'
    return ""


# === Sub-agent task tool ===


def _fmt_task(result: Any) -> str:
    """Sub-agent ``task`` returns a polymorphic dict — pick the best key.

    ``is not None`` is used (instead of truthy) because some keys carry an
    empty dict / empty list and that still signals "this kind of task ran".
    """
    if not isinstance(result, dict):
        return ""
    if result.get("sql_file_path"):
        return "SQL file generated"
    if result.get("sql"):
        return "SQL generated"
    semantic_models = result.get("semantic_models")
    if isinstance(semantic_models, list):
        return f"{pluralize(len(semantic_models), 'semantic model')} generated"
    if result.get("sql_summary_file"):
        return "SQL summary saved"
    if result.get("ext_knowledge_file"):
        return "knowledge saved"
    if result.get("report_result") is not None:
        return "report ready"
    skill_name = result.get("skill_name")
    if result.get("skill_path"):
        return f'skill "{skill_name}" generated' if skill_name else "skill generated"
    if result.get("dashboard_result") is not None:
        return "dashboard updated"
    if result.get("scheduler_result") is not None:
        return "scheduler updated"
    if result.get("items_saved") is not None:
        return "feedback saved"
    response = result.get("response")
    if isinstance(response, str) and response.strip():
        return f'"{truncate_text(response, 40)}"'
    return ""


# === Platform doc search ===


def _fmt_list_document_nav(result: Any) -> str:
    if isinstance(result, dict):
        platform = result.get("platform") or ""
        total = result.get("total_docs")
        if isinstance(total, int):
            base = pluralize(total, "doc")
            return f"{platform}: {base}" if platform else base
    return ""


def _fmt_get_document(result: Any) -> str:
    if isinstance(result, dict):
        platform = result.get("platform") or ""
        chunks = result.get("chunk_count")
        if chunks is None and isinstance(result.get("chunks"), list):
            chunks = len(result["chunks"])
        if isinstance(chunks, int):
            base = pluralize(chunks, "chunk")
            return f"{platform}: {base}" if platform else base
    return ""


def _fmt_search_document(result: Any) -> str:
    if isinstance(result, dict):
        n = result.get("doc_count")
        if n is None and isinstance(result.get("docs"), list):
            n = len(result["docs"])
        if isinstance(n, int):
            return pluralize(n, "doc match") if n == 1 else f"{n} doc matches"
    return ""


def _fmt_web_search_document(result: Any) -> str:
    if isinstance(result, list):
        return pluralize(len(result), "web result")
    if isinstance(result, dict):
        n = result.get("doc_count")
        if n is None and isinstance(result.get("docs"), list):
            n = len(result["docs"])
        if isinstance(n, int):
            return pluralize(n, "web result")
    return ""


# ── Registry ────────────────────────────────────────────────────────────


FormatterFn = Callable[[Any], str]


class ToolSummaryRegistry:
    """Centralized per-tool success-summary registry.

    Failure summaries are produced uniformly by :func:`format_failure`;
    per-tool formatters are invoked only when the payload indicates
    success and the unwrapped ``result`` is non-empty.
    """

    def __init__(self) -> None:
        self._formatters: Dict[str, FormatterFn] = {}

    def register(self, tool_name: str, fn: FormatterFn) -> None:
        self._formatters[tool_name] = fn

    def has(self, tool_name: str) -> bool:
        return tool_name in self._formatters

    def names(self) -> list:
        return sorted(self._formatters.keys())

    def summarize_dict(self, data: Any, tool_name: str = "") -> str:
        """Build a one-line summary from a FuncToolResult-shaped dict.

        Priority:
          1. Failure state (``success`` in (0, False) or non-empty ``error``)
          2. Tool-specific formatter from the registry
          3. Canonical FuncToolListResult envelope and common count fields
          4. Generic shape fallbacks (list/int/str/dict)
        """
        if not isinstance(data, dict):
            return format_generic_result(data) if data is not None else "Empty result"

        if looks_like_failure(data):
            return format_failure(data)

        result_value = data["result"] if "result" in data else data

        if is_empty_result(result_value):
            return "Empty result"

        formatter = self._formatters.get(tool_name)
        if formatter is not None:
            try:
                summary = formatter(result_value)
                if summary:
                    return summary
            except Exception as fmt_err:  # pragma: no cover - defensive
                logger.debug(f"Tool summary formatter for {tool_name} raised: {fmt_err}")

        return format_generic_result(result_value)

    def summarize_content(self, content: str, tool_name: str = "") -> str:
        """Build a summary from a tool result string (MCP / legacy adapters)."""
        if not content:
            return "Empty result"

        try:
            data = json.loads(content)
        except (TypeError, ValueError):
            return truncate_text(content)

        if isinstance(data, dict):
            return self.summarize_dict(data, tool_name)
        if isinstance(data, list):
            return pluralize(len(data), "item")
        if isinstance(data, bool):
            return "OK" if data else "Failed"
        if isinstance(data, int):
            return pluralize(data, "row")
        return truncate_text(str(data))


def _register_builtins(registry: ToolSummaryRegistry) -> None:
    """Register every built-in tool formatter."""
    builtins: Dict[str, FormatterFn] = {
        # Database tools
        "read_query": _fmt_read_query,
        "query": _fmt_read_query,
        "execute_write": _fmt_execute_write,
        "execute_ddl": _fmt_execute_ddl,
        "describe_table": _fmt_describe_table,
        "get_table_ddl": _fmt_get_table_ddl,
        "list_tables": _fmt_list_tables,
        "table_overview": _fmt_list_tables,
        "list_databases": _fmt_list_databases,
        "list_schemas": _fmt_list_schemas,
        "search_table": _fmt_search_table,
        "transfer_query_result": _fmt_transfer_query_result,
        # BI tools
        "list_dashboards": _fmt_list_dashboards,
        "get_dashboard": _fmt_get_dashboard,
        "list_charts": _fmt_list_charts,
        "get_chart": _fmt_get_chart,
        "get_chart_data": _fmt_get_chart_data,
        "list_datasets": _fmt_list_datasets,
        "create_dashboard": _fmt_create_dashboard,
        "update_dashboard": _fmt_update_dashboard,
        "delete_dashboard": _fmt_delete_dashboard,
        "create_chart": _fmt_create_chart,
        "update_chart": _fmt_update_chart,
        "add_chart_to_dashboard": _fmt_add_chart_to_dashboard,
        "delete_chart": _fmt_delete_chart,
        "create_dataset": _fmt_create_dataset,
        "list_bi_databases": _fmt_list_bi_databases,
        "delete_dataset": _fmt_delete_dataset,
        "write_query": _fmt_write_query,
        # Semantic tools
        "list_metrics": _fmt_list_metrics,
        "get_dimensions": _fmt_get_dimensions,
        "query_metrics": _fmt_query_metrics,
        "validate_semantic": _fmt_validate_semantic,
        "attribution_analyze": _fmt_attribution_analyze,
        "search_metrics": _fmt_search_metrics,
        "search_reference_sql": _fmt_search_reference_sql,
        "search_semantic_objects": _fmt_search_semantic_objects,
        "search_knowledge": _fmt_search_knowledge,
        # Generation / semantic-model-gen
        "check_semantic_object_exists": _fmt_check_semantic_object_exists,
        "check_semantic_model_exists": _fmt_check_semantic_model_exists,
        "end_semantic_model_generation": _fmt_end_semantic_model_generation,
        "end_metric_generation": _fmt_end_metric_generation,
        "generate_sql_summary_id": _fmt_generate_sql_summary_id,
        "analyze_table_relationships": _fmt_analyze_table_relationships,
        "analyze_column_usage_patterns": _fmt_analyze_column_usage_patterns,
        "get_multiple_tables_ddl": _fmt_get_multiple_tables_ddl,
        # Scheduler tools
        "submit_sql_job": _fmt_submit_sql_job,
        "submit_sparksql_job": _fmt_submit_sparksql_job,
        "trigger_scheduler_job": _fmt_trigger_scheduler_job,
        "get_scheduler_job": _fmt_get_scheduler_job,
        "list_scheduler_jobs": _fmt_list_scheduler_jobs,
        "pause_job": _fmt_pause_job,
        "resume_job": _fmt_resume_job,
        "delete_job": _fmt_delete_job,
        "delete_scheduler_job": _fmt_delete_job,
        "update_job": _fmt_update_job,
        "list_job_runs": _fmt_list_job_runs,
        "get_run_log": _fmt_get_run_log,
        "list_scheduler_connections": _fmt_list_scheduler_connections,
        # Context search
        "list_subject_tree": _fmt_list_subject_tree,
        "get_metrics": _fmt_get_metrics,
        "get_reference_sql": _fmt_get_reference_sql,
        "get_knowledge": _fmt_get_knowledge,
        # Reference templates
        "search_reference_template": _fmt_search_reference_template,
        "get_reference_template": _fmt_get_reference_template,
        "render_reference_template": _fmt_render_reference_template,
        "execute_reference_template": _fmt_execute_reference_template,
        # Filesystem
        "read_file": _fmt_read_file,
        "write_file": _fmt_write_file,
        "edit_file": _fmt_edit_file,
        "glob": _fmt_glob,
        "grep": _fmt_grep,
        # Plan / todo
        "todo_read": _fmt_todo_read,
        "todo_write": _fmt_todo_write,
        "todo_update": _fmt_todo_update,
        # Date / session
        "parse_temporal_expressions": _fmt_parse_temporal_expressions,
        "get_current_date": _fmt_get_current_date,
        "search_skill_usage": _fmt_search_skill_usage,
        # Skill
        "load_skill": _fmt_load_skill,
        "validate_skill": _fmt_validate_skill,
        # Ask user
        "ask_user": _fmt_ask_user,
        # Sub-agent task
        "task": _fmt_task,
        # Platform doc search
        "list_document_nav": _fmt_list_document_nav,
        "get_document": _fmt_get_document,
        "search_document": _fmt_search_document,
        "web_search_document": _fmt_web_search_document,
    }
    for name, fn in builtins.items():
        registry.register(name, fn)


TOOL_SUMMARY_REGISTRY = ToolSummaryRegistry()
_register_builtins(TOOL_SUMMARY_REGISTRY)
