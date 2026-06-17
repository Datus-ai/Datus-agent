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

All non-filesystem summaries are clipped to ``SUMMARY_TEXT_MAX_CHARS``
characters at the registry exit; filesystem tools (``read_file``,
``write_file``, ``edit_file``, ``glob``, ``grep``) bypass the clip.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


SUMMARY_TEXT_MAX_CHARS = 19
SUMMARY_ERROR_MAX_CHARS = 19

FS_TOOLS_NO_CLIP = frozenset({"read_file", "write_file", "edit_file", "glob", "grep"})


# ── Generic helpers (public API) ────────────────────────────────────────


def pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def truncate_text(text: str, limit: int = SUMMARY_TEXT_MAX_CHARS) -> str:
    first_line = next((line for line in text.splitlines() if line.strip()), "").strip()
    if not first_line:
        return "Empty result"
    if len(first_line) <= limit:
        return first_line
    return first_line[: limit - 1].rstrip() + "…"


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
    return _envelope_with_label(value, "item", "items")


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


# ── Per-tool helpers ────────────────────────────────────────────────────


def _envelope_with_label(value: Any, singular: str, plural: str) -> str:
    """Render a ``FuncToolListResult`` payload with a tool-specific noun.

    Compact format: ``"N noun"`` / ``"N/total noun"`` / ``"... noun+"``
    when ``has_more`` is set.
    """
    if not isinstance(value, dict) or "items" not in value:
        return ""
    items = value.get("items") or []
    n = len(items)
    noun = singular if n == 1 else plural
    total = value.get("total")
    if isinstance(total, int) and total != n:
        base = f"{n}/{total} {noun}"
    else:
        base = f"{n} {noun}"
    if value.get("has_more"):
        base = f"{base}+"
    return base


def _list_count(value: Any, singular: str, plural: str) -> str:
    """Render a plain ``list`` payload (no envelope)."""
    if not isinstance(value, list):
        return ""
    n = len(value)
    return f"{n} {singular}" if n == 1 else f"{n} {plural}"


def _clip_short(text: str, tool_name: str = "", limit: int = SUMMARY_TEXT_MAX_CHARS) -> str:
    """Final-stage clip applied at registry exit.

    Filesystem tools (``read_file``, ``write_file``, ``edit_file``,
    ``glob``, ``grep``) are exempt — their summaries are returned
    verbatim because users want full path / count visibility there.
    """
    if not isinstance(text, str):
        return text
    if tool_name in FS_TOOLS_NO_CLIP:
        return text
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# ── Tool-specific formatters ────────────────────────────────────────────
#
# Each formatter takes the unwrapped ``result`` field of a FuncToolResult
# (success path only) and returns a one-line summary, or ``""`` to fall
# back to the generic formatter. The registry exit applies _clip_short.


# === Database tools ===


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
            return f"{rows}×{cols} rows"
        if isinstance(rows, int):
            return pluralize(rows, "row")
    if isinstance(result, list):
        return pluralize(len(result), "row")
    return ""


def _fmt_execute_write(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("row_count", "affected_rows", "rows_affected"):
            if isinstance(result.get(key), int):
                return f"+{pluralize(result[key], 'row')}"
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
        n = len(columns)
        return f"{n} col" if n == 1 else f"{n} cols"
    return ""


def _fmt_list_tables(result: Any) -> str:
    if isinstance(result, list):
        return _list_count(result, "table", "tables")
    return ""


def _fmt_list_databases(result: Any) -> str:
    if isinstance(result, list):
        n = len(result)
        return f"{n} db" if n == 1 else f"{n} dbs"
    return ""


def _fmt_list_schemas(result: Any) -> str:
    if isinstance(result, list):
        return _list_count(result, "schema", "schemas")
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
    if n == 0 and sample_rows == 0:
        return "no matches"
    tbl_label = "tbl" if n == 1 else "tbls"
    if sample_rows:
        return f"{n} {tbl_label}, {sample_rows} rows"
    return f"{n} {tbl_label}"


def _fmt_transfer_query_result(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("row_count", "rows_transferred", "rows", "affected_rows"):
            if isinstance(result.get(key), int):
                rows = result[key]
                target = result.get("target_table")
                if target:
                    return f"moved {rows}→{target}"
                return f"moved {pluralize(rows, 'row')}"
    return ""


# === BI tools ===


def _fmt_list_dashboards(result: Any) -> str:
    return _envelope_with_label(result, "dashboard", "dashboards")


def _fmt_get_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title") or result.get("name")
        charts = result.get("charts")
        if title and isinstance(charts, list):
            return f"dash: {title} ({len(charts)})"
        if title:
            return f"dash: {title}"
        dash_id = result.get("dashboard_id") or result.get("id")
        if dash_id:
            return f"dash {dash_id}"
    return ""


def _fmt_list_charts(result: Any) -> str:
    return _envelope_with_label(result, "chart", "charts")


def _fmt_get_chart(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        if name:
            return f"chart: {name}"
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
            return f"{rows}r × {len(cols)}c"
        if isinstance(rows, int):
            return pluralize(rows, "row")
    return ""


def _fmt_list_datasets(result: Any) -> str:
    return _envelope_with_label(result, "dataset", "datasets")


def _fmt_create_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title") or result.get("name")
        dash_id = result.get("dashboard_id") or result.get("id")
        if title:
            return f"created: {title}"
        if dash_id:
            return f"created: {dash_id}"
    return ""


def _fmt_update_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title") or result.get("name")
        dash_id = result.get("dashboard_id") or result.get("id")
        if title:
            return f"updated: {title}"
        if dash_id:
            return f"updated: {dash_id}"
    return ""


def _fmt_delete_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title") or result.get("name")
        deleted = result.get("deleted")
        dash_id = result.get("dashboard_id") or result.get("id")
        if deleted is False and dash_id:
            return f"not deleted: {dash_id}"
        if title:
            return f"deleted: {title}"
        if dash_id:
            return f"deleted: {dash_id}"
        if deleted:
            return "deleted dashboard"
    return ""


def _fmt_create_chart(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        chart_id = result.get("chart_id") or result.get("id")
        if name:
            return f"created: {name}"
        if chart_id:
            return f"created: {chart_id}"
    return ""


def _fmt_update_chart(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        chart_id = result.get("chart_id") or result.get("id")
        if name:
            return f"updated: {name}"
        if chart_id:
            return f"updated: {chart_id}"
    return ""


def _fmt_add_chart_to_dashboard(result: Any) -> str:
    if isinstance(result, dict):
        chart_id = result.get("chart_id")
        dash_id = result.get("dashboard_id")
        if chart_id and dash_id:
            return f"chart {chart_id}→dash {dash_id}"
    return ""


def _fmt_delete_chart(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        chart_id = result.get("chart_id") or result.get("id")
        if name:
            return f"deleted: {name}"
        if chart_id:
            return f"deleted: {chart_id}"
        if result.get("deleted"):
            return "deleted chart"
    return ""


def _fmt_create_dataset(result: Any) -> str:
    if isinstance(result, dict):
        name = result.get("name")
        dataset_id = result.get("dataset_id") or result.get("id")
        if name:
            return f"created: {name}"
        if dataset_id:
            return f"created: {dataset_id}"
    return ""


def _fmt_list_bi_databases(result: Any) -> str:
    if isinstance(result, list):
        n = len(result)
        return f"{n} BI db" if n == 1 else f"{n} BI dbs"
    return ""


def _fmt_delete_dataset(result: Any) -> str:
    if isinstance(result, dict):
        dataset_id = result.get("dataset_id") or result.get("id")
        if dataset_id:
            return f"deleted: {dataset_id}"
        if result.get("deleted"):
            return "deleted dataset"
    return ""


def _fmt_write_query(result: Any) -> str:
    if isinstance(result, dict):
        rows = result.get("rows_written")
        table = result.get("table_name")
        if isinstance(rows, int) and table:
            return f"+{rows}→{table}"
        if isinstance(rows, int):
            return f"+{pluralize(rows, 'row')}"
    return ""


# === Semantic tools ===


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
            return f"{rows}r × {len(cols)}c"
        if isinstance(rows, int):
            return pluralize(rows, "row")
        if isinstance(cols, list):
            n = len(cols)
            return f"{n} col" if n == 1 else f"{n} cols"
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
            return f"sel {n_sel}/{n_rank} dims"
        if n_sel:
            return f"sel {n_sel} dim" if n_sel == 1 else f"sel {n_sel} dims"
    return ""


def _fmt_search_metrics(result: Any) -> str:
    if isinstance(result, list):
        n = len(result)
        return f"{n} metric hit" if n == 1 else f"{n} metric hits"
    return ""


def _fmt_search_reference_sql(result: Any) -> str:
    if isinstance(result, list):
        n = len(result)
        return f"{n} SQL hit" if n == 1 else f"{n} SQL hits"
    return ""


def _fmt_search_semantic_objects(result: Any) -> str:
    return _list_count(result, "object", "objects")


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
            n = len(files)
            return f"{n} semantic file" if n == 1 else f"{n} semantic files"
    return ""


def _fmt_end_metric_generation(result: Any) -> str:
    if isinstance(result, dict):
        sync = result.get("sync") or {}
        if isinstance(sync, dict) and sync.get("success"):
            return "metric synced"
        if result.get("metric_file"):
            return "metric generated"
    return ""


def _fmt_generate_sql_summary_id(result: Any) -> str:
    if isinstance(result, str) and result:
        return f"id: {result}"
    return ""


def _fmt_analyze_table_relationships(result: Any) -> str:
    if isinstance(result, dict):
        relationships = result.get("relationships")
        if isinstance(relationships, list) and relationships:
            n = len(relationships)
            return f"{n} rel" if n == 1 else f"{n} rels"
        summary = result.get("summary")
        if isinstance(summary, str) and summary:
            return summary
        if isinstance(relationships, list):
            return "0 rels"
    return ""


def _fmt_analyze_column_usage_patterns(result: Any) -> str:
    if isinstance(result, dict):
        patterns = result.get("column_patterns")
        if isinstance(patterns, dict) and patterns:
            n = len(patterns)
            return f"{n} col analyzed" if n == 1 else f"{n} cols analyzed"
        summary = result.get("summary")
        if isinstance(summary, str) and summary:
            return summary
    return ""


def _fmt_get_multiple_tables_ddl(result: Any) -> str:
    if isinstance(result, list):
        n = len(result)
        return f"DDL of {n} table" if n == 1 else f"DDL of {n} tables"
    return ""


def _fmt_analyze_metric_candidates_from_history(result: Any) -> str:
    if isinstance(result, dict):
        candidates = result.get("metric_candidates")
        if isinstance(candidates, list):
            n = len(candidates)
            suffix = ""
            if result.get("query_classification") == "metric_plus_derived_datasource":
                suffix = " + datasource"
            return (f"{n} metric cand" if n == 1 else f"{n} metric cands") + suffix
        summary = result.get("summary")
        if isinstance(summary, str) and summary:
            return summary
    return ""


# === Scheduler tools ===


def _fmt_submit_sql_job(result: Any) -> str:
    if isinstance(result, dict):
        job_id = result.get("job_id")
        if job_id:
            return f"+job {job_id}"
    return ""


def _fmt_submit_sparksql_job(result: Any) -> str:
    if isinstance(result, dict):
        job_id = result.get("job_id")
        if job_id:
            return f"+spark {job_id}"
    return ""


def _fmt_trigger_scheduler_job(result: Any) -> str:
    if isinstance(result, dict):
        run_id = result.get("run_id")
        job_id = result.get("job_id")
        if job_id and run_id:
            return f"{job_id}→{run_id}"
        if job_id:
            return f"trig {job_id}"
    return ""


def _fmt_get_scheduler_job(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("found") is False:
            return f"{result.get('job_id', '?')} not found"
        job_name = result.get("job_name")
        status = result.get("status")
        if job_name and status:
            return f"{job_name}: {status}"
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
            return f"{run_id}: {lines} lines" if lines != 1 else f"{run_id}: 1 line"
        if run_id:
            return f"log: {run_id}"
    return ""


def _fmt_list_scheduler_connections(result: Any) -> str:
    if isinstance(result, dict) and isinstance(result.get("total"), int):
        n = result["total"]
        return f"{n} connection" if n == 1 else f"{n} connections"
    return ""


# === Context search tools ===


def _fmt_list_subject_tree(result: Any) -> str:
    """Walk the nested taxonomy and return the total leaf count."""
    if not isinstance(result, dict):
        return ""

    leaf_keys = {"metrics", "reference_sql", "reference_template"}
    total = 0

    def walk(node: Any) -> None:
        nonlocal total
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in leaf_keys:
                if isinstance(value, list):
                    total += len(value)
            elif isinstance(value, dict):
                walk(value)

    walk(result)
    if total == 0:
        return "subject tree empty"
    return f"{total} item" if total == 1 else f"{total} items"


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
            return f"SQL: {name}"
    if isinstance(result, list):
        n = len(result)
        return f"{n} SQL" if n == 1 else f"{n} SQLs"
    return ""


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
            return f"{rows} rows: {name}"
        if name:
            return f'executed "{name}"'
    return ""


# === Filesystem tools (NOT clipped at exit) ===


def _fmt_read_file(result: Any) -> str:
    if isinstance(result, str):
        line_count = result.count("\n") + (1 if result and not result.endswith("\n") else 0)
        return f"read {pluralize(line_count, 'line')}"
    return ""


def _fmt_write_file(result: Any) -> str:
    if isinstance(result, str):
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


def _fmt_todo_list(result: Any) -> str:
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            total = result.get("total", len(items))
            completed = result.get(
                "completed", sum(1 for it in items if isinstance(it, dict) and it.get("status") == "completed")
            )
            return f"{completed}/{total} todos"
    return ""


def _fmt_todo_read(result: Any) -> str:
    if isinstance(result, dict):
        title = result.get("title")
        status = result.get("status")
        if title and status:
            return f"{title}: {status}"
    return ""


def _fmt_todo_write(result: Any) -> str:
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            return f"{pluralize(len(items), 'todo')}"
    return ""


def _fmt_todo_update(result: Any) -> str:
    if isinstance(result, dict):
        item = result.get("updated_item") or {}
        if isinstance(item, dict):
            status = item.get("status")
            title = item.get("title")
            if status and title:
                return f"{title}: {status}"
            if status:
                return f"todo: {status}"
    return ""


# === Date / session tools ===


def _fmt_parse_temporal_expressions(result: Any) -> str:
    if isinstance(result, dict):
        dates = result.get("extracted_dates")
        if isinstance(dates, list):
            return f"parsed {len(dates)} dates"
    return ""


def _fmt_get_current_date(result: Any) -> str:
    if isinstance(result, dict) and result.get("current_date"):
        return str(result["current_date"])
    return ""


def _fmt_search_skill_usage(result: Any) -> str:
    if isinstance(result, dict):
        matches = result.get("matches")
        if isinstance(matches, list):
            return _list_count(matches, "session", "sessions")
    return ""


# === Skill tools ===


def _fmt_load_skill(result: Any) -> str:
    if isinstance(result, dict):
        metadata = result.get("metadata") or {}
        name = metadata.get("name") or result.get("name")
        if name:
            return f"+{name}"
    return ""


def _fmt_validate_skill(result: Any) -> str:
    if isinstance(result, dict):
        skill_name = result.get("skill_name") or "skill"
        warnings = result.get("warnings", 0)
        if warnings:
            return f"{skill_name} valid ({warnings} warns)"
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
                    return f"{preview} +{len(decoded) - 1}"
                return f'"{preview}"'
        return f"{len(decoded)} answers"
    if isinstance(text, str):
        return f'"{text}"'
    return ""


# === Sub-agent task tool ===


def _fmt_task(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("sql_file_path"):
        return "SQL file generated"
    if result.get("sql"):
        return "SQL generated"
    semantic_models = result.get("semantic_models")
    if isinstance(semantic_models, list):
        n = len(semantic_models)
        return f"{n} semantic model" if n == 1 else f"{n} semantic models"
    if result.get("sql_summary_file"):
        return "SQL summary saved"
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
        return f'"{response}"'
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

    The registry exit applies :func:`_clip_short` so every non-filesystem
    summary is bounded to ``SUMMARY_TEXT_MAX_CHARS`` characters.
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
        """Build a one-line summary from a FuncToolResult-shaped dict."""
        if not isinstance(data, dict):
            raw = format_generic_result(data) if data is not None else "Empty result"
            return _clip_short(raw, tool_name)

        if looks_like_failure(data):
            return _clip_short(format_failure(data), tool_name)

        result_value = data["result"] if "result" in data else data

        if is_empty_result(result_value):
            return _clip_short("Empty result", tool_name)

        formatter = self._formatters.get(tool_name)
        if formatter is not None:
            try:
                summary = formatter(result_value)
                if summary:
                    return _clip_short(summary, tool_name)
            except Exception as fmt_err:  # pragma: no cover - defensive
                logger.debug(f"Tool summary formatter for {tool_name} raised: {fmt_err}")

        return _clip_short(format_generic_result(result_value), tool_name)

    def summarize_content(self, content: str, tool_name: str = "") -> str:
        """Build a summary from a tool result string (MCP / legacy adapters)."""
        if not content:
            return _clip_short("Empty result", tool_name)

        try:
            data = json.loads(content)
        except (TypeError, ValueError):
            return _clip_short(truncate_text(content), tool_name)

        if isinstance(data, dict):
            return self.summarize_dict(data, tool_name)
        if isinstance(data, list):
            return _clip_short(pluralize(len(data), "item"), tool_name)
        if isinstance(data, bool):
            return _clip_short("OK" if data else "Failed", tool_name)
        if isinstance(data, int):
            return _clip_short(pluralize(data, "row"), tool_name)
        return _clip_short(truncate_text(str(data)), tool_name)


def _register_builtins(registry: ToolSummaryRegistry) -> None:
    """Register every built-in tool formatter."""
    builtins: Dict[str, FormatterFn] = {
        # Database tools
        "read_query": _fmt_read_query,
        "query": _fmt_read_query,
        "execute_write": _fmt_execute_write,
        "execute_ddl": _fmt_execute_ddl,
        "describe_table": _fmt_describe_table,
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
        # Generation / semantic discovery
        "check_semantic_object_exists": _fmt_check_semantic_object_exists,
        "check_semantic_model_exists": _fmt_check_semantic_model_exists,
        "end_semantic_model_generation": _fmt_end_semantic_model_generation,
        "end_metric_generation": _fmt_end_metric_generation,
        "generate_sql_summary_id": _fmt_generate_sql_summary_id,
        "analyze_table_relationships": _fmt_analyze_table_relationships,
        "analyze_column_usage_patterns": _fmt_analyze_column_usage_patterns,
        "analyze_metric_candidates_from_history": _fmt_analyze_metric_candidates_from_history,
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
        "todo_list": _fmt_todo_list,
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
