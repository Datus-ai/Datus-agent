# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the unified per-tool summary registry.

These tests pin the wording produced by every registered formatter so a
future formatter regression is caught before it reaches the CLI compact
line or the SSE ``shortDesc`` payload.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from datus.schemas.tool_summary import (
    TOOL_SUMMARY_REGISTRY,
    ToolSummaryRegistry,
    format_failure,
    format_generic_result,
    format_list_envelope,
    is_empty_result,
    looks_like_failure,
    pluralize,
    truncate_text,
)

# ── Helpers (public API) ────────────────────────────────────────────────


class TestPublicHelpers:
    def test_pluralize(self):
        assert pluralize(0, "row") == "0 rows"
        assert pluralize(1, "row") == "1 row"
        assert pluralize(2, "row") == "2 rows"

    def test_truncate_text_short_returns_first_nonempty_line(self):
        assert truncate_text("\n\nhello world\nfoo") == "hello world"

    def test_truncate_text_long_uses_ellipsis(self):
        long = "x" * 90
        out = truncate_text(long, limit=80)
        assert out.endswith("…")
        assert len(out) <= 81

    def test_truncate_text_empty_returns_sentinel(self):
        assert truncate_text("\n  \n") == "Empty result"

    def test_looks_like_failure_recognizes_zero_success(self):
        assert looks_like_failure({"success": 0}) is True
        assert looks_like_failure({"success": False}) is True

    def test_looks_like_failure_recognizes_non_empty_error(self):
        assert looks_like_failure({"error": "boom"}) is True
        assert looks_like_failure({"error": "  "}) is False

    def test_looks_like_failure_default_false(self):
        assert looks_like_failure({"success": 1}) is False
        assert looks_like_failure({}) is False

    def test_format_failure_with_message(self):
        assert format_failure({"error": "table missing"}) == "Failed: table missing"

    def test_format_failure_without_message(self):
        assert format_failure({"success": 0}) == "Failed"

    def test_is_empty_result(self):
        assert is_empty_result(None) is True
        assert is_empty_result([]) is True
        assert is_empty_result({}) is True
        assert is_empty_result("") is True
        assert is_empty_result([1]) is False
        assert is_empty_result(0) is False

    def test_format_list_envelope(self):
        assert format_list_envelope({"items": [1, 2, 3]}) == "3 items"
        assert format_list_envelope({"items": [1], "total": 10}) == "1 item of 10"
        assert format_list_envelope({"items": [1, 2], "has_more": True}) == "2 items (+more)"

    def test_format_generic_result_dict_envelope(self):
        assert format_generic_result({"items": [1, 2]}) == "2 items"

    def test_format_generic_result_dict_count_keys(self):
        assert format_generic_result({"row_count": 7}) == "7 rows"
        assert format_generic_result({"affected_rows": 1}) == "1 row"
        assert format_generic_result({"count": 4}) == "4 items"
        assert format_generic_result({"rows": 2}) == "2 rows"

    def test_format_generic_result_misc_types(self):
        assert format_generic_result([1, 2]) == "2 items"
        assert format_generic_result(True) == "OK"
        assert format_generic_result(False) == "Failed"
        assert format_generic_result(5) == "5 rows"
        assert format_generic_result("hi") == "hi"
        assert format_generic_result({"foo": "bar"}) == "OK"


# ── Registry routing ────────────────────────────────────────────────────


class TestRegistryRouting:
    def test_failure_short_circuits_per_tool_formatter(self):
        # Even with a registered tool name, failure path skips the formatter.
        out = TOOL_SUMMARY_REGISTRY.summarize_dict(
            {"success": 0, "error": "missing", "result": {"original_rows": 99}},
            "read_query",
        )
        assert out == "Failed: missing"

    def test_unknown_tool_falls_back_to_generic(self):
        out = TOOL_SUMMARY_REGISTRY.summarize_dict(
            {"success": 1, "result": {"items": [1, 2]}},
            "tool_that_does_not_exist",
        )
        assert out == "2 items"

    def test_empty_envelope_passes_through_to_formatter(self):
        # The list-envelope shape ``{"items": [], "total": 0}`` is *not*
        # the same as a missing payload — the formatter still has the
        # noun, so we get a meaningful "0 metrics" rather than the
        # generic "Empty result" sentinel.
        out = TOOL_SUMMARY_REGISTRY.summarize_dict({"success": 1, "result": {"items": []}}, "list_metrics")
        assert out == "0 metrics"

    def test_truly_empty_result_returns_sentinel(self):
        # No payload at all → sentinel.
        out = TOOL_SUMMARY_REGISTRY.summarize_dict({"success": 1, "result": {}}, "list_metrics")
        assert out == "Empty result"

    def test_formatter_returning_empty_falls_back(self):
        # Register a formatter that always returns ""; routing must fall back.
        local = ToolSummaryRegistry()
        local.register("dummy", lambda _r: "")
        out = local.summarize_dict({"success": 1, "result": {"row_count": 4}}, "dummy")
        assert out == "4 rows"

    def test_formatter_exception_falls_back(self):
        local = ToolSummaryRegistry()

        def boom(_r: Any) -> str:
            raise RuntimeError("formatter bug")

        local.register("dummy", boom)
        out = local.summarize_dict({"success": 1, "result": {"row_count": 4}}, "dummy")
        assert out == "4 rows"

    def test_summarize_content_decodes_json(self):
        payload = json.dumps({"success": 1, "result": {"original_rows": 9}})
        assert TOOL_SUMMARY_REGISTRY.summarize_content(payload, "read_query") == "9 rows"

    def test_summarize_content_plain_string(self):
        assert TOOL_SUMMARY_REGISTRY.summarize_content("just a note", "tool") == "just a note"

    def test_summarize_content_empty_string(self):
        assert TOOL_SUMMARY_REGISTRY.summarize_content("", "tool") == "Empty result"

    def test_summarize_content_json_list(self):
        assert TOOL_SUMMARY_REGISTRY.summarize_content("[1, 2, 3]", "tool") == "3 items"

    def test_singleton_covers_all_planned_tools(self):
        names = set(TOOL_SUMMARY_REGISTRY.names())
        # Pin a representative slice across all categories so a missing
        # registration breaks loudly. The exact count may grow over time.
        must_have = {
            # database
            "read_query",
            "execute_write",
            "execute_ddl",
            "describe_table",
            "get_table_ddl",
            "list_tables",
            "list_databases",
            "list_schemas",
            "search_table",
            # bi
            "list_dashboards",
            "create_dashboard",
            "write_query",
            # semantic
            "list_metrics",
            "query_metrics",
            "validate_semantic",
            # gen / scheduler
            "submit_sql_job",
            "list_scheduler_jobs",
            "get_run_log",
            # context / template
            "list_subject_tree",
            "search_reference_template",
            "execute_reference_template",
            # filesystem / plan
            "read_file",
            "glob",
            "todo_read",
            # misc
            "ask_user",
            "task",
            "load_skill",
            "list_document_nav",
        }
        missing = must_have - names
        assert not missing, f"Registry missing formatters for: {sorted(missing)}"
        assert len(names) >= 80, f"Expected ~85 formatters, only got {len(names)}"


# ── Per-tool formatters (success path) ─────────────────────────────────


def _summarize(tool: str, payload: dict) -> str:
    return TOOL_SUMMARY_REGISTRY.summarize_dict(payload, tool)


class TestDatabaseFormatters:
    def test_read_query_rows_only(self):
        assert _summarize("read_query", {"success": 1, "result": {"original_rows": 3}}) == "3 rows"

    def test_read_query_rows_and_columns(self):
        out = _summarize(
            "read_query",
            {"success": 1, "result": {"original_rows": 7, "column_count": 5}},
        )
        assert out == "7 × 5 result"

    def test_read_query_columns_inferred_from_compressed(self):
        out = _summarize(
            "read_query",
            {"success": 1, "result": {"original_rows": 2, "compressed_data": "a,b,c\n1,2,3\n4,5,6"}},
        )
        assert out == "2 × 3 result"

    def test_query_alias_routes_to_read_query(self):
        # 'query' is a registered alias of read_query.
        assert _summarize("query", {"success": 1, "result": {"original_rows": 1}}) == "1 row"

    def test_execute_write(self):
        assert _summarize("execute_write", {"success": 1, "result": {"row_count": 5}}) == "wrote 5 rows"

    def test_execute_ddl(self):
        out = _summarize("execute_ddl", {"success": 1, "result": {"message": "DDL executed"}})
        assert out == "DDL OK"

    def test_describe_table(self):
        cols = [{"name": "id"}, {"name": "name"}, {"name": "email"}]
        assert _summarize("describe_table", {"success": 1, "result": {"columns": cols}}) == "3 columns"

    def test_get_table_ddl(self):
        out = _summarize(
            "get_table_ddl",
            {"success": 1, "result": {"identifier": "public.orders", "definition": "CREATE TABLE..."}},
        )
        assert out == "DDL of public.orders"

    def test_list_tables_with_preview(self):
        out = _summarize(
            "list_tables",
            {"success": 1, "result": [{"name": "orders"}, {"name": "customers"}, {"name": "products"}]},
        )
        assert out == "3 tables: orders, customers, products"

    def test_list_tables_truncated_preview(self):
        out = _summarize(
            "list_tables",
            {"success": 1, "result": [{"name": str(i)} for i in range(5)]},
        )
        # Shows first 3 names + "..." marker when more remain.
        assert out == "5 tables: 0, 1, 2, ..."

    def test_list_databases(self):
        assert _summarize("list_databases", {"success": 1, "result": ["a", "b"]}) == "2 databases: a, b"

    def test_list_schemas_singular(self):
        assert _summarize("list_schemas", {"success": 1, "result": ["public"]}) == "1 schema: public"

    def test_search_table_with_samples(self):
        out = _summarize(
            "search_table",
            {
                "success": 1,
                "result": {
                    "metadata": [{"table_name": "orders"}, {"table_name": "customers"}],
                    "sample_data": {"original_rows": 5},
                },
            },
        )
        assert out == "2 tables and 5 sample rows"

    def test_search_table_no_samples(self):
        out = _summarize(
            "search_table",
            {"success": 1, "result": {"metadata": [{"name": "t"}], "sample_data": []}},
        )
        assert out == "1 table"

    def test_search_table_empty(self):
        out = _summarize("search_table", {"success": 1, "result": {"metadata": [], "sample_data": []}})
        assert out == "no tables matched"


class TestBIFormatters:
    def test_list_dashboards_with_preview(self):
        out = _summarize(
            "list_dashboards",
            {
                "success": 1,
                "result": {"items": [{"title": "Sales"}, {"title": "Ops"}], "total": 2, "has_more": False},
            },
        )
        assert out == "2 dashboards: Sales, Ops"

    def test_get_dashboard(self):
        out = _summarize(
            "get_dashboard",
            {"success": 1, "result": {"title": "Sales", "charts": [1, 2, 3]}},
        )
        assert out == 'dashboard "Sales" (3 charts)'

    def test_get_chart(self):
        out = _summarize(
            "get_chart",
            {"success": 1, "result": {"name": "Revenue trend", "chart_type": "line"}},
        )
        assert out == 'chart "Revenue trend" (line)'

    def test_get_chart_data_with_columns(self):
        out = _summarize(
            "get_chart_data",
            {"success": 1, "result": {"row_count": 30, "column_names": ["d", "v"]}},
        )
        assert out == "30 rows × 2 cols"

    def test_create_dashboard(self):
        out = _summarize("create_dashboard", {"success": 1, "result": {"title": "Q1", "dashboard_id": "42"}})
        assert out == 'created dashboard "Q1" (42)'

    def test_delete_dashboard(self):
        out = _summarize(
            "delete_dashboard",
            {"success": 1, "result": {"deleted": True, "dashboard_id": "42"}},
        )
        assert out == "deleted dashboard 42"

    def test_create_chart_with_dashboard(self):
        out = _summarize(
            "create_chart",
            {"success": 1, "result": {"name": "Trend", "dashboard_id": "42", "chart_id": "c1"}},
        )
        assert out == 'created chart "Trend" → dashboard 42'

    def test_write_query(self):
        out = _summarize(
            "write_query",
            {"success": 1, "result": {"table_name": "facts.sales", "rows_written": 100}},
        )
        assert out == "wrote 100 rows → facts.sales"


class TestSemanticFormatters:
    def test_list_metrics_with_total_and_more(self):
        out = _summarize(
            "list_metrics",
            {
                "success": 1,
                "result": {"items": [{"name": "revenue"}, {"name": "gmv"}], "total": 5, "has_more": True},
            },
        )
        assert out == "2 metrics of 5 (+more): revenue, gmv"

    def test_get_dimensions(self):
        out = _summarize(
            "get_dimensions",
            {"success": 1, "result": {"items": [{"name": "region"}], "total": 1, "has_more": False}},
        )
        assert out == "1 dimension: region"

    def test_query_metrics_rows_and_cols(self):
        out = _summarize(
            "query_metrics",
            {"success": 1, "result": {"columns": ["a", "b", "c"], "data": {"original_rows": 12}}},
        )
        assert out == "12 rows × 3 cols"

    def test_validate_semantic_valid(self):
        out = _summarize("validate_semantic", {"success": 1, "result": {"valid": True, "issues": []}})
        assert out == "valid"

    def test_validate_semantic_invalid_with_issues(self):
        out = _summarize(
            "validate_semantic",
            {"success": 1, "result": {"valid": False, "issues": [1, 2, 3]}},
        )
        assert out == "3 issues"

    def test_attribution_analyze(self):
        out = _summarize(
            "attribution_analyze",
            {"success": 1, "result": {"dimension_ranking": list(range(8)), "selected_dimensions": list(range(3))}},
        )
        assert out == "selected 3 of 8 dimensions"


class TestGenerationFormatters:
    def test_check_object_exists_true(self):
        out = _summarize(
            "check_semantic_object_exists",
            {"success": 1, "result": {"exists": True, "kind": "table"}},
        )
        assert out == "table exists"

    def test_check_object_exists_false(self):
        out = _summarize(
            "check_semantic_object_exists",
            {"success": 1, "result": {"exists": False, "kind": "metric"}},
        )
        assert out == "metric not found"

    def test_end_semantic_model_generation(self):
        out = _summarize(
            "end_semantic_model_generation",
            {"success": 1, "result": {"semantic_model_files": ["a.yml", "b.yml"]}},
        )
        assert out == "validated 2 semantic files"

    def test_end_metric_generation(self):
        out = _summarize(
            "end_metric_generation",
            {"success": 1, "result": {"metric_file": "m.yml", "sync": {"success": True}}},
        )
        assert out == "metric generated and synced"

    def test_generate_sql_summary_id(self):
        out = _summarize("generate_sql_summary_id", {"success": 1, "result": "abc12345"})
        assert out == "id: abc12345"

    def test_analyze_table_relationships_uses_inline_summary(self):
        out = _summarize(
            "analyze_table_relationships",
            {"success": 1, "result": {"relationships": [], "summary": "Found 4 relationships across 3 tables"}},
        )
        assert out == "Found 4 relationships across 3 tables"

    def test_get_multiple_tables_ddl(self):
        out = _summarize(
            "get_multiple_tables_ddl",
            {"success": 1, "result": [{"table_name": "a"}, {"table_name": "b"}]},
        )
        assert out == "DDL of 2 tables"


class TestSchedulerFormatters:
    def test_submit_sql_job(self):
        out = _summarize(
            "submit_sql_job",
            {"success": 1, "result": {"job_id": "dag_42", "job_name": "daily_etl", "status": "submitted"}},
        )
        assert out == 'submitted "daily_etl" (dag_42)'

    def test_submit_sparksql_job(self):
        out = _summarize(
            "submit_sparksql_job",
            {"success": 1, "result": {"job_id": "spark_1", "job_name": "agg"}},
        )
        assert out == 'submitted spark "agg" (spark_1)'

    def test_trigger_scheduler_job(self):
        out = _summarize(
            "trigger_scheduler_job",
            {"success": 1, "result": {"run_id": "r99", "job_id": "dag_42", "status": "running"}},
        )
        assert out == "triggered dag_42 → run r99"

    def test_get_scheduler_job_found(self):
        out = _summarize(
            "get_scheduler_job",
            {"success": 1, "result": {"found": True, "job_id": "j", "job_name": "daily", "status": "active"}},
        )
        assert out == 'job "daily" (active)'

    def test_get_scheduler_job_not_found(self):
        out = _summarize(
            "get_scheduler_job",
            {"success": 1, "result": {"found": False, "job_id": "ghost"}},
        )
        assert out == "job ghost not found"

    def test_list_scheduler_jobs(self):
        out = _summarize(
            "list_scheduler_jobs",
            {"success": 1, "result": {"items": [{"job_name": "a"}, {"job_name": "b"}], "total": 2, "has_more": False}},
        )
        assert out == "2 jobs: a, b"

    def test_pause_resume_delete_update(self):
        for tool in ("pause_job", "resume_job", "delete_job", "update_job"):
            assert _summarize(tool, {"success": 1, "result": {"job_id": "x", "status": "paused"}}).endswith("x")

    def test_get_run_log_with_lines(self):
        log = "line1\nline2\nline3"
        out = _summarize("get_run_log", {"success": 1, "result": {"run_id": "r1", "log": log}})
        assert out == "log of run r1 (3 lines)"

    def test_list_scheduler_connections(self):
        out = _summarize(
            "list_scheduler_connections",
            {"success": 1, "result": {"total": 4, "connections": []}},
        )
        assert out == "4 scheduler connections"


class TestContextSearchFormatters:
    def test_list_subject_tree_aggregates_counts(self):
        tree = {
            "Finance": {
                "Revenue": {"metrics": ["revenue", "gmv"], "reference_sql": ["q1"]},
                "Cost": {"metrics": ["cogs"]},
            },
            "Marketing": {"knowledge": ["k1", "k2"]},
        }
        out = _summarize("list_subject_tree", {"success": 1, "result": tree})
        assert out == "3 metrics, 1 SQLs, 2 knowledge"

    def test_list_subject_tree_empty(self):
        # Empty result short-circuits to sentinel.
        out = _summarize("list_subject_tree", {"success": 1, "result": {}})
        assert out == "Empty result"

    def test_get_metrics_single(self):
        out = _summarize("get_metrics", {"success": 1, "result": {"name": "revenue"}})
        assert out == 'metric "revenue"'

    def test_get_reference_sql_single(self):
        out = _summarize("get_reference_sql", {"success": 1, "result": {"name": "top_users"}})
        assert out == 'reference SQL "top_users"'

    def test_get_knowledge_list(self):
        out = _summarize("get_knowledge", {"success": 1, "result": [{"name": "a"}, {"name": "b"}]})
        assert out == "2 knowledge entries"


class TestReferenceTemplateFormatters:
    def test_search_reference_template(self):
        out = _summarize(
            "search_reference_template",
            {"success": 1, "result": [{"name": "t1"}, {"name": "t2"}]},
        )
        assert out == "2 templates"

    def test_get_reference_template(self):
        assert _summarize("get_reference_template", {"success": 1, "result": {"name": "tpl"}}) == 'template "tpl"'

    def test_render_reference_template(self):
        out = _summarize(
            "render_reference_template",
            {"success": 1, "result": {"rendered_sql": "SELECT 1", "template_name": "x"}},
        )
        assert out == 'rendered "x"'

    def test_execute_reference_template_with_rows(self):
        out = _summarize(
            "execute_reference_template",
            {
                "success": 1,
                "result": {
                    "rendered_sql": "SELECT 1",
                    "template_name": "x",
                    "query_result": {"original_rows": 9},
                },
            },
        )
        assert out == '9 rows from "x"'


class TestFilesystemFormatters:
    def test_read_file_string_lines(self):
        out = _summarize("read_file", {"success": 1, "result": "line1\nline2\nline3\n"})
        assert out == "read 3 lines"

    def test_write_file(self):
        out = _summarize("write_file", {"success": 1, "result": "File written successfully: /tmp/x.txt"})
        assert out == "wrote /tmp/x.txt"

    def test_edit_file(self):
        out = _summarize("edit_file", {"success": 1, "result": "File edited successfully: /tmp/y.py"})
        assert out == "edited /tmp/y.py"

    def test_glob(self):
        out = _summarize(
            "glob",
            {"success": 1, "result": {"files": ["a.py", "b.py", "c.py"], "truncated": False}},
        )
        assert out == "3 files"

    def test_glob_truncated(self):
        out = _summarize(
            "glob",
            {"success": 1, "result": {"files": ["a"] * 200, "truncated": True}},
        )
        assert out == "200 files (truncated)"

    def test_grep(self):
        out = _summarize(
            "grep",
            {"success": 1, "result": {"matches": [{"file": "a", "line": 1}], "truncated": False}},
        )
        assert out == "1 match"


class TestPlanFormatters:
    def test_todo_read_progress(self):
        out = _summarize(
            "todo_read",
            {
                "success": 1,
                "result": {
                    "lists": [{"items": [{"status": "completed"}, {"status": "completed"}, {"status": "pending"}]}],
                    "total_lists": 1,
                },
            },
        )
        assert out == "2/3 todos"

    def test_todo_write(self):
        out = _summarize(
            "todo_write",
            {"success": 1, "result": {"todo_list": {"items": [1, 2, 3]}}},
        )
        assert out == "3 todos"

    def test_todo_update(self):
        out = _summarize(
            "todo_update",
            {"success": 1, "result": {"updated_item": {"content": "Run report", "status": "completed"}}},
        )
        assert out == '"Run report" → completed'


class TestDateAndSessionFormatters:
    def test_parse_temporal_expressions(self):
        out = _summarize(
            "parse_temporal_expressions",
            {"success": 1, "result": {"extracted_dates": [1, 2]}},
        )
        assert out == "parsed 2 expressions"

    def test_get_current_date(self):
        out = _summarize("get_current_date", {"success": 1, "result": {"current_date": "2026-04-25"}})
        assert out == "2026-04-25"

    def test_search_skill_usage(self):
        out = _summarize(
            "search_skill_usage",
            {"success": 1, "result": {"matches": [{"session_id": "s1"}, {"session_id": "s2"}]}},
        )
        assert out == "2 session matches"


class TestSkillAskUserAndTask:
    def test_load_skill(self):
        out = _summarize(
            "load_skill",
            {"success": 1, "result": {"metadata": {"name": "sql-optimization"}}},
        )
        assert out == 'loaded "sql-optimization"'

    def test_validate_skill_clean(self):
        out = _summarize(
            "validate_skill",
            {"success": 1, "result": {"skill_name": "x", "warnings": 0}},
        )
        assert out == "x valid"

    def test_validate_skill_with_warnings(self):
        out = _summarize(
            "validate_skill",
            {"success": 1, "result": {"skill_name": "x", "warnings": 2}},
        )
        assert out == "x valid (2 warnings)"

    def test_ask_user_json_array(self):
        payload = json.dumps([{"question": "DB?", "answer": "PostgreSQL"}])
        out = _summarize("ask_user", {"success": 1, "result": payload})
        assert out == '"PostgreSQL"'

    def test_ask_user_multiple_questions(self):
        payload = json.dumps(
            [
                {"question": "DB?", "answer": "PostgreSQL"},
                {"question": "Schema?", "answer": "public"},
            ]
        )
        out = _summarize("ask_user", {"success": 1, "result": payload})
        assert out == '"PostgreSQL" (+1 more)'

    def test_task_with_sql(self):
        out = _summarize("task", {"success": 1, "result": {"sql": "SELECT 1", "response": "done"}})
        assert out == "SQL generated"

    def test_task_with_dashboard(self):
        out = _summarize(
            "task",
            {"success": 1, "result": {"dashboard_result": {"id": 7}, "response": "ok"}},
        )
        assert out == "dashboard updated"

    def test_task_generic_response(self):
        out = _summarize("task", {"success": 1, "result": {"response": "Found 3 tables"}})
        assert out == '"Found 3 tables"'


class TestPlatformDocFormatters:
    def test_list_document_nav(self):
        out = _summarize(
            "list_document_nav",
            {"success": 1, "result": {"platform": "snowflake", "total_docs": 42, "nav_tree": {}}},
        )
        assert out == "snowflake: 42 docs"

    def test_get_document(self):
        out = _summarize(
            "get_document",
            {"success": 1, "result": {"platform": "duckdb", "chunk_count": 3, "chunks": []}},
        )
        assert out == "duckdb: 3 chunks"

    def test_search_document(self):
        out = _summarize(
            "search_document",
            {"success": 1, "result": {"docs": [1, 2], "doc_count": 2}},
        )
        assert out == "2 doc matches"


# ── Failure path covers all tools uniformly ─────────────────────────────


@pytest.mark.parametrize(
    "tool",
    [
        "read_query",
        "list_metrics",
        "submit_sql_job",
        "ask_user",
        "task",
        "list_document_nav",
    ],
)
def test_failure_path_uniform(tool: str):
    out = TOOL_SUMMARY_REGISTRY.summarize_dict({"success": 0, "error": "boom"}, tool)
    assert out == "Failed: boom"


# ── Per-tool fallback / branch coverage ─────────────────────────────────
#
# Each formatter has multiple "graceful degradation" branches that fire
# when the expected fields are missing. Exercising them keeps the
# formatters resilient against minor schema drift in the underlying
# tools without introducing a thicket of one-off test methods.


@pytest.mark.parametrize(
    "tool, payload, expected",
    [
        # BI: get_dashboard / get_chart / create_* / update_* — fall through
        # to id-only or no-title variants.
        ("get_dashboard", {"title": "Sales", "charts": "broken"}, 'dashboard "Sales"'),
        ("get_dashboard", {"dashboard_id": "d99"}, "dashboard d99"),
        ("get_chart", {"name": "Trend"}, 'chart "Trend"'),
        ("get_chart", {"chart_id": "c7"}, "chart c7"),
        ("get_chart_data", {"row_count": 4}, "4 rows"),
        ("get_chart_data", {"rows": [1, 2], "column_names": ["a", "b"]}, "2 rows × 2 cols"),
        ("create_dashboard", {"title": "Sales"}, 'created dashboard "Sales"'),
        ("create_dashboard", {"dashboard_id": "d1"}, "created dashboard d1"),
        ("update_dashboard", {"title": "X"}, 'updated dashboard "X"'),
        ("update_dashboard", {"dashboard_id": "d2"}, "updated dashboard d2"),
        ("delete_dashboard", {"deleted": False, "dashboard_id": "d3"}, "dashboard d3 (not deleted)"),
        ("delete_dashboard", {"deleted": True}, "deleted dashboard"),
        ("create_chart", {"name": "T"}, 'created chart "T"'),
        ("create_chart", {"chart_id": "c"}, "created chart c"),
        ("update_chart", {"chart_id": "c1"}, "updated chart c1"),
        ("add_chart_to_dashboard", {"chart_id": "c", "dashboard_id": "d"}, "chart c → dashboard d"),
        ("delete_chart", {"chart_id": "c5"}, "deleted chart c5"),
        ("delete_chart", {"deleted": True}, "deleted chart"),
        ("create_dataset", {"name": "ds"}, 'created dataset "ds"'),
        ("create_dataset", {"dataset_id": "7"}, "created dataset 7"),
        ("delete_dataset", {"dataset_id": "9"}, "deleted dataset 9"),
        ("delete_dataset", {"deleted": True}, "deleted dataset"),
        ("write_query", {"rows_written": 5}, "wrote 5 rows"),
        ("list_bi_databases", [{"id": 1}, {"id": 2}], "2 BI databases"),
        # Database: search_table partial / read_query partial.
        ("read_query", [1, 2, 3], "3 rows"),
        ("describe_table", {"schema": [{"name": "id"}, {"name": "v"}]}, "2 columns"),
        ("get_table_ddl", {"table_name": "orders", "definition": "CREATE TABLE..."}, "DDL of orders"),
        ("list_tables", [{"name": "orders"}], "1 table: orders"),
        ("list_databases", ["mydb"], "1 database: mydb"),
        # Semantic: validate / attribution / get_dimensions partial.
        ("validate_semantic", {"valid": False, "issues": []}, "invalid"),
        ("attribution_analyze", {"selected_dimensions": [1]}, "selected 1 dimension"),
        ("query_metrics", {"columns": ["a", "b"]}, "2 columns"),
        ("query_metrics", {"data": {"original_rows": 7}}, "7 rows"),
        # Generation
        ("check_semantic_object_exists", {"exists": True}, "object exists"),
        ("check_semantic_model_exists", {"exists": True}, "table exists"),
        ("check_semantic_model_exists", {"exists": False}, "table not found"),
        ("end_metric_generation", {"metric_file": "m.yml"}, "metric generated"),
        ("analyze_table_relationships", {"relationships": [{}, {}, {}]}, "3 relationships"),
        (
            "analyze_column_usage_patterns",
            {"summary": "Analyzed 5 columns from 10 SQLs"},
            "Analyzed 5 columns from 10 SQLs",
        ),
        ("analyze_column_usage_patterns", {"column_patterns": {"a": {}, "b": {}}}, "2 columns analyzed"),
        # Scheduler
        ("submit_sql_job", {"job_id": "j"}, "submitted j"),
        ("submit_sparksql_job", {"job_id": "j2"}, "submitted spark j2"),
        ("trigger_scheduler_job", {"job_id": "j3"}, "triggered j3"),
        ("get_scheduler_job", {"job_id": "j4"}, "job j4"),
        ("get_run_log", {"run_id": "r2"}, "log of run r2"),
        # Context search
        ("get_metrics", [{"name": "a"}, {"name": "b"}], "2 metrics"),
        ("get_reference_sql", [{"name": "s"}], "1 reference SQL"),
        # Reference template
        ("execute_reference_template", {"template_name": "x"}, 'executed "x"'),
        # Filesystem fallback
        ("read_file", "single line no newline", "read 1 line"),
        ("write_file", "Some other format", "Some other format"),
        ("edit_file", "Some other edit msg", "Some other edit msg"),
        ("grep", {"matches": [{"f": "a"}, {"f": "b"}], "truncated": True}, "2 matches (truncated)"),
        # Plan
        ("todo_read", {"lists": [], "total_lists": 0}, "no todos"),
        ("todo_update", {"updated_item": {"status": "pending"}}, "todo → pending"),
        # Skill / ask user / date
        ("validate_skill", {"warnings": 1}, "skill valid (1 warning)"),
        ("ask_user", "free-text answer", '"free-text answer"'),
        ("ask_user", {"answer": "yes"}, '"yes"'),
        # Sub-agent task
        ("task", {"sql_file_path": "/tmp/q.sql"}, "SQL file generated"),
        ("task", {"semantic_models": ["a.yml", "b.yml"]}, "2 semantic models generated"),
        ("task", {"sql_summary_file": "x"}, "SQL summary saved"),
        ("task", {"ext_knowledge_file": "k"}, "knowledge saved"),
        ("task", {"report_result": {}}, "report ready"),
        ("task", {"skill_path": "/tmp", "skill_name": "x"}, 'skill "x" generated'),
        ("task", {"scheduler_result": {}}, "scheduler updated"),
        ("task", {"items_saved": 3}, "feedback saved"),
        # Platform docs
        ("list_document_nav", {"total_docs": 5}, "5 docs"),
        ("get_document", {"chunks": [1, 2, 3]}, "3 chunks"),
        ("search_document", {"docs": [1]}, "1 doc match"),
        ("web_search_document", [1, 2, 3], "3 web results"),
        ("web_search_document", {"docs": [1, 2]}, "2 web results"),
    ],
)
def test_per_tool_fallback_branches(tool: str, payload: Any, expected: str):
    out = TOOL_SUMMARY_REGISTRY.summarize_dict({"success": 1, "result": payload}, tool)
    assert out == expected, f"tool={tool!r}, payload={payload!r}, got={out!r}"


def test_summarize_dict_passes_non_dict_to_generic():
    """When ``data`` itself isn't a dict, fall back to the generic formatter."""
    assert TOOL_SUMMARY_REGISTRY.summarize_dict([1, 2], "any") == "2 items"


def test_summarize_dict_none_returns_sentinel():
    assert TOOL_SUMMARY_REGISTRY.summarize_dict(None, "any") == "Empty result"


def test_summarize_content_int_and_bool():
    assert TOOL_SUMMARY_REGISTRY.summarize_content("true", "x") == "OK"
    assert TOOL_SUMMARY_REGISTRY.summarize_content("false", "x") == "Failed"
    assert TOOL_SUMMARY_REGISTRY.summarize_content("7", "x") == "7 rows"


def test_registry_register_and_has():
    local = ToolSummaryRegistry()
    assert local.has("foo") is False
    local.register("foo", lambda r: "custom")
    assert local.has("foo") is True
    assert local.summarize_dict({"success": 1, "result": {"x": 1}}, "foo") == "custom"
