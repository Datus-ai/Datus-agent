# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/action_display/tool_content.py."""

import json
import uuid
from datetime import datetime, timedelta

import pytest

from datus.cli.action_display.tool_content import (
    ToolCallContent,
    ToolCallContentBuilder,
    _build_analyze_columns,
    _build_analyze_relationships,
    _build_ask_user,
    _build_attribution_analyze,
    _build_check_exists,
    _build_describe_table,
    _build_doc_search_result,
    _build_end_generation,
    _build_end_metric_generation,
    _build_execute_command,
    _build_generate_sql_summary_id,
    _build_get_detail,
    _build_get_dimensions,
    _build_get_document,
    _build_get_knowledge,
    _build_get_metrics,
    _build_get_multiple_ddl,
    _build_get_reference_sql,
    _build_get_table_ddl,
    _build_glob,
    _build_grep,
    _build_list_databases,
    _build_list_document_nav,
    _build_list_metrics_semantic,
    _build_list_schemas,
    _build_list_subject_tree,
    _build_list_tables,
    _build_load_skill,
    _build_parse_dates,
    _build_query_metrics,
    _build_read_file,
    _build_read_query,
    _build_search_documents,
    _build_search_external_knowledge,
    _build_search_knowledge,
    _build_search_metrics,
    _build_search_reference_sql,
    _build_search_semantic_objects,
    _build_search_table,
    _build_simple_action,
    _build_simple_list,
    _build_todo_read,
    _build_todo_update,
    _build_todo_write,
    _build_validate_semantic,
    _build_write_file,
    _format_csv_preview,
    _format_describe_table_output_verbose,
    _format_get_table_ddl_output_verbose,
    _format_read_query_output_verbose,
    _format_result_only_markup,
    _format_value_markup,
    calc_duration,
    extract_args,
    format_generic_preview,
    format_output_verbose,
    make_base_content,
    parse_output_data,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


def _make(
    status=ActionStatus.SUCCESS,
    messages="",
    input_data=None,
    output_data=None,
    start_time=None,
    end_time=None,
) -> ActionHistory:
    return ActionHistory(
        action_id=str(uuid.uuid4()),
        role=ActionRole.TOOL,
        messages=messages,
        action_type="test",
        input=input_data,
        output=output_data,
        status=status,
        start_time=start_time or datetime.now(),
        end_time=end_time,
        depth=0,
    )


# ── ToolCallContent dataclass ──────────────────────────────────────


@pytest.mark.ci
class TestToolCallContent:
    def test_defaults(self):
        tc = ToolCallContent(label="t", status_mark="\u2713", duration_str="")
        assert tc.output_preview == ""
        assert tc.args_lines == []
        assert tc.output_lines == []

    def test_all_fields(self):
        tc = ToolCallContent(
            label="search_table",
            status_mark="\u2713",
            duration_str=" (1.2s)",
            output_preview="\u2713 3 tables",
            args_lines=["query: SELECT 1"],
            output_lines=["success: true"],
        )
        assert tc.duration_str == " (1.2s)"
        assert len(tc.args_lines) == 1


# ── Shared helpers ─────────────────────────────────────────────────


@pytest.mark.ci
class TestSharedHelpers:
    def test_calc_duration_with_times(self):
        now = datetime.now()
        a = _make(start_time=now, end_time=now + timedelta(seconds=2.5))
        assert "2.5s" in calc_duration(a)

    def test_calc_duration_no_end(self):
        assert calc_duration(_make()) == ""

    def test_extract_args_dict(self):
        a = _make(input_data={"function_name": "f", "arguments": {"a": 1, "b": "x"}})
        lines = extract_args(a)
        assert lines == ["a: 1", "b: x"]

    def test_extract_args_json_string_parsed_to_kv(self):
        """Verify JSON string args are parsed into key: value lines."""
        a = _make(input_data={"function_name": "f", "arguments": '{"table_name": "t1", "database": "db"}'})
        lines = extract_args(a)
        assert lines == ["table_name: t1", "database: db"]

    def test_extract_args_non_json_string(self):
        """Verify non-JSON string args fall back to 'args: ...' format."""
        a = _make(input_data={"function_name": "f", "arguments": "raw"})
        assert extract_args(a) == ["args: raw"]

    def test_extract_args_empty(self):
        assert extract_args(_make(input_data={})) == []

    def test_make_base_content_success(self):
        a = _make(input_data={"function_name": "list_tables"})
        tc = make_base_content(a)
        assert tc.label == "list_tables"
        assert tc.status_mark == "\u2713"

    def test_make_base_content_failed(self):
        a = _make(status=ActionStatus.FAILED, messages="fallback_msg", input_data={})
        tc = make_base_content(a)
        assert tc.label == "fallback_msg"
        assert tc.status_mark == "\u2717"

    def test_parse_output_data_none(self):
        assert parse_output_data(None) is None
        assert parse_output_data("") is None

    def test_parse_output_data_string_json(self):
        assert parse_output_data('{"k": "v"}') == {"k": "v"}

    def test_parse_output_data_string_invalid(self):
        assert parse_output_data("not json") is None

    def test_parse_output_data_non_dict(self):
        assert parse_output_data(42) is None

    def test_parse_output_data_raw_output(self):
        d = parse_output_data({"raw_output": '{"a": 1}'})
        assert d == {"a": 1}

    def test_parse_output_data_raw_output_invalid(self):
        assert parse_output_data({"raw_output": "not json"}) is None

    def test_parse_output_data_raw_output_non_dict(self):
        assert parse_output_data({"raw_output": "[1,2]"}) is None


# ── format_output_verbose ──────────────────────────────────────────


@pytest.mark.ci
class TestFormatOutputVerbose:
    def test_empty(self):
        assert format_output_verbose(None) == []

    def test_string_json(self):
        lines = format_output_verbose('{"k": "v"}')
        assert any("k: v" in line for line in lines)

    def test_string_invalid(self):
        assert format_output_verbose("bad") == ["output: bad"]

    def test_non_dict(self):
        assert format_output_verbose(42) == ["output: 42"]

    def test_raw_output_invalid_string(self):
        lines = format_output_verbose({"raw_output": "bad"})
        assert lines == ["output: bad"]

    def test_raw_output_non_dict_parsed(self):
        lines = format_output_verbose({"raw_output": "[1]"})
        assert lines == ["output: [1, 2]"] or "output:" in lines[0]

    def test_multiline_value(self):
        lines = format_output_verbose({"key": "a\nb"})
        assert any("key:" in line for line in lines)
        assert any("a" in line for line in lines)
        assert any("b" in line for line in lines)

    def test_non_dict_inner_data(self):
        # raw_output parses to a list, not dict
        lines = format_output_verbose({"raw_output": '"just a string"'})
        assert any("output:" in line for line in lines)


# ── format_generic_preview ─────────────────────────────────────────


@pytest.mark.ci
class TestFormatGenericPreview:
    def test_empty(self):
        assert format_generic_preview(None) == ""
        assert format_generic_preview("") == ""

    def test_unparseable(self):
        assert "preview unavailable" in format_generic_preview("not json")

    def test_non_dict(self):
        assert "preview unavailable" in format_generic_preview(42)

    def test_failed_with_error(self):
        r = format_generic_preview({"success": False, "error": "timeout"})
        assert "Failed" in r and "timeout" in r

    def test_failed_long_error(self):
        r = format_generic_preview({"success": False, "error": "x" * 100})
        assert "..." in r

    def test_failed_no_error(self):
        assert "Failed" in format_generic_preview({"success": False})

    def test_list_items(self):
        r = format_generic_preview({"result": [1, 2, 3]})
        assert "3 items" in r

    def test_text_json_array(self):
        r = format_generic_preview({"text": '[{"a":1}]'})
        assert "1 items" in r

    def test_text_plain_short(self):
        assert format_generic_preview({"text": "hello"}, enable_truncation=True) == "hello"

    def test_text_plain_long_truncated(self):
        r = format_generic_preview({"text": "x" * 100}, enable_truncation=True)
        assert r.endswith("...")

    def test_success_fallback(self):
        assert "Success" in format_generic_preview({"success": True})

    def test_completed_fallback(self):
        assert "Completed" in format_generic_preview({"some_key": 1})

    def test_raw_output_unparseable(self):
        r = format_generic_preview({"raw_output": "bad"})
        assert "preview unavailable" in r


# ── Per-tool builders ──────────────────────────────────────────────


@pytest.mark.ci
class TestBuildListTables:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "list_tables"},
            output_data={"result": [1, 2]},
        )
        tc = _build_list_tables(a, verbose=False)
        assert "2 tables" in tc.compact_result

    def test_verbose(self):
        a = _make(
            input_data={"function_name": "list_tables", "arguments": {"db": "main"}},
            output_data={"raw_output": '{"success": 1, "result": [{"type": "table", "name": "t1"}]}'},
        )
        tc = _build_list_tables(a, verbose=True)
        assert len(tc.args_lines) == 1
        assert len(tc.output_lines) > 0
        # Should not contain success/error metadata
        assert not any("success" in line.lower() for line in tc.output_lines)
        assert tc.output_preview == ""

    def test_no_items(self):
        a = _make(input_data={"function_name": "list_tables"}, output_data={"some": "data"})
        tc = _build_list_tables(a, verbose=False)
        assert tc.output_preview == ""


@pytest.mark.ci
class TestBuildDescribeTable:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "describe_table"},
            output_data={"result": ["c1", "c2", "c3"]},
        )
        tc = _build_describe_table(a, verbose=False)
        assert "3 columns" in tc.compact_result


@pytest.mark.ci
class TestBuildReadQuery:
    def test_compact_original_rows(self):
        a = _make(
            input_data={"function_name": "read_query"},
            output_data={"original_rows": 42},
        )
        tc = _build_read_query(a, verbose=False)
        assert "42 rows" in tc.compact_result

    def test_compact_list_fallback(self):
        a = _make(
            input_data={"function_name": "read_query"},
            output_data={"result": [1, 2]},
        )
        tc = _build_read_query(a, verbose=False)
        assert "2 items" in tc.compact_result

    def test_compact_no_data(self):
        a = _make(input_data={"function_name": "read_query"}, output_data={"k": "v"})
        tc = _build_read_query(a, verbose=False)
        assert tc.output_preview == ""

    def test_compact_infers_cols_from_compressed_data(self):
        """When column_count is absent, infer column count from the first CSV header row."""
        a = _make(
            input_data={"function_name": "read_query"},
            output_data={
                "raw_output": (
                    '{"success": 1, "result": {"original_rows": 7, '
                    '"compressed_data": "id,name,created_at\\n1,a,2024\\n2,b,2025"}}'
                )
            },
        )
        tc = _build_read_query(a, verbose=False)
        assert tc.compact_result == "7 \u00d7 3 result"


@pytest.mark.ci
class TestBuildSearchTable:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "search_table"},
            output_data={"metadata": [1, 2], "sample_data": [3]},
        )
        tc = _build_search_table(a, verbose=False)
        assert "2 tables" in tc.compact_result
        assert "1 sample row" in tc.compact_result

    def test_compact_with_compressed_sample_data(self):
        a = _make(
            input_data={"function_name": "search_table"},
            output_data={
                "metadata": [1, 2],
                "sample_data": {
                    "original_rows": 1,
                    "original_columns": ["sample_rows"],
                    "is_compressed": False,
                    "compressed_data": "index,sample_rows\n0,[{'id': 1}]",
                    "removed_columns": [],
                    "compression_type": "none",
                },
            },
        )
        tc = _build_search_table(a, verbose=False)
        assert "2 tables" in tc.compact_result
        assert "1 sample row" in tc.compact_result

    def test_compact_no_data(self):
        a = _make(input_data={"function_name": "search_table"})
        tc = _build_search_table(a, verbose=False)
        assert tc.output_preview == ""


@pytest.mark.ci
class TestBuildSearchGeneric:
    """Test search_metrics, search_reference_sql, search_external_knowledge, search_documents."""

    @pytest.mark.parametrize(
        "fn, builder, unit",
        [
            ("search_metrics", _build_search_metrics, "metrics"),
            ("search_reference_sql", _build_search_reference_sql, "reference SQLs"),
            ("search_external_knowledge", _build_search_external_knowledge, "knowledge entries"),
            ("search_documents", _build_search_documents, "documents"),
        ],
    )
    def test_compact_with_items(self, fn, builder, unit):
        a = _make(input_data={"function_name": fn}, output_data={"result": [1, 2, 3]})
        tc = builder(a, verbose=False)
        assert f"3 {unit}" in tc.compact_result

    @pytest.mark.parametrize(
        "fn, builder",
        [
            ("search_metrics", _build_search_metrics),
            ("search_documents", _build_search_documents),
        ],
    )
    def test_compact_no_items(self, fn, builder):
        a = _make(input_data={"function_name": fn}, output_data={"k": "v"})
        tc = builder(a, verbose=False)
        # Shape-aware: "0 <noun(s)> matched" so a stray "0 " elsewhere won't pass.
        assert tc.compact_result.startswith("0 ")
        assert tc.compact_result.endswith(" matched")


# ── Builder class ──────────────────────────────────────────────────


@pytest.mark.ci
class TestToolCallContentBuilderDefault:
    """Test the default (unregistered) build path."""

    def test_compact_success(self):
        now = datetime.now()
        a = _make(
            input_data={"function_name": "unknown_tool"},
            output_data={"success": True},
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        builder = ToolCallContentBuilder()
        tc = builder.build(a, verbose=False)
        assert tc.label == "unknown_tool"
        assert "Success" in tc.compact_result
        assert "1.0s" in tc.duration_str

    def test_compact_failed_no_preview(self):
        a = _make(
            status=ActionStatus.FAILED,
            input_data={"function_name": "unknown"},
            output_data={"error": "boom"},
        )
        tc = ToolCallContentBuilder().build(a, verbose=False)
        assert tc.output_preview == ""

    def test_verbose(self):
        a = _make(
            input_data={"function_name": "f", "arguments": {"sql": "SELECT 1"}},
            output_data={"raw_output": '{"ok": true}'},
        )
        tc = ToolCallContentBuilder().build(a, verbose=True)
        # SQL key produces header line + value line (2 lines)
        assert len(tc.args_lines) >= 1
        assert any("sql" in line for line in tc.args_lines)
        assert any("SELECT 1" in line for line in tc.args_lines)
        assert len(tc.output_lines) > 0
        assert tc.output_preview == ""


@pytest.mark.ci
class TestToolCallContentBuilderRegistry:
    def test_builtin_registered(self):
        """Built-in tools are auto-registered."""
        builder = ToolCallContentBuilder()
        assert "list_tables" in builder._registry
        assert "search_table" in builder._registry

    def test_custom_builder_called(self):
        def custom(action, verbose):
            return ToolCallContent(label="custom", status_mark="*", duration_str="")

        builder = ToolCallContentBuilder()
        builder.register("my_tool", custom)
        a = _make(input_data={"function_name": "my_tool"})
        tc = builder.build(a, verbose=False)
        assert tc.label == "custom"

    def test_unregistered_uses_default(self):
        a = _make(input_data={"function_name": "brand_new_tool"}, output_data={"success": True})
        tc = ToolCallContentBuilder().build(a, verbose=False)
        assert tc.label == "brand_new_tool"
        assert "Success" in tc.compact_result

    def test_override_builtin(self):
        """Can override a built-in registration."""
        builder = ToolCallContentBuilder()
        builder.register(
            "list_tables", lambda a, v: ToolCallContent(label="overridden", status_mark="", duration_str="")
        )
        a = _make(input_data={"function_name": "list_tables"})
        tc = builder.build(a, verbose=False)
        assert tc.label == "overridden"

    def test_builtin_dispatches_correctly(self):
        """list_tables dispatched to _build_list_tables producing table count."""
        a = _make(input_data={"function_name": "list_tables"}, output_data={"result": [1, 2]})
        tc = ToolCallContentBuilder().build(a, verbose=False)
        assert "2 tables" in tc.compact_result


# ── Parity: main/sub produce same content ──────────────────────────


@pytest.mark.ci
class TestMainSubParity:
    def test_same_content_compact(self):
        from datus.cli.action_display.renderers import ActionContentGenerator

        cg = ActionContentGenerator(enable_truncation=True)
        now = datetime.now()
        a = _make(
            input_data={"function_name": "list_tables"},
            output_data={"result": [1, 2, 3]},
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        tc1 = cg.tool_content_builder.build(a, verbose=False)
        tc2 = cg.tool_content_builder.build(a, verbose=False)
        assert tc1.label == tc2.label
        assert tc1.output_preview == tc2.output_preview

    def test_same_content_verbose(self):
        from datus.cli.action_display.renderers import ActionContentGenerator

        cg = ActionContentGenerator(enable_truncation=True)
        now = datetime.now()
        a = _make(
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1"}},
            output_data={"raw_output": '{"success": true}'},
            start_time=now,
            end_time=now + timedelta(seconds=0.3),
        )
        tc1 = cg.tool_content_builder.build(a, verbose=True)
        tc2 = cg.tool_content_builder.build(a, verbose=True)
        assert tc1.args_lines == tc2.args_lines
        assert tc1.output_lines == tc2.output_lines


# ── _format_csv_preview ───────────────────────────────────────────


@pytest.mark.ci
class TestFormatCsvPreview:
    """Tests for _format_csv_preview: CSV text -> readable table."""

    def test_simple_csv_produces_header_separator_and_data(self):
        """Verify basic CSV renders with header, separator, and data rows."""
        csv = "name,age\nAlice,30\nBob,25"
        lines = _format_csv_preview(csv)
        assert lines[0] == "data:"
        assert "name" in lines[1]
        assert "age" in lines[1]
        # tabulate uses "---" separator
        assert "---" in lines[2]
        assert "Alice" in lines[3]
        assert "Bob" in lines[4]

    def test_placeholder_rows_rendered_as_ellipsis(self):
        """Verify '...' placeholder rows are rendered as ellipsis."""
        csv = "a,b\n1,2\n...,...\n3,4"
        lines = _format_csv_preview(csv)
        assert any("..." in line and "1" not in line and "3" not in line for line in lines[3:])

    def test_placeholder_at_start_is_skipped(self):
        """Verify placeholder rows before any data row are skipped."""
        csv = "a,b\n...,...\n1,2"
        lines = _format_csv_preview(csv)
        assert any("1" in line for line in lines)
        # Only one data row (placeholder at start skipped)
        data_lines = [
            line
            for line in lines
            if line.strip()
            and not line.strip().startswith("---")
            and "data:" not in line
            and "a" not in line.split()[0]
            if len(line.split()) > 0
        ]
        assert len(data_lines) >= 1

    def test_max_rows_truncation(self):
        """Verify large datasets show truncation message."""
        rows = ["col"] + [str(i) for i in range(20)]
        csv = "\n".join(rows)
        lines = _format_csv_preview(csv, max_rows=5)
        assert any("20 rows total" in line for line in lines)

    def test_max_cols_truncation(self):
        """Verify wide datasets show column truncation message."""
        header = ",".join([f"c{i}" for i in range(12)])
        data = ",".join(["v"] * 12)
        csv = f"{header}\n{data}"
        lines = _format_csv_preview(csv)
        assert any("more columns not shown" in line for line in lines)

    def test_column_value_truncation(self):
        """Verify long column values are truncated with '...'."""
        long_val = "x" * 50
        csv = f"col\n{long_val}"
        lines = _format_csv_preview(csv, max_col_width=10)
        assert any("..." in line for line in lines[2:])

    def test_empty_csv_returns_empty(self):
        """Verify empty input returns empty list."""
        assert _format_csv_preview("") == []

    def test_header_only_csv(self):
        """Verify CSV with only header (no data) renders correctly."""
        lines = _format_csv_preview("a,b,c")
        assert lines[0] == "data:"
        assert "a" in lines[1]


# ── _format_read_query_output_verbose ─────────────────────────────


@pytest.mark.ci
class TestFormatReadQueryOutputVerbose:
    """Tests for structured read_query verbose output."""

    def test_shows_only_data_preview(self):
        """Verify read_query verbose only shows CSV data preview."""
        output = {
            "raw_output": '{"success": 1, "error": null, "result": '
            '{"original_rows": 3, "original_columns": ["id", "name"], '
            '"compressed_data": "index,id,name\\n0,1,Alice\\n1,2,Bob\\n2,3,Carol", '
            '"removed_columns": ["age"], "compression_type": "columns"}}'
        }
        lines = _format_read_query_output_verbose(output)
        assert lines[0] == "data:"
        assert any("Alice" in line for line in lines)
        # Should NOT contain metadata lines
        assert not any("success:" in line for line in lines)
        assert not any("rows:" in line for line in lines)
        assert not any("columns:" in line for line in lines)
        assert not any("removed_columns:" in line for line in lines)

    def test_error_in_output(self):
        """Verify error message is shown when present."""
        output = {"raw_output": '{"success": 0, "error": "table not found", "result": null}'}
        lines = _format_read_query_output_verbose(output)
        assert lines == ["error: table not found"]

    def test_non_dict_result_fallback(self):
        """Verify non-dict result is shown as raw string."""
        output = {"raw_output": '{"success": 1, "result": "some text result"}'}
        lines = _format_read_query_output_verbose(output)
        assert lines == ["result: some text result"]

    def test_unparseable_falls_back_to_generic(self):
        """Verify unparseable output falls back to format_output_verbose."""
        lines = _format_read_query_output_verbose("not json")
        assert len(lines) > 0


# ── _format_describe_table_output_verbose ─────────────────────────


@pytest.mark.ci
class TestFormatDescribeTableOutputVerbose:
    """Tests for structured describe_table verbose output."""

    def test_columns_displayed_with_alignment(self):
        """Verify columns are formatted with name, type, and comment aligned."""
        output = {
            "raw_output": '{"success": 1, "result": {"columns": ['
            '{"name": "id", "type": "int", "comment": "primary key"},'
            '{"name": "username", "type": "varchar(100)", "comment": "user name"}'
            "]}}"
        }
        lines = _format_describe_table_output_verbose(output)
        assert "success: 1" in lines[0]
        assert "columns (2):" in lines[1]
        # Check alignment: longer name 'username' should pad 'id'
        id_line = lines[2]
        username_line = lines[3]
        assert "id" in id_line
        assert "int" in id_line
        assert "primary key" in id_line
        assert "username" in username_line
        assert "varchar(100)" in username_line

    def test_column_without_comment(self):
        """Verify columns without comment are displayed without trailing spaces issue."""
        output = {
            "raw_output": '{"success": 1, "result": {"columns": [{"name": "col1", "type": "int", "comment": ""}]}}'
        }
        lines = _format_describe_table_output_verbose(output)
        assert "columns (1):" in lines[1]
        col_line = lines[2]
        assert "col1" in col_line
        assert "int" in col_line

    def test_non_dict_columns_fallback(self):
        """Verify non-dict column entries are rendered as raw strings."""
        output = {"raw_output": '{"success": 1, "result": {"columns": ["col1", "col2"]}}'}
        lines = _format_describe_table_output_verbose(output)
        assert any("col1" in line for line in lines)
        assert any("col2" in line for line in lines)

    def test_result_without_columns_key(self):
        """Verify result without 'columns' key falls back to key-value pairs."""
        output = {"raw_output": '{"success": 1, "result": {"table_name": "t1", "row_count": 100}}'}
        lines = _format_describe_table_output_verbose(output)
        assert any("table_name: t1" in line for line in lines)
        assert any("row_count: 100" in line for line in lines)

    def test_non_dict_result_fallback(self):
        """Verify non-dict result returns only success line."""
        output = {"raw_output": '{"success": 1, "result": "some string"}'}
        lines = _format_describe_table_output_verbose(output)
        assert lines == ["success: 1"]

    def test_error_shown(self):
        """Verify error message is included in output."""
        output = {"raw_output": '{"success": 0, "error": "access denied", "result": null}'}
        lines = _format_describe_table_output_verbose(output)
        assert any("error: access denied" in line for line in lines)


# ── _format_get_table_ddl_output_verbose ──────────────────────────


@pytest.mark.ci
class TestFormatGetTableDdlOutputVerbose:
    """Tests for structured get_table_ddl verbose output."""

    def test_full_ddl_output(self):
        """Verify DDL output shows identifier, type, and multi-line definition."""
        output = {
            "raw_output": '{"success": 1, "result": '
            '{"identifier": "db.schema.my_table", "table_type": "table", '
            '"definition": "CREATE TABLE my_table (\\n  id INT,\\n  name VARCHAR(50)\\n)"}}'
        }
        lines = _format_get_table_ddl_output_verbose(output)
        assert "success: 1" in lines[0]
        assert "table: db.schema.my_table" in lines[1]
        assert "type: table" in lines[2]
        assert "definition:" in lines[3]
        # DDL lines should be indented
        assert "  CREATE TABLE my_table (" in lines[4]
        assert "    id INT," in lines[5]

    def test_no_identifier_no_type(self):
        """Verify missing identifier and table_type are gracefully omitted."""
        output = {"raw_output": '{"success": 1, "result": {"definition": "CREATE TABLE t (id INT)"}}'}
        lines = _format_get_table_ddl_output_verbose(output)
        assert "success: 1" in lines[0]
        assert "definition:" in lines[1]
        assert not any("table:" in line for line in lines)
        assert not any("type:" in line for line in lines)

    def test_non_dict_result(self):
        """Verify non-dict result shows as raw value."""
        output = {"raw_output": '{"success": 1, "result": "DDL not available"}'}
        lines = _format_get_table_ddl_output_verbose(output)
        assert any("result: DDL not available" in line for line in lines)

    def test_error_output(self):
        """Verify error is shown for failed DDL retrieval."""
        output = {"raw_output": '{"success": 0, "error": "table not found", "result": null}'}
        lines = _format_get_table_ddl_output_verbose(output)
        assert any("error: table not found" in line for line in lines)

    def test_unparseable_falls_back(self):
        """Verify unparseable output falls back to generic format."""
        lines = _format_get_table_ddl_output_verbose("not json")
        assert len(lines) > 0


# ── _build_get_table_ddl ──────────────────────────────────────────


@pytest.mark.ci
class TestBuildGetTableDdl:
    """Tests for the get_table_ddl builder function."""

    def test_compact_shows_identifier(self):
        """Verify compact mode shows table identifier."""
        a = _make(
            input_data={"function_name": "get_table_ddl"},
            output_data={
                "raw_output": '{"success": 1, "result": '
                '{"identifier": "catalog.db.my_table", "definition": "CREATE TABLE ..."}}'
            },
        )
        tc = _build_get_table_ddl(a, verbose=False)
        assert "catalog.db.my_table" in tc.compact_result

    def test_compact_no_identifier_fallback(self):
        """When only the DDL body is present, show its character count."""
        a = _make(
            input_data={"function_name": "get_table_ddl"},
            output_data={"raw_output": '{"success": 1, "result": {"definition": "CREATE TABLE ..."}}'},
        )
        tc = _build_get_table_ddl(a, verbose=False)
        assert tc.compact_result == "16 chars"

    def test_compact_no_data(self):
        """Verify compact mode produces empty preview with no output."""
        a = _make(input_data={"function_name": "get_table_ddl"})
        tc = _build_get_table_ddl(a, verbose=False)
        assert tc.output_preview == ""

    def test_verbose_shows_ddl(self):
        """Verify verbose mode formats DDL output with markup."""
        a = _make(
            input_data={"function_name": "get_table_ddl", "arguments": {"table_name": "t1"}},
            output_data={
                "raw_output": '{"success": 1, "result": '
                '{"identifier": "db.t1", "table_type": "table", '
                '"definition": "CREATE TABLE t1 (id INT)"}}'
            },
        )
        tc = _build_get_table_ddl(a, verbose=True)
        assert any("table_name" in line for line in tc.args_lines)
        assert any("db.t1" in line for line in tc.output_lines)
        assert any("definition" in line for line in tc.output_lines)


# ── _build_write_file ─────────────────────────────────────────────


@pytest.mark.ci
class TestBuildWriteFile:
    """Tests for the write_file builder function."""

    def test_compact_success_with_result_message(self):
        """Verify compact mode shows 'File written' on success result string."""
        a = _make(
            input_data={"function_name": "write_file"},
            output_data={"raw_output": '{"success": 1, "result": "File written successfully: /tmp/f.sql"}'},
        )
        tc = _build_write_file(a, verbose=False)
        assert "File written" in tc.compact_result

    def test_compact_success_with_success_flag(self):
        """Verify compact mode shows 'File written' when success flag is true."""
        a = _make(
            input_data={"function_name": "write_file"},
            output_data={"raw_output": '{"success": 1, "result": {"path": "/tmp/f.sql"}}'},
        )
        tc = _build_write_file(a, verbose=False)
        assert "File written" in tc.compact_result

    def test_compact_no_data(self):
        """Verify compact mode with no output produces empty preview."""
        a = _make(input_data={"function_name": "write_file"})
        tc = _build_write_file(a, verbose=False)
        assert tc.output_preview == ""

    def test_verbose_truncates_long_content(self):
        """Verify verbose mode truncates content with more than 5 lines."""
        long_content = "\n".join([f"line {i}" for i in range(10)])
        a = _make(
            input_data={
                "function_name": "write_file",
                "arguments": {"path": "/tmp/test.sql", "content": long_content, "file_type": "sql"},
            },
            output_data={"raw_output": '{"success": 1}'},
        )
        tc = _build_write_file(a, verbose=True)
        assert any("/tmp/test.sql" in line for line in tc.args_lines)
        assert any("file_type" in line and "sql" in line for line in tc.args_lines)
        assert any("content" in line and "10 lines" in line for line in tc.args_lines)
        assert any("5 more lines" in line for line in tc.args_lines)

    def test_verbose_short_content_shown_fully(self):
        """Verify verbose mode shows short content fully without truncation."""
        content = "SELECT 1;\nSELECT 2;"
        a = _make(
            input_data={
                "function_name": "write_file",
                "arguments": {"path": "/tmp/q.sql", "content": content},
            },
            output_data={"raw_output": '{"success": 1}'},
        )
        tc = _build_write_file(a, verbose=True)
        assert any("content" in line for line in tc.args_lines)
        assert any("SELECT 1;" in line for line in tc.args_lines)
        assert any("SELECT 2;" in line for line in tc.args_lines)

    def test_verbose_no_arguments(self):
        """Verify verbose mode handles missing arguments gracefully."""
        a = _make(input_data={"function_name": "write_file"})
        tc = _build_write_file(a, verbose=True)
        assert tc.args_lines == []


# ── Builder registry: new tools registered ────────────────────────


@pytest.mark.ci
class TestNewToolsRegistered:
    """Verify newly added tools are registered in the builder."""

    def test_get_table_ddl_registered(self):
        """Verify get_table_ddl is auto-registered."""
        builder = ToolCallContentBuilder()
        assert "get_table_ddl" in builder._registry

    def test_write_file_registered(self):
        """Verify write_file is auto-registered."""
        builder = ToolCallContentBuilder()
        assert "write_file" in builder._registry

    def test_get_table_ddl_dispatches_correctly(self):
        """Verify get_table_ddl dispatched via builder produces correct output."""
        a = _make(
            input_data={"function_name": "get_table_ddl"},
            output_data={
                "raw_output": '{"success": 1, "result": {"identifier": "db.t", "definition": "CREATE TABLE t"}}'
            },
        )
        tc = ToolCallContentBuilder().build(a, verbose=False)
        assert "db.t" in tc.compact_result

    def test_write_file_dispatches_correctly(self):
        """Verify write_file dispatched via builder produces correct output."""
        a = _make(
            input_data={"function_name": "write_file"},
            output_data={"raw_output": '{"success": 1, "result": "File written successfully"}'},
        )
        tc = ToolCallContentBuilder().build(a, verbose=False)
        assert "File written" in tc.compact_result


# ── read_query compact: nested original_rows ──────────────────────


@pytest.mark.ci
class TestBuildReadQueryNestedRows:
    """Tests for read_query compact mode with nested original_rows in result."""

    def test_compact_nested_original_rows(self):
        """Verify compact mode finds original_rows inside nested result dict."""
        a = _make(
            input_data={"function_name": "read_query"},
            output_data={"raw_output": '{"success": 1, "result": {"original_rows": 15, "compressed_data": "a\\n1"}}'},
        )
        tc = _build_read_query(a, verbose=False)
        # Column count is inferred from the compressed_data header line.
        assert tc.compact_result == "15 \u00d7 1 result"

    def test_verbose_shows_only_data(self):
        """Verify verbose mode only shows CSV data preview with markup."""
        a = _make(
            input_data={"function_name": "read_query", "arguments": {"sql": "SELECT 1", "database": "db"}},
            output_data={
                "raw_output": '{"success": 1, "result": '
                '{"original_rows": 2, "original_columns": ["id", "val"], '
                '"compressed_data": "index,id,val\\n0,1,a\\n1,2,b", '
                '"removed_columns": [], "compression_type": "none"}}'
            },
        )
        tc = _build_read_query(a, verbose=True)
        assert any("data" in line for line in tc.output_lines)
        assert any("a" in line for line in tc.output_lines)
        assert not any("rows:" in line for line in tc.output_lines)

    def test_describe_table_verbose_structured(self):
        """Verify describe_table verbose uses structured column display with markup."""
        a = _make(
            input_data={"function_name": "describe_table", "arguments": {"table_name": "t1"}},
            output_data={
                "raw_output": '{"success": 1, "result": {"columns": [{"name": "id", "type": "int", "comment": "pk"}]}}'
            },
        )
        tc = _build_describe_table(a, verbose=True)
        assert any("columns" in line and "(1)" in line for line in tc.output_lines)
        assert any("id" in line and "int" in line and "pk" in line for line in tc.output_lines)


# ── _format_result_only_markup ────────────────────────────────────


@pytest.mark.ci
class TestFormatResultOnlyMarkup:
    """Tests for _format_result_only_markup — skips success/error metadata."""

    def test_skips_success_and_null_error(self):
        output = {"raw_output": '{"success": 1, "error": null, "result": [{"name": "t1"}]}'}
        lines = _format_result_only_markup(output)
        assert not any("success" in line.lower() for line in lines)
        assert not any("error" in line.lower() for line in lines)
        assert any("t1" in line for line in lines)

    def test_shows_error_when_present(self):
        output = {"raw_output": '{"success": 0, "error": "access denied", "result": null}'}
        lines = _format_result_only_markup(output)
        assert any("access denied" in line for line in lines)

    def test_dict_result_shows_keys(self):
        output = {"raw_output": '{"success": 1, "result": {"name": "metric1", "description": "test"}}'}
        lines = _format_result_only_markup(output)
        assert any("name" in line and "metric1" in line for line in lines)
        assert any("description" in line and "test" in line for line in lines)

    def test_string_result(self):
        output = {"raw_output": '{"success": 1, "result": "File created"}'}
        lines = _format_result_only_markup(output)
        assert any("File created" in line for line in lines)

    def test_no_result_key_shows_non_metadata(self):
        output = {"raw_output": '{"success": 1, "message": "done"}'}
        lines = _format_result_only_markup(output)
        assert any("message" in line and "done" in line for line in lines)
        assert not any("success" in line.lower() for line in lines)

    def test_unparseable_falls_back(self):
        lines = _format_result_only_markup("not json")
        assert len(lines) > 0


@pytest.mark.ci
class TestFormatValueMarkup:
    """Tests for _format_value_markup helper."""

    def test_list_of_dicts(self):
        lines = _format_value_markup([{"name": "a"}, {"name": "b"}])
        assert len(lines) == 2
        assert any("a" in line for line in lines)

    def test_dict_with_sql(self):
        lines = _format_value_markup({"sql": "SELECT 1"})
        assert any("SELECT 1" in line for line in lines)

    def test_string_value(self):
        lines = _format_value_markup("hello world")
        assert lines == ["hello world"]

    def test_multiline_string(self):
        lines = _format_value_markup("line1\nline2")
        assert len(lines) == 2


# ── Shared builder patterns ───────────────────────────────────────


@pytest.mark.ci
class TestBuildSimpleList:
    """Tests for _build_simple_list shared builder."""

    def test_compact_with_list_result(self):
        a = _make(
            input_data={"function_name": "test"},
            output_data={"raw_output": '{"success": 1, "result": ["a", "b", "c"]}'},
        )
        tc = _build_simple_list(a, verbose=False, unit="items")
        assert "3 items" in tc.compact_result

    def test_compact_no_list(self):
        a = _make(input_data={"function_name": "test"}, output_data={"raw_output": '{"success": 1}'})
        tc = _build_simple_list(a, verbose=False, unit="items")
        assert tc.output_preview == ""

    def test_verbose(self):
        a = _make(
            input_data={"function_name": "test", "arguments": {"x": 1}},
            output_data={"raw_output": '{"success": 1, "result": ["a"]}'},
        )
        tc = _build_simple_list(a, verbose=True, unit="items")
        assert len(tc.args_lines) >= 1
        assert len(tc.output_lines) > 0


@pytest.mark.ci
class TestBuildGetDetail:
    """Tests for _build_get_detail shared builder."""

    def test_compact_with_name(self):
        a = _make(
            input_data={"function_name": "test"},
            output_data={"raw_output": '{"success": 1, "result": {"name": "my_metric", "description": "desc"}}'},
        )
        tc = _build_get_detail(a, verbose=False)
        assert "my_metric" in tc.compact_result

    def test_compact_no_name(self):
        a = _make(
            input_data={"function_name": "test"},
            output_data={"raw_output": '{"success": 1, "result": {"description": "no name"}}'},
        )
        tc = _build_get_detail(a, verbose=False)
        assert "Retrieved" in tc.compact_result


@pytest.mark.ci
class TestBuildSimpleAction:
    """Tests for _build_simple_action shared builder."""

    def test_compact_success(self):
        a = _make(
            input_data={"function_name": "test"},
            output_data={"raw_output": '{"success": 1, "result": "ok"}'},
        )
        tc = _build_simple_action(a, verbose=False, success_label="Done")
        assert "Done" in tc.compact_result

    def test_compact_no_data(self):
        a = _make(input_data={"function_name": "test"})
        tc = _build_simple_action(a, verbose=False, success_label="Done")
        assert tc.output_preview == ""


@pytest.mark.ci
class TestBuildDocSearchResult:
    """Tests for _build_doc_search_result shared builder."""

    def test_compact_with_doc_count(self):
        a = _make(
            input_data={"function_name": "search_document"},
            output_data={"raw_output": '{"success": 1, "result": {"docs": [], "doc_count": 5}}'},
        )
        tc = _build_doc_search_result(a, verbose=False)
        assert "5 docs found" in tc.compact_result

    def test_compact_fallback_list(self):
        a = _make(
            input_data={"function_name": "search_document"},
            output_data={"result": [1, 2]},
        )
        tc = _build_doc_search_result(a, verbose=False)
        assert "2 docs found" in tc.compact_result


# ── Database tools ────────────────────────────────────────────────


@pytest.mark.ci
class TestBuildListDatabases:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "list_databases"},
            output_data={"raw_output": '{"success": 1, "result": ["db1", "db2"]}'},
        )
        tc = _build_list_databases(a, verbose=False)
        assert "2 databases" in tc.compact_result


@pytest.mark.ci
class TestBuildListSchemas:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "list_schemas"},
            output_data={"raw_output": '{"success": 1, "result": ["public", "dbo"]}'},
        )
        tc = _build_list_schemas(a, verbose=False)
        assert "2 schemas" in tc.compact_result


# ── Context search tools ──────────────────────────────────────────


@pytest.mark.ci
class TestBuildListSubjectTree:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "list_subject_tree"},
            output_data={"raw_output": '{"success": 1, "result": {"domain1": {}, "domain2": {}}}'},
        )
        tc = _build_list_subject_tree(a, verbose=False)
        assert "2 domains" in tc.compact_result

    def test_compact_no_result(self):
        a = _make(input_data={"function_name": "list_subject_tree"})
        tc = _build_list_subject_tree(a, verbose=False)
        assert tc.output_preview == ""

    def test_verbose_tree_structure(self):
        import json

        tree = {
            "Game_Analytics": {
                "User_Classification": {
                    "metrics": ["dau", "retention"],
                    "reference_sql": ["top_users"],
                    "knowledge": ["bitmap_rule"],
                }
            },
            "Sales": {"Revenue": {"metrics": ["gmv"]}},
        }
        a = _make(
            input_data={"function_name": "list_subject_tree"},
            output_data={"raw_output": json.dumps({"success": 1, "result": tree})},
        )
        tc = _build_list_subject_tree(a, verbose=True)
        lines = tc.output_lines
        # Tree connectors present
        assert any("\u251c" in line or "\u2514" in line for line in lines)
        # Domain names
        assert any("Game_Analytics" in line for line in lines)
        assert any("Sales" in line for line in lines)
        # Leaf counts
        assert any("metrics" in line and "2" in line for line in lines)
        assert any("reference_sql" in line and "1" in line for line in lines)
        assert any("knowledge" in line and "1" in line for line in lines)


@pytest.mark.ci
class TestBuildGetMetrics:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "get_metrics"},
            output_data={"raw_output": '{"success": 1, "result": {"name": "revenue", "description": "total"}}'},
        )
        tc = _build_get_metrics(a, verbose=False)
        assert "revenue" in tc.compact_result


@pytest.mark.ci
class TestBuildGetReferenceSQL:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "get_reference_sql"},
            output_data={"raw_output": '{"success": 1, "result": {"name": "top_sales", "sql": "SELECT 1"}}'},
        )
        tc = _build_get_reference_sql(a, verbose=False)
        assert "top_sales" in tc.compact_result


@pytest.mark.ci
class TestBuildSearchSemanticObjects:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "search_semantic_objects"},
            output_data={"result": [{"kind": "table", "name": "t1"}, {"kind": "metric", "name": "m1"}]},
        )
        tc = _build_search_semantic_objects(a, verbose=False)
        assert "2 semantic objects" in tc.compact_result


@pytest.mark.ci
class TestBuildSearchKnowledge:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "search_knowledge"},
            output_data={"result": [{"search_text": "q1", "explanation": "e1"}]},
        )
        tc = _build_search_knowledge(a, verbose=False)
        assert "1 knowledge entry matched" in tc.compact_result


@pytest.mark.ci
class TestBuildGetKnowledge:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "get_knowledge"},
            output_data={"result": {"name": "sla_policy"}},
        )
        tc = _build_get_knowledge(a, verbose=False)
        assert "sla_policy" in tc.compact_result


# ── Semantic tools ────────────────────────────────────────────────


@pytest.mark.ci
class TestBuildListMetricsSemantic:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "list_metrics"},
            output_data={"result": [{"name": "m1"}, {"name": "m2"}, {"name": "m3"}]},
        )
        tc = _build_list_metrics_semantic(a, verbose=False)
        assert "3 metrics" in tc.compact_result


@pytest.mark.ci
class TestBuildGetDimensions:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "get_dimensions"},
            output_data={"raw_output": '{"success": 1, "result": ["dim1", "dim2"]}'},
        )
        tc = _build_get_dimensions(a, verbose=False)
        assert "2 dimensions" in tc.compact_result


@pytest.mark.ci
class TestBuildQueryMetrics:
    def test_compact_with_metadata_rows(self):
        a = _make(
            input_data={"function_name": "query_metrics"},
            output_data={
                "raw_output": '{"success": 1, "result": {"columns": ["a"], '
                '"data": "a\\n1\\n2", "metadata": {"row_count": 2}}}'
            },
        )
        tc = _build_query_metrics(a, verbose=False)
        assert "2 rows" in tc.compact_result

    def test_compact_fallback(self):
        a = _make(
            input_data={"function_name": "query_metrics"},
            output_data={"raw_output": '{"success": 1, "result": {"columns": ["a"], "data": "a\\n1"}}'},
        )
        tc = _build_query_metrics(a, verbose=False)
        assert "Query completed" in tc.compact_result

    def test_verbose_csv_preview(self):
        a = _make(
            input_data={"function_name": "query_metrics", "arguments": {"metrics": ["revenue"]}},
            output_data={
                "raw_output": '{"success": 1, "result": {"columns": ["a", "b"], '
                '"data": "a,b\\n1,2\\n3,4", "metadata": {}}}'
            },
        )
        tc = _build_query_metrics(a, verbose=True)
        assert any("data" in line for line in tc.output_lines)


@pytest.mark.ci
class TestBuildValidateSemantic:
    def test_compact_valid(self):
        a = _make(
            input_data={"function_name": "validate_semantic"},
            output_data={"raw_output": '{"success": 1, "result": {"valid": true, "issues": []}}'},
        )
        tc = _build_validate_semantic(a, verbose=False)
        assert "Valid" in tc.compact_result

    def test_compact_invalid(self):
        a = _make(
            input_data={"function_name": "validate_semantic"},
            output_data={
                "raw_output": '{"success": 1, "result": {"valid": false, "issues": '
                '[{"severity": "error", "message": "bad"}, {"severity": "warn", "message": "meh"}]}}'
            },
        )
        tc = _build_validate_semantic(a, verbose=False)
        assert "2 validation errors" in tc.compact_result


@pytest.mark.ci
class TestBuildAttributionAnalyze:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "attribution_analyze"},
            output_data={
                "raw_output": '{"success": 1, "result": {"dimension_ranking": ["d1", "d2"], '
                '"selected_dimensions": ["d1"], "top_dimension_values": []}}'
            },
        )
        tc = _build_attribution_analyze(a, verbose=False)
        assert "1 dimensions analyzed" in tc.compact_result


# ── Filesystem tools ──────────────────────────────────────────────


@pytest.mark.ci
class TestBuildReadFile:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "read_file"},
            output_data={"raw_output": '{"success": 1, "result": "line1\\nline2\\nline3"}'},
        )
        tc = _build_read_file(a, verbose=False)
        assert "3 lines" in tc.compact_result

    def test_verbose_truncates(self):
        content = "\\n".join([f"line {i}" for i in range(30)])
        a = _make(
            input_data={"function_name": "read_file", "arguments": {"path": "/tmp/f.txt"}},
            output_data={"raw_output": f'{{"success": 1, "result": "{content}"}}'},
        )
        tc = _build_read_file(a, verbose=True)
        assert any("content" in line for line in tc.output_lines)


@pytest.mark.ci
class TestBuildGlob:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "glob"},
            output_data={"raw_output": '{"success": 1, "result": {"files": ["/a.py", "/b.py"], "truncated": false}}'},
        )
        tc = _build_glob(a, verbose=False)
        assert "2 files found" in tc.compact_result


@pytest.mark.ci
class TestBuildGrep:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "grep"},
            output_data={
                "raw_output": '{"success": 1, "result": {"matches": ['
                '{"file": "/a.py", "line": 1, "content": "hello"}], "truncated": false}}'
            },
        )
        tc = _build_grep(a, verbose=False)
        assert "1 matches" in tc.compact_result

    def test_verbose_shows_matches(self):
        a = _make(
            input_data={"function_name": "grep", "arguments": {"pattern": "hello"}},
            output_data={
                "raw_output": '{"success": 1, "result": {"matches": ['
                '{"file": "/a.py", "line": 1, "content": "hello world"}], "truncated": false}}'
            },
        )
        tc = _build_grep(a, verbose=True)
        assert any("/a.py" in line for line in tc.output_lines)


# ── Platform document tools ───────────────────────────────────────


@pytest.mark.ci
class TestBuildListDocumentNav:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "list_document_nav"},
            output_data={
                "raw_output": '{"success": 1, "result": '
                '{"platform": "spark", "version": "3.5", "nav_tree": {}, "total_docs": 42}}'
            },
        )
        tc = _build_list_document_nav(a, verbose=False)
        assert "spark" in tc.compact_result
        assert "42 docs" in tc.compact_result


@pytest.mark.ci
class TestBuildGetDocument:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "get_document"},
            output_data={
                "raw_output": '{"success": 1, "result": '
                '{"platform": "spark", "title": "Overview", "chunk_count": 3, "chunks": []}}'
            },
        )
        tc = _build_get_document(a, verbose=False)
        assert "Overview" in tc.compact_result
        assert "3 chunks" in tc.compact_result


# ── Plan tools ────────────────────────────────────────────────────


@pytest.mark.ci
class TestBuildTodoRead:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "todo_read"},
            output_data={"raw_output": '{"success": 1, "result": {"message": "ok", "lists": [], "total_lists": 3}}'},
        )
        tc = _build_todo_read(a, verbose=False)
        assert "3 todo lists" in tc.compact_result


@pytest.mark.ci
class TestBuildTodoWrite:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "todo_write"},
            output_data={"raw_output": '{"success": 1, "result": {"message": "saved"}}'},
        )
        tc = _build_todo_write(a, verbose=False)
        assert "Todo list updated" in tc.compact_result


@pytest.mark.ci
class TestBuildTodoUpdate:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "todo_update"},
            output_data={
                "raw_output": '{"success": 1, "result": '
                '{"message": "ok", "updated_item": {"id": "1", "content": "task", "status": "completed"}}}'
            },
        )
        tc = _build_todo_update(a, verbose=False)
        assert "completed" in tc.compact_result

    def test_compact_no_status(self):
        a = _make(
            input_data={"function_name": "todo_update"},
            output_data={"raw_output": '{"success": 1, "result": {"message": "ok", "updated_item": {}}}'},
        )
        tc = _build_todo_update(a, verbose=False)
        assert "Updated" in tc.compact_result


# ── Generation tools ──────────────────────────────────────────────


@pytest.mark.ci
class TestBuildCheckExists:
    def test_compact_exists(self):
        a = _make(
            input_data={"function_name": "check_semantic_object_exists"},
            output_data={"raw_output": '{"success": 1, "result": {"exists": true, "name": "t1"}}'},
        )
        tc = _build_check_exists(a, verbose=False)
        assert "Exists" in tc.compact_result

    def test_compact_not_found(self):
        a = _make(
            input_data={"function_name": "check_semantic_object_exists"},
            output_data={"raw_output": '{"success": 1, "result": {"exists": false, "name": "t1"}}'},
        )
        tc = _build_check_exists(a, verbose=False)
        assert "Not found" in tc.compact_result


@pytest.mark.ci
class TestBuildEndGeneration:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "end_semantic_model_generation"},
            output_data={
                "raw_output": '{"success": 1, "result": '
                '{"message": "done", "semantic_model_files": ["a.yml", "b.yml"]}}'
            },
        )
        tc = _build_end_generation(a, verbose=False)
        assert "2 semantic models generated" in tc.compact_result


@pytest.mark.ci
class TestBuildEndMetricGeneration:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "end_metric_generation"},
            output_data={"raw_output": '{"success": 1, "result": {"message": "done"}}'},
        )
        tc = _build_end_metric_generation(a, verbose=False)
        assert "Metric generated" in tc.compact_result


@pytest.mark.ci
class TestBuildGenerateSqlSummaryId:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "generate_sql_summary_id"},
            output_data={"raw_output": '{"success": 1, "result": "abc123"}'},
        )
        tc = _build_generate_sql_summary_id(a, verbose=False)
        assert "ID generated" in tc.compact_result


# ── Date parsing tools ────────────────────────────────────────────


@pytest.mark.ci
class TestBuildParseDates:
    def test_compact_single_date(self):
        """Single extracted date with only a start → show the date itself."""
        a = _make(
            input_data={"function_name": "parse_temporal_expressions"},
            output_data={
                "raw_output": '{"success": 1, "result": {"extracted_dates": '
                '[{"field_name": "order_date", "start_date": "2024-01-01"}], "date_context": "monthly"}}'
            },
        )
        tc = _build_parse_dates(a, verbose=False)
        assert tc.compact_result == "2024-01-01"

    def test_compact_date_range(self):
        """Single extracted date with start + end → show ``start → end``."""
        a = _make(
            input_data={"function_name": "parse_temporal_expressions"},
            output_data={
                "raw_output": '{"success": 1, "result": {"extracted_dates": '
                '[{"start_date": "2024-01-01", "end_date": "2024-12-31"}]}}'
            },
        )
        tc = _build_parse_dates(a, verbose=False)
        assert tc.compact_result == "2024-01-01 \u2192 2024-12-31"

    def test_compact_multiple(self):
        """Multiple extracted dates → fall back to the count summary."""
        a = _make(
            input_data={"function_name": "parse_temporal_expressions"},
            output_data={
                "raw_output": '{"success": 1, "result": {"extracted_dates": '
                '[{"start_date": "2024-01-01"}, {"start_date": "2024-02-01"}]}}'
            },
        )
        tc = _build_parse_dates(a, verbose=False)
        assert tc.compact_result == "2 date expressions"


# ── Semantic model generation tools ───────────────────────────────


@pytest.mark.ci
class TestBuildAnalyzeRelationships:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "analyze_table_relationships"},
            output_data={
                "raw_output": '{"success": 1, "result": {"relationships": ['
                '{"source_table": "a", "target_table": "b"}], "summary": "ok"}}'
            },
        )
        tc = _build_analyze_relationships(a, verbose=False)
        assert "1 relationships found" in tc.compact_result


@pytest.mark.ci
class TestBuildGetMultipleDdl:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "get_multiple_tables_ddl"},
            output_data={
                "raw_output": '{"success": 1, "result": ['
                '{"table_name": "t1", "definition": "CREATE TABLE t1 (id INT)"},'
                '{"table_name": "t2", "definition": "CREATE TABLE t2 (id INT)"}]}'
            },
        )
        tc = _build_get_multiple_ddl(a, verbose=False)
        assert "2 DDLs retrieved" in tc.compact_result

    def test_verbose_shows_ddl(self):
        a = _make(
            input_data={"function_name": "get_multiple_tables_ddl", "arguments": {"tables": ["t1"]}},
            output_data={
                "raw_output": '{"success": 1, "result": ['
                '{"table_name": "t1", "definition": "CREATE TABLE t1 (id INT)"}]}'
            },
        )
        tc = _build_get_multiple_ddl(a, verbose=True)
        assert any("t1" in line for line in tc.output_lines)
        assert any("CREATE TABLE" in line for line in tc.output_lines)

    def test_verbose_shows_error(self):
        a = _make(
            input_data={"function_name": "get_multiple_tables_ddl", "arguments": {"tables": ["t1"]}},
            output_data={"raw_output": '{"success": 1, "result": [{"table_name": "t1", "error": "not found"}]}'},
        )
        tc = _build_get_multiple_ddl(a, verbose=True)
        assert any("not found" in line for line in tc.output_lines)


@pytest.mark.ci
class TestBuildAnalyzeColumns:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "analyze_column_usage_patterns"},
            output_data={
                "raw_output": '{"success": 1, "result": {"column_patterns": '
                '{"col1": {"usage_count": 5}, "col2": {"usage_count": 3}}, "summary": "ok"}}'
            },
        )
        tc = _build_analyze_columns(a, verbose=False)
        assert "2 columns analyzed" in tc.compact_result


# ── Skill tools ───────────────────────────────────────────────────


@pytest.mark.ci
class TestBuildExecuteCommand:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "execute_command"},
            output_data={"raw_output": '{"success": 1, "result": "output text"}'},
        )
        tc = _build_execute_command(a, verbose=False)
        assert "Command executed" in tc.compact_result


@pytest.mark.ci
class TestBuildLoadSkill:
    def test_compact(self):
        a = _make(
            input_data={"function_name": "load_skill"},
            output_data={"raw_output": '{"success": 1, "result": "skill content"}'},
        )
        tc = _build_load_skill(a, verbose=False)
        assert "Skill loaded" in tc.compact_result


class TestBuildAskUser:
    def test_compact_success(self):
        result_json = json.dumps([{"question": "Which DB?", "answer": "MySQL"}])
        a = _make(
            input_data={
                "function_name": "ask_user",
                "arguments": {"questions": [{"question": "Which DB?", "options": ["MySQL", "PostgreSQL"]}]},
            },
            output_data={"raw_output": json.dumps({"success": 1, "result": result_json})},
        )
        tc = _build_ask_user(a, verbose=False)
        assert "1 answer" in tc.compact_result

    def test_compact_error(self):
        a = _make(
            input_data={"function_name": "ask_user", "arguments": {"questions": []}},
            output_data={"raw_output": json.dumps({"success": 0, "error": "questions must be a non-empty list"})},
        )
        tc = _build_ask_user(a, verbose=False)
        assert "non-empty" in tc.compact_result

    def test_verbose_shows_formatted_questions(self):
        result_json = json.dumps(
            [
                {"question": "Which DB?", "answer": "MySQL"},
                {"question": "Time range?", "answer": "Last 7 days"},
            ]
        )
        a = _make(
            input_data={
                "function_name": "ask_user",
                "arguments": {
                    "questions": [
                        {"question": "Which DB?", "options": ["MySQL", "PostgreSQL"]},
                        {"question": "Time range?"},
                    ]
                },
            },
            output_data={"raw_output": json.dumps({"success": 1, "result": result_json})},
        )
        tc = _build_ask_user(a, verbose=True)
        # Args: formatted Q1/Q2 with options
        assert any("Q1" in line for line in tc.args_lines)
        assert any("Which DB?" in line for line in tc.args_lines)
        assert any("options" in line for line in tc.args_lines)
        # Output: Q→A pairs
        assert any("MySQL" in line for line in tc.output_lines)
        assert any("\u2192" in line for line in tc.output_lines)

    def test_verbose_fallback_on_bad_args(self):
        """When arguments can't be parsed, fallback to standard extract_args_markup."""
        a = _make(
            input_data={"function_name": "ask_user", "arguments": "not valid json {{{"},
            output_data={"raw_output": json.dumps({"success": 0, "error": "parse error"})},
        )
        tc = _build_ask_user(a, verbose=True)
        # Should fallback to standard args display, not be empty
        assert len(tc.args_lines) >= 1
        assert len(tc.output_lines) >= 1


# ── Registry completeness ─────────────────────────────────────────


@pytest.mark.ci
class TestAllToolsRegistered:
    """Verify all new tools are registered in the builder."""

    EXPECTED_TOOLS = [
        # Database
        "list_tables",
        "table_overview",
        "describe_table",
        "read_query",
        "query",
        "search_table",
        "get_table_ddl",
        "list_databases",
        "list_schemas",
        # Context search
        "search_metrics",
        "search_reference_sql",
        "search_external_knowledge",
        "search_knowledge",
        "get_knowledge",
        "search_documents",
        "search_document",
        "list_subject_tree",
        "get_metrics",
        "get_reference_sql",
        "search_semantic_objects",
        # Semantic
        "list_metrics",
        "get_dimensions",
        "query_metrics",
        "validate_semantic",
        "attribution_analyze",
        # Filesystem
        "edit_file",
        "write_file",
        "read_file",
        "glob",
        "grep",
        # Platform doc
        "list_document_nav",
        "get_document",
        "web_search_document",
        # Plan
        "todo_read",
        "todo_write",
        "todo_update",
        # Generation
        "check_semantic_object_exists",
        "check_semantic_model_exists",
        "end_semantic_model_generation",
        "end_metric_generation",
        "generate_sql_summary_id",
        # Date
        "parse_temporal_expressions",
        # Semantic model gen
        "analyze_table_relationships",
        "get_multiple_tables_ddl",
        "analyze_column_usage_patterns",
        # Skill
        "execute_command",
        "skill_execute_command",
        "load_skill",
        # Interaction
        "ask_user",
    ]

    def test_all_expected_tools_registered(self):
        builder = ToolCallContentBuilder()
        for tool in self.EXPECTED_TOOLS:
            assert tool in builder._registry, f"{tool} not registered"

    def test_registry_count(self):
        builder = ToolCallContentBuilder()
        assert len(builder._registry) >= len(self.EXPECTED_TOOLS)


# ═══════════════════════════════════════════════════════════════════════════
# Compact-layout helpers and per-tool formatters
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestCompactHelpers:
    """Coverage for the helpers that back the header + └─ compact layout."""

    def test_collapse_whitespace_flattens_newlines_and_tabs(self):
        from datus.cli.action_display.tool_content import _collapse_whitespace

        assert _collapse_whitespace("a\n b\tc\r\nd") == "a b c d"
        assert _collapse_whitespace("   ") == ""

    def test_truncate_middle_short_stays_unchanged(self):
        from datus.cli.action_display.tool_content import _truncate_middle

        assert _truncate_middle("short", max_len=60) == "short"

    def test_truncate_middle_long_gets_ellipsis(self):
        from datus.cli.action_display.tool_content import _truncate_middle

        result = _truncate_middle("x" * 100, max_len=20)
        assert "..." in result
        assert len(result) <= 20

    def test_truncate_middle_flattens_embedded_newlines(self):
        from datus.cli.action_display.tool_content import _truncate_middle

        result = _truncate_middle("SELECT\n a,\n b\n FROM t", max_len=60)
        assert "\n" not in result

    def test_format_positional_picks_first_nonempty_key(self):
        from datus.cli.action_display.tool_content import _format_positional

        assert _format_positional({"x": "", "y": "val"}, "x", "y") == '"val"'
        assert _format_positional({}, "x") == ""
        # None / empty list / empty dict are all treated as missing.
        assert _format_positional({"x": None, "y": [], "z": "v"}, "x", "y", "z") == '"v"'

    def test_format_kw_joins_nonempty_pairs(self):
        from datus.cli.action_display.tool_content import _format_kw

        assert _format_kw({"pattern": "*.py", "path": "src"}, "pattern", "path") == ('pattern: "*.py", path: "src"')
        assert _format_kw({"pattern": "*.py"}, "pattern", "path") == 'pattern: "*.py"'

    def test_fallback_args_summary_limits_pairs(self):
        from datus.cli.action_display.tool_content import _fallback_args_summary

        args = {"a": "1", "b": "2", "c": "3"}
        result = _fallback_args_summary(args, max_kv=2)
        # Order follows dict iteration; only two pairs shown.
        assert result.count(":") == 2

    def test_extract_error_message_reads_error_field(self):
        from datus.cli.action_display.tool_content import _extract_error_message

        assert _extract_error_message({"raw_output": '{"error": "boom"}'}) == "boom"
        assert _extract_error_message({"raw_output": "{}"}) == ""
        assert _extract_error_message(None) == ""

    def test_set_default_args_summary_uses_fallback(self):
        from datus.cli.action_display.tool_content import (
            ToolCallContent,
            set_default_args_summary,
        )

        tc = ToolCallContent(label="x", status_mark="\u2713", duration_str="")
        action = _make(input_data={"function_name": "custom", "arguments": {"pattern": "*.py"}})
        set_default_args_summary(tc, action)
        assert tc.args_summary == 'pattern: "*.py"'

    def test_set_default_args_summary_noop_when_prefilled(self):
        from datus.cli.action_display.tool_content import (
            ToolCallContent,
            set_default_args_summary,
        )

        tc = ToolCallContent(label="x", status_mark="\u2713", duration_str="", args_summary='"preset"')
        action = _make(input_data={"function_name": "custom", "arguments": {"q": "v"}})
        set_default_args_summary(tc, action)
        assert tc.args_summary == '"preset"'

    def test_set_error_as_result_populates_on_failure(self):
        from datus.cli.action_display.tool_content import (
            ToolCallContent,
            set_error_as_result,
        )

        tc = ToolCallContent(label="x", status_mark="\u2717", duration_str="")
        action = _make(output_data={"raw_output": '{"error": "file not found"}'})
        action.status = ActionStatus.FAILED
        set_error_as_result(tc, action)
        assert "file not found" in tc.compact_result

    def test_set_error_as_result_noop_on_success(self):
        from datus.cli.action_display.tool_content import (
            ToolCallContent,
            set_error_as_result,
        )

        tc = ToolCallContent(label="x", status_mark="\u2713", duration_str="")
        action = _make(output_data={"raw_output": '{"error": null}'})
        set_error_as_result(tc, action)
        assert tc.compact_result == ""

    def test_strip_legacy_preview_removes_known_prefixes(self):
        from datus.cli.action_display.tool_content import _strip_legacy_preview

        assert _strip_legacy_preview("\u2713 3 tables") == "3 tables"
        assert _strip_legacy_preview("\u2717 failed") == "failed"
        assert _strip_legacy_preview("") == ""
        assert _strip_legacy_preview("plain text") == "plain text"

    def test_item_name_extracts_from_dict_or_str(self):
        from datus.cli.action_display.tool_content import _item_name

        assert _item_name("orders") == "orders"
        assert _item_name({"name": "n1"}) == "n1"
        assert _item_name({"table_name": "t1"}) == "t1"
        assert _item_name({"other": "x"}) == ""

    def test_fmt_count_with_preview_lists_first_items(self):
        from datus.cli.action_display.tool_content import _fmt_count_with_preview

        result = _fmt_count_with_preview(3, "table", "tables", ["orders", "customers", "products"])
        assert result == "3 tables: orders, customers, products"

    def test_fmt_count_with_preview_adds_ellipsis_when_truncated(self):
        from datus.cli.action_display.tool_content import _fmt_count_with_preview

        result = _fmt_count_with_preview(5, "table", "tables", ["a", "b", "c", "d", "e"], max_show=3)
        assert result == "5 tables: a, b, c, ..."

    def test_fmt_count_with_preview_singular_noun(self):
        from datus.cli.action_display.tool_content import _fmt_count_with_preview

        assert _fmt_count_with_preview(1, "table", "tables", ["solo"]) == "1 table: solo"

    def test_fmt_count_with_preview_without_items(self):
        from datus.cli.action_display.tool_content import _fmt_count_with_preview

        assert _fmt_count_with_preview(7, "file", "files") == "7 files"


@pytest.mark.ci
class TestExtractFileContent:
    """Coverage for the robust file-body extraction used by _build_read_file."""

    def test_none_returns_none(self):
        from datus.cli.action_display.tool_content import _extract_file_content

        assert _extract_file_content(None) is None

    def test_plain_string_is_body(self):
        from datus.cli.action_display.tool_content import _extract_file_content

        assert _extract_file_content("hello\nworld") == "hello\nworld"

    def test_raw_output_dict_result_key(self):
        from datus.cli.action_display.tool_content import _extract_file_content

        out = {"raw_output": {"success": 1, "result": "abc\n"}}
        assert _extract_file_content(out) == "abc\n"

    def test_raw_output_json_string(self):
        from datus.cli.action_display.tool_content import _extract_file_content

        out = {"raw_output": '{"success": 1, "result": "line1\\nline2"}'}
        assert _extract_file_content(out) == "line1\nline2"

    def test_raw_output_plain_string(self):
        from datus.cli.action_display.tool_content import _extract_file_content

        out = {"raw_output": "not json, just content"}
        assert _extract_file_content(out) == "not json, just content"

    def test_top_level_content_key(self):
        from datus.cli.action_display.tool_content import _extract_file_content

        out = {"content": "body"}
        assert _extract_file_content(out) == "body"

    def test_top_level_generic_key_is_ignored(self):
        """Top-level ``text`` / ``data`` / ``output`` often hold wrappers, not bodies."""
        from datus.cli.action_display.tool_content import _extract_file_content

        # ``text`` at the top level must not be mistaken for the file body.
        assert _extract_file_content({"text": '{"success": 1}'}) is None
        assert _extract_file_content({"data": "wrapper"}) is None
        assert _extract_file_content({"output": "wrapper"}) is None

    def test_raw_output_wins_over_top_level_text(self):
        """When both a wrapper ``text`` field and ``raw_output`` are present,
        the real body inside ``raw_output`` takes precedence."""
        from datus.cli.action_display.tool_content import _extract_file_content

        out = {
            "text": '{"success": 1, "result": "envelope-only"}',
            "raw_output": {"success": 1, "result": "real body"},
        }
        assert _extract_file_content(out) == "real body"

    def test_non_dict_non_string_returns_none(self):
        from datus.cli.action_display.tool_content import _extract_file_content

        assert _extract_file_content(42) is None


@pytest.mark.ci
class TestToolArgsFormatters:
    """Spot-check per-tool arg summary formatters registered in the table."""

    def _args(self, **kwargs) -> dict:
        return kwargs

    def test_list_tables_kw_args(self):
        from datus.cli.action_display.tool_content import _TOOL_ARGS_FORMATTERS

        fmt = _TOOL_ARGS_FORMATTERS["list_tables"]
        assert fmt(self._args(catalog="sales")) == 'catalog: "sales"'
        assert fmt(self._args()) == ""

    def test_describe_table_positional(self):
        from datus.cli.action_display.tool_content import _TOOL_ARGS_FORMATTERS

        fmt = _TOOL_ARGS_FORMATTERS["describe_table"]
        assert fmt(self._args(table_name="orders")) == '"orders"'

    def test_grep_kw_args(self):
        from datus.cli.action_display.tool_content import _TOOL_ARGS_FORMATTERS

        fmt = _TOOL_ARGS_FORMATTERS["grep"]
        assert fmt(self._args(pattern="TODO", path="src")) == 'pattern: "TODO", path: "src"'

    def test_list_databases_is_empty(self):
        from datus.cli.action_display.tool_content import _TOOL_ARGS_FORMATTERS

        fmt = _TOOL_ARGS_FORMATTERS["list_databases"]
        assert fmt({}) == ""

    def test_skill_execute_command_kw(self):
        from datus.cli.action_display.tool_content import _TOOL_ARGS_FORMATTERS

        fmt = _TOOL_ARGS_FORMATTERS["skill_execute_command"]
        assert fmt(self._args(skill_name="s", command="c")) == 'skill_name: "s", command: "c"'


@pytest.mark.ci
class TestBuilderPostProcessing:
    """End-to-end check that the registry populates args_summary / compact_result."""

    def test_registered_tool_gets_custom_args_summary(self):
        builder = ToolCallContentBuilder()
        action = _make(
            input_data={"function_name": "describe_table", "arguments": {"table_name": "orders"}},
            output_data={"result": ["c1", "c2"]},
        )
        tc = builder.build(action, verbose=False)
        assert tc.args_summary == '"orders"'
        assert "2 columns" in tc.compact_result

    def test_unknown_tool_falls_back_to_kv_pairs(self):
        builder = ToolCallContentBuilder()
        action = _make(
            input_data={"function_name": "custom_tool", "arguments": {"x": "1", "y": "2"}},
            output_data={"success": True},
        )
        tc = builder.build(action, verbose=False)
        # Fallback: first two k:v pairs joined by ", "
        assert 'x: "1"' in tc.args_summary
        assert 'y: "2"' in tc.args_summary

    def test_failed_action_sets_error_as_compact_result(self):
        builder = ToolCallContentBuilder()
        action = _make(
            input_data={"function_name": "unknown_tool", "arguments": {}},
            output_data={"raw_output": '{"error": "bad thing"}'},
            status=ActionStatus.FAILED,
        )
        tc = builder.build(action, verbose=False)
        assert "bad thing" in tc.compact_result
