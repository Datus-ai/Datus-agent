# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from datus.storage.reference_template.template_file_processor import (
    _find_effective_semicolon_j2,
    _is_block_closing_tag,
    _is_block_opening_tag,
    _update_jinja_state,
    extract_template_parameters,
    log_invalid_entries,
    parse_template_blocks,
    process_template_files,
    process_template_items,
    validate_template,
)
from datus.utils.exceptions import DatusException


class TestExtractTemplateParameters:
    def test_simple_variables(self):
        template = "SELECT * FROM table1 WHERE dt > {{start_date}} AND region = {{region}}"
        params = extract_template_parameters(template)
        assert len(params) == 2
        names = [p["name"] for p in params]
        assert "start_date" in names
        assert "region" in names

    def test_no_variables(self):
        template = "SELECT * FROM table1 WHERE active = 1"
        params = extract_template_parameters(template)
        assert params == []

    def test_duplicate_variables(self):
        template = "SELECT {{col}} FROM table1 WHERE {{col}} IS NOT NULL AND dt = {{dt}}"
        params = extract_template_parameters(template)
        names = [p["name"] for p in params]
        assert "col" in names
        assert "dt" in names
        assert len(params) == 2  # No duplicates

    def test_variables_with_whitespace(self):
        template = "SELECT * FROM t WHERE dt > {{ start_date }} AND dt <= {{ end_date }}"
        params = extract_template_parameters(template)
        names = [p["name"] for p in params]
        assert "start_date" in names
        assert "end_date" in names

    def test_variables_sorted(self):
        template = "{{z_var}} {{a_var}} {{m_var}}"
        params = extract_template_parameters(template)
        names = [p["name"] for p in params]
        assert names == ["a_var", "m_var", "z_var"]

    def test_fallback_regex_on_syntax_error(self):
        """When Jinja2 AST parsing fails, should fall back to regex extraction."""
        # This template has invalid Jinja2 syntax but still has {{ var }} patterns
        template = "SELECT {{col}} FROM t {% if broken"
        params = extract_template_parameters(template)
        names = [p["name"] for p in params]
        assert "col" in names


class TestValidateTemplate:
    def test_valid_template(self):
        template = "SELECT * FROM {{table}} WHERE id = {{id}}"
        is_valid, error = validate_template(template)
        assert is_valid
        assert error == ""

    def test_valid_template_with_blocks(self):
        template = "{% if condition %}SELECT 1{% endif %}"
        is_valid, error = validate_template(template)
        assert is_valid
        assert error == ""

    def test_invalid_template(self):
        template = "{% if condition %}SELECT 1"  # Missing endif
        is_valid, error = validate_template(template)
        assert not is_valid
        assert "syntax error" in error.lower() or "Unexpected" in error

    def test_plain_sql_is_valid(self):
        template = "SELECT col1, col2 FROM table1 WHERE active = 1"
        is_valid, error = validate_template(template)
        assert is_valid


class TestFindEffectiveSemicolonJ2:
    def test_simple_semicolon(self):
        assert _find_effective_semicolon_j2("SELECT 1;", 0, False) == 8

    def test_no_semicolon(self):
        assert _find_effective_semicolon_j2("SELECT 1", 0, False) == -1

    def test_semicolon_in_single_quote(self):
        assert _find_effective_semicolon_j2("WHERE name = 'a;b'", 0, False) == -1

    def test_semicolon_after_single_quote(self):
        assert _find_effective_semicolon_j2("WHERE name = 'ab';", 0, False) == 17

    def test_semicolon_in_double_quote(self):
        assert _find_effective_semicolon_j2('WHERE "col;name" = 1', 0, False) == -1

    def test_semicolon_after_double_quote(self):
        assert _find_effective_semicolon_j2('WHERE "col" = 1;', 0, False) == 15

    def test_sql_line_comment(self):
        assert _find_effective_semicolon_j2("SELECT 1 -- comment; here", 0, False) == -1

    def test_in_jinja_comment(self):
        assert _find_effective_semicolon_j2("anything;", 0, True) == -1

    def test_in_jinja_block(self):
        assert _find_effective_semicolon_j2("anything;", 1, False) == -1

    def test_jinja_expression_then_semicolon(self):
        assert _find_effective_semicolon_j2("WHERE x = {{val}};", 0, False) >= 0

    def test_jinja_block_tag_then_semicolon(self):
        pos = _find_effective_semicolon_j2("{% if x %}SELECT 1;", 0, False)
        assert pos >= 0  # Semicolon after closed block tag

    def test_unclosed_jinja_expression(self):
        assert _find_effective_semicolon_j2("WHERE x = {{val", 0, False) == -1

    def test_unclosed_jinja_block(self):
        assert _find_effective_semicolon_j2("{% if x", 0, False) == -1

    def test_unclosed_jinja_comment(self):
        assert _find_effective_semicolon_j2("{# comment without end", 0, False) == -1

    def test_jinja_comment_then_semicolon(self):
        pos = _find_effective_semicolon_j2("{# comment #};", 0, False)
        assert pos >= 0

    def test_semicolon_after_closing_single_quote(self):
        assert _find_effective_semicolon_j2("WHERE name = 'abc';", 0, False) == 18

    def test_semicolon_after_closing_double_quote(self):
        assert _find_effective_semicolon_j2('WHERE "col" = 1;', 0, False) == 15


class TestUpdateJinjaState:
    def test_no_jinja(self):
        depth, in_comment = _update_jinja_state("SELECT 1", 0, False)
        assert depth == 0
        assert in_comment is False

    def test_open_if_block(self):
        depth, in_comment = _update_jinja_state("{% if condition %}", 0, False)
        assert depth == 1
        assert in_comment is False

    def test_close_if_block(self):
        depth, in_comment = _update_jinja_state("{% endif %}", 1, False)
        assert depth == 0
        assert in_comment is False

    def test_open_for_block(self):
        depth, in_comment = _update_jinja_state("{% for x in items %}", 0, False)
        assert depth == 1

    def test_nested_blocks(self):
        depth, _ = _update_jinja_state("{% if x %}{% for y in z %}", 0, False)
        assert depth == 2

    def test_jinja_comment_start(self):
        depth, in_comment = _update_jinja_state("{# this is a comment #}", 0, False)
        assert in_comment is False  # Comment opened and closed on same line

    def test_jinja_comment_unclosed(self):
        depth, in_comment = _update_jinja_state("{# unclosed comment", 0, False)
        assert in_comment is True

    def test_jinja_comment_continuation(self):
        depth, in_comment = _update_jinja_state("still in comment #}", 0, True)
        assert in_comment is False

    def test_jinja_comment_continuation_still_open(self):
        depth, in_comment = _update_jinja_state("still in comment", 0, True)
        assert in_comment is True

    def test_unclosed_block_tag(self):
        # {% without closing %} on the line
        depth, in_comment = _update_jinja_state("{% if condition", 0, False)
        # Should increment because it skips past (i += 2)
        assert depth == 0  # No closing %} so no tag content parsed

    def test_close_below_zero(self):
        depth, _ = _update_jinja_state("{% endif %}", 0, False)
        assert depth == 0  # max(0, -1) = 0


class TestParseTemplateBlocks:
    def test_single_template_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write("SELECT * FROM table1 WHERE dt > {{start_date}}")
            f.flush()
            try:
                blocks = parse_template_blocks(f.name)
                assert len(blocks) == 1
                assert "{{start_date}}" in blocks[0][1]
            finally:
                os.unlink(f.name)

    def test_multi_template_file_with_semicolons(self):
        content = (
            "SELECT * FROM table1 WHERE dt > {{start_date}};\nSELECT count(*) FROM table2 WHERE region = {{region}}"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                blocks = parse_template_blocks(f.name)
                assert len(blocks) == 2
                assert "{{start_date}}" in blocks[0][1]
                assert "{{region}}" in blocks[1][1]
            finally:
                os.unlink(f.name)

    def test_template_with_jinja_block(self):
        content = "SELECT dt, col1\nFROM table1\n{% if has_filter %}\nWHERE region = {{region}}\n{% endif %}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                blocks = parse_template_blocks(f.name)
                assert len(blocks) == 1
                assert "{% if has_filter %}" in blocks[0][1]
            finally:
                os.unlink(f.name)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write("")
            f.flush()
            try:
                blocks = parse_template_blocks(f.name)
                assert len(blocks) == 0
            finally:
                os.unlink(f.name)

    def test_semicolon_inside_jinja_block(self):
        """Semicolons inside Jinja2 blocks should not split templates."""
        content = "{% for i in items %}\nSELECT {{i}};\n{% endfor %}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                blocks = parse_template_blocks(f.name)
                # The semicolon is inside a for block, so it should be one template
                assert len(blocks) == 1
            finally:
                os.unlink(f.name)

    def test_trailing_semicolon_stripped(self):
        content = "SELECT * FROM t;"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                blocks = parse_template_blocks(f.name)
                assert len(blocks) == 1
                assert not blocks[0][1].endswith(";")
            finally:
                os.unlink(f.name)

    def test_gbk_encoding(self):
        """Files with GBK encoding should be readable."""
        with tempfile.NamedTemporaryFile(suffix=".j2", delete=False) as f:
            content = "SELECT * FROM t WHERE name = '测试'"
            f.write(content.encode("gbk"))
            f.flush()
            try:
                blocks = parse_template_blocks(f.name)
                assert len(blocks) == 1
                assert "测试" in blocks[0][1]
            finally:
                os.unlink(f.name)


class TestProcessTemplateItems:
    def test_valid_items(self):
        items = [
            {
                "template": "SELECT * FROM t WHERE dt > {{start_date}}",
                "filepath": "/tmp/test.j2",
                "comment": "",
                "line_number": 1,
            }
        ]
        valid, invalid = process_template_items(items)
        assert len(valid) == 1
        assert len(invalid) == 0
        assert "parameters" in valid[0]
        params = json.loads(valid[0]["parameters"])
        assert any(p["name"] == "start_date" for p in params)

    def test_invalid_items(self):
        items = [
            {
                "template": "{% if x %}SELECT 1",  # Invalid J2
                "filepath": "/tmp/bad.j2",
                "comment": "",
                "line_number": 1,
            }
        ]
        valid, invalid = process_template_items(items)
        assert len(valid) == 0
        assert len(invalid) == 1
        assert "error" in invalid[0]

    def test_empty_template_skipped(self):
        items = [
            {
                "template": "",
                "filepath": "/tmp/empty.j2",
                "comment": "",
                "line_number": 1,
            }
        ]
        valid, invalid = process_template_items(items)
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_line_number_removed_from_valid(self):
        items = [
            {
                "template": "SELECT 1",
                "filepath": "/tmp/test.j2",
                "comment": "test",
                "line_number": 5,
            }
        ]
        valid, _ = process_template_items(items)
        assert "line_number" not in valid[0]

    def test_error_field_removed_from_valid(self):
        items = [
            {
                "template": "SELECT 1",
                "filepath": "/tmp/test.j2",
                "comment": "",
                "line_number": 1,
                "error": "previous error",
            }
        ]
        valid, _ = process_template_items(items)
        assert "error" not in valid[0]


class TestProcessTemplateFiles:
    def test_directory_with_j2_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "query1.j2"), "w") as f:
                f.write("SELECT * FROM users WHERE created_at > {{start_date}}")
            with open(os.path.join(tmpdir, "query2.jinja2"), "w") as f:
                f.write("SELECT count(*) FROM orders WHERE region = {{region}}")

            valid, invalid = process_template_files(tmpdir)
            assert len(valid) == 2
            assert len(invalid) == 0

    def test_single_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as f:
            f.write("SELECT * FROM t WHERE x = {{val}}")
            f.flush()
            try:
                valid, invalid = process_template_files(f.name)
                assert len(valid) == 1
            finally:
                os.unlink(f.name)

    def test_no_template_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(DatusException):
                process_template_files(tmpdir)

    def test_nonexistent_directory_raises(self):
        with pytest.raises(DatusException):
            process_template_files("/nonexistent/path")

    def test_non_j2_file_raises(self):
        """A file with non-.j2 extension should raise DatusException."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
            f.write("SELECT 1")
            f.flush()
            try:
                with pytest.raises(DatusException):
                    process_template_files(f.name)
            finally:
                os.unlink(f.name)

    @patch("datus.storage.reference_template.template_file_processor.log_invalid_entries")
    def test_directory_with_invalid_templates(self, mock_log):
        """Invalid templates should be reported in invalid list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "good.j2"), "w") as f:
                f.write("SELECT {{val}}")
            with open(os.path.join(tmpdir, "bad.j2"), "w") as f:
                f.write("{% if broken")

            valid, invalid = process_template_files(tmpdir)
            assert len(valid) == 1
            assert len(invalid) == 1
            mock_log.assert_called_once()


class TestLogInvalidEntries:
    def test_log_invalid_entries(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        entries = [
            {
                "filepath": "/tmp/bad.j2",
                "comment": "test",
                "error": "syntax error",
                "template": "{% if broken",
                "line_number": 1,
            }
        ]
        log_invalid_entries(entries)
        log_file = tmp_path / "template_processing_errors.log"
        assert log_file.exists()
        with open(log_file) as f:
            content = f.read()
        assert "syntax error" in content
        assert "/tmp/bad.j2" in content

    def test_log_without_line_number(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        entries = [
            {
                "filepath": "/tmp/bad.j2",
                "comment": "",
                "error": "error",
                "template": "broken",
            }
        ]
        log_invalid_entries(entries)
        assert (tmp_path / "template_processing_errors.log").exists()


class TestJinjaBlockTags:
    def test_opening_tags(self):
        assert _is_block_opening_tag("for item in items")
        assert _is_block_opening_tag("if condition")
        assert _is_block_opening_tag("block content")
        assert _is_block_opening_tag("macro render()")
        assert _is_block_opening_tag("call something()")
        assert _is_block_opening_tag("filter upper")
        assert not _is_block_opening_tag("set x = 1")  # Assignment form
        assert _is_block_opening_tag("set")  # Block form (no =)

    def test_closing_tags(self):
        assert _is_block_closing_tag("endfor")
        assert _is_block_closing_tag("endif")
        assert _is_block_closing_tag("endblock")
        assert _is_block_closing_tag("endmacro")
        assert _is_block_closing_tag("endcall")
        assert _is_block_closing_tag("endfilter")
        assert _is_block_closing_tag("endset")
        assert not _is_block_closing_tag("for item in items")
        assert not _is_block_closing_tag("if condition")

    def test_empty_tag(self):
        assert not _is_block_opening_tag("")
        assert not _is_block_closing_tag("")

    def test_unknown_tag(self):
        assert not _is_block_opening_tag("extends 'base.html'")
        assert not _is_block_closing_tag("include 'header.html'")


class TestAnalyzeTemplateParameters:
    """Tests for analyze_template_parameters — the sqlglot-based enrichment function."""

    def test_no_params_returns_empty(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        result = analyze_template_parameters("SELECT 1 FROM t")
        assert result == []

    def test_dimension_param_in_quoted_context(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM orders WHERE region = '{{region}}'"
        result = analyze_template_parameters(sql)
        assert len(result) == 1
        entry = result[0]
        assert entry["name"] == "region"
        assert entry["type"] == "dimension"

    def test_number_param_in_limit(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM orders LIMIT {{n}}"
        result = analyze_template_parameters(sql)
        assert len(result) == 1
        assert result[0]["name"] == "n"
        assert result[0]["type"] == "number"

    def test_number_param_in_comparison(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM t WHERE score > {{threshold}}"
        result = analyze_template_parameters(sql)
        assert len(result) == 1
        assert result[0]["name"] == "threshold"
        assert result[0]["type"] == "number"

    def test_keyword_param_after_order_by(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM t ORDER BY score {{sort_dir}}"
        result = analyze_template_parameters(sql)
        assert len(result) == 1
        entry = result[0]
        assert entry["name"] == "sort_dir"
        assert entry["type"] == "keyword"
        assert "ASC" in entry.get("allowed_values", [])
        assert "DESC" in entry.get("allowed_values", [])

    def test_column_param_in_group_by(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT {{col}}, count(*) FROM t GROUP BY {{col}}"
        result = analyze_template_parameters(sql)
        names = {e["name"] for e in result}
        assert "col" in names
        col_entry = next(e for e in result if e["name"] == "col")
        assert col_entry["type"] == "column"

    def test_column_param_in_select(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT {{col}} FROM t"
        result = analyze_template_parameters(sql)
        assert len(result) == 1
        assert result[0]["name"] == "col"
        assert result[0]["type"] == "column"

    def test_unknown_param_unquoted_no_special_context(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM t WHERE {{something}} = 1"
        result = analyze_template_parameters(sql)
        assert len(result) == 1
        assert result[0]["type"] == "unknown"

    def test_dimension_param_resolves_column_ref_via_sqlglot(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM schools WHERE county = '{{county}}'"
        result = analyze_template_parameters(sql, dialect="sqlite")
        assert len(result) == 1
        entry = result[0]
        assert entry["name"] == "county"
        assert entry["type"] == "dimension"
        assert "column_ref" in entry
        assert "county" in entry["column_ref"]

    def test_multiple_params_mixed_types(self):
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM t WHERE region = '{{region}}' ORDER BY score {{dir}} LIMIT {{n}}"
        result = analyze_template_parameters(sql)
        by_name = {e["name"]: e for e in result}
        assert by_name["region"]["type"] == "dimension"
        assert by_name["dir"]["type"] == "keyword"
        assert by_name["n"]["type"] == "number"

    def test_dialect_none_uses_fallback_chain(self):
        """When dialect=None, should still parse without error."""
        from datus.storage.reference_template.template_file_processor import analyze_template_parameters

        sql = "SELECT * FROM t WHERE x = '{{val}}'"
        result = analyze_template_parameters(sql, dialect=None)
        assert len(result) == 1
        assert result[0]["type"] == "dimension"


class TestResolveDimensionColumns:
    """Tests for _resolve_dimension_columns — the sqlglot AST resolution function."""

    def test_resolves_simple_where_eq(self):
        from datus.storage.reference_template.template_file_processor import _resolve_dimension_columns

        sql = "SELECT * FROM schools WHERE county = '{{county}}'"
        refs, tables = _resolve_dimension_columns(sql, {"county"}, dialect="sqlite")
        assert "county" in refs
        assert "county" in refs["county"]
        assert "schools" in tables

    def test_resolves_aliased_table(self):
        from datus.storage.reference_template.template_file_processor import _resolve_dimension_columns

        sql = "SELECT * FROM schools s WHERE s.county = '{{county}}'"
        refs, tables = _resolve_dimension_columns(sql, {"county"}, dialect="sqlite")
        assert "county" in refs
        assert "schools" in refs["county"]

    def test_resolves_multiple_tables(self):
        from datus.storage.reference_template.template_file_processor import _resolve_dimension_columns

        sql = "SELECT * FROM a JOIN b ON a.id = b.id WHERE a.name = '{{name}}'"
        refs, tables = _resolve_dimension_columns(sql, {"name"}, dialect="sqlite")
        assert "a" in tables and "b" in tables

    def test_returns_empty_on_parse_failure(self):
        from datus.storage.reference_template.template_file_processor import _resolve_dimension_columns

        # A completely invalid SQL that no dialect can parse
        refs, tables = _resolve_dimension_columns("NOT SQL AT ALL !!!###@@@", {"p"}, dialect="sqlite")
        assert refs == {}

    def test_empty_quoted_params(self):
        from datus.storage.reference_template.template_file_processor import _resolve_dimension_columns

        sql = "SELECT * FROM t WHERE x = 1"
        refs, tables = _resolve_dimension_columns(sql, set(), dialect="sqlite")
        assert refs == {}
        assert "t" in tables

    def test_single_from_table_used_when_no_alias(self):
        from datus.storage.reference_template.template_file_processor import _resolve_dimension_columns

        sql = "SELECT * FROM orders WHERE status = '{{status}}'"
        refs, tables = _resolve_dimension_columns(sql, {"status"}, dialect="sqlite")
        assert "status" in refs
        assert "orders" in refs["status"]
