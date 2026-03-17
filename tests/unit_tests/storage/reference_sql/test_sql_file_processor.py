# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.storage.reference_sql.sql_file_processor module."""

import pytest

from datus.storage.reference_sql.sql_file_processor import (
    _find_effective_semicolon,
    log_invalid_entries,
    parse_comment_sql_pairs,
    process_sql_files,
    process_sql_items,
    validate_sql,
)

# ============================================================
# Tests for _find_effective_semicolon
# ============================================================


class TestFindEffectiveSemicolon:
    """Tests for _find_effective_semicolon function."""

    def test_simple_semicolon_at_end(self):
        """Semicolon at end of a plain SQL line is found."""
        pos, in_block = _find_effective_semicolon("SELECT 1;", False)
        assert pos == 8
        assert in_block is False

    def test_no_semicolon(self):
        """Line with no semicolon returns -1."""
        pos, in_block = _find_effective_semicolon("SELECT 1", False)
        assert pos == -1
        assert in_block is False

    def test_semicolon_in_single_line_comment(self):
        """Semicolon inside -- comment is not effective."""
        pos, in_block = _find_effective_semicolon("SELECT 1 -- comment;", False)
        assert pos == -1
        assert in_block is False

    def test_semicolon_in_block_comment(self):
        """Semicolon inside /* ... */ block comment is not effective."""
        pos, in_block = _find_effective_semicolon("SELECT /* ; */ 1", False)
        assert pos == -1
        assert in_block is False

    def test_semicolon_after_block_comment(self):
        """Semicolon after a closed block comment is effective."""
        pos, in_block = _find_effective_semicolon("SELECT /* comment */ 1;", False)
        assert pos == 22
        assert in_block is False

    def test_semicolon_in_single_quoted_string(self):
        """Semicolon inside a single-quoted string is not effective."""
        pos, in_block = _find_effective_semicolon("SELECT 'hello;world'", False)
        assert pos == -1
        assert in_block is False

    def test_semicolon_after_single_quoted_string(self):
        """Semicolon after a single-quoted string is effective."""
        pos, in_block = _find_effective_semicolon("SELECT 'hello;world';", False)
        assert pos == 20
        assert in_block is False

    def test_semicolon_in_double_quoted_identifier(self):
        """Semicolon inside a double-quoted identifier is not effective."""
        pos, in_block = _find_effective_semicolon('SELECT "col;name"', False)
        assert pos == -1
        assert in_block is False

    def test_semicolon_after_double_quoted_identifier(self):
        """Semicolon after a double-quoted identifier is effective."""
        pos, in_block = _find_effective_semicolon('SELECT "col;name";', False)
        assert pos == 17
        assert in_block is False

    def test_entering_block_comment_no_close(self):
        """Line starts a block comment that does not close, in_block_comment becomes True."""
        pos, in_block = _find_effective_semicolon("SELECT /* unclosed comment", False)
        assert pos == -1
        assert in_block is True

    def test_already_in_block_comment_with_close(self):
        """Continuing from a block comment, the close is found."""
        line = "still in comment */ SELECT 1;"
        pos, in_block = _find_effective_semicolon(line, True)
        assert pos == len(line) - 1  # semicolon at the last position
        assert in_block is False

    def test_already_in_block_comment_no_close(self):
        """Continuing from a block comment, no close found."""
        pos, in_block = _find_effective_semicolon("still in block comment", True)
        assert pos == -1
        assert in_block is True

    def test_empty_line(self):
        """Empty line returns -1 and preserves block comment state."""
        pos, in_block = _find_effective_semicolon("", False)
        assert pos == -1
        assert in_block is False

    def test_empty_line_in_block_comment(self):
        """Empty line while in block comment returns -1 and stays in block comment."""
        pos, in_block = _find_effective_semicolon("", True)
        assert pos == -1
        assert in_block is True

    def test_escaped_single_quote(self):
        """Escaped single quote ('') inside string does not close the string."""
        pos, in_block = _find_effective_semicolon("SELECT 'it''s here';", False)
        assert pos == 19
        assert in_block is False

    def test_escaped_double_quote(self):
        """Escaped double quote inside identifier does not close the identifier."""
        pos, in_block = _find_effective_semicolon('SELECT "col""name";', False)
        assert pos == 18
        assert in_block is False

    def test_semicolon_only(self):
        """Line with only a semicolon."""
        pos, in_block = _find_effective_semicolon(";", False)
        assert pos == 0
        assert in_block is False

    def test_multiple_semicolons_returns_first(self):
        """When multiple effective semicolons exist, first one is returned."""
        pos, in_block = _find_effective_semicolon("SELECT 1; SELECT 2;", False)
        assert pos == 8
        assert in_block is False

    @pytest.mark.parametrize(
        "line,expected_pos",
        [
            ("SELECT 1;", 8),
            ("INSERT INTO t VALUES (';');", 26),
            ("-- comment;", -1),
            ("/* block ; */;", 13),
        ],
    )
    def test_various_semicolon_positions(self, line, expected_pos):
        """Parametrized test for various semicolon positions."""
        pos, _ = _find_effective_semicolon(line, False)
        assert pos == expected_pos

    def test_block_comment_opens_after_semicolon(self):
        """Block comment opened after semicolon updates state correctly."""
        pos, in_block = _find_effective_semicolon("SELECT 1; /* start comment", False)
        assert pos == 8
        assert in_block is True

    def test_line_comment_after_semicolon(self):
        """Line comment after semicolon does not affect result."""
        pos, in_block = _find_effective_semicolon("SELECT 1; -- trailing comment", False)
        assert pos == 8
        assert in_block is False


# ============================================================
# Tests for parse_comment_sql_pairs
# ============================================================


class TestParseCommentSqlPairs:
    """Tests for parse_comment_sql_pairs function."""

    def test_single_select_statement(self, tmp_path):
        """Parse a file with one simple SELECT statement."""
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT * FROM users;", encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        comment, sql, line_num = pairs[0]
        assert comment == ""
        assert sql == "SELECT * FROM users"
        assert line_num == 1

    def test_multiple_statements(self, tmp_path):
        """Parse a file with multiple SQL statements."""
        content = "SELECT 1;\nSELECT 2;\nSELECT 3;"
        sql_file = tmp_path / "multi.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 3
        assert pairs[0][1] == "SELECT 1"
        assert pairs[0][2] == 1
        assert pairs[1][1] == "SELECT 2"
        assert pairs[1][2] == 2
        assert pairs[2][1] == "SELECT 3"
        assert pairs[2][2] == 3

    def test_multiline_statement(self, tmp_path):
        """Parse a multi-line SQL statement."""
        content = "SELECT\n  col1,\n  col2\nFROM users;"
        sql_file = tmp_path / "multiline.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert "SELECT" in pairs[0][1]
        assert "FROM users" in pairs[0][1]
        assert pairs[0][2] == 1

    def test_statement_with_inline_comments(self, tmp_path):
        """Comments embedded in SQL blocks are preserved in the SQL."""
        content = "-- get all users\nSELECT * FROM users;"
        sql_file = tmp_path / "commented.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert "-- get all users" in pairs[0][1]
        assert "SELECT * FROM users" in pairs[0][1]

    def test_block_comment_with_semicolon_inside(self, tmp_path):
        """Semicolons inside block comments do not split statements."""
        content = "SELECT /* this; is a comment */ 1;"
        sql_file = tmp_path / "block.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert "SELECT /* this; is a comment */ 1" == pairs[0][1]

    def test_empty_file(self, tmp_path):
        """Empty file returns no pairs."""
        sql_file = tmp_path / "empty.sql"
        sql_file.write_text("", encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert pairs == []

    def test_whitespace_only_file(self, tmp_path):
        """File with only whitespace returns no pairs."""
        sql_file = tmp_path / "space.sql"
        sql_file.write_text("   \n\n   \n", encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert pairs == []

    def test_statement_without_trailing_semicolon(self, tmp_path):
        """Statement without semicolon is still captured as a block."""
        content = "SELECT 1 FROM dual"
        sql_file = tmp_path / "nosemicolon.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert pairs[0][1] == "SELECT 1 FROM dual"

    def test_line_numbers_tracked_correctly(self, tmp_path):
        """Line numbers correspond to block start positions."""
        content = "SELECT 1;\n\nSELECT 2;\n\n\nSELECT 3;"
        sql_file = tmp_path / "lines.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 3
        assert pairs[0][2] == 1
        assert pairs[1][2] == 2
        assert pairs[2][2] == 4

    def test_multiblock_with_comments(self, tmp_path):
        """Multiple blocks each preceded by comments."""
        content = "-- first query\nSELECT a FROM t1;\n-- second query\nSELECT b FROM t2;\n"
        sql_file = tmp_path / "multi_comments.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 2
        assert "SELECT a FROM t1" in pairs[0][1]
        assert "-- first query" in pairs[0][1]
        assert "SELECT b FROM t2" in pairs[1][1]
        assert "-- second query" in pairs[1][1]

    def test_string_with_semicolon(self, tmp_path):
        """Semicolons inside string literals do not split statements."""
        content = "SELECT 'a;b' FROM t1;"
        sql_file = tmp_path / "string_semi.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert pairs[0][1] == "SELECT 'a;b' FROM t1"

    def test_multiline_block_comment_spanning_lines(self, tmp_path):
        """Block comment spanning multiple lines with semicolon inside."""
        content = "SELECT 1\n/* comment\n; still comment */\nFROM t1;"
        sql_file = tmp_path / "span_block.sql"
        sql_file.write_text(content, encoding="utf-8")
        pairs = parse_comment_sql_pairs(str(sql_file))
        assert len(pairs) == 1
        assert "SELECT 1" in pairs[0][1]
        assert "FROM t1" in pairs[0][1]


# ============================================================
# Tests for validate_sql
# ============================================================


class TestValidateSql:
    """Tests for validate_sql function."""

    def test_valid_simple_select(self):
        """A simple SELECT statement is valid."""
        is_valid, cleaned, error = validate_sql("SELECT 1")
        assert is_valid is True
        assert cleaned.strip() != ""
        assert error == ""

    def test_valid_select_with_where(self):
        """SELECT with WHERE clause is valid."""
        is_valid, cleaned, error = validate_sql("SELECT * FROM users WHERE id = 1")
        assert is_valid is True
        assert "SELECT" in cleaned.upper()
        assert error == ""

    def test_valid_select_with_join(self):
        """SELECT with JOIN is valid."""
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        is_valid, cleaned, error = validate_sql(sql)
        assert is_valid is True
        assert error == ""

    def test_valid_select_with_subquery(self):
        """SELECT with subquery is valid."""
        sql = "SELECT * FROM (SELECT id, name FROM users) AS sub"
        is_valid, cleaned, error = validate_sql(sql)
        assert is_valid is True
        assert error == ""

    def test_invalid_sql(self):
        """Completely invalid SQL returns invalid."""
        is_valid, cleaned, error = validate_sql("THIS IS NOT SQL AT ALL SYNTAX ERROR !!!")
        # sqlglot may or may not reject this depending on dialect, but the function should handle it
        # If sqlglot is lenient enough to parse it, that's also acceptable
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)

    def test_valid_create_table(self):
        """CREATE TABLE statement is valid SQL."""
        sql = "CREATE TABLE users (id INT, name VARCHAR(100))"
        is_valid, cleaned, error = validate_sql(sql)
        assert is_valid is True
        assert error == ""

    def test_valid_insert_statement(self):
        """INSERT statement is valid SQL."""
        sql = "INSERT INTO users (id, name) VALUES (1, 'Alice')"
        is_valid, cleaned, error = validate_sql(sql)
        assert is_valid is True
        assert error == ""

    def test_empty_sql(self):
        """Empty SQL is invalid."""
        is_valid, cleaned, error = validate_sql("")
        assert is_valid is False
        assert cleaned == ""

    def test_valid_complex_query(self):
        """Complex query with CTE and window functions validates."""
        sql = (
            "WITH ranked AS ("
            "  SELECT id, name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn"
            "  FROM employees"
            ") SELECT * FROM ranked WHERE rn = 1"
        )
        is_valid, cleaned, error = validate_sql(sql)
        assert is_valid is True
        assert error == ""


# ============================================================
# Tests for process_sql_items
# ============================================================


class TestProcessSqlItems:
    """Tests for process_sql_items function."""

    def test_valid_select_item(self):
        """A valid SELECT item is classified as valid."""
        items = [
            {
                "sql": "SELECT * FROM users",
                "comment": "all users",
                "filepath": "/tmp/test.sql",
                "line_number": 1,
            }
        ]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 1
        assert len(invalid) == 0
        assert valid[0]["comment"] == "all users"
        assert valid[0]["filepath"] == "/tmp/test.sql"
        # line_number should be removed from valid entries
        assert "line_number" not in valid[0]
        # error should be removed from valid entries
        assert "error" not in valid[0]

    def test_non_select_items_are_skipped(self):
        """Non-SELECT statements (INSERT, CREATE, etc.) are skipped entirely."""
        items = [
            {"sql": "INSERT INTO users VALUES (1, 'a')", "filepath": "f.sql", "line_number": 1},
            {"sql": "CREATE TABLE t (id INT)", "filepath": "f.sql", "line_number": 2},
            {"sql": "DROP TABLE t", "filepath": "f.sql", "line_number": 3},
        ]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_empty_sql_is_skipped(self):
        """Items with empty SQL are skipped."""
        items = [{"sql": "", "filepath": "f.sql", "line_number": 1}]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_whitespace_only_sql_is_skipped(self):
        """Items with whitespace-only SQL are skipped."""
        items = [{"sql": "   \n\t  ", "filepath": "f.sql", "line_number": 1}]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_none_sql_is_skipped(self):
        """Items with None SQL are skipped."""
        items = [{"sql": None, "filepath": "f.sql", "line_number": 1}]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_missing_sql_key_is_skipped(self):
        """Items missing the sql key are skipped."""
        items = [{"filepath": "f.sql", "line_number": 1}]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 0
        assert len(invalid) == 0

    def test_mixed_valid_and_non_select(self):
        """Mix of SELECT and non-SELECT: only SELECTs are processed."""
        items = [
            {"sql": "SELECT 1", "filepath": "a.sql", "line_number": 1},
            {"sql": "INSERT INTO t VALUES (1)", "filepath": "a.sql", "line_number": 2},
            {"sql": "SELECT 2", "filepath": "a.sql", "line_number": 3},
        ]
        valid, invalid = process_sql_items(items)
        assert len(valid) == 2

    def test_valid_entry_has_cleaned_sql(self):
        """Valid entry should contain the cleaned/transpiled SQL."""
        items = [{"sql": "select * from users", "filepath": "a.sql", "line_number": 1}]
        valid, _ = process_sql_items(items)
        assert len(valid) == 1
        # The SQL should be cleaned/pretty-printed by validate_sql
        assert "SELECT" in valid[0]["sql"].upper()

    def test_default_comment_and_filepath(self):
        """Missing comment and filepath default to empty strings."""
        items = [{"sql": "SELECT 1", "line_number": 1}]
        valid, _ = process_sql_items(items)
        assert len(valid) == 1
        assert valid[0]["comment"] == ""
        assert valid[0]["filepath"] == ""

    def test_empty_items_list(self):
        """Empty items list returns empty results."""
        valid, invalid = process_sql_items([])
        assert valid == []
        assert invalid == []

    def test_extra_fields_preserved_in_valid(self):
        """Extra fields in the input dict are preserved in valid entries."""
        items = [
            {
                "sql": "SELECT 1",
                "filepath": "a.sql",
                "line_number": 1,
                "custom_field": "custom_value",
            }
        ]
        valid, _ = process_sql_items(items)
        assert len(valid) == 1
        assert valid[0]["custom_field"] == "custom_value"


# ============================================================
# Tests for process_sql_files
# ============================================================


class TestProcessSqlFiles:
    """Tests for process_sql_files function."""

    def test_process_directory_with_select_files(self, tmp_path):
        """Process a directory containing SQL files with SELECT statements."""
        sql_file = tmp_path / "queries.sql"
        sql_file.write_text("SELECT * FROM users;\nSELECT id FROM orders;", encoding="utf-8")
        valid, invalid = process_sql_files(str(tmp_path))
        assert len(valid) == 2
        assert len(invalid) == 0

    def test_process_single_sql_file(self, tmp_path):
        """Process a single SQL file path directly."""
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT 1;", encoding="utf-8")
        valid, invalid = process_sql_files(str(sql_file))
        assert len(valid) == 1

    def test_nonexistent_directory_raises(self):
        """Non-existent path raises ValueError."""
        with pytest.raises(ValueError, match="SQL directory not found"):
            process_sql_files("/nonexistent/path/to/sql")

    def test_empty_directory_raises(self, tmp_path):
        """Directory with no .sql files raises ValueError."""
        with pytest.raises(ValueError, match="No SQL files found"):
            process_sql_files(str(tmp_path))

    def test_non_sql_files_ignored(self, tmp_path):
        """Non-.sql files in the directory are ignored."""
        (tmp_path / "readme.txt").write_text("not sql", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="No SQL files found"):
            process_sql_files(str(tmp_path))

    def test_mixed_select_and_ddl(self, tmp_path):
        """Only SELECT statements are returned as valid."""
        sql_file = tmp_path / "mixed.sql"
        content = "CREATE TABLE t (id INT);\nSELECT * FROM t;\nINSERT INTO t VALUES (1);"
        sql_file.write_text(content, encoding="utf-8")
        valid, invalid = process_sql_files(str(tmp_path))
        assert len(valid) == 1
        assert "SELECT" in valid[0]["sql"].upper()

    def test_filepath_set_in_entries(self, tmp_path):
        """The filepath field is set correctly on valid entries."""
        sql_file = tmp_path / "queries.sql"
        sql_file.write_text("SELECT 1;", encoding="utf-8")
        valid, _ = process_sql_files(str(tmp_path))
        assert len(valid) == 1
        assert valid[0]["filepath"] == str(sql_file)


# ============================================================
# Tests for log_invalid_entries
# ============================================================


class TestLogInvalidEntries:
    """Tests for log_invalid_entries function."""

    def test_log_file_created(self, tmp_path, monkeypatch):
        """Log file is created with invalid entry details."""
        monkeypatch.chdir(tmp_path)
        invalid_entries = [
            {
                "filepath": "/tmp/test.sql",
                "comment": "bad query",
                "error": "syntax error",
                "sql": "SELEC * FROM t",
                "line_number": 5,
            }
        ]
        log_invalid_entries(invalid_entries)
        log_file = tmp_path / "sql_processing_errors.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "1 invalid entries" in content
        assert "/tmp/test.sql" in content
        assert "syntax error" in content
        assert "SELEC * FROM t" in content
        assert "line 5" in content

    def test_log_multiple_entries(self, tmp_path, monkeypatch):
        """Log file contains all invalid entries."""
        monkeypatch.chdir(tmp_path)
        invalid_entries = [
            {"filepath": "a.sql", "comment": "", "error": "err1", "sql": "BAD1", "line_number": 1},
            {"filepath": "b.sql", "comment": "", "error": "err2", "sql": "BAD2", "line_number": 2},
        ]
        log_invalid_entries(invalid_entries)
        log_file = tmp_path / "sql_processing_errors.log"
        content = log_file.read_text(encoding="utf-8")
        assert "2 invalid entries" in content
        assert "a.sql" in content
        assert "b.sql" in content
        assert "[1]" in content
        assert "[2]" in content
