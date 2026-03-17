# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Extended unit tests for datus/utils/sql_utils.py - covering uncovered lines."""

from unittest.mock import patch

import pytest

from datus.tools.db_tools import connector_registry
from datus.utils.constants import DBType, SQLType
from datus.utils.sql_utils import (
    _fallback_sql_type,
    _first_statement,
    _is_escaped,
    _match_dollar_tag,
    _metadata_pattern,
    format_sql_to_pretty,
    normalize_sql,
    parse_context_switch,
    parse_dialect,
    parse_metadata_from_ddl,
    parse_read_dialect,
    parse_sql_type,
    parse_table_name_parts,
    parse_table_names_parts,
    strip_sql_comments,
)


@pytest.fixture(autouse=True)
def _register_test_capabilities():
    """Register capabilities for dialects used in tests."""
    connector_registry.register_handlers("mysql", capabilities={"database"})
    connector_registry.register_handlers("starrocks", capabilities={"catalog", "database"})
    connector_registry.register_handlers("oracle", capabilities={"database", "schema"})
    connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
    connector_registry.register_handlers("snowflake", capabilities={"catalog", "database", "schema"})
    yield


# ---------------------------------------------------------------------------
# parse_read_dialect
# ---------------------------------------------------------------------------


class TestParseReadDialect:
    def test_postgres(self):
        assert parse_read_dialect("postgres") == "postgres"
        assert parse_read_dialect("postgresql") == "postgres"
        assert parse_read_dialect("redshift") == "postgres"
        assert parse_read_dialect("greenplum") == "postgres"

    def test_hive(self):
        assert parse_read_dialect("spark") == "hive"
        assert parse_read_dialect("databricks") == "hive"
        assert parse_read_dialect("hive") == "hive"
        assert parse_read_dialect("starrocks") == "hive"

    def test_tsql(self):
        assert parse_read_dialect("mssql") == "tsql"
        assert parse_read_dialect("sqlserver") == "tsql"

    def test_passthrough(self):
        assert parse_read_dialect("snowflake") == "snowflake"
        assert parse_read_dialect("mysql") == "mysql"

    def test_empty_string(self):
        assert parse_read_dialect("") == ""

    def test_whitespace_trimmed(self):
        assert parse_read_dialect("  postgres  ") == "postgres"


# ---------------------------------------------------------------------------
# parse_dialect
# ---------------------------------------------------------------------------


class TestParseDialect:
    def test_postgres(self):
        assert parse_dialect("postgres") == "postgres"
        assert parse_dialect("postgresql") == "postgres"

    def test_tsql(self):
        assert parse_dialect("mssql") == "tsql"
        assert parse_dialect("sqlserver") == "tsql"

    def test_passthrough(self):
        assert parse_dialect("snowflake") == "snowflake"
        assert parse_dialect("mysql") == "mysql"
        assert parse_dialect("duckdb") == "duckdb"


# ---------------------------------------------------------------------------
# strip_sql_comments
# ---------------------------------------------------------------------------


class TestStripSqlComments:
    def test_removes_block_comments(self):
        sql = "SELECT /* this is a comment */ 1"
        result = strip_sql_comments(sql)
        assert "comment" not in result
        assert "SELECT" in result

    def test_removes_line_comments(self):
        sql = "SELECT 1 -- this is a comment\nFROM t"
        result = strip_sql_comments(sql)
        assert "comment" not in result
        assert "FROM t" in result

    def test_multiline_block_comment(self):
        sql = "SELECT /* multi\nline\ncomment */ 1"
        result = strip_sql_comments(sql)
        assert "multi" not in result


# ---------------------------------------------------------------------------
# _is_escaped
# ---------------------------------------------------------------------------


class TestIsEscaped:
    def test_not_escaped(self):
        assert _is_escaped("abc'def", 3) is False

    def test_escaped_by_one_backslash(self):
        assert _is_escaped("abc\\'def", 4) is True

    def test_escaped_by_two_backslashes_not_escaped(self):
        # Two backslashes before = not escaped (even number)
        assert _is_escaped("abc\\\\'def", 5) is False

    def test_index_at_start(self):
        assert _is_escaped("'test", 0) is False


# ---------------------------------------------------------------------------
# _match_dollar_tag
# ---------------------------------------------------------------------------


class TestMatchDollarTag:
    def test_matches_simple_tag(self):
        text = "$$hello$$"
        tag = _match_dollar_tag(text, 0)
        assert tag == "$$"

    def test_matches_named_tag(self):
        text = "$body$hello$body$"
        tag = _match_dollar_tag(text, 0)
        assert tag == "$body$"

    def test_no_match(self):
        text = "abc"
        assert _match_dollar_tag(text, 0) is None

    def test_matches_at_offset(self):
        text = "  $$content$$"
        assert _match_dollar_tag(text, 2) == "$$"


# ---------------------------------------------------------------------------
# _first_statement - extended cases
# ---------------------------------------------------------------------------


class TestFirstStatementExtended:
    def test_backtick_quotes(self):
        sql = "SELECT `col;name` FROM t; SELECT 2"
        result = _first_statement(sql)
        assert result == "SELECT `col;name` FROM t"

    def test_bracket_quotes(self):
        sql = "SELECT [col;name] FROM t; SELECT 2"
        result = _first_statement(sql)
        assert result == "SELECT [col;name] FROM t"

    def test_escaped_single_quote(self):
        sql = "INSERT INTO t VALUES ('it\\'s ok'); SELECT 1"
        result = _first_statement(sql)
        assert "INSERT" in result

    def test_double_single_quote_escape(self):
        sql = "INSERT INTO t VALUES ('it''s ok'); SELECT 1"
        result = _first_statement(sql)
        assert "INSERT INTO t VALUES" in result

    def test_double_double_quote_escape(self):
        sql = 'INSERT INTO t VALUES ("it""s ok"); SELECT 1'
        result = _first_statement(sql)
        assert "INSERT" in result

    def test_double_backtick_escape(self):
        sql = "SELECT `col``name` FROM t; SELECT 2"
        result = _first_statement(sql)
        assert "SELECT" in result

    def test_empty_sql(self):
        assert _first_statement("") == ""

    def test_no_semicolon(self):
        sql = "SELECT 1"
        assert _first_statement(sql) == "SELECT 1"


# ---------------------------------------------------------------------------
# _fallback_sql_type
# ---------------------------------------------------------------------------


class TestFallbackSqlType:
    def test_select(self):
        assert _fallback_sql_type("SELECT * FROM t") == SQLType.SELECT

    def test_insert(self):
        assert _fallback_sql_type("INSERT INTO t VALUES (1)") == SQLType.INSERT

    def test_update(self):
        assert _fallback_sql_type("UPDATE t SET x=1") == SQLType.UPDATE

    def test_delete(self):
        assert _fallback_sql_type("DELETE FROM t") == SQLType.DELETE

    def test_ddl_create(self):
        assert _fallback_sql_type("CREATE TABLE t (id INT)") == SQLType.DDL

    def test_ddl_drop(self):
        assert _fallback_sql_type("DROP TABLE t") == SQLType.DDL

    def test_ddl_alter(self):
        assert _fallback_sql_type("ALTER TABLE t ADD COLUMN c INT") == SQLType.DDL

    def test_show(self):
        assert _fallback_sql_type("SHOW TABLES") == SQLType.METADATA_SHOW

    def test_explain(self):
        assert _fallback_sql_type("EXPLAIN SELECT * FROM t") == SQLType.EXPLAIN

    def test_use(self):
        assert _fallback_sql_type("USE mydb") == SQLType.CONTENT_SET

    def test_with_select(self):
        result = _fallback_sql_type("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert result == SQLType.SELECT

    def test_with_insert(self):
        result = _fallback_sql_type("WITH cte AS (SELECT 1) INSERT INTO t SELECT * FROM cte")
        assert result == SQLType.INSERT

    def test_empty_string(self):
        assert _fallback_sql_type("") is None

    def test_unknown_keyword(self):
        assert _fallback_sql_type("FOOBAR * FROM t") is None

    def test_values_keyword(self):
        assert _fallback_sql_type("VALUES (1, 2, 3)") == SQLType.SELECT

    def test_replace_keyword(self):
        assert _fallback_sql_type("REPLACE INTO t VALUES (1)") == SQLType.INSERT


# ---------------------------------------------------------------------------
# _metadata_pattern
# ---------------------------------------------------------------------------


class TestMetadataPattern:
    def test_matches_show(self):
        pattern = _metadata_pattern()
        assert pattern.match("SHOW TABLES")

    def test_matches_describe(self):
        pattern = _metadata_pattern()
        assert pattern.match("DESCRIBE t")

    def test_matches_pragma(self):
        pattern = _metadata_pattern()
        assert pattern.match("PRAGMA table_info(t)")

    def test_no_match_select(self):
        pattern = _metadata_pattern()
        assert not pattern.match("SELECT 1")

    def test_singleton(self):
        p1 = _metadata_pattern()
        p2 = _metadata_pattern()
        assert p1 is p2


# ---------------------------------------------------------------------------
# parse_sql_type - additional cases
# ---------------------------------------------------------------------------


class TestParseSqlTypeExtended:
    def test_empty_string(self):
        assert parse_sql_type("", "duckdb") == SQLType.UNKNOWN

    def test_none_like_input(self):
        assert parse_sql_type(None, "duckdb") == SQLType.UNKNOWN

    def test_only_whitespace(self):
        assert parse_sql_type("   ", "duckdb") == SQLType.UNKNOWN

    def test_insert_statement(self):
        assert parse_sql_type("INSERT INTO t VALUES (1)", "duckdb") == SQLType.INSERT

    def test_update_statement(self):
        assert parse_sql_type("UPDATE t SET x = 1", "duckdb") == SQLType.UPDATE

    def test_delete_statement(self):
        assert parse_sql_type("DELETE FROM t WHERE id = 1", "duckdb") == SQLType.DELETE

    def test_create_table(self):
        assert parse_sql_type("CREATE TABLE t (id INT)", "duckdb") == SQLType.DDL

    def test_drop_table(self):
        assert parse_sql_type("DROP TABLE t", "duckdb") == SQLType.DDL

    def test_alter_table(self):
        assert parse_sql_type("ALTER TABLE t ADD COLUMN c INT", "duckdb") == SQLType.DDL

    def test_use_statement_mysql(self):
        assert parse_sql_type("USE mydb", "mysql") == SQLType.CONTENT_SET

    def test_describe_duckdb(self):
        result = parse_sql_type("DESCRIBE t", "duckdb")
        assert result == SQLType.METADATA_SHOW

    def test_pragma_sqlite(self):
        result = parse_sql_type("PRAGMA table_info(t)", DBType.SQLITE)
        assert result == SQLType.METADATA_SHOW

    def test_sql_with_comments(self):
        sql = "-- This is a comment\nSELECT * FROM t"
        assert parse_sql_type(sql, "duckdb") == SQLType.SELECT

    def test_fallback_when_parse_exception(self):
        with patch("datus.utils.sql_utils.sqlglot.parse_one", side_effect=Exception("parse error")):
            result = parse_sql_type("SELECT * FROM t", "duckdb")
        # Falls back to keyword-based detection
        assert result == SQLType.SELECT

    def test_starrocks_metadata_returns_none_from_parser(self):
        with patch("datus.utils.sql_utils.sqlglot.parse_one", return_value=None):
            result = parse_sql_type("SHOW DATABASES", "starrocks")
        assert result == SQLType.METADATA_SHOW


# ---------------------------------------------------------------------------
# normalize_sql
# ---------------------------------------------------------------------------


class TestNormalizeSql:
    def test_removes_newlines(self):
        sql = "SELECT *\nFROM t\nWHERE id = 1"
        result = normalize_sql(sql)
        assert "\n" not in result
        assert "SELECT * FROM t WHERE id = 1" == result

    def test_removes_tabs(self):
        sql = "SELECT\t*\tFROM\tt"
        result = normalize_sql(sql)
        assert "\t" not in result

    def test_collapses_spaces(self):
        sql = "SELECT   *   FROM   t"
        result = normalize_sql(sql)
        assert "  " not in result

    def test_strips_leading_trailing(self):
        sql = "  SELECT 1  "
        result = normalize_sql(sql)
        assert result == "SELECT 1"

    def test_mixed_whitespace(self):
        sql = "SELECT\r\n*\r\nFROM\r\nt"
        result = normalize_sql(sql)
        assert "\r" not in result
        assert "\n" not in result


# ---------------------------------------------------------------------------
# format_sql_to_pretty
# ---------------------------------------------------------------------------


class TestFormatSqlToPretty:
    def test_formats_valid_sql(self):
        sql = "SELECT * FROM t WHERE id=1"
        result = format_sql_to_pretty(sql, "duckdb")
        assert "SELECT" in result

    def test_empty_string_passthrough(self):
        assert format_sql_to_pretty("", "duckdb") == ""

    def test_none_passthrough(self):
        assert format_sql_to_pretty(None, "duckdb") is None

    def test_fallback_on_exception(self):
        sql = "SELECT * FROM t"
        with patch("datus.utils.sql_utils.sqlglot.transpile", side_effect=Exception("parse error")):
            result = format_sql_to_pretty(sql, "duckdb")
        assert result == sql

    def test_fallback_on_empty_transpile_result(self):
        sql = "SELECT 1"
        with patch("datus.utils.sql_utils.sqlglot.transpile", return_value=[]):
            result = format_sql_to_pretty(sql, "duckdb")
        assert result == sql


# ---------------------------------------------------------------------------
# parse_table_names_parts (plural)
# ---------------------------------------------------------------------------


class TestParseTableNamesParts:
    def test_single_table(self):
        result = parse_table_names_parts(["schema.table"], dialect=DBType.DUCKDB)
        assert len(result) == 1
        assert result[0]["table_name"] == "table"

    def test_multiple_tables(self):
        result = parse_table_names_parts(["db.schema.table1", "db.schema.table2"], dialect="snowflake")
        assert len(result) == 2

    def test_empty_list(self):
        result = parse_table_names_parts([])
        assert result == []


# ---------------------------------------------------------------------------
# parse_table_name_parts - extended
# ---------------------------------------------------------------------------


class TestParseTableNamePartsExtended:
    def test_mysql_single_part(self):
        result = parse_table_name_parts("mytable", dialect="mysql")
        assert result["table_name"] == "mytable"

    def test_duckdb_three_parts(self):
        result = parse_table_name_parts("mydb.myschema.mytable", dialect=DBType.DUCKDB)
        assert result["database_name"] == "mydb"
        assert result["schema_name"] == "myschema"
        assert result["table_name"] == "mytable"

    def test_empty_string(self):
        result = parse_table_name_parts("", dialect="snowflake")
        assert result["table_name"] == ""

    def test_bracket_quoted(self):
        result = parse_table_name_parts("[mydb].[schema].[table]", dialect="sqlserver")
        assert result["table_name"] == "table"

    def test_snowflake_four_parts_excess(self):
        # More parts than expected - takes last N
        result = parse_table_name_parts("extra.catalog.db.schema.table", dialect="snowflake")
        assert result["table_name"] == "table"

    def test_unknown_dialect_fallback(self):
        # Unknown dialect falls through to default behavior
        result = parse_table_name_parts("a.b.c.d", dialect="unknown_dialect_xyz")
        assert result["table_name"] == "d"


# ---------------------------------------------------------------------------
# parse_context_switch - additional cases
# ---------------------------------------------------------------------------


class TestParseContextSwitchExtended:
    def test_returns_none_for_empty_input(self):
        assert parse_context_switch("", "duckdb") is None
        assert parse_context_switch(None, "duckdb") is None

    def test_returns_none_for_non_use_set(self):
        assert parse_context_switch("SELECT 1", "duckdb") is None

    def test_returns_none_for_invalid_use_expr(self):
        # USE with an expression that sqlglot can't parse as Use
        result = parse_context_switch("USE", "duckdb")
        assert result is None

    def test_set_database(self):
        result = parse_context_switch("SET DATABASE mydb", "duckdb")
        assert result is not None
        assert result["command"] == "SET"
        assert result["target"] == "database"
        assert result["database_name"] == "mydb"

    def test_set_schema_duckdb(self):
        result = parse_context_switch("SET SCHEMA main", "duckdb")
        assert result is not None
        assert result["target"] == "schema"

    def test_set_catalog(self):
        result = parse_context_switch("SET CATALOG mycat", "duckdb")
        assert result is not None
        assert result["target"] == "catalog"

    def test_set_with_equals(self):
        result = parse_context_switch("SET CATALOG = mycat", "snowflake")
        assert result is not None
        assert result["catalog_name"] == "mycat"

    def test_set_with_to(self):
        result = parse_context_switch("SET CATALOG TO mycat", "snowflake")
        assert result is not None
        assert result["catalog_name"] == "mycat"

    def test_set_empty_remainder_returns_none(self):
        result = parse_context_switch("SET CATALOG", "snowflake")
        assert result is None

    def test_use_catalog_keyword(self):
        result = parse_context_switch("USE CATALOG my_catalog", "starrocks")
        assert result is not None
        assert result["target"] == "catalog"

    def test_use_database_keyword(self):
        result = parse_context_switch("USE DATABASE mydb", "snowflake")
        assert result is not None
        assert result["target"] == "database"

    def test_use_schema_keyword_snowflake(self):
        result = parse_context_switch("USE SCHEMA myschema", "snowflake")
        assert result is not None
        assert result["target"] == "schema"
        assert result["schema_name"] == "myschema"

    def test_set_session_database(self):
        result = parse_context_switch("SET SESSION DATABASE mydb", "snowflake")
        assert result is not None
        assert result["target"] == "database"
        assert result["database_name"] == "mydb"

    def test_generic_use_fallback(self):
        # For postgres (has schema capability), generic fallback applies
        result = parse_context_switch("USE myschema", "postgres")
        assert result is not None

    def test_starrocks_use_catalog_dot_db(self):
        result = parse_context_switch("USE my_catalog.my_db", "starrocks")
        assert result is not None
        assert result["target"] == "database"
        assert result["catalog_name"] == "my_catalog"
        assert result["database_name"] == "my_db"


# ---------------------------------------------------------------------------
# parse_metadata_from_ddl - edge cases
# ---------------------------------------------------------------------------


class TestParseMetadataFromDDLExtended:
    def test_malformed_sql_returns_empty(self):
        result = parse_metadata_from_ddl("NOT VALID SQL AT ALL !!!!")
        assert result["table"]["name"] == ""

    def test_empty_string_returns_empty(self):
        result = parse_metadata_from_ddl("")
        assert result["table"]["name"] == ""

    def test_mysql_backtick_names(self):
        ddl = "CREATE TABLE `my_table` (`col1` INT, `col2` VARCHAR(255))"
        result = parse_metadata_from_ddl(ddl, "mysql")
        assert result["table"]["name"] == "my_table"
        assert result["columns"][0]["name"] == "col1"

    def test_column_comment(self):
        ddl = """CREATE TABLE t (
            id INT,
            name VARCHAR(100) COMMENT 'The name field'
        )"""
        result = parse_metadata_from_ddl(ddl, "mysql")
        assert result["table"]["name"] == "t"
        name_col = next((c for c in result["columns"] if c["name"] == "name"), None)
        assert name_col is not None
