# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.storage.scoped_filter — LanceDB WHERE filter builder."""

from datus.storage.conditions import build_where
from datus.storage.scoped_filter import (
    ScopedFilterBuilder,
    _build_id_condition,
    _replace_wildcard,
    _split_csv,
    _subject_condition_for_parts,
    _table_condition_for_token,
    _value_condition,
)
from datus.utils.constants import DBType

# ---------------------------------------------------------------------------
# TestReplaceWildcard
# ---------------------------------------------------------------------------


class TestReplaceWildcard:
    """Tests for _replace_wildcard helper."""

    def test_replace_single_wildcard(self):
        """Asterisk is replaced with SQL LIKE percent."""
        assert _replace_wildcard("orders*") == "orders%"

    def test_replace_multiple_wildcards(self):
        """Multiple asterisks are all replaced."""
        assert _replace_wildcard("*orders*") == "%orders%"

    def test_no_wildcard_unchanged(self):
        """Strings without asterisk pass through."""
        assert _replace_wildcard("orders") == "orders"


# ---------------------------------------------------------------------------
# TestSplitCsv
# ---------------------------------------------------------------------------


class TestSplitCsv:
    """Tests for _split_csv helper."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert _split_csv("") == []

    def test_none_value(self):
        """None returns empty list."""
        assert _split_csv(None) == []

    def test_single_token(self):
        """Single value is returned as a list."""
        assert _split_csv("public.users") == ["public.users"]

    def test_comma_separated(self):
        """Comma-separated tokens are split and trimmed."""
        result = _split_csv("public.users, orders, products")
        assert result == ["public.users", "orders", "products"]

    def test_newline_separated(self):
        """Newlines are treated as separators."""
        result = _split_csv("public.users\norders\nproducts")
        assert result == ["public.users", "orders", "products"]

    def test_deduplication(self):
        """Duplicate tokens are removed, preserving order."""
        result = _split_csv("users, orders, users")
        assert result == ["users", "orders"]
        assert len(result) == 2

    def test_whitespace_trimming(self):
        """Extra whitespace around tokens is trimmed."""
        result = _split_csv("  users  ,  orders  ")
        assert result == ["users", "orders"]

    def test_empty_tokens_filtered(self):
        """Empty tokens from consecutive commas are filtered out."""
        result = _split_csv("users,,orders")
        assert result == ["users", "orders"]


# ---------------------------------------------------------------------------
# TestBuildIdCondition
# ---------------------------------------------------------------------------


class TestBuildIdCondition:
    """Tests for _build_id_condition helper."""

    def test_single_id(self):
        """Single ID produces an eq condition."""
        node = _build_id_condition([42])
        clause = build_where(node)
        assert clause == "subject_node_id = 42"

    def test_multiple_ids(self):
        """Multiple IDs produce an OR condition."""
        node = _build_id_condition([1, 2, 3])
        clause = build_where(node)
        assert "subject_node_id = 1" in clause
        assert "subject_node_id = 2" in clause
        assert "subject_node_id = 3" in clause
        assert "OR" in clause

    def test_duplicate_ids_deduplicated(self):
        """Duplicate IDs are deduplicated."""
        node = _build_id_condition([5, 5, 5])
        clause = build_where(node)
        assert clause == "subject_node_id = 5"


# ---------------------------------------------------------------------------
# TestValueCondition
# ---------------------------------------------------------------------------


class TestValueCondition:
    """Tests for _value_condition helper."""

    def test_exact_match(self):
        """Value without wildcard produces eq condition."""
        node = _value_condition("name", "revenue")
        clause = build_where(node)
        assert clause == "name = 'revenue'"

    def test_wildcard_produces_like(self):
        """Value with asterisk produces LIKE condition."""
        node = _value_condition("name", "rev*")
        clause = build_where(node)
        assert "LIKE" in clause
        assert "rev%" in clause

    def test_empty_value(self):
        """Empty string produces eq with empty string."""
        node = _value_condition("name", "")
        clause = build_where(node)
        assert clause == "name = ''"

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace is trimmed."""
        node = _value_condition("name", "  revenue  ")
        clause = build_where(node)
        assert clause == "name = 'revenue'"


# ---------------------------------------------------------------------------
# TestTableConditionForToken
# ---------------------------------------------------------------------------


class TestTableConditionForToken:
    """Tests for _table_condition_for_token parser."""

    def test_simple_table_name(self):
        """Simple table name without dialect maps to table_name."""
        node = _table_condition_for_token("users")
        clause = build_where(node)
        assert "table_name = 'users'" in clause

    def test_schema_dot_table_with_postgres(self):
        """schema.table for PostgreSQL maps to schema_name and table_name."""
        node = _table_condition_for_token("public.users", DBType.POSTGRESQL)
        clause = build_where(node)
        assert "schema_name = 'public'" in clause
        assert "table_name = 'users'" in clause

    def test_database_dot_table_with_sqlite(self):
        """database.table for SQLite maps to database_name and table_name."""
        node = _table_condition_for_token("main.users", DBType.SQLITE)
        clause = build_where(node)
        assert "database_name = 'main'" in clause
        assert "table_name = 'users'" in clause

    def test_empty_token(self):
        """Empty token returns None."""
        assert _table_condition_for_token("") is None

    def test_whitespace_only_parts(self):
        """Token with only dots/whitespace returns None."""
        assert _table_condition_for_token(". . .") is None

    def test_wildcard_in_table_name(self):
        """Wildcard in table name produces LIKE condition."""
        node = _table_condition_for_token("order*")
        clause = build_where(node)
        assert "LIKE" in clause
        assert "order%" in clause

    def test_three_part_with_snowflake(self):
        """Three-part name with Snowflake maps catalog.database.schema.table."""
        node = _table_condition_for_token("mydb.public.users", DBType.SNOWFLAKE)
        clause = build_where(node)
        assert "table_name = 'users'" in clause
        assert "schema_name = 'public'" in clause

    def test_single_part_no_dialect(self):
        """Single part with no dialect maps only to table_name."""
        node = _table_condition_for_token("orders", "")
        clause = build_where(node)
        assert "table_name = 'orders'" in clause


# ---------------------------------------------------------------------------
# TestBuildTableFilter
# ---------------------------------------------------------------------------


class TestBuildTableFilter:
    """Tests for ScopedFilterBuilder.build_table_filter."""

    def test_single_table(self):
        """Single table token produces a single condition."""
        node = ScopedFilterBuilder.build_table_filter("users")
        clause = build_where(node)
        assert "table_name = 'users'" in clause

    def test_multiple_tables_produce_or(self):
        """Multiple table tokens produce an OR condition."""
        node = ScopedFilterBuilder.build_table_filter("users, orders")
        clause = build_where(node)
        assert "users" in clause
        assert "orders" in clause
        assert "OR" in clause

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert ScopedFilterBuilder.build_table_filter("") is None

    def test_with_dialect(self):
        """Dialect is passed through to table condition builder."""
        node = ScopedFilterBuilder.build_table_filter("public.users", DBType.POSTGRESQL)
        clause = build_where(node)
        assert "schema_name = 'public'" in clause
        assert "table_name = 'users'" in clause


# ---------------------------------------------------------------------------
# TestBuildSubjectFilter
# ---------------------------------------------------------------------------


class _FakeSubjectTree:
    """Fake subject tree for testing subject filter resolution."""

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def get_matched_children_id(self, parts, descendant):
        key = (".".join(parts), descendant)
        return self._mapping.get(key, [])


class TestBuildSubjectFilter:
    """Tests for ScopedFilterBuilder.build_subject_filter."""

    def test_empty_returns_none(self):
        """Empty paths string returns None."""
        tree = _FakeSubjectTree({})
        assert ScopedFilterBuilder.build_subject_filter("", tree) is None

    def test_single_path_full_match(self):
        """Single path with full match returns id condition."""
        tree = _FakeSubjectTree({("Finance.Revenue", True): [10]})
        node = ScopedFilterBuilder.build_subject_filter("Finance.Revenue", tree)
        clause = build_where(node)
        assert "subject_node_id = 10" in clause

    def test_multiple_paths_produce_or(self):
        """Multiple paths with matches produce OR conditions."""
        tree = _FakeSubjectTree(
            {
                ("Finance", True): [1],
                ("Sales", True): [2],
            }
        )
        node = ScopedFilterBuilder.build_subject_filter("Finance, Sales", tree)
        clause = build_where(node)
        assert "subject_node_id = 1" in clause
        assert "subject_node_id = 2" in clause
        assert "OR" in clause

    def test_no_match_returns_none(self):
        """Paths that don't match anything return None."""
        tree = _FakeSubjectTree({})
        assert ScopedFilterBuilder.build_subject_filter("Nonexistent", tree) is None


# ---------------------------------------------------------------------------
# TestSubjectConditionForParts
# ---------------------------------------------------------------------------


class TestSubjectConditionForParts:
    """Tests for _subject_condition_for_parts two-pass resolution."""

    def test_pass1_full_path_match(self):
        """Pass 1: full path matches with descendant=True."""
        tree = _FakeSubjectTree({("Finance.Revenue", True): [10, 20]})
        node = _subject_condition_for_parts(["Finance", "Revenue"], tree)
        clause = build_where(node)
        assert "subject_node_id = 10" in clause
        assert "subject_node_id = 20" in clause

    def test_pass2_name_fallback(self):
        """Pass 2: fallback splits last part as name filter."""
        tree = _FakeSubjectTree(
            {
                ("Finance.total_revenue", True): [],
                ("Finance", False): [5],
            }
        )
        node = _subject_condition_for_parts(["Finance", "total_revenue"], tree)
        clause = build_where(node)
        assert "subject_node_id = 5" in clause
        assert "name = 'total_revenue'" in clause

    def test_no_match_at_all(self):
        """Both passes fail returns None."""
        tree = _FakeSubjectTree({})
        assert _subject_condition_for_parts(["Unknown"], tree) is None

    def test_single_part_no_fallback(self):
        """Single part with no match skips pass 2 (no fallback for len==1)."""
        tree = _FakeSubjectTree({("X", True): []})
        assert _subject_condition_for_parts(["X"], tree) is None

    def test_pass2_wildcard_name(self):
        """Pass 2 with wildcard name produces LIKE condition."""
        tree = _FakeSubjectTree(
            {
                ("Sales.rev*", True): [],
                ("Sales", False): [7],
            }
        )
        node = _subject_condition_for_parts(["Sales", "rev*"], tree)
        clause = build_where(node)
        assert "subject_node_id = 7" in clause
        assert "LIKE" in clause
        assert "rev%" in clause
