# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/node_utils.py"""

import pytest

from datus.utils.node_utils import build_database_context


class TestBuildDatabaseContext:
    """Tests for build_database_context."""

    def test_dialect_only(self):
        """Only db_type produces a single Dialect entry."""
        result = build_database_context("sqlite")
        assert "**Dialect**: sqlite" in result
        assert "**Catalog**" not in result
        assert "**Database**" not in result
        assert "**Schema**" not in result

    def test_all_fields(self):
        """All fields present are included."""
        result = build_database_context("snowflake", catalog="my_cat", database="my_db", schema="my_schema")
        assert "**Dialect**: snowflake" in result
        assert "**Catalog**: my_cat" in result
        assert "**Database**: my_db" in result
        assert "**Schema**: my_schema" in result

    def test_partial_fields(self):
        """Only non-empty fields are included."""
        result = build_database_context("postgresql", database="prod_db", schema="public")
        assert "**Dialect**: postgresql" in result
        assert "**Catalog**" not in result
        assert "**Database**: prod_db" in result
        assert "**Schema**: public" in result

    def test_none_values_excluded(self):
        """None values are treated like missing fields."""
        result = build_database_context("mysql", catalog=None, database="mydb", schema=None)
        assert "**Catalog**" not in result
        assert "**Database**: mydb" in result
        assert "**Schema**" not in result

    def test_empty_string_excluded(self):
        """Empty strings are treated like missing fields."""
        result = build_database_context("duckdb", catalog="", database="", schema="")
        assert "**Dialect**: duckdb" in result
        assert "**Catalog**" not in result
        assert "**Database**" not in result
        assert "**Schema**" not in result

    def test_return_prefix(self):
        """Return string starts with 'Database Context:'."""
        result = build_database_context("sqlite")
        assert result.startswith("Database Context:")

    def test_db_type_enum_like(self):
        """db_type works with str-enum values (DBType inherits str)."""
        result = build_database_context("bigquery", catalog="proj", database="dataset")
        assert "**Dialect**: bigquery" in result
        assert "**Catalog**: proj" in result

    @pytest.mark.parametrize(
        "catalog,database,schema,expected_count",
        [
            (None, None, None, 1),
            ("c", None, None, 2),
            (None, "d", None, 2),
            (None, None, "s", 2),
            ("c", "d", "s", 4),
        ],
    )
    def test_field_count(self, catalog, database, schema, expected_count):
        """Number of context parts matches number of non-empty fields + dialect."""
        result = build_database_context("sqlite", catalog=catalog, database=database, schema=schema)
        # Parts are comma-separated in the output
        assert result.count("**") == expected_count * 2  # each part has opening and closing **
