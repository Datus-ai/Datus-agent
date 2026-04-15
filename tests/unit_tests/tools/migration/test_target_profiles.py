"""Tests for target database profiles (datus/tools/migration/target_profiles.py)."""

import pytest

from datus.tools.migration.target_profiles import (
    GreenplumProfile,
    StarRocksProfile,
    build_target_ddl,
)
from datus.tools.migration.type_mapping import UnsupportedTypeError


class TestGreenplumProfile:
    """Test Greenplum profile behavior."""

    def test_default_schema(self):
        profile = GreenplumProfile()
        assert profile.schema_name == "public"

    def test_custom_schema(self):
        profile = GreenplumProfile(schema_name="staging")
        assert profile.schema_name == "staging"

    def test_format_table_name_with_schema(self):
        profile = GreenplumProfile(schema_name="staging")
        assert profile.format_table_name("users") == "staging.users"

    def test_format_table_name_already_qualified(self):
        profile = GreenplumProfile(schema_name="staging")
        assert profile.format_table_name("public.users") == "public.users"

    def test_ddl_suffix_empty(self):
        profile = GreenplumProfile()
        assert profile.ddl_suffix() == ""

    def test_format_table_name_default_schema(self):
        """Unqualified table name should get 'public' prefix by default."""
        profile = GreenplumProfile()
        assert profile.format_table_name("users") == "public.users"


class TestStarRocksProfile:
    """Test StarRocks profile behavior."""

    def test_default_values(self):
        profile = StarRocksProfile(database="test")
        assert profile.catalog == "default_catalog"
        assert profile.database == "test"

    def test_format_table_name(self):
        profile = StarRocksProfile(database="demo")
        assert profile.format_table_name("users") == "demo.users"

    def test_format_table_name_already_qualified(self):
        profile = StarRocksProfile(database="demo")
        assert profile.format_table_name("other_db.users") == "other_db.users"

    def test_select_key_columns_id_priority(self):
        """Columns with 'id' or '_id' suffix should be selected first."""
        profile = StarRocksProfile(database="test")
        columns = [
            {"name": "user_id", "type": "BIGINT", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
            {"name": "order_id", "type": "INT", "nullable": False},
            {"name": "amount", "type": "DECIMAL", "nullable": True},
        ]
        keys = profile.select_key_columns(columns)
        assert set(keys) == {"user_id", "order_id"}
        assert len(keys) == 2

    def test_select_key_columns_int_fallback(self):
        """If no id columns, prefer INT/BIGINT columns."""
        profile = StarRocksProfile(database="test")
        columns = [
            {"name": "code", "type": "BIGINT", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
            {"name": "status", "type": "INT", "nullable": False},
        ]
        keys = profile.select_key_columns(columns)
        assert "code" in keys
        assert "status" in keys
        assert "name" not in keys
        assert len(keys) == 2

    def test_select_key_columns_non_nullable_preferred(self):
        """Non-nullable columns should be preferred over nullable ones."""
        profile = StarRocksProfile(database="test")
        columns = [
            {"name": "a", "type": "INT", "nullable": True},
            {"name": "b", "type": "INT", "nullable": False},
        ]
        keys = profile.select_key_columns(columns)
        assert keys[0] == "b"
        assert len(keys) == 2
        assert "a" in keys

    def test_select_key_columns_fallback_first(self):
        """When no good candidates, fallback to first column."""
        profile = StarRocksProfile(database="test")
        columns = [
            {"name": "desc", "type": "VARCHAR", "nullable": True},
            {"name": "notes", "type": "TEXT", "nullable": True},
        ]
        keys = profile.select_key_columns(columns)
        assert keys == ["desc"]

    def test_select_key_columns_max_three(self):
        """At most 3 key columns should be selected."""
        profile = StarRocksProfile(database="test")
        columns = [
            {"name": "id1", "type": "INT", "nullable": False},
            {"name": "id2", "type": "INT", "nullable": False},
            {"name": "id3", "type": "INT", "nullable": False},
            {"name": "id4", "type": "INT", "nullable": False},
        ]
        keys = profile.select_key_columns(columns)
        assert len(keys) <= 3

    def test_ddl_suffix(self):
        profile = StarRocksProfile(database="test")
        suffix = profile.ddl_suffix(["user_id", "order_id"])
        assert "DUPLICATE KEY" in suffix
        assert "user_id" in suffix
        assert "order_id" in suffix
        assert "DISTRIBUTED BY HASH" in suffix
        assert "BUCKETS" in suffix

    def test_select_key_columns_empty_input(self):
        """Empty columns list should return empty list."""
        profile = StarRocksProfile(database="test")
        assert profile.select_key_columns([]) == []

    @pytest.mark.parametrize(
        "key_columns",
        [
            ["user_id"],
            ["user_id", "order_id"],
            ["user_id", "order_id", "product_id"],
        ],
    )
    def test_ddl_suffix_key_count_variants(self, key_columns):
        """DUPLICATE KEY and DISTRIBUTED BY HASH must list exactly the given keys."""
        profile = StarRocksProfile(database="test")
        suffix = profile.ddl_suffix(key_columns)
        assert "DUPLICATE KEY" in suffix
        assert "DISTRIBUTED BY HASH" in suffix
        for col in key_columns:
            assert col in suffix

    def test_custom_catalog_attribute(self):
        """catalog attribute should be settable and format_table_name still works."""
        profile = StarRocksProfile(database="demo", catalog="my_catalog")
        assert profile.catalog == "my_catalog"
        assert profile.format_table_name("orders") == "demo.orders"

    def test_custom_catalog_already_qualified(self):
        """Already-qualified table name is returned as-is even with custom catalog."""
        profile = StarRocksProfile(database="demo", catalog="my_catalog")
        assert profile.format_table_name("other_db.orders") == "other_db.orders"


class TestBuildTargetDDLGreenplum:
    """Test DDL generation for Greenplum."""

    def test_basic_create_table(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR(100)", "nullable": True},
            {"name": "amount", "type": "DECIMAL(10,2)", "nullable": True},
        ]
        profile = GreenplumProfile(schema_name="public")
        ddl = build_target_ddl(columns, "duckdb", "greenplum", "users", profile)
        assert "CREATE TABLE" in ddl
        assert "public.users" in ddl
        assert "INTEGER" in ddl
        assert "VARCHAR(100)" in ddl
        assert "NUMERIC(10,2)" in ddl
        assert "NOT NULL" in ddl

    def test_nullable_columns(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
        ]
        profile = GreenplumProfile()
        ddl = build_target_ddl(columns, "duckdb", "greenplum", "t", profile)
        # The non-nullable column line must contain NOT NULL
        id_line = next(line for line in ddl.splitlines() if "id" in line and "name" not in line)
        assert "NOT NULL" in id_line
        # The nullable column line must NOT contain NOT NULL
        name_line = next(line for line in ddl.splitlines() if "name" in line)
        assert "NOT NULL" not in name_line


class TestBuildTargetDDLStarRocks:
    """Test DDL generation for StarRocks."""

    def test_basic_create_table(self):
        columns = [
            {"name": "user_id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR(100)", "nullable": True},
            {"name": "is_active", "type": "BOOLEAN", "nullable": True},
        ]
        profile = StarRocksProfile(database="demo")
        ddl = build_target_ddl(columns, "duckdb", "starrocks", "users", profile)
        assert "CREATE TABLE" in ddl
        assert "demo.users" in ddl
        assert "INT" in ddl
        assert "VARCHAR(100)" in ddl
        assert "BOOLEAN" in ddl
        assert "DUPLICATE KEY" in ddl
        assert "DISTRIBUTED BY HASH" in ddl

    def test_key_selection_in_ddl(self):
        columns = [
            {"name": "order_id", "type": "BIGINT", "nullable": False},
            {"name": "product_id", "type": "INTEGER", "nullable": False},
            {"name": "quantity", "type": "INTEGER", "nullable": True},
        ]
        profile = StarRocksProfile(database="demo")
        ddl = build_target_ddl(columns, "duckdb", "starrocks", "orders", profile)
        assert "order_id" in ddl
        assert "product_id" in ddl


class TestBuildTargetDDLUnsupportedTypes:
    """Test DDL generation with unsupported types."""

    def test_unsupported_type_raises_error(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "data", "type": "STRUCT", "nullable": True},
        ]
        profile = GreenplumProfile()
        with pytest.raises(UnsupportedTypeError):
            build_target_ddl(columns, "duckdb", "greenplum", "t", profile)


class TestBuildTargetDDLWithoutProfile:
    """Test DDL generation without explicit profile."""

    def test_greenplum_without_profile(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
        ]
        ddl = build_target_ddl(columns, "duckdb", "greenplum", "public.users")
        assert "CREATE TABLE" in ddl
        assert "public.users" in ddl

    def test_starrocks_without_profile(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
        ]
        ddl = build_target_ddl(columns, "duckdb", "starrocks", "test.users")
        assert "CREATE TABLE" in ddl
        assert "DUPLICATE KEY" in ddl
        # Default profile should produce DUPLICATE KEY and preserve the table name
        assert "test.users" in ddl

    def test_starrocks_empty_database_produces_valid_name(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
        ]
        ddl = build_target_ddl(columns, "duckdb", "starrocks", "users")
        assert "CREATE TABLE" in ddl
        # Should not produce ".users" (leading dot)
        assert ".users" not in ddl or "users" in ddl.split("TABLE")[1].strip().split("(")[0]
