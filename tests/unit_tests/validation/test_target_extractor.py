# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.validation.target_extractor``."""

from __future__ import annotations

from datus.validation.target_extractor import extract_ddl_target, extract_dml_target


class TestExtractDDLTarget:
    def test_basic_create_table(self):
        t = extract_ddl_target("CREATE TABLE staging.users (id INT)", "main")
        assert t is not None
        assert t.database == "main"
        assert t.db_schema == "staging"
        assert t.table == "users"

    def test_if_not_exists(self):
        t = extract_ddl_target("CREATE TABLE IF NOT EXISTS orders (id INT)", "main")
        assert t is not None
        assert t.table == "orders"
        assert t.db_schema is None

    def test_create_or_replace(self):
        t = extract_ddl_target("CREATE OR REPLACE TABLE analytics.revenue (m TEXT)", "prod")
        assert t is not None
        assert t.database == "prod"
        assert t.db_schema == "analytics"
        assert t.table == "revenue"

    def test_temporary(self):
        t = extract_ddl_target("CREATE TEMPORARY TABLE tmp_foo (x INT)", "db1")
        assert t is not None
        assert t.table == "tmp_foo"

    def test_ctas(self):
        t = extract_ddl_target(
            "CREATE TABLE analytics.revenue_monthly AS SELECT * FROM staging.sales",
            "db1",
        )
        assert t is not None
        assert t.db_schema == "analytics"
        assert t.table == "revenue_monthly"

    def test_quoted_identifier(self):
        t = extract_ddl_target('CREATE TABLE "My Schema"."My Table" (x INT)', "db1")
        assert t is not None
        assert t.db_schema == "My Schema"
        assert t.table == "My Table"

    def test_three_part_identifier(self):
        """``db.schema.table`` — the SQL-level db wins over the tool's default."""
        t = extract_ddl_target("CREATE TABLE mydb.myschema.mytable (x INT)", "default_db")
        assert t is not None
        assert t.database == "mydb"
        assert t.db_schema == "myschema"
        assert t.table == "mytable"

    def test_drop_table_returns_none(self):
        assert extract_ddl_target("DROP TABLE foo", "db1") is None

    def test_alter_table_returns_none(self):
        assert extract_ddl_target("ALTER TABLE foo ADD COLUMN x INT", "db1") is None

    def test_create_schema_returns_none(self):
        assert extract_ddl_target("CREATE SCHEMA foo", "db1") is None

    def test_create_view_returns_none(self):
        """Views aren't targets we can run row-count checks against."""
        assert extract_ddl_target("CREATE VIEW v AS SELECT 1", "db1") is None

    def test_parser_error_returns_none(self):
        assert extract_ddl_target("not valid sql at all", "db1") is None

    def test_empty_returns_none(self):
        assert extract_ddl_target("", "db1") is None
        assert extract_ddl_target("   ", "db1") is None


class TestExtractDMLTarget:
    def test_insert(self):
        t = extract_dml_target("INSERT INTO staging.users (id) VALUES (1)", "main")
        assert t is not None
        assert t.db_schema == "staging"
        assert t.table == "users"

    def test_update(self):
        t = extract_dml_target("UPDATE orders SET status = 'done' WHERE id = 1", "db")
        assert t is not None
        assert t.table == "orders"

    def test_delete(self):
        t = extract_dml_target("DELETE FROM orders WHERE id = 1", "db")
        assert t is not None
        assert t.table == "orders"

    def test_select_returns_none(self):
        """SELECT is not a mutating tool — no target."""
        assert extract_dml_target("SELECT * FROM t", "db") is None

    def test_parser_error_returns_none(self):
        assert extract_dml_target("invalid", "db") is None
