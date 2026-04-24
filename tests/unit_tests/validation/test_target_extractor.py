# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.validation.target_extractor``."""

from __future__ import annotations

import pytest

from datus.tools.db_tools import connector_registry
from datus.validation.target_extractor import extract_ddl_target, extract_dml_target


@pytest.fixture(autouse=True)
def _register_test_capabilities():
    """Register capability sets for dialects referenced by these tests.

    The validation layer consults ``connector_registry.support_catalog`` to
    decide whether a three-part identifier should populate ``catalog``.
    Unit-test harnesses only load sqlite / duckdb connectors by default, so
    we register bare capability records for the engines exercised here —
    matching the pattern used in ``test_sql_utils.py`` and
    ``test_dashboard_assembler.py``.

    ``connector_registry`` holds three class-level dicts that
    ``register_handlers`` can mutate: ``_capabilities``, ``_uri_builders``,
    and ``_context_resolvers``. Snapshot all three and restore on teardown
    so future ``register_handlers`` calls (or changes to the method itself)
    can't leak into unrelated tests via execution order.
    """
    snapshot_attrs = ("_capabilities", "_uri_builders", "_context_resolvers")
    snapshots = {
        attr: {k: (set(v) if isinstance(v, set) else v) for k, v in getattr(connector_registry, attr).items()}
        for attr in snapshot_attrs
    }
    connector_registry.register_handlers("starrocks", capabilities={"catalog", "database"})
    connector_registry.register_handlers("postgres", capabilities={"database", "schema"})
    try:
        yield
    finally:
        for attr, saved in snapshots.items():
            live = getattr(connector_registry, attr)
            live.clear()
            live.update(saved)


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

    def test_three_part_identifier_default_dialect(self):
        """Empty dialect defaults to catalog-tier semantics (the safer bet —
        misreading a StarRocks three-part identifier as db.schema.table made
        validator flag real tables as missing). Leading component becomes
        ``catalog``, middle becomes ``database``, ``db_schema`` stays None."""
        t = extract_ddl_target("CREATE TABLE mydb.myschema.mytable (x INT)", "default_db")
        assert t is not None
        assert t.catalog == "mydb"
        assert t.database == "myschema"
        assert t.db_schema is None
        assert t.table == "mytable"

    def test_three_part_identifier_starrocks_has_catalog(self):
        """On StarRocks ``default_catalog.ac_manage.stats`` is
        ``catalog.database.table`` (no schema tier). The middle component
        must land on ``database``, not on ``schema`` — otherwise builtin
        ``describe_table`` would query the wrong namespace and report the
        table as missing (reviewer P2-A)."""
        t = extract_ddl_target(
            "CREATE TABLE default_catalog.ac_manage.stats (id INT)",
            "default_db",
            dialect="starrocks",
        )
        assert t is not None
        assert t.catalog == "default_catalog"
        assert t.database == "ac_manage"
        assert t.db_schema is None
        assert t.table == "stats"

    def test_three_part_identifier_postgres_no_catalog(self):
        """On Postgres the three-part form is ``db.schema.table`` — the
        leading component is the database, NOT a catalog. Catalog stays
        unset so validator prompts don't render a spurious ``Catalog:``
        line."""
        t = extract_ddl_target(
            "CREATE TABLE mydb.public.users (id INT)",
            "default_db",
            dialect="postgres",
        )
        assert t is not None
        assert t.catalog is None
        assert t.database == "mydb"
        assert t.db_schema == "public"
        assert t.table == "users"

    def test_two_part_identifier_catalog_is_none(self):
        """Two-part ``schema.table`` leaves catalog unset."""
        t = extract_ddl_target("CREATE TABLE analytics.users (x INT)", "db1")
        assert t is not None
        assert t.catalog is None

    def test_one_part_identifier_catalog_is_none(self):
        """Bare table name leaves catalog unset."""
        t = extract_ddl_target("CREATE TABLE users (x INT)", "db1")
        assert t is not None
        assert t.catalog is None

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
