# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SQLite MigrationTargetMixin implementation."""

import pytest

from datus.tools.db_tools._migration_compat import MigrationTargetMixin
from datus.tools.db_tools.sqlite_connector import SQLiteConnector


@pytest.fixture
def connector():
    return SQLiteConnector.__new__(SQLiteConnector)


class TestMixinInheritance:
    def test_sqlite_is_migration_target(self, connector):
        assert isinstance(connector, MigrationTargetMixin)


class TestDescribeMigrationCapabilities:
    def test_supported_true(self, connector):
        assert connector.describe_migration_capabilities()["supported"] is True

    def test_dialect_family_sqlite(self, connector):
        assert connector.describe_migration_capabilities()["dialect_family"] == "sqlite"

    def test_no_hard_requirements(self, connector):
        assert connector.describe_migration_capabilities()["requires"] == []

    def test_type_hints_mention_affinity(self, connector):
        """SQLite uses type affinity — VARCHAR must be flagged as the affinity-driven case."""
        hints = connector.describe_migration_capabilities()["type_hints"]
        # The VARCHAR hint is the canonical spot where affinity is documented —
        # pin that specific mapping instead of scanning the whole dict.
        assert "affinity" in hints["VARCHAR"].lower()
        assert hints["VARCHAR"].startswith("TEXT")


class TestValidateDdl:
    def test_accepts_standard_sqlite_ddl(self, connector):
        ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"
        assert connector.validate_ddl(ddl) == []

    def test_rejects_starrocks_syntax(self, connector):
        ddl = "CREATE TABLE t (id INTEGER) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        errors = connector.validate_ddl(ddl)
        assert any("DUPLICATE KEY" in e.upper() or "STARROCKS" in e.upper() for e in errors)

    def test_rejects_engine_clickhouse(self, connector):
        ddl = "CREATE TABLE t (id INTEGER) ENGINE = MergeTree() ORDER BY id"
        errors = connector.validate_ddl(ddl)
        assert any("ENGINE" in e.upper() or "CLICKHOUSE" in e.upper() for e in errors)


class TestSuggestTableLayout:
    def test_returns_empty_dict(self, connector):
        columns = [{"name": "id", "type": "INTEGER", "nullable": False}]
        assert connector.suggest_table_layout(columns) == {}


class TestMapSourceType:
    def test_varchar_to_text(self, connector):
        """SQLite uses TEXT affinity — all string types map to TEXT."""
        assert connector.map_source_type("mysql", "VARCHAR") == "TEXT"

    def test_decimal_to_numeric(self, connector):
        assert connector.map_source_type("mysql", "DECIMAL(10,2)") == "NUMERIC"

    def test_double_to_real(self, connector):
        assert connector.map_source_type("mysql", "DOUBLE") == "REAL"

    def test_boolean_to_integer(self, connector):
        """SQLite has no native BOOLEAN; use INTEGER 0/1."""
        assert connector.map_source_type("postgresql", "BOOLEAN") == "INTEGER"

    def test_unknown_int_family_returns_none(self, connector):
        """INTEGER passes through unchanged."""
        assert connector.map_source_type("mysql", "INTEGER") is None
