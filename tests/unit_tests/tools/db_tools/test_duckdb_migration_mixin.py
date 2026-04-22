# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for DuckDB MigrationTargetMixin implementation."""

import pytest

from datus.tools.db_tools._migration_compat import MigrationTargetMixin
from datus.tools.db_tools.duckdb_connector import DuckdbConnector


@pytest.fixture
def connector():
    return DuckdbConnector.__new__(DuckdbConnector)


class TestMixinInheritance:
    def test_duckdb_is_migration_target(self, connector):
        assert isinstance(connector, MigrationTargetMixin)


class TestDescribeMigrationCapabilities:
    def test_supported_true(self, connector):
        assert connector.describe_migration_capabilities()["supported"] is True

    def test_dialect_family_duckdb(self, connector):
        assert connector.describe_migration_capabilities()["dialect_family"] == "duckdb"

    def test_no_hard_requirements(self, connector):
        """DuckDB is OLTP-like for our purposes; no distribution/partition required."""
        assert connector.describe_migration_capabilities()["requires"] == []

    def test_type_hints_mention_complex_types(self, connector):
        """DuckDB's native LIST / STRUCT / MAP hints must all be documented."""
        hints = connector.describe_migration_capabilities()["type_hints"]
        # The source dict keys are the canonical place to pin these — checking
        # every key independently makes the contract explicit rather than
        # relying on substring scans across concatenated values.
        assert "LIST<T>" in hints
        assert "STRUCT" in hints
        assert "MAP" in hints

    def test_example_ddl_has_create_table(self, connector):
        ddl = connector.describe_migration_capabilities()["example_ddl"].upper()
        assert "CREATE TABLE" in ddl


class TestValidateDdl:
    def test_accepts_standard_duckdb_ddl(self, connector):
        ddl = "CREATE TABLE main.t (id BIGINT, name VARCHAR, created_at TIMESTAMP)"
        assert connector.validate_ddl(ddl) == []

    def test_rejects_starrocks_syntax(self, connector):
        ddl = "CREATE TABLE t (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 10"
        errors = connector.validate_ddl(ddl)
        assert any("DUPLICATE KEY" in e.upper() or "STARROCKS" in e.upper() for e in errors)

    def test_rejects_engine_clickhouse(self, connector):
        ddl = "CREATE TABLE t (id BIGINT) ENGINE = MergeTree() ORDER BY id"
        errors = connector.validate_ddl(ddl)
        assert any("ENGINE" in e.upper() or "CLICKHOUSE" in e.upper() for e in errors)


class TestSuggestTableLayout:
    def test_returns_empty_dict(self, connector):
        """DuckDB is in-memory/single-node — no distribution keys needed."""
        columns = [{"name": "id", "type": "BIGINT", "nullable": False}]
        assert connector.suggest_table_layout(columns) == {}


class TestMapSourceType:
    def test_unknown_returns_none(self, connector):
        """DuckDB supports most standard types natively; let LLM decide."""
        assert connector.map_source_type("postgresql", "VARCHAR") is None

    def test_json_to_json(self, connector):
        """DuckDB has native JSON type."""
        assert connector.map_source_type("postgresql", "JSONB") == "JSON"

    def test_variant_to_json(self, connector):
        """Snowflake VARIANT → DuckDB JSON."""
        assert connector.map_source_type("snowflake", "VARIANT") == "JSON"
