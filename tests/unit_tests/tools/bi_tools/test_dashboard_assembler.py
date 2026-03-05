# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for DashboardAssembler._can_qualify_table."""

import pytest

from datus.tools.bi_tools.dashboard_assembler import DashboardAssembler
from datus.tools.db_tools import connector_registry
from datus.utils.constants import DBType


@pytest.fixture(autouse=True)
def _register_test_capabilities():
    """Register external dialect capabilities needed by tests."""
    connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
    connector_registry.register_handlers("mysql", capabilities={"database"})
    connector_registry.register_handlers("snowflake", capabilities={"catalog", "database", "schema"})
    yield


class TestCanQualifyTable:
    """Test DashboardAssembler._can_qualify_table method."""

    @pytest.fixture
    def assembler(self):
        # _can_qualify_table doesn't use the adaptor, so None is fine
        return DashboardAssembler(adaptor=None)

    def test_sqlite_with_database(self, assembler):
        assert assembler._can_qualify_table(DBType.SQLITE, "", "main", "") is True

    def test_sqlite_without_database(self, assembler):
        assert assembler._can_qualify_table(DBType.SQLITE, "", "", "") is False

    def test_duckdb_with_database_and_schema(self, assembler):
        assert assembler._can_qualify_table(DBType.DUCKDB, "", "memory", "main") is True

    def test_duckdb_missing_schema(self, assembler):
        assert assembler._can_qualify_table(DBType.DUCKDB, "", "memory", "") is False

    def test_external_dialect_with_schema_support(self, assembler):
        # PostgreSQL supports schema — needs both database and schema
        assert assembler._can_qualify_table("postgresql", "", "mydb", "public") is True
        assert assembler._can_qualify_table("postgresql", "", "mydb", "") is False

    def test_external_dialect_database_only(self, assembler):
        # MySQL supports database but not schema
        assert assembler._can_qualify_table("mysql", "", "mydb", "") is True
        assert assembler._can_qualify_table("mysql", "", "", "") is False

    def test_unknown_dialect_fallback(self, assembler):
        # Unknown dialect: returns True if database or schema is present
        assert assembler._can_qualify_table("unknown_db", "", "db", "") is True
        assert assembler._can_qualify_table("unknown_db", "", "", "sch") is True
        assert assembler._can_qualify_table("unknown_db", "", "", "") is False

    def test_none_dialect(self, assembler):
        assert assembler._can_qualify_table(None, "", "db", "") is True
