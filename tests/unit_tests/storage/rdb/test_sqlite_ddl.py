# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SQLite DDL generation covering all branches."""

import pytest
from datus_storage_base.rdb.base import ColumnDef, IndexDef, TableDefinition

from datus.storage.rdb.sqlite_backend import (
    SqliteRdbDatabase,
    _sqlite_col_ddl,
    _sqlite_map_type,
)


class TestSqliteMapType:
    """Tests for SQLite type mapping."""

    @pytest.mark.parametrize(
        "input_type,expected",
        [
            ("BOOLEAN", "INTEGER"),
            ("TIMESTAMP", "TEXT"),
            ("JSONB", "JSONB"),  # unknown type passes through as-is
            ("REAL", "REAL"),
            ("boolean", "INTEGER"),  # case-insensitive
            ("integer", "INTEGER"),  # case-insensitive
        ],
    )
    def test_type_mapping(self, input_type, expected):
        """SQLite type mapping produces correct output for all known and unknown types."""
        assert _sqlite_map_type(input_type) == expected


class TestSqliteColDdl:
    """Tests for _sqlite_col_ddl covering all branches."""

    def test_autoincrement(self):
        """SQLite auto-increment generates INTEGER PRIMARY KEY AUTOINCREMENT."""
        col = ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True)
        result = _sqlite_col_ddl(col)
        assert "INTEGER PRIMARY KEY AUTOINCREMENT" in result

    def test_primary_key_without_autoincrement(self):
        """Primary key without autoincrement."""
        col = ColumnDef(name="id", col_type="INTEGER", primary_key=True)
        result = _sqlite_col_ddl(col)
        assert "PRIMARY KEY" in result
        assert "AUTOINCREMENT" not in result

    def test_unique_column(self):
        """Column with unique=True generates UNIQUE constraint."""
        col = ColumnDef(name="email", col_type="TEXT", unique=True)
        result = _sqlite_col_ddl(col)
        assert "UNIQUE" in result

    def test_not_null(self):
        """Column with nullable=False generates NOT NULL."""
        col = ColumnDef(name="name", col_type="TEXT", nullable=False)
        result = _sqlite_col_ddl(col)
        assert "NOT NULL" in result

    def test_string_default(self):
        """String default value is quoted."""
        col = ColumnDef(name="status", col_type="TEXT", default="active")
        result = _sqlite_col_ddl(col)
        assert "DEFAULT 'active'" in result

    def test_numeric_default(self):
        """Numeric default value is unquoted."""
        col = ColumnDef(name="score", col_type="INTEGER", default=0)
        result = _sqlite_col_ddl(col)
        assert "DEFAULT 0" in result


class TestSqliteGenerateDdl:
    """Tests for SqliteRdbDatabase._generate_ddl with constraints and indices."""

    @pytest.fixture()
    def backend(self, tmp_path):
        import os

        return SqliteRdbDatabase(
            os.path.join(str(tmp_path), "ddl_test.db"),
        )

    def test_create_table_with_constraints(self, backend):
        """Table with constraints includes them in DDL."""
        table_def = TableDefinition(
            table_name="items",
            columns=[
                ColumnDef(name="id", col_type="INTEGER", primary_key=True),
                ColumnDef(name="name", col_type="TEXT"),
            ],
            constraints=["UNIQUE(name)"],
        )
        ddl = backend._generate_ddl(table_def)
        assert len(ddl) >= 1
        assert "UNIQUE(name)" in ddl[0]

    def test_create_table_with_index(self, backend):
        """Table with indices generates CREATE INDEX statements."""
        table_def = TableDefinition(
            table_name="items",
            columns=[
                ColumnDef(name="id", col_type="INTEGER"),
                ColumnDef(name="category", col_type="TEXT"),
            ],
            indices=[IndexDef(name="idx_cat", columns=["category"])],
        )
        ddl = backend._generate_ddl(table_def)
        assert len(ddl) == 2
        assert "CREATE INDEX IF NOT EXISTS idx_cat" in ddl[1]

    def test_unique_index(self, backend):
        """Unique index generates CREATE UNIQUE INDEX."""
        table_def = TableDefinition(
            table_name="items",
            columns=[ColumnDef(name="code", col_type="TEXT")],
            indices=[IndexDef(name="idx_code", columns=["code"], unique=True)],
        )
        ddl = backend._generate_ddl(table_def)
        assert "CREATE UNIQUE INDEX" in ddl[1]
