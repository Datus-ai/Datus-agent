# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SqliteRdbTable (concrete RdbTable implementation)."""

import os
from dataclasses import dataclass

import pytest
from datus_storage_base.rdb.base import ColumnDef, IndexDef, TableDefinition, WhereOp

from datus.storage.rdb.sqlite_backend import SqliteRdbDatabase, SqliteRdbTable


@dataclass
class _Item:
    """Test record model."""

    name: str = ""
    value: str = ""
    id: int = None


@pytest.fixture
def database(tmp_path):
    """Create an initialized SqliteRdbDatabase."""
    db_file = os.path.join(str(tmp_path), "test.db")
    return SqliteRdbDatabase(
        db_file,
    )


@pytest.fixture
def table(database):
    """Create a table handle via ensure_table."""
    table_def = TableDefinition(
        table_name="my_table",
        columns=[
            ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
            ColumnDef(name="name", col_type="TEXT", nullable=False),
            ColumnDef(name="value", col_type="TEXT"),
        ],
        indices=[IndexDef(name="idx_name", columns=["name"], unique=True)],
    )
    return database.ensure_table(table_def)


class TestRdbTable:
    """Tests for SqliteRdbTable CRUD operations."""

    def test_table_name_property(self, table):
        """table_name returns the bound table name."""
        assert table.table_name == "my_table"

    def test_is_sqlite_rdb_table(self, table):
        """ensure_table returns a SqliteRdbTable instance."""
        assert isinstance(table, SqliteRdbTable)

    def test_insert_and_query(self, table):
        """insert() inserts a record; query() retrieves it."""
        row_id = table.insert(_Item(name="a", value="b"))
        assert row_id >= 1
        rows = table.query(_Item)
        assert len(rows) == 1
        assert rows[0].name == "a"
        assert rows[0].value == "b"

    def test_query_with_where(self, table):
        """query() with where clause filters correctly."""
        table.insert(_Item(name="a", value="v1"))
        table.insert(_Item(name="b", value="v2"))
        rows = table.query(_Item, where={"name": "a"})
        assert len(rows) == 1
        assert rows[0].name == "a"

    def test_query_with_order_by(self, table):
        """query() with order_by sorts correctly."""
        table.insert(_Item(name="b", value="v2"))
        table.insert(_Item(name="a", value="v1"))
        rows = table.query(_Item, order_by=["name"])
        assert rows[0].name == "a"
        assert rows[1].name == "b"

    def test_update(self, table):
        """update() modifies rows and returns affected count."""
        table.insert(_Item(name="a", value="v1"))
        count = table.update({"value": "v2"}, where={"name": "a"})
        assert count == 1
        rows = table.query(_Item, where={"name": "a"})
        assert rows[0].value == "v2"

    def test_delete(self, table):
        """delete() removes rows and returns affected count."""
        table.insert(_Item(name="a", value="v1"))
        count = table.delete(where={"name": "a"})
        assert count == 1
        rows = table.query(_Item)
        assert rows == []

    def test_delete_with_operator(self, table):
        """delete() with WhereOp filters correctly."""
        table.insert(_Item(name="a", value="v1"))
        count = table.delete(where=[("name", WhereOp.EQ, "a")])
        assert count == 1

    def test_upsert(self, table):
        """upsert() inserts or replaces a record."""
        table.upsert(_Item(name="k1", value="v1"), conflict_columns=["name"])
        table.upsert(_Item(name="k1", value="v2"), conflict_columns=["name"])
        rows = table.query(_Item, where={"name": "k1"})
        assert len(rows) == 1
        assert rows[0].value == "v2"
