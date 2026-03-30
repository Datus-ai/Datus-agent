# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SQLite RDB backend CRUD interface."""

import os
from dataclasses import dataclass

import pytest
from datus_storage_base.rdb.base import (
    ColumnDef,
    IndexDef,
    IntegrityError,
    TableDefinition,
    UniqueViolationError,
    WhereOp,
)

from datus.storage.rdb.sqlite_backend import SqliteRdbDatabase
from datus.utils.exceptions import DatusException


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
def table_def():
    """Standard test table definition."""
    return TableDefinition(
        table_name="test_items",
        columns=[
            ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
            ColumnDef(name="name", col_type="TEXT", nullable=False),
            ColumnDef(name="value", col_type="TEXT"),
        ],
        indices=[IndexDef(name="idx_name", columns=["name"], unique=True)],
    )


class TestSqliteRdbDatabaseInit:
    """Tests for initialization."""

    def test_initialize_creates_directory(self, tmp_path):
        """SqliteRdbDatabase creates the parent directory for the db file."""
        db_file = os.path.join(str(tmp_path / "subdir"), "test.db")
        db = SqliteRdbDatabase(
            db_file,
        )
        assert os.path.isdir(str(tmp_path / "subdir"))
        assert db.db_file.endswith("test.db")

    def test_close_is_noop(self, database):
        """close() does nothing and doesn't raise."""
        database.close()


class TestSqliteRdbDatabaseInsert:
    """Tests for insert() via table handle."""

    def test_insert_returns_lastrowid(self, database, table_def):
        """insert() returns the lastrowid."""
        table = database.ensure_table(table_def)
        row_id = table.insert(_Item(name="a", value="b"))
        assert row_id >= 1

    def test_insert_duplicate_raises_integrity_error(self, database, table_def):
        """insert() raises IntegrityError on unique constraint violation."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="unique_key", value="v1"))
        with pytest.raises(IntegrityError):
            table.insert(_Item(name="unique_key", value="v2"))

    def test_insert_duplicate_raises_unique_violation_error(self, database, table_def):
        """insert() raises UniqueViolationError (an IntegrityError subclass) on unique constraint violation."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="unique_key", value="v1"))
        with pytest.raises(UniqueViolationError):
            table.insert(_Item(name="unique_key", value="v2"))


class TestSqliteRdbDatabaseQuery:
    """Tests for query() via table handle."""

    def test_query_empty_table(self, database, table_def):
        """query() returns empty list for empty table."""
        table = database.ensure_table(table_def)
        rows = table.query(_Item)
        assert rows == []

    def test_query_returns_model_instances(self, database, table_def):
        """query() returns typed model instances."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="b"))
        rows = table.query(_Item)
        assert len(rows) == 1
        assert isinstance(rows[0], _Item)
        assert rows[0].name == "a"
        assert rows[0].value == "b"

    def test_query_with_dict_where(self, database, table_def):
        """query() with dict WHERE clause filters correctly."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="v1"))
        table.insert(_Item(name="b", value="v2"))
        rows = table.query(_Item, where={"name": "a"})
        assert len(rows) == 1
        assert rows[0].name == "a"

    def test_query_with_tuple_where(self, database, table_def):
        """query() with tuple list WHERE clause filters correctly."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="v1"))
        table.insert(_Item(name="b", value="v2"))
        rows = table.query(_Item, where=[("name", WhereOp.NE, "a")])
        assert len(rows) == 1
        assert rows[0].name == "b"

    def test_query_with_columns(self, database, table_def):
        """query() with columns parameter selects specific columns."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="v1"))

        @dataclass
        class _NameOnly:
            name: str = ""

        rows = table.query(_NameOnly, columns=["name"])
        assert len(rows) == 1
        assert rows[0].name == "a"

    def test_query_with_order_by_asc(self, database, table_def):
        """query() with order_by sorts ascending."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="b", value="v2"))
        table.insert(_Item(name="a", value="v1"))
        rows = table.query(_Item, order_by=["name"])
        assert rows[0].name == "a"
        assert rows[1].name == "b"

    def test_query_with_order_by_desc(self, database, table_def):
        """query() with -prefix sorts descending."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="v1"))
        table.insert(_Item(name="b", value="v2"))
        rows = table.query(_Item, order_by=["-name"])
        assert rows[0].name == "b"
        assert rows[1].name == "a"

    def test_query_where_lt(self, database, table_def):
        """query() with WhereOp.LT filters correctly."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="1"))
        table.insert(_Item(name="b", value="2"))
        rows = table.query(_Item, where=[("value", WhereOp.LT, "2")])
        assert len(rows) == 1
        assert rows[0].name == "a"

    def test_query_where_is_null(self, database, table_def):
        """query() with WhereOp.IS_NULL filters correctly."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a"))  # value defaults to ""
        rows = table.query(_Item, where=[("value", WhereOp.IS_NULL, None)])
        # value is "" not NULL, so no results
        assert len(rows) == 0


class TestSqliteRdbDatabaseUpdate:
    """Tests for update() via table handle."""

    def test_update_returns_affected_count(self, database, table_def):
        """update() returns affected row count."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="v1"))
        count = table.update({"value": "v2"}, where={"name": "a"})
        assert count == 1

    def test_update_empty_data_returns_zero(self, database, table_def):
        """update() with empty data returns 0."""
        table = database.ensure_table(table_def)
        count = table.update({})
        assert count == 0

    def test_update_nonexistent_returns_zero(self, database, table_def):
        """update() returns 0 when no rows match."""
        table = database.ensure_table(table_def)
        count = table.update({"value": "v2"}, where={"name": "nonexistent"})
        assert count == 0


class TestSqliteRdbDatabaseDelete:
    """Tests for delete() via table handle."""

    def test_delete_returns_affected_count(self, database, table_def):
        """delete() returns affected row count."""
        table = database.ensure_table(table_def)
        table.insert(_Item(name="a", value="v1"))
        count = table.delete(where={"name": "a"})
        assert count == 1

    def test_delete_nonexistent_returns_zero(self, database, table_def):
        """delete() returns 0 when no rows match."""
        table = database.ensure_table(table_def)
        count = table.delete(where={"name": "nonexistent"})
        assert count == 0


class TestSqliteRdbDatabaseUpsert:
    """Tests for upsert() via table handle."""

    def test_upsert_inserts_new(self, database, table_def):
        """upsert() inserts a new record."""
        table = database.ensure_table(table_def)
        table.upsert(_Item(name="k1", value="v1"), ["name"])
        rows = table.query(_Item, where={"name": "k1"})
        assert len(rows) == 1
        assert rows[0].value == "v1"

    def test_upsert_replaces_existing(self, database, table_def):
        """upsert() replaces existing rows with INSERT OR REPLACE."""
        table = database.ensure_table(table_def)
        table.upsert(_Item(name="k1", value="v1"), ["name"])
        table.upsert(_Item(name="k1", value="v2"), ["name"])
        rows = table.query(_Item, where={"name": "k1"})
        assert len(rows) == 1
        assert rows[0].value == "v2"


class TestSqliteRdbDatabaseTransaction:
    """Tests for transaction()."""

    def test_transaction_commits_on_success(self, database, table_def):
        """transaction() commits on successful exit."""
        table = database.ensure_table(table_def)
        with database.transaction():
            table.insert(_Item(name="a", value="v1"))
            table.insert(_Item(name="b", value="v2"))
        rows = table.query(_Item)
        assert len(rows) == 2

    def test_transaction_rollback_on_exception(self, database, table_def):
        """transaction() rolls back on exception."""
        table = database.ensure_table(table_def)
        with pytest.raises(ValueError):
            with database.transaction():
                table.insert(_Item(name="a", value="v1"))
                raise ValueError("test error")
        rows = table.query(_Item)
        assert len(rows) == 0


class TestSqliteRdbDatabaseEdgeCases:
    """Tests for error paths."""

    def test_connection_error_raises_datus_exception(self, tmp_path):
        """Connecting to an invalid db file path raises DatusException."""
        bad_path = str(tmp_path / "is_a_dir")
        os.makedirs(bad_path)
        db = SqliteRdbDatabase.__new__(SqliteRdbDatabase)
        db._db_file = bad_path
        db._local = __import__("threading").local()
        with pytest.raises(DatusException):
            with db._auto_conn():
                pass

    def test_ensure_table_error_raises_with_ddl(self, database):
        """ensure_table raises with DDL text when creation fails."""
        bad_table = TableDefinition(
            table_name="bad;;table",
            columns=[ColumnDef(name="id", col_type="INVALID_TYPE")],
        )
        with pytest.raises(DatusException) as exc_info:
            database.ensure_table(bad_table)
        assert "bad;;table" in str(exc_info.value)


class TestMigrateMissingColumns:
    """Tests for _migrate_missing_columns auto-migration."""

    def test_adds_missing_column(self, tmp_path):
        """Adds a column that exists in definition but not in the live table."""
        db_file = os.path.join(str(tmp_path), "migrate.db")
        db = SqliteRdbDatabase(db_file)

        # Create table with only 2 columns
        old_def = TableDefinition(
            table_name="items",
            columns=[
                ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
                ColumnDef(name="name", col_type="TEXT"),
            ],
        )
        db.ensure_table(old_def)

        # Now ensure_table with an extra column — should auto-migrate
        new_def = TableDefinition(
            table_name="items",
            columns=[
                ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
                ColumnDef(name="name", col_type="TEXT"),
                ColumnDef(name="description", col_type="TEXT", default=""),
            ],
        )
        table = db.ensure_table(new_def)

        # Verify the new column works
        @dataclass
        class _ExtItem:
            name: str = ""
            description: str = ""
            id: int = None

        table.insert(_ExtItem(name="test", description="added"))
        rows = table.query(_ExtItem, where={"name": "test"})
        assert len(rows) == 1
        assert rows[0].description == "added"

    def test_already_up_to_date_is_noop(self, tmp_path):
        """No ALTER TABLE when all columns already exist."""
        db_file = os.path.join(str(tmp_path), "noop.db")
        db = SqliteRdbDatabase(db_file)

        table_def = TableDefinition(
            table_name="items",
            columns=[
                ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
                ColumnDef(name="name", col_type="TEXT"),
                ColumnDef(name="value", col_type="TEXT"),
            ],
        )
        db.ensure_table(table_def)
        # Second call — no migration needed, no error
        table = db.ensure_table(table_def)
        table.insert(_Item(name="ok", value="v1"))
        assert len(table.query(_Item)) == 1

    def test_ensure_table_idempotent(self, tmp_path):
        """Calling ensure_table multiple times with same definition is safe."""
        db_file = os.path.join(str(tmp_path), "idempotent.db")
        db = SqliteRdbDatabase(db_file)

        table_def = TableDefinition(
            table_name="items",
            columns=[
                ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
                ColumnDef(name="name", col_type="TEXT"),
                ColumnDef(name="value", col_type="TEXT"),
            ],
        )
        db.ensure_table(table_def)
        db.ensure_table(table_def)
        table = db.ensure_table(table_def)
        table.insert(_Item(name="test", value="v1"))
        assert len(table.query(_Item)) == 1


class TestSqliteRdbBackendConnect:
    """Tests for SqliteRdbBackend lifecycle and connect()."""

    def test_connect_physical_with_namespace(self, tmp_path):
        """PHYSICAL: connect() uses datus_db_{namespace}/ directory."""
        from datus.storage.rdb.sqlite_backend import SqliteRdbBackend

        b = SqliteRdbBackend()
        b.initialize({"data_dir": str(tmp_path), "isolation": "physical"})
        db = b.connect("my_ns", "test")
        assert isinstance(db, SqliteRdbDatabase)
        assert db.db_file == os.path.join(str(tmp_path), "datus_db_my_ns", "test.db")

    def test_connect_physical_no_namespace(self, tmp_path):
        """PHYSICAL: connect() with empty namespace uses datus_db/ directory."""
        from datus.storage.rdb.sqlite_backend import SqliteRdbBackend

        b = SqliteRdbBackend()
        b.initialize({"data_dir": str(tmp_path), "isolation": "physical"})
        db = b.connect("", "test")
        assert db.db_file == os.path.join(str(tmp_path), "datus_db", "test.db")

    def test_connect_logical_shares_datus_db(self, tmp_path):
        """LOGICAL: connect() always uses datus_db/ regardless of namespace."""
        from datus.storage.rdb.sqlite_backend import SqliteRdbBackend

        b = SqliteRdbBackend()
        b.initialize({"data_dir": str(tmp_path), "isolation": "logical"})
        db_a = b.connect("ns_a", "test")
        db_b = b.connect("ns_b", "test")
        assert db_a.db_file == os.path.join(str(tmp_path), "datus_db", "test.db")
        assert db_b.db_file == db_a.db_file

    def test_connect_default_is_physical(self, tmp_path):
        """Default isolation is PHYSICAL."""
        from datus.storage.rdb.sqlite_backend import SqliteRdbBackend

        b = SqliteRdbBackend()
        b.initialize({"data_dir": str(tmp_path)})
        db = b.connect("ns", "test")
        assert "datus_db_ns" in db.db_file


# ---------------------------------------------------------------------------
# LOGICAL isolation tests
# ---------------------------------------------------------------------------


@dataclass
class _TenantItem:
    """Test record model with datasource_id."""

    name: str = ""
    value: str = ""
    datasource_id: str = ""
    id: int = None


@pytest.fixture
def logical_table_def():
    """Table definition with datasource_id column for LOGICAL isolation tests."""
    return TableDefinition(
        table_name="tenant_items",
        columns=[
            ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
            ColumnDef(name="name", col_type="TEXT", nullable=False),
            ColumnDef(name="value", col_type="TEXT"),
            ColumnDef(name="datasource_id", col_type="TEXT", default=""),
        ],
        indices=[],
    )


class TestSqliteLogicalIsolation:
    """Tests for LOGICAL mode datasource_id auto-injection and filtering."""

    def _make_db(self, tmp_path, namespace: str) -> SqliteRdbDatabase:
        db_file = os.path.join(str(tmp_path), "shared.db")
        return SqliteRdbDatabase(db_file, isolation="logical", datasource_id=namespace)

    def test_insert_injects_datasource_id(self, tmp_path, logical_table_def):
        """In LOGICAL mode, insert() auto-injects datasource_id."""
        db = self._make_db(tmp_path, "tenant_a")
        table = db.ensure_table(logical_table_def)
        table.insert(_TenantItem(name="item1", value="v1"))

        # Query without isolation to see raw data
        raw_db = SqliteRdbDatabase(os.path.join(str(tmp_path), "shared.db"))
        raw_table = raw_db.ensure_table(logical_table_def)
        rows = raw_table.query(_TenantItem)
        assert len(rows) == 1
        assert rows[0].datasource_id == "tenant_a"

    def test_query_filters_by_datasource_id(self, tmp_path, logical_table_def):
        """In LOGICAL mode, query() only returns rows for the current datasource."""
        # Insert via raw DB (no isolation) to set up test data
        raw_db = SqliteRdbDatabase(os.path.join(str(tmp_path), "shared.db"))
        raw_table = raw_db.ensure_table(logical_table_def)
        raw_table.insert(_TenantItem(name="a_item", value="v1", datasource_id="tenant_a"))
        raw_table.insert(_TenantItem(name="b_item", value="v2", datasource_id="tenant_b"))

        # Query via LOGICAL db for tenant_a
        db_a = self._make_db(tmp_path, "tenant_a")
        table_a = db_a.ensure_table(logical_table_def)
        rows = table_a.query(_TenantItem)
        assert len(rows) == 1
        assert rows[0].name == "a_item"

    def test_update_scoped_to_datasource(self, tmp_path, logical_table_def):
        """In LOGICAL mode, update() only affects rows for the current datasource."""
        raw_db = SqliteRdbDatabase(os.path.join(str(tmp_path), "shared.db"))
        raw_table = raw_db.ensure_table(logical_table_def)
        raw_table.insert(_TenantItem(name="shared", value="old_a", datasource_id="tenant_a"))
        raw_table.insert(_TenantItem(name="shared", value="old_b", datasource_id="tenant_b"))

        # Update via tenant_a — should only change tenant_a's row
        db_a = self._make_db(tmp_path, "tenant_a")
        table_a = db_a.ensure_table(logical_table_def)
        table_a.update({"value": "new_a"}, where={"name": "shared"})

        # Verify tenant_b's row is untouched
        rows = raw_table.query(_TenantItem, order_by=["datasource_id"])
        assert rows[0].value == "new_a"  # tenant_a updated
        assert rows[1].value == "old_b"  # tenant_b untouched

    def test_delete_scoped_to_datasource(self, tmp_path, logical_table_def):
        """In LOGICAL mode, delete() only removes rows for the current datasource."""
        raw_db = SqliteRdbDatabase(os.path.join(str(tmp_path), "shared.db"))
        raw_table = raw_db.ensure_table(logical_table_def)
        raw_table.insert(_TenantItem(name="item", value="v1", datasource_id="tenant_a"))
        raw_table.insert(_TenantItem(name="item", value="v2", datasource_id="tenant_b"))

        # Delete via tenant_a
        db_a = self._make_db(tmp_path, "tenant_a")
        table_a = db_a.ensure_table(logical_table_def)
        count = table_a.delete(where={"name": "item"})
        assert count == 1

        # tenant_b's row still exists
        rows = raw_table.query(_TenantItem)
        assert len(rows) == 1
        assert rows[0].datasource_id == "tenant_b"

    def test_upsert_injects_datasource_id(self, tmp_path, logical_table_def):
        """In LOGICAL mode, upsert() auto-injects datasource_id."""
        db = self._make_db(tmp_path, "tenant_a")
        table = db.ensure_table(logical_table_def)
        table.upsert(_TenantItem(name="k1", value="v1"), ["name"])

        raw_db = SqliteRdbDatabase(os.path.join(str(tmp_path), "shared.db"))
        raw_table = raw_db.ensure_table(logical_table_def)
        rows = raw_table.query(_TenantItem)
        assert len(rows) == 1
        assert rows[0].datasource_id == "tenant_a"

    def test_physical_mode_no_injection(self, tmp_path, logical_table_def):
        """In PHYSICAL mode, no datasource_id injection or filtering happens."""
        db_file = os.path.join(str(tmp_path), "physical.db")
        db = SqliteRdbDatabase(db_file, isolation="physical", datasource_id="tenant_a")
        table = db.ensure_table(logical_table_def)
        table.insert(_TenantItem(name="item", value="v1"))

        rows = table.query(_TenantItem)
        assert len(rows) == 1
        assert rows[0].datasource_id == ""  # not injected
