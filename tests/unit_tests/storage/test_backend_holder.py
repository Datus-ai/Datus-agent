# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for backend_holder module covering RDB singleton and non-sqlite path."""

from datus_storage_base.backend_config import StorageBackendConfig
from datus_storage_base.rdb.base import BaseRdbBackend, RdbDatabase, RdbTable

from datus.storage.backend_holder import create_rdb_for_store, init_backends
from datus.storage.rdb.sqlite_backend import SqliteRdbDatabase


class _StubRdbTable(RdbTable):
    """Minimal concrete RdbTable for testing."""

    def __init__(self, name):
        self._name = name

    @property
    def table_name(self):
        return self._name

    def insert(self, record):
        return 0

    def query(self, model, where=None, columns=None, order_by=None):
        return []

    def update(self, data, where=None):
        return 0

    def delete(self, where=None):
        return 0

    def upsert(self, record, conflict_columns):
        pass


class _StubRdbDatabase(RdbDatabase):
    """Minimal concrete RdbDatabase for testing."""

    def ensure_table(self, table_def):
        return _StubRdbTable(table_def.table_name)

    def transaction(self):
        pass

    def close(self):
        pass


class _StubRdbBackend(BaseRdbBackend):
    """Stub backend for testing registry-based creation."""

    last_config = None

    def initialize(self, config):
        _StubRdbBackend.last_config = config

    def connect(self, datasource, store_db_name):
        return _StubRdbDatabase()

    def close(self):
        pass


class TestCreateRdbForStoreSqlite:
    """Tests for SQLite path in create_rdb_for_store."""

    def test_sqlite_creates_database(self, tmp_path):
        """create_rdb_for_store builds a project-scoped SqliteRdbDatabase."""
        init_backends(StorageBackendConfig(), data_dir=str(tmp_path))
        db = create_rdb_for_store("test", "proj_a")
        assert isinstance(db, SqliteRdbDatabase)
        assert db.db_file.endswith("test.db")
        assert "proj_a" in db.db_file

    def test_single_backend_reused_across_projects(self, tmp_path):
        """Backend instance is stateless: one init_backends call serves many projects."""
        init_backends(StorageBackendConfig(), data_dir=str(tmp_path))
        db_a = create_rdb_for_store("s", "proj_a")
        db_b = create_rdb_for_store("s", "proj_b")
        assert "proj_a" in db_a.db_file
        assert "proj_b" in db_b.db_file
        assert db_a.db_file != db_b.db_file
