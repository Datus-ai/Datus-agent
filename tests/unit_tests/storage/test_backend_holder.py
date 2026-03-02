# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for backend_holder module covering RDB singleton and non-sqlite path."""

import pytest

from datus.storage.backend_config import RdbBackendConfig, StorageBackendConfig
from datus.storage.backend_holder import create_rdb_for_store, init_backends, reset_backends
from datus.storage.rdb.base import BaseRdbBackend, RdbDatabase, RdbTable
from datus.storage.rdb.registry import RdbRegistry
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

    def connect(self, namespace, store_db_name):
        return _StubRdbDatabase()

    def close(self):
        pass


@pytest.fixture(autouse=True)
def clean_state():
    """Reset backend holder and registry between tests."""
    reset_backends()
    RdbRegistry.reset()
    yield
    reset_backends()
    RdbRegistry.reset()


class TestCreateRdbForStoreSqlite:
    """Tests for SQLite path in create_rdb_for_store."""

    def test_sqlite_creates_database(self, tmp_path):
        """SQLite config creates a SqliteRdbDatabase with correct db_file."""
        init_backends(StorageBackendConfig(), data_dir=str(tmp_path), namespace="test")
        db = create_rdb_for_store("test")
        assert isinstance(db, SqliteRdbDatabase)
        assert db.db_file.endswith("test.db")


class TestCreateRdbForStoreNonSqlite:
    """Tests for non-sqlite path in create_rdb_for_store."""

    def test_non_sqlite_uses_registry(self, tmp_path):
        """Non-sqlite config creates backend via RdbRegistry and calls connect()."""
        RdbRegistry.register("stub", _StubRdbBackend)
        config = StorageBackendConfig(
            rdb=RdbBackendConfig(type="stub", params={"host": "myhost", "port": 3306}),
        )
        init_backends(config, data_dir=str(tmp_path), namespace="ns")
        db = create_rdb_for_store("store")
        assert isinstance(db, _StubRdbDatabase)
        assert _StubRdbBackend.last_config["host"] == "myhost"
        assert _StubRdbBackend.last_config["data_dir"] == str(tmp_path)
