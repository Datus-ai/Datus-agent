# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for RDB registry backend discovery and creation."""

import pytest

from datus.storage.rdb.base import BaseRdbBackend, RdbDatabase, RdbTable
from datus.storage.rdb.registry import RdbRegistry
from datus.utils.exceptions import DatusException


class _DummyRdbTable(RdbTable):
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


class _DummyRdbDatabase(RdbDatabase):
    """Minimal concrete RdbDatabase for testing."""

    def ensure_table(self, table_def):
        return _DummyRdbTable(table_def.table_name)

    def transaction(self):
        pass

    def close(self):
        pass


class _DummyRdbBackend(BaseRdbBackend):
    """Minimal concrete backend for testing."""

    initialized_config = None

    def initialize(self, config):
        self.initialized_config = config

    def connect(self, namespace, store_db_name):
        return _DummyRdbDatabase()

    def close(self):
        pass


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure registry is clean before and after each test."""
    RdbRegistry.reset()
    yield
    RdbRegistry.reset()


class TestRdbRegistryRegister:
    """Tests for register() and is_registered()."""

    def test_register_backend_class(self):
        """Register a backend class and verify it is discoverable."""
        RdbRegistry.register("dummy", _DummyRdbBackend)
        assert RdbRegistry.is_registered("dummy")
        assert RdbRegistry.is_registered("DUMMY")

    def test_register_with_factory(self):
        """Register a backend with a custom factory function."""
        factory_called = {}

        def factory(config):
            factory_called["config"] = config
            return _DummyRdbBackend()

        RdbRegistry.register("custom", _DummyRdbBackend, factory=factory)
        backend = RdbRegistry.create_backend("custom", {"key": "val"})
        assert isinstance(backend, _DummyRdbBackend)
        assert factory_called["config"] == {"key": "val"}

    def test_is_registered_unknown(self):
        """Unknown backend type returns False."""
        assert not RdbRegistry.is_registered("nonexistent")


class TestRdbRegistryCreateBackend:
    """Tests for create_backend()."""

    def test_create_backend_initializes(self):
        """create_backend instantiates and initializes the backend."""
        RdbRegistry.register("dummy", _DummyRdbBackend)
        config = {"host": "localhost"}
        backend = RdbRegistry.create_backend("dummy", config)
        assert isinstance(backend, _DummyRdbBackend)
        assert backend.initialized_config == config

    def test_create_backend_unknown_raises(self):
        """create_backend raises DatusException for unknown types."""
        with pytest.raises(DatusException) as exc_info:
            RdbRegistry.create_backend("nonexistent_xyz", {})
        assert "not found" in str(exc_info.value)

    def test_create_backend_case_insensitive(self):
        """Backend type lookup is case-insensitive."""
        RdbRegistry.register("MyBackend", _DummyRdbBackend)
        backend = RdbRegistry.create_backend("MYBACKEND", {})
        assert isinstance(backend, _DummyRdbBackend)


class TestRdbRegistryPublicAPI:
    """Tests for registered_types() and get_backend_class()."""

    def test_registered_types_returns_all(self):
        """registered_types() returns all registered backend names."""
        RdbRegistry.register("dummy", _DummyRdbBackend)
        RdbRegistry.register("another", _DummyRdbBackend)
        types = RdbRegistry.registered_types()
        assert "dummy" in types
        assert "another" in types

    def test_registered_types_triggers_discovery(self):
        """registered_types() triggers discover_adapters(), so sqlite is always present."""
        types = RdbRegistry.registered_types()
        assert "sqlite" in types

    def test_get_backend_class_found(self):
        """get_backend_class() returns the registered class."""
        RdbRegistry.register("dummy", _DummyRdbBackend)
        cls = RdbRegistry.get_backend_class("dummy")
        assert cls is _DummyRdbBackend

    def test_get_backend_class_case_insensitive(self):
        """get_backend_class() is case-insensitive."""
        RdbRegistry.register("Dummy", _DummyRdbBackend)
        assert RdbRegistry.get_backend_class("DUMMY") is _DummyRdbBackend
        assert RdbRegistry.get_backend_class("dummy") is _DummyRdbBackend

    def test_get_backend_class_not_found(self):
        """get_backend_class() returns None for unknown types."""
        assert RdbRegistry.get_backend_class("nonexistent") is None


class TestRdbRegistryReset:
    """Tests for reset()."""

    def test_reset_clears_all(self):
        """reset() clears backends, factories, and initialized flag."""
        RdbRegistry.register("dummy", _DummyRdbBackend)
        assert RdbRegistry.is_registered("dummy")

        RdbRegistry.reset()
        assert not RdbRegistry.is_registered("dummy")
        assert not RdbRegistry._initialized
