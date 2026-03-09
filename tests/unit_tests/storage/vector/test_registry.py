# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for Vector registry backend discovery and creation."""

import pytest

from datus.storage.vector.base import BaseVectorBackend
from datus.storage.vector.registry import VectorRegistry
from datus.utils.exceptions import DatusException


class _DummyVectorBackend(BaseVectorBackend):
    """Minimal concrete backend for testing."""

    initialized_config = None

    def initialize(self, config):
        self.initialized_config = config

    def connect(self, db_path, namespace=""):
        return None

    def drop_table(self, db_handle, table_name, ignore_missing=False):
        pass

    def table_exists(self, db_handle, table_name):
        return False

    def table_names(self, db_handle, limit=100):
        return []

    def create_table(
        self,
        db_handle,
        table_name,
        schema=None,
        embedding_function=None,
        vector_column="",
        source_column="",
        exist_ok=True,
    ):
        return None

    def open_table(self, db_handle, table_name, embedding_function=None, vector_column="", source_column=""):
        return None

    def add(self, table_handle, data):
        pass

    def merge_insert(self, table_handle, data, on_column):
        pass

    def delete(self, table_handle, where):
        pass

    def update(self, table_handle, where, values):
        pass

    def search_vector(self, table_handle, query_text, vector_column, top_n, where=None, select_fields=None):
        return None

    def search_hybrid(self, table_handle, query_text, vector_source_column, top_n, where=None, select_fields=None):
        return None

    def search_all(self, table_handle, where=None, select_fields=None, limit=None):
        return None

    def count_rows(self, table_handle, where=None):
        return 0

    def create_vector_index(self, table_handle, column, metric="cosine", **kwargs):
        pass

    def create_fts_index(self, table_handle, field_names):
        pass

    def create_scalar_index(self, table_handle, column):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure registry is clean before and after each test."""
    VectorRegistry.reset()
    yield
    VectorRegistry.reset()


class TestVectorRegistryRegister:
    """Tests for register() and is_registered()."""

    def test_register_backend_class(self):
        """Register a backend class and verify it is discoverable."""
        VectorRegistry.register("dummy", _DummyVectorBackend)
        assert VectorRegistry.is_registered("dummy")
        assert VectorRegistry.is_registered("DUMMY")

    def test_register_with_factory(self):
        """Register a backend with a custom factory function."""
        factory_called = {}

        def factory(config):
            factory_called["config"] = config
            return _DummyVectorBackend()

        VectorRegistry.register("custom", _DummyVectorBackend, factory=factory)
        backend = VectorRegistry.create_backend("custom", {"key": "val"})
        assert isinstance(backend, _DummyVectorBackend)
        assert factory_called["config"] == {"key": "val"}

    def test_is_registered_unknown(self):
        """Unknown backend type returns False."""
        assert not VectorRegistry.is_registered("nonexistent")


class TestVectorRegistryCreateBackend:
    """Tests for create_backend()."""

    def test_create_backend_initializes(self):
        """create_backend instantiates and initializes the backend."""
        VectorRegistry.register("dummy", _DummyVectorBackend)
        config = {"host": "localhost"}
        backend = VectorRegistry.create_backend("dummy", config)
        assert isinstance(backend, _DummyVectorBackend)
        assert backend.initialized_config == config

    def test_create_backend_unknown_raises(self):
        """create_backend raises DatusException for unknown types."""
        with pytest.raises(DatusException) as exc_info:
            VectorRegistry.create_backend("nonexistent_xyz", {})
        assert "not found" in str(exc_info.value)

    def test_create_backend_case_insensitive(self):
        """Backend type lookup is case-insensitive."""
        VectorRegistry.register("MyBackend", _DummyVectorBackend)
        backend = VectorRegistry.create_backend("MYBACKEND", {})
        assert isinstance(backend, _DummyVectorBackend)


class TestVectorRegistryPublicAPI:
    """Tests for registered_types() and get_backend_class()."""

    def test_registered_types_returns_all(self):
        """registered_types() returns all registered backend names."""
        VectorRegistry.register("dummy", _DummyVectorBackend)
        VectorRegistry.register("another", _DummyVectorBackend)
        types = VectorRegistry.registered_types()
        assert "dummy" in types
        assert "another" in types

    def test_registered_types_triggers_discovery(self):
        """registered_types() triggers discover_adapters(), so lance is always present."""
        types = VectorRegistry.registered_types()
        assert "lance" in types

    def test_get_backend_class_found(self):
        """get_backend_class() returns the registered class."""
        VectorRegistry.register("dummy", _DummyVectorBackend)
        cls = VectorRegistry.get_backend_class("dummy")
        assert cls is _DummyVectorBackend

    def test_get_backend_class_case_insensitive(self):
        """get_backend_class() is case-insensitive."""
        VectorRegistry.register("Dummy", _DummyVectorBackend)
        assert VectorRegistry.get_backend_class("DUMMY") is _DummyVectorBackend
        assert VectorRegistry.get_backend_class("dummy") is _DummyVectorBackend

    def test_get_backend_class_not_found(self):
        """get_backend_class() returns None for unknown types."""
        assert VectorRegistry.get_backend_class("nonexistent") is None


class TestVectorRegistryReset:
    """Tests for reset()."""

    def test_reset_clears_all(self):
        """reset() clears backends, factories, and initialized flag."""
        VectorRegistry.register("dummy", _DummyVectorBackend)
        assert VectorRegistry.is_registered("dummy")

        VectorRegistry.reset()
        assert not VectorRegistry.is_registered("dummy")
        assert not VectorRegistry._initialized
