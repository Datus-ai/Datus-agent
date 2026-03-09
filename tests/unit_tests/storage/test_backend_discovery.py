# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for backend discovery with lifecycle management."""

from unittest.mock import MagicMock, patch

import pytest

from tests.unit_tests.storage._backend_discovery import (
    BackendTestConfig,
    _discover_via_registry,
    _teardown_callbacks,
    cleanup_test_environments,
    discover_test_backends,
)


@pytest.fixture(autouse=True)
def _clean_teardown_callbacks():
    """Ensure teardown callbacks are clean before and after each test."""
    _teardown_callbacks.clear()
    yield
    _teardown_callbacks.clear()


class TestDiscoverTestBackends:
    """Tests for the main discover_test_backends() function."""

    def test_always_includes_default(self):
        """sqlite+lance is always first in the returned list."""
        backends = discover_test_backends()
        assert len(backends) >= 1
        assert backends[0].rdb_type == "sqlite"
        assert backends[0].vector_type == "lance"
        assert backends[0].id == "sqlite+lance"

    def test_calls_registry_discovery(self):
        """discover_test_backends() calls _discover_via_registry()."""
        with patch(
            "tests.unit_tests.storage._backend_discovery._discover_via_registry",
            return_value=[],
        ) as mock_discover:
            backends = discover_test_backends()
            mock_discover.assert_called_once()
            assert len(backends) == 1  # only default


class TestDiscoverViaRegistry:
    """Tests for _discover_via_registry() with lifecycle management."""

    def test_paired_backend(self):
        """When both RDB and vector provide config with same name, they are paired."""
        mock_rdb_cls = MagicMock()
        mock_rdb_cls.setup_test_env.return_value = {"host": "localhost", "port": 5432}

        mock_vec_cls = MagicMock()
        mock_vec_cls.setup_test_env.return_value = {"host": "localhost", "port": 5432}

        with (
            patch("datus.storage.rdb.registry.RdbRegistry.registered_types", return_value=["sqlite", "postgres"]),
            patch("datus.storage.vector.registry.VectorRegistry.registered_types", return_value=["lance", "postgres"]),
            patch("datus.storage.rdb.registry.RdbRegistry.get_backend_class", return_value=mock_rdb_cls),
            patch("datus.storage.vector.registry.VectorRegistry.get_backend_class", return_value=mock_vec_cls),
        ):
            configs = _discover_via_registry()

        assert len(configs) == 1
        assert configs[0].rdb_type == "postgres"
        assert configs[0].vector_type == "postgres"
        assert configs[0].rdb_params == {"host": "localhost", "port": 5432}
        assert configs[0].vector_params == {"host": "localhost", "port": 5432}

    def test_rdb_only_pairs_with_lance(self):
        """RDB-only backend pairs with default lance vector."""
        mock_rdb_cls = MagicMock()
        mock_rdb_cls.setup_test_env.return_value = {"host": "localhost"}

        with (
            patch("datus.storage.rdb.registry.RdbRegistry.registered_types", return_value=["sqlite", "mysql"]),
            patch("datus.storage.vector.registry.VectorRegistry.registered_types", return_value=["lance"]),
            patch("datus.storage.rdb.registry.RdbRegistry.get_backend_class", return_value=mock_rdb_cls),
            patch("datus.storage.vector.registry.VectorRegistry.get_backend_class", return_value=None),
        ):
            configs = _discover_via_registry()

        assert len(configs) == 1
        assert configs[0].rdb_type == "mysql"
        assert configs[0].vector_type == "lance"

    def test_vector_only_pairs_with_sqlite(self):
        """Vector-only backend pairs with default sqlite RDB."""
        mock_vec_cls = MagicMock()
        mock_vec_cls.setup_test_env.return_value = {"host": "localhost"}

        with (
            patch("datus.storage.rdb.registry.RdbRegistry.registered_types", return_value=["sqlite"]),
            patch("datus.storage.vector.registry.VectorRegistry.registered_types", return_value=["lance", "milvus"]),
            patch("datus.storage.rdb.registry.RdbRegistry.get_backend_class", return_value=None),
            patch("datus.storage.vector.registry.VectorRegistry.get_backend_class", return_value=mock_vec_cls),
        ):
            configs = _discover_via_registry()

        assert len(configs) == 1
        assert configs[0].rdb_type == "sqlite"
        assert configs[0].vector_type == "milvus"

    def test_setup_returns_none_skips(self):
        """Backend is skipped when setup_test_env() returns None."""
        mock_rdb_cls = MagicMock()
        mock_rdb_cls.setup_test_env.return_value = None

        with (
            patch("datus.storage.rdb.registry.RdbRegistry.registered_types", return_value=["sqlite", "postgres"]),
            patch("datus.storage.vector.registry.VectorRegistry.registered_types", return_value=["lance"]),
            patch("datus.storage.rdb.registry.RdbRegistry.get_backend_class", return_value=mock_rdb_cls),
            patch("datus.storage.vector.registry.VectorRegistry.get_backend_class", return_value=None),
        ):
            configs = _discover_via_registry()

        assert len(configs) == 0

    def test_setup_raises_exception_skips(self):
        """Backend is skipped gracefully when setup_test_env() raises."""
        mock_rdb_cls = MagicMock()
        mock_rdb_cls.setup_test_env.side_effect = RuntimeError("Docker not available")

        with (
            patch("datus.storage.rdb.registry.RdbRegistry.registered_types", return_value=["sqlite", "postgres"]),
            patch("datus.storage.vector.registry.VectorRegistry.registered_types", return_value=["lance"]),
            patch("datus.storage.rdb.registry.RdbRegistry.get_backend_class", return_value=mock_rdb_cls),
            patch("datus.storage.vector.registry.VectorRegistry.get_backend_class", return_value=None),
        ):
            configs = _discover_via_registry()

        assert len(configs) == 0

    def test_get_backend_class_returns_none_skips(self):
        """Backend is skipped when get_backend_class() returns None."""
        with (
            patch("datus.storage.rdb.registry.RdbRegistry.registered_types", return_value=["sqlite", "ghost"]),
            patch("datus.storage.vector.registry.VectorRegistry.registered_types", return_value=["lance"]),
            patch("datus.storage.rdb.registry.RdbRegistry.get_backend_class", return_value=None),
            patch("datus.storage.vector.registry.VectorRegistry.get_backend_class", return_value=None),
        ):
            configs = _discover_via_registry()

        assert len(configs) == 0


class TestCleanupTestEnvironments:
    """Tests for cleanup_test_environments()."""

    def test_teardown_called_on_cleanup(self):
        """Verify teardown_test_env() is called during cleanup."""
        mock_teardown = MagicMock()
        _teardown_callbacks.append(mock_teardown)

        cleanup_test_environments()

        mock_teardown.assert_called_once()
        assert len(_teardown_callbacks) == 0

    def test_teardown_called_in_reverse_order(self):
        """Teardown callbacks are called in reverse registration order."""
        call_order = []
        _teardown_callbacks.append(lambda: call_order.append("first"))
        _teardown_callbacks.append(lambda: call_order.append("second"))

        cleanup_test_environments()

        assert call_order == ["second", "first"]
        assert len(_teardown_callbacks) == 0

    def test_teardown_exception_does_not_stop_others(self):
        """An exception in one teardown does not prevent others from running."""
        call_order = []

        def failing_teardown():
            raise RuntimeError("teardown failed")

        _teardown_callbacks.append(lambda: call_order.append("first"))
        _teardown_callbacks.append(failing_teardown)
        _teardown_callbacks.append(lambda: call_order.append("third"))

        cleanup_test_environments()

        # "third" runs first (reverse), then failing_teardown, then "first"
        assert "first" in call_order
        assert "third" in call_order
        assert len(_teardown_callbacks) == 0

    def test_teardown_registered_during_discovery(self):
        """Teardown callbacks are registered when setup_test_env() succeeds."""
        mock_rdb_cls = MagicMock()
        mock_rdb_cls.setup_test_env.return_value = {"host": "localhost"}

        with (
            patch("datus.storage.rdb.registry.RdbRegistry.registered_types", return_value=["sqlite", "postgres"]),
            patch("datus.storage.vector.registry.VectorRegistry.registered_types", return_value=["lance"]),
            patch("datus.storage.rdb.registry.RdbRegistry.get_backend_class", return_value=mock_rdb_cls),
            patch("datus.storage.vector.registry.VectorRegistry.get_backend_class", return_value=None),
        ):
            _discover_via_registry()

        assert len(_teardown_callbacks) == 1

        # Cleanup calls teardown_test_env
        cleanup_test_environments()
        mock_rdb_cls.teardown_test_env.assert_called_once_with({"host": "localhost"})


class TestBackendTestConfig:
    """Tests for BackendTestConfig dataclass."""

    def test_default_values(self):
        """Default config is sqlite+lance with empty params."""
        cfg = BackendTestConfig()
        assert cfg.rdb_type == "sqlite"
        assert cfg.vector_type == "lance"
        assert cfg.rdb_params == {}
        assert cfg.vector_params == {}

    def test_id_property(self):
        """id property returns 'rdb+vector' format."""
        cfg = BackendTestConfig(rdb_type="postgres", vector_type="pgvector")
        assert cfg.id == "postgres+pgvector"
