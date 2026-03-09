# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for backend discovery with TestEnv-based lifecycle management."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.testing import RdbTestEnv, TestEnvConfig, VectorTestEnv
from tests.unit_tests.storage._backend_discovery import (
    BackendTestConfig,
    _active_rdb_envs,
    _active_vector_envs,
    _discover_via_entry_points,
    cleanup_test_environments,
    discover_test_backends,
)


@pytest.fixture(autouse=True)
def _clean_active_envs():
    """Ensure active env dicts are clean before and after each test."""
    _active_rdb_envs.clear()
    _active_vector_envs.clear()
    yield
    _active_rdb_envs.clear()
    _active_vector_envs.clear()


def _make_mock_rdb_env(backend_type="postgresql", params=None):
    """Create a mock RdbTestEnv that returns the given config."""
    env = MagicMock(spec=RdbTestEnv)
    env.get_config.return_value = TestEnvConfig(
        backend_type=backend_type,
        params=params or {"host": "localhost", "port": 5432},
    )
    return env


def _make_mock_vector_env(backend_type="postgresql", params=None):
    """Create a mock VectorTestEnv that returns the given config."""
    env = MagicMock(spec=VectorTestEnv)
    env.get_config.return_value = TestEnvConfig(
        backend_type=backend_type,
        params=params or {"host": "localhost", "port": 5432},
    )
    return env


def _make_mock_factory(env):
    """Create a factory function that returns the given env."""
    return MagicMock(return_value=env)


class TestDiscoverTestBackends:
    """Tests for the main discover_test_backends() function."""

    def test_always_includes_default(self):
        """sqlite+lance is always first in the returned list."""
        backends = discover_test_backends()
        assert len(backends) >= 1
        assert backends[0].rdb_type == "sqlite"
        assert backends[0].vector_type == "lance"
        assert backends[0].id == "sqlite+lance"

    def test_default_has_builtin_test_envs(self):
        """Default config has built-in TestEnv instances."""
        backends = discover_test_backends()
        assert backends[0].rdb_test_env is not None
        assert backends[0].vector_test_env is not None

    def test_calls_entry_point_discovery(self):
        """discover_test_backends() calls _discover_via_entry_points()."""
        with patch(
            "tests.unit_tests.storage._backend_discovery._discover_via_entry_points",
            return_value=[],
        ) as mock_discover:
            backends = discover_test_backends()
            mock_discover.assert_called_once()
            assert len(backends) == 1  # only default


class TestDiscoverViaEntryPoints:
    """Tests for _discover_via_entry_points() with TestEnv-based lifecycle."""

    def test_paired_backend(self):
        """When both RDB and vector have same entry point name, they are paired."""
        rdb_env = _make_mock_rdb_env()
        vec_env = _make_mock_vector_env()

        rdb_factories = {"postgresql": _make_mock_factory(rdb_env)}
        vec_factories = {"postgresql": _make_mock_factory(vec_env)}

        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: rdb_factories if "rdb" in group else vec_factories,
        ):
            configs = _discover_via_entry_points()

        assert len(configs) == 1
        assert configs[0].rdb_type == "postgresql"
        assert configs[0].vector_type == "postgresql"
        assert configs[0].rdb_test_env is rdb_env
        assert configs[0].vector_test_env is vec_env
        rdb_env.setup.assert_called_once()
        vec_env.setup.assert_called_once()

    def test_rdb_only_pairs_with_lance(self):
        """RDB-only entry point pairs with default lance vector."""
        rdb_env = _make_mock_rdb_env(backend_type="mysql", params={"host": "localhost"})
        rdb_factories = {"mysql": _make_mock_factory(rdb_env)}

        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: rdb_factories if "rdb" in group else {},
        ):
            configs = _discover_via_entry_points()

        assert len(configs) == 1
        assert configs[0].rdb_type == "mysql"
        assert configs[0].vector_type == "lance"
        assert configs[0].rdb_test_env is rdb_env
        assert configs[0].vector_test_env is None

    def test_vector_only_pairs_with_sqlite(self):
        """Vector-only entry point pairs with default sqlite RDB."""
        vec_env = _make_mock_vector_env(backend_type="milvus", params={"host": "localhost"})
        vec_factories = {"milvus": _make_mock_factory(vec_env)}

        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: {} if "rdb" in group else vec_factories,
        ):
            configs = _discover_via_entry_points()

        assert len(configs) == 1
        assert configs[0].rdb_type == "sqlite"
        assert configs[0].vector_type == "milvus"
        assert configs[0].rdb_test_env is None
        assert configs[0].vector_test_env is vec_env

    def test_setup_raises_exception_skips(self):
        """Entry point is skipped gracefully when setup() raises."""
        rdb_env = _make_mock_rdb_env()
        rdb_env.setup.side_effect = RuntimeError("Docker not available")
        rdb_factories = {"postgresql": _make_mock_factory(rdb_env)}

        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: rdb_factories if "rdb" in group else {},
        ):
            configs = _discover_via_entry_points()

        assert len(configs) == 0

    def test_factory_raises_exception_skips(self):
        """Entry point is skipped gracefully when factory() raises."""
        factory = MagicMock(side_effect=RuntimeError("Import error"))
        rdb_factories = {"postgresql": factory}

        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: rdb_factories if "rdb" in group else {},
        ):
            configs = _discover_via_entry_points()

        assert len(configs) == 0


class TestCleanupTestEnvironments:
    """Tests for cleanup_test_environments()."""

    def test_teardown_called_on_cleanup(self):
        """Verify teardown() is called during cleanup."""
        rdb_env = _make_mock_rdb_env()
        _active_rdb_envs["postgresql"] = rdb_env

        cleanup_test_environments()

        rdb_env.teardown.assert_called_once()
        assert len(_active_rdb_envs) == 0
        assert len(_active_vector_envs) == 0

    def test_teardown_called_for_both_types(self):
        """Both RDB and vector environments are torn down."""
        rdb_env = _make_mock_rdb_env()
        vec_env = _make_mock_vector_env()
        _active_rdb_envs["postgresql"] = rdb_env
        _active_vector_envs["postgresql"] = vec_env

        cleanup_test_environments()

        rdb_env.teardown.assert_called_once()
        vec_env.teardown.assert_called_once()
        assert len(_active_rdb_envs) == 0
        assert len(_active_vector_envs) == 0

    def test_teardown_exception_does_not_stop_others(self):
        """An exception in one teardown does not prevent others from running."""
        rdb_env1 = _make_mock_rdb_env()
        rdb_env2 = _make_mock_rdb_env()
        rdb_env1.teardown.side_effect = RuntimeError("teardown failed")
        _active_rdb_envs["first"] = rdb_env1
        _active_rdb_envs["second"] = rdb_env2

        cleanup_test_environments()

        rdb_env1.teardown.assert_called_once()
        rdb_env2.teardown.assert_called_once()
        assert len(_active_rdb_envs) == 0

    def test_teardown_registered_during_discovery(self):
        """Test environments are registered in _active_*_envs during discovery."""
        rdb_env = _make_mock_rdb_env()
        rdb_factories = {"postgresql": _make_mock_factory(rdb_env)}

        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: rdb_factories if "rdb" in group else {},
        ):
            _discover_via_entry_points()

        assert "postgresql" in _active_rdb_envs

        # Cleanup calls teardown
        cleanup_test_environments()
        rdb_env.teardown.assert_called_once()


class TestBackendTestConfig:
    """Tests for BackendTestConfig dataclass."""

    def test_default_values(self):
        """Default config is sqlite+lance with empty params."""
        cfg = BackendTestConfig()
        assert cfg.rdb_type == "sqlite"
        assert cfg.vector_type == "lance"
        assert cfg.rdb_params == {}
        assert cfg.vector_params == {}
        assert cfg.rdb_test_env is None
        assert cfg.vector_test_env is None

    def test_id_property(self):
        """id property returns 'rdb+vector' format."""
        cfg = BackendTestConfig(rdb_type="postgres", vector_type="pgvector")
        assert cfg.id == "postgres+pgvector"

    def test_with_test_env(self):
        """BackendTestConfig can hold TestEnv references."""
        rdb_env = _make_mock_rdb_env()
        vec_env = _make_mock_vector_env()
        cfg = BackendTestConfig(
            rdb_type="postgresql",
            vector_type="postgresql",
            rdb_test_env=rdb_env,
            vector_test_env=vec_env,
        )
        assert cfg.rdb_test_env is rdb_env
        assert cfg.vector_test_env is vec_env
