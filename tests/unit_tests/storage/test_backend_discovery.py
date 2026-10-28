# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for side-effect-free backend discovery and fixture-owned lifecycle."""

from unittest.mock import MagicMock, patch

import pytest
from datus_storage_base.testing import RdbTestEnv, TestEnvConfig, VectorTestEnv

from tests.unit_tests.storage._backend_discovery import (
    BackendTestConfig,
    BackendTestSpec,
    _discover_via_entry_points,
    _LanceTestEnv,
    _SqliteTestEnv,
    discover_test_backends,
    setup_test_backend,
    teardown_test_backend,
)


def _make_mock_rdb_env(backend_type="postgresql", params=None):
    env = MagicMock(spec=RdbTestEnv)
    env.get_config.return_value = TestEnvConfig(
        backend_type=backend_type,
        params=params or {"host": "localhost", "port": 5432},
    )
    return env


def _make_mock_vector_env(backend_type="postgresql", params=None):
    env = MagicMock(spec=VectorTestEnv)
    env.get_config.return_value = TestEnvConfig(
        backend_type=backend_type,
        params=params or {"host": "localhost", "port": 5432},
    )
    return env


class TestDiscoverTestBackends:
    def test_always_includes_default(self):
        with patch(
            "tests.unit_tests.storage._backend_discovery._discover_via_entry_points",
            return_value=[],
        ):
            backends = discover_test_backends()

        assert backends == [
            BackendTestSpec(
                rdb_factory=_SqliteTestEnv,
                vector_factory=_LanceTestEnv,
            )
        ]

    def test_entry_point_discovery_does_not_construct_environments(self):
        rdb_factory = MagicMock()
        vector_factory = MagicMock()

        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: {"postgresql": rdb_factory} if "rdb" in group else {"postgresql": vector_factory},
        ):
            specs = _discover_via_entry_points()

        assert [spec.id for spec in specs] == ["postgresql+postgresql"]
        rdb_factory.assert_not_called()
        vector_factory.assert_not_called()

    def test_rdb_only_pairs_with_lance(self):
        rdb_factory = MagicMock()
        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: {"mysql": rdb_factory} if "rdb" in group else {},
        ):
            specs = _discover_via_entry_points()

        assert specs == [BackendTestSpec(rdb_type="mysql", vector_type="lance", rdb_factory=rdb_factory)]

    def test_vector_only_pairs_with_sqlite(self):
        vector_factory = MagicMock()
        with patch(
            "tests.unit_tests.storage._backend_discovery._load_entry_points",
            side_effect=lambda group: {} if "rdb" in group else {"milvus": vector_factory},
        ):
            specs = _discover_via_entry_points()

        assert specs == [BackendTestSpec(rdb_type="sqlite", vector_type="milvus", vector_factory=vector_factory)]


class TestBackendLifecycle:
    def test_setup_and_teardown_paired_backend(self):
        teardown_order = []
        rdb_env = _make_mock_rdb_env()
        vector_env = _make_mock_vector_env()
        rdb_env.teardown.side_effect = lambda: teardown_order.append("rdb")
        vector_env.teardown.side_effect = lambda: teardown_order.append("vector")
        spec = BackendTestSpec(
            rdb_type="postgresql",
            vector_type="postgresql",
            rdb_factory=MagicMock(return_value=rdb_env),
            vector_factory=MagicMock(return_value=vector_env),
        )

        backend = setup_test_backend(spec)
        teardown_test_backend(backend)

        rdb_env.setup.assert_called_once()
        vector_env.setup.assert_called_once()
        rdb_env.teardown.assert_called_once()
        vector_env.teardown.assert_called_once()
        assert teardown_order == ["vector", "rdb"]
        assert backend.id == "postgresql+postgresql"

    def test_partial_setup_failure_tears_down_started_environments(self):
        teardown_order = []
        rdb_env = _make_mock_rdb_env()
        vector_env = _make_mock_vector_env()
        vector_env.setup.side_effect = RuntimeError("Docker unavailable")
        rdb_env.teardown.side_effect = lambda: teardown_order.append("rdb")
        vector_env.teardown.side_effect = lambda: teardown_order.append("vector")
        spec = BackendTestSpec(
            rdb_type="postgresql",
            vector_type="postgresql",
            rdb_factory=MagicMock(return_value=rdb_env),
            vector_factory=MagicMock(return_value=vector_env),
        )

        with pytest.raises(RuntimeError, match="Docker unavailable"):
            setup_test_backend(spec)

        rdb_env.teardown.assert_called_once()
        vector_env.teardown.assert_called_once()
        assert teardown_order == ["vector", "rdb"]

    def test_keyboard_interrupt_tears_down_started_environments(self):
        teardown_order = []
        rdb_env = _make_mock_rdb_env()
        vector_env = _make_mock_vector_env()
        vector_env.setup.side_effect = KeyboardInterrupt
        rdb_env.teardown.side_effect = lambda: teardown_order.append("rdb")
        vector_env.teardown.side_effect = lambda: teardown_order.append("vector")
        spec = BackendTestSpec(
            rdb_type="postgresql",
            vector_type="postgresql",
            rdb_factory=MagicMock(return_value=rdb_env),
            vector_factory=MagicMock(return_value=vector_env),
        )

        with pytest.raises(KeyboardInterrupt):
            setup_test_backend(spec)

        rdb_env.teardown.assert_called_once()
        vector_env.teardown.assert_called_once()
        assert teardown_order == ["vector", "rdb"]

    def test_factory_failure_does_not_leak_previous_environment(self):
        rdb_env = _make_mock_rdb_env()
        spec = BackendTestSpec(
            rdb_type="postgresql",
            vector_type="postgresql",
            rdb_factory=MagicMock(return_value=rdb_env),
            vector_factory=MagicMock(side_effect=RuntimeError("factory failed")),
        )

        with pytest.raises(RuntimeError, match="factory failed"):
            setup_test_backend(spec)

        rdb_env.teardown.assert_called_once()

    def test_teardown_exception_does_not_stop_other_cleanup(self):
        teardown_order = []
        rdb_env = _make_mock_rdb_env()
        vector_env = _make_mock_vector_env()

        def fail_vector_teardown():
            teardown_order.append("vector")
            raise RuntimeError("teardown failed")

        rdb_env.teardown.side_effect = lambda: teardown_order.append("rdb")
        vector_env.teardown.side_effect = fail_vector_teardown
        backend = BackendTestConfig(rdb_test_env=rdb_env, vector_test_env=vector_env)

        teardown_test_backend(backend)

        vector_env.teardown.assert_called_once()
        rdb_env.teardown.assert_called_once()
        assert teardown_order == ["vector", "rdb"]

    def test_teardown_interrupt_does_not_stop_other_cleanup(self):
        teardown_order = []
        rdb_env = _make_mock_rdb_env()
        vector_env = _make_mock_vector_env()

        def interrupt_vector_teardown():
            teardown_order.append("vector")
            raise KeyboardInterrupt

        rdb_env.teardown.side_effect = lambda: teardown_order.append("rdb")
        vector_env.teardown.side_effect = interrupt_vector_teardown
        backend = BackendTestConfig(rdb_test_env=rdb_env, vector_test_env=vector_env)

        with pytest.raises(KeyboardInterrupt):
            teardown_test_backend(backend)

        vector_env.teardown.assert_called_once()
        rdb_env.teardown.assert_called_once()
        assert teardown_order == ["vector", "rdb"]


class TestBackendTestConfig:
    def test_default_values(self):
        config = BackendTestConfig()

        assert config.id == "sqlite+lance"
        assert config.rdb_params == {}
        assert config.vector_params == {}
        assert config.rdb_test_env is None
        assert config.vector_test_env is None
