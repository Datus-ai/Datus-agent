# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Discover available storage backends for parameterized testing.

Test environments are discovered via entry points:
    datus.storage.rdb.testing    -- RDB test environment providers
    datus.storage.vector.testing -- Vector test environment providers

Each entry point references a factory function returning a TestEnv instance.
Built-in backends (sqlite, lance) use no-op TestEnv implementations.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datus_storage_base.testing import RdbTestEnv, TestEnvConfig, VectorTestEnv

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_DEFAULT_RDB = "sqlite"
_DEFAULT_VECTOR = "lance"

# Active test environments keyed by entry point name
_active_rdb_envs: Dict[str, RdbTestEnv] = {}
_active_vector_envs: Dict[str, VectorTestEnv] = {}


# ---------------------------------------------------------------------------
# Built-in no-op TestEnv implementations for file-based backends
# ---------------------------------------------------------------------------


class _SqliteTestEnv(RdbTestEnv):
    """No-op test environment for SQLite (file-based, relies on tmp_path)."""

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def clear_data(self, datasource: str) -> None:
        pass  # tmp_path provides isolation

    def get_config(self) -> TestEnvConfig:
        return TestEnvConfig(backend_type="sqlite", params={})


class _LanceTestEnv(VectorTestEnv):
    """No-op test environment for LanceDB (file-based, relies on tmp_path)."""

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def clear_data(self, datasource: str) -> None:
        pass  # tmp_path provides isolation

    def get_config(self) -> TestEnvConfig:
        return TestEnvConfig(backend_type="lance", params={})


# Singleton instances for built-in backends
_builtin_sqlite_env = _SqliteTestEnv()
_builtin_lance_env = _LanceTestEnv()


# ---------------------------------------------------------------------------
# BackendTestConfig
# ---------------------------------------------------------------------------


@dataclass
class BackendTestConfig:
    """Describes one rdb+vector backend combination for parameterized tests."""

    rdb_type: str = "sqlite"
    vector_type: str = "lance"
    rdb_params: Dict[str, Any] = field(default_factory=dict)
    vector_params: Dict[str, Any] = field(default_factory=dict)
    rdb_test_env: Optional[RdbTestEnv] = field(default=None, repr=False)
    vector_test_env: Optional[VectorTestEnv] = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return f"{self.rdb_type}+{self.vector_type}"


# ---------------------------------------------------------------------------
# Entry-point-based discovery
# ---------------------------------------------------------------------------


def _load_entry_points(group: str) -> Dict[str, Any]:
    """Load entry points for the given group, returning {name: loaded_object}."""
    results: Dict[str, Any] = {}
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group=group)
        for ep in eps:
            try:
                results[ep.name] = ep.load()
            except Exception as e:
                logger.debug(f"Failed to load entry point '{ep.name}' from '{group}': {e}")
    except Exception as e:
        logger.debug(f"Failed to scan entry points for group '{group}': {e}")
    return results


def _discover_via_entry_points() -> List[BackendTestConfig]:
    """Auto-discover test environments via entry points.

    Discovery flow:
        1. Scan entry_points("datus.storage.rdb.testing")
        2. For each: factory = ep.load(), env = factory(), env.setup()
        3. Scan entry_points("datus.storage.vector.testing")
        4. Pair by entry point name (same name -> pair together)
        5. Single-side entries pair with the default backend
    """
    configs: List[BackendTestConfig] = []

    # Discover RDB test environments
    rdb_factories = _load_entry_points("datus.storage.rdb.testing")
    for name, factory in rdb_factories.items():
        try:
            env = factory()
            env.setup()
            _active_rdb_envs[name] = env
            logger.info(f"RDB test env '{name}' ready")
        except Exception as e:
            logger.debug(f"RDB test env '{name}' setup failed: {e}")

    # Discover Vector test environments
    vector_factories = _load_entry_points("datus.storage.vector.testing")
    for name, factory in vector_factories.items():
        try:
            env = factory()
            env.setup()
            _active_vector_envs[name] = env
            logger.info(f"Vector test env '{name}' ready")
        except Exception as e:
            logger.debug(f"Vector test env '{name}' setup failed: {e}")

    # Pair by common name
    all_names = set(_active_rdb_envs.keys()) | set(_active_vector_envs.keys())
    for name in sorted(all_names):
        rdb_env = _active_rdb_envs.get(name)
        vec_env = _active_vector_envs.get(name)

        if rdb_env is not None:
            rdb_cfg = rdb_env.get_config()
            rdb_type = rdb_cfg.backend_type
            rdb_params = rdb_cfg.params
        else:
            rdb_type = _DEFAULT_RDB
            rdb_params = {}

        if vec_env is not None:
            vec_cfg = vec_env.get_config()
            vec_type = vec_cfg.backend_type
            vec_params = vec_cfg.params
        else:
            vec_type = _DEFAULT_VECTOR
            vec_params = {}

        if rdb_type == _DEFAULT_RDB and vec_type == _DEFAULT_VECTOR:
            continue

        configs.append(
            BackendTestConfig(
                rdb_type=rdb_type,
                vector_type=vec_type,
                rdb_params=rdb_params,
                vector_params=vec_params,
                rdb_test_env=rdb_env,
                vector_test_env=vec_env,
            )
        )
        logger.info(f"Auto-discovered backend combo: {rdb_type}+{vec_type}")

    return configs


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_test_environments() -> None:
    """Tear down all active test environments."""
    for name in reversed(list(_active_vector_envs.keys())):
        try:
            _active_vector_envs[name].teardown()
        except Exception as e:
            logger.debug(f"Vector test env '{name}' teardown failed: {e}")
    _active_vector_envs.clear()

    for name in reversed(list(_active_rdb_envs.keys())):
        try:
            _active_rdb_envs[name].teardown()
        except Exception as e:
            logger.debug(f"RDB test env '{name}' teardown failed: {e}")
    _active_rdb_envs.clear()


# Safety net: ensure cleanup even if pytest crashes
atexit.register(cleanup_test_environments)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_test_backends() -> List[BackendTestConfig]:
    """Return the list of backend configs to parameterize storage tests with.

    1. Always includes the default sqlite+lance config (with built-in TestEnv).
    2. Auto-discovers additional backends via entry points.
    """
    backends = [
        BackendTestConfig(
            rdb_test_env=_builtin_sqlite_env,
            vector_test_env=_builtin_lance_env,
        )
    ]
    backends.extend(_discover_via_entry_points())
    return backends
