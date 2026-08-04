# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Discover and provision storage backends for parameterized tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from datus_storage_base.testing import RdbTestEnv, TestEnvConfig, VectorTestEnv

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_DEFAULT_RDB = "sqlite"
_DEFAULT_VECTOR = "lance"

RdbTestEnvFactory = Callable[[], RdbTestEnv]
VectorTestEnvFactory = Callable[[], VectorTestEnv]


class _SqliteTestEnv(RdbTestEnv):
    """No-op test environment for SQLite (file-based, relies on tmp_path)."""

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def clear_data(self, datasource: str) -> None:
        pass

    def get_config(self) -> TestEnvConfig:
        return TestEnvConfig(backend_type=_DEFAULT_RDB, params={})


class _LanceTestEnv(VectorTestEnv):
    """No-op test environment for LanceDB (file-based, relies on tmp_path)."""

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def clear_data(self, datasource: str) -> None:
        pass

    def get_config(self) -> TestEnvConfig:
        return TestEnvConfig(backend_type=_DEFAULT_VECTOR, params={})


@dataclass(frozen=True)
class BackendTestSpec:
    """A side-effect-free description of one backend test combination."""

    rdb_type: str = _DEFAULT_RDB
    vector_type: str = _DEFAULT_VECTOR
    rdb_factory: RdbTestEnvFactory | None = field(default=None, repr=False)
    vector_factory: VectorTestEnvFactory | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return f"{self.rdb_type}+{self.vector_type}"


@dataclass
class BackendTestConfig:
    """A provisioned rdb+vector backend combination."""

    rdb_type: str = _DEFAULT_RDB
    vector_type: str = _DEFAULT_VECTOR
    rdb_params: dict[str, Any] = field(default_factory=dict)
    vector_params: dict[str, Any] = field(default_factory=dict)
    rdb_test_env: RdbTestEnv | None = field(default=None, repr=False)
    vector_test_env: VectorTestEnv | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return f"{self.rdb_type}+{self.vector_type}"


def _load_entry_points(group: str) -> dict[str, Any]:
    """Load entry points for the given group, returning {name: loaded_object}."""
    results: dict[str, Any] = {}
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


def _discover_via_entry_points() -> list[BackendTestSpec]:
    """Discover backend factories without constructing or starting environments."""
    rdb_factories = _load_entry_points("datus.storage.rdb.testing")
    vector_factories = _load_entry_points("datus.storage.vector.testing")
    specs: list[BackendTestSpec] = []

    for name in sorted(set(rdb_factories) | set(vector_factories)):
        rdb_factory = rdb_factories.get(name)
        vector_factory = vector_factories.get(name)
        rdb_type = name if rdb_factory is not None else _DEFAULT_RDB
        vector_type = name if vector_factory is not None else _DEFAULT_VECTOR
        if rdb_type == _DEFAULT_RDB and vector_type == _DEFAULT_VECTOR:
            continue
        specs.append(
            BackendTestSpec(
                rdb_type=rdb_type,
                vector_type=vector_type,
                rdb_factory=rdb_factory,
                vector_factory=vector_factory,
            )
        )
    return specs


def discover_test_backends() -> list[BackendTestSpec]:
    """Return backend specifications without starting external resources."""
    backends = [
        BackendTestSpec(
            rdb_factory=_SqliteTestEnv,
            vector_factory=_LanceTestEnv,
        )
    ]
    backends.extend(_discover_via_entry_points())
    return backends


def _teardown_environments(environments: list[RdbTestEnv | VectorTestEnv]) -> None:
    for env in reversed(environments):
        try:
            env.teardown()
        except Exception as e:
            logger.debug(f"Test environment teardown failed: {e}")


def setup_test_backend(spec: BackendTestSpec) -> BackendTestConfig:
    """Provision one backend specification and clean partial setup on failure."""
    started: list[RdbTestEnv | VectorTestEnv] = []
    rdb_env: RdbTestEnv | None = None
    vector_env: VectorTestEnv | None = None
    try:
        if spec.rdb_factory is not None:
            rdb_env = spec.rdb_factory()
            started.append(rdb_env)
            rdb_env.setup()
            rdb_config = rdb_env.get_config()
        else:
            rdb_config = TestEnvConfig(backend_type=spec.rdb_type, params={})

        if spec.vector_factory is not None:
            vector_env = spec.vector_factory()
            started.append(vector_env)
            vector_env.setup()
            vector_config = vector_env.get_config()
        else:
            vector_config = TestEnvConfig(backend_type=spec.vector_type, params={})
    except BaseException:
        _teardown_environments(started)
        raise

    return BackendTestConfig(
        rdb_type=rdb_config.backend_type,
        vector_type=vector_config.backend_type,
        rdb_params=rdb_config.params,
        vector_params=vector_config.params,
        rdb_test_env=rdb_env,
        vector_test_env=vector_env,
    )


def teardown_test_backend(backend: BackendTestConfig) -> None:
    """Tear down a provisioned backend in reverse setup order."""
    environments = [env for env in (backend.rdb_test_env, backend.vector_test_env) if env is not None]
    _teardown_environments(environments)
