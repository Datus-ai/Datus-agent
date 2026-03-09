# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Discover available storage backends for parameterized testing.

Backends self-manage test environments via setup_test_env() / teardown_test_env().
No environment variables needed -- installed adapters are auto-discovered.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_DEFAULT_RDB = "sqlite"
_DEFAULT_VECTOR = "lance"

# Cleanup callbacks registered during discovery
_teardown_callbacks: List[Callable[[], None]] = []


@dataclass
class BackendTestConfig:
    """Describes one rdb+vector backend combination for parameterized tests."""

    rdb_type: str = "sqlite"
    vector_type: str = "lance"
    rdb_params: Dict[str, Any] = field(default_factory=dict)
    vector_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.rdb_type}+{self.vector_type}"


def _discover_via_registry() -> List[BackendTestConfig]:
    """Auto-discover backends via registry, calling setup_test_env() for each.

    Pairing: same name in both registries -> pair together;
    RDB-only -> pair with lance; Vector-only -> pair with sqlite.
    """
    configs: List[BackendTestConfig] = []
    try:
        from datus.storage.rdb.registry import RdbRegistry
        from datus.storage.vector.registry import VectorRegistry

        # Collect non-default backends that provide test environments
        rdb_configs: Dict[str, Dict[str, Any]] = {}
        for rdb_type in RdbRegistry.registered_types():
            if rdb_type == _DEFAULT_RDB:
                continue
            backend_cls = RdbRegistry.get_backend_class(rdb_type)
            if backend_cls is None:
                continue
            try:
                cfg = backend_cls.setup_test_env()
            except Exception as e:
                logger.debug(f"setup_test_env() failed for RDB '{rdb_type}': {e}")
                continue
            if cfg is not None:
                rdb_configs[rdb_type] = cfg
                # Register teardown callback
                cls_ref = backend_cls
                cfg_ref = cfg
                _teardown_callbacks.append(lambda c=cls_ref, p=cfg_ref: c.teardown_test_env(p))
                logger.info(f"RDB backend '{rdb_type}' test env ready")

        vector_configs: Dict[str, Dict[str, Any]] = {}
        for vec_type in VectorRegistry.registered_types():
            if vec_type == _DEFAULT_VECTOR:
                continue
            backend_cls = VectorRegistry.get_backend_class(vec_type)
            if backend_cls is None:
                continue
            try:
                cfg = backend_cls.setup_test_env()
            except Exception as e:
                logger.debug(f"setup_test_env() failed for vector '{vec_type}': {e}")
                continue
            if cfg is not None:
                vector_configs[vec_type] = cfg
                cls_ref = backend_cls
                cfg_ref = cfg
                _teardown_callbacks.append(lambda c=cls_ref, p=cfg_ref: c.teardown_test_env(p))
                logger.info(f"Vector backend '{vec_type}' test env ready")

        # Pair by common name
        all_names = set(rdb_configs.keys()) | set(vector_configs.keys())
        for name in sorted(all_names):
            rdb_type = name if name in rdb_configs else _DEFAULT_RDB
            vec_type = name if name in vector_configs else _DEFAULT_VECTOR
            if rdb_type == _DEFAULT_RDB and vec_type == _DEFAULT_VECTOR:
                continue
            configs.append(
                BackendTestConfig(
                    rdb_type=rdb_type,
                    vector_type=vec_type,
                    rdb_params=rdb_configs.get(name, {}),
                    vector_params=vector_configs.get(name, {}),
                )
            )
            logger.info(f"Auto-discovered backend combo: {rdb_type}+{vec_type}")

    except Exception as e:
        logger.debug(f"Registry-based backend discovery failed: {e}")

    return configs


def cleanup_test_environments() -> None:
    """Tear down all test environments created during discovery."""
    for callback in reversed(_teardown_callbacks):
        try:
            callback()
        except Exception as e:
            logger.debug(f"teardown_test_env() failed: {e}")
    _teardown_callbacks.clear()


# Safety net: ensure cleanup even if pytest crashes
atexit.register(cleanup_test_environments)


def discover_test_backends() -> List[BackendTestConfig]:
    """Return the list of backend configs to parameterize storage tests with.

    1. Always includes the default sqlite+lance config.
    2. Auto-discovers additional backends via registry.
    """
    backends = [BackendTestConfig()]  # default: sqlite + lance
    backends.extend(_discover_via_registry())
    return backends
