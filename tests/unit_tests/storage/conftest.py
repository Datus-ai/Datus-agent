"""Shared fixtures for storage tests.

The autouse ``_init_storage_backends`` fixture is parameterized across all
discovered backends so that store-level tests automatically repeat on every
available rdb+vector combination.
"""

from collections.abc import Iterator

import pytest
from datus_storage_base.backend_config import RdbBackendConfig, StorageBackendConfig, VectorBackendConfig

from datus.storage.backend_holder import init_backends, reset_backends
from datus.storage.registry import clear_storage_registry
from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager
from tests.unit_tests.storage._backend_discovery import (
    BackendTestConfig,
    discover_test_backends,
    setup_test_backend,
    teardown_test_backend,
)

_BACKEND_SPECS = discover_test_backends()


@pytest.fixture(scope="session", params=_BACKEND_SPECS, ids=lambda backend: backend.id)
def _storage_backend_environment(request) -> Iterator[BackendTestConfig]:
    """Own one backend environment for the lifetime of its parameterized test session."""
    try:
        backend = setup_test_backend(request.param)
    except Exception as exc:
        pytest.skip(  # audit-noqa: try_except_skip - external backend availability is optional
            f"Storage backend {request.param.id} is unavailable: {exc}"
        )
    try:
        yield backend
    finally:
        teardown_test_backend(backend)


@pytest.fixture
def storage_test_project():
    """Override in subdirectory conftest to customize the test project identifier.

    Used for backend-test environment plumbing (``clear_data``) and passed to
    ``get_storage`` / ``create_rdb_for_store`` via tests. Must be non-empty —
    backends now reject empty project identifiers.
    """
    return "test"


@pytest.fixture(autouse=True)
def _init_storage_backends(_storage_backend_environment, tmp_path, storage_test_project) -> Iterator[BackendTestConfig]:
    """Ensure storage backends are configured with a valid data_dir for every storage test."""
    backend = _storage_backend_environment
    config = StorageBackendConfig(
        rdb=RdbBackendConfig(type=backend.rdb_type, params=backend.rdb_params),
        vector=VectorBackendConfig(type=backend.vector_type, params=backend.vector_params),
    )
    init_backends(config=config, data_dir=str(tmp_path))
    # Install a path-manager context so implicit ``StorageBase(db=None)``
    # callers see a non-empty project_name.
    pm = DatusPathManager(datus_home=tmp_path, project_name=storage_test_project, project_root=tmp_path)
    token = set_current_path_manager(pm)
    try:
        yield backend
    finally:
        reset_path_manager(token)
        # 1. Clear cache and reset backends (close connection pools)
        clear_storage_registry()
        reset_backends()
        # 2. Clear server-side data (after connection pools are closed)
        if backend.rdb_test_env is not None:
            try:
                backend.rdb_test_env.clear_data(storage_test_project)
            except Exception:
                pass
        if backend.vector_test_env is not None:
            try:
                backend.vector_test_env.clear_data(storage_test_project)
            except Exception:
                pass


@pytest.fixture
def agent_storage_config(_init_storage_backends):
    """Keep real AgentConfig fixtures on the backend selected by parameterization."""
    backend = _init_storage_backends
    return {
        "rdb": {"type": backend.rdb_type, **backend.rdb_params},
        "vector": {"type": backend.vector_type, **backend.vector_params},
    }


@pytest.fixture
def agent_project_name(storage_test_project):
    """Use the same safe project identifier for AgentConfig and backend cleanup."""
    return storage_test_project


def pytest_runtest_setup(item):
    """Skip tests based on backend_specific marker."""
    for marker in item.iter_markers("backend_specific"):
        required = marker.args[0] if marker.args else None
        if not required:
            continue
        backend_spec = None
        if hasattr(item, "callspec") and "_storage_backend_environment" in item.callspec.params:
            backend_spec = item.callspec.params["_storage_backend_environment"]
        if backend_spec and required != backend_spec.rdb_type and required != backend_spec.vector_type:
            pytest.skip(f"Requires {required} backend")
