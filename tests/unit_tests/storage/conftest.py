"""Shared fixtures for storage tests.

The autouse ``_init_storage_backends`` fixture is parameterized across all
discovered backends so that store-level tests automatically repeat on every
available rdb+vector combination.
"""

import pytest
from datus_storage_base.backend_config import RdbBackendConfig, StorageBackendConfig, VectorBackendConfig

from datus.storage.backend_holder import init_backends, reset_backends
from datus.storage.registry import clear_storage_registry
from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager
from tests.unit_tests.storage._backend_discovery import discover_test_backends

_BACKENDS = discover_test_backends()


@pytest.fixture
def storage_test_project():
    """Override in subdirectory conftest to customize the test project identifier.

    Used for backend-test environment plumbing (``clear_data``) and passed to
    ``get_storage`` / ``create_rdb_for_store`` via tests. Must be non-empty —
    backends now reject empty project identifiers.
    """
    return "test"


@pytest.fixture(autouse=True, params=_BACKENDS, ids=lambda b: b.id)
def _init_storage_backends(request, tmp_path, storage_test_project):
    """Ensure storage backends are configured with a valid data_dir for every storage test."""
    backend = request.param
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


def pytest_sessionfinish(session, exitstatus):
    """Clean up backend test environments at session end."""
    from tests.unit_tests.storage._backend_discovery import cleanup_test_environments

    cleanup_test_environments()


def pytest_runtest_setup(item):
    """Skip tests based on backend_specific marker."""
    for marker in item.iter_markers("backend_specific"):
        required = marker.args[0] if marker.args else None
        if not required:
            continue
        # Find the backend config from the _init_storage_backends param
        backend = None
        if hasattr(item, "callspec") and "_init_storage_backends" in item.callspec.params:
            backend = item.callspec.params["_init_storage_backends"]
        if backend and required != backend.rdb_type and required != backend.vector_type:
            pytest.skip(f"Requires {required} backend")
