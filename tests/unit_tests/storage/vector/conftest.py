"""Override parent's parameterized fixture. vector/ tests are Lance-specific."""

import pytest

from datus.storage.backend_holder import init_backends
from datus.storage.registry import clear_storage_registry


@pytest.fixture(autouse=True)
def _init_storage_backends(tmp_path):
    """Always use default backends — no parameterization for backend-specific tests."""
    init_backends(data_dir=str(tmp_path))
    yield
    clear_storage_registry()
