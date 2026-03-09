"""Override parent's parameterized fixture. rdb/ tests are SQLite-specific."""

import pytest

from datus.storage.backend_holder import init_backends, reset_backends
from datus.storage.cache import clear_cache


@pytest.fixture(autouse=True)
def _init_storage_backends(tmp_path):
    """Always use default backends — no parameterization for backend-specific tests."""
    init_backends(data_dir=str(tmp_path))
    yield
    clear_cache()
    reset_backends()
