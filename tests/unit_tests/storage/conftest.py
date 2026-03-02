"""Shared fixtures for storage tests."""

import pytest

from datus.storage.backend_holder import init_backends, reset_backends
from datus.storage.cache import clear_cache


@pytest.fixture(autouse=True)
def _init_storage_backends(tmp_path):
    """Ensure storage backends are configured with a valid data_dir for every storage test."""
    init_backends(data_dir=str(tmp_path))
    yield
    clear_cache()
    reset_backends()
