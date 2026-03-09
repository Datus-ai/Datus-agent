"""Fixtures for task store tests."""

import pytest


@pytest.fixture
def storage_test_namespace():
    """Task store requires a namespace for data isolation."""
    return "test"
