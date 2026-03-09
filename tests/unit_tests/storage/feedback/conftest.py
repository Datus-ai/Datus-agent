"""Fixtures for feedback store tests."""

import pytest


@pytest.fixture
def storage_test_namespace():
    """Feedback store requires a namespace for data isolation."""
    return "test"
