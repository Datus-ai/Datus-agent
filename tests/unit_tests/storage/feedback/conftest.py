"""Fixtures for feedback store tests."""

import pytest


@pytest.fixture
def storage_test_project():
    """Feedback store requires a project for data isolation."""
    return "test"
