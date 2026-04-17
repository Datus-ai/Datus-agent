"""Fixtures for task store tests."""

import pytest


@pytest.fixture
def storage_test_project():
    """Task store requires a project for data isolation."""
    return "test"
