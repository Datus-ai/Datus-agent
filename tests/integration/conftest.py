import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests under integration/ with the 'integration' marker."""
    for item in items:
        if "integration" in str(item.fspath) and "unit_tests" not in str(item.fspath):
            item.add_marker(pytest.mark.integration)
