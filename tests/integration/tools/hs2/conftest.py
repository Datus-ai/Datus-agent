"""The dedicated suite is collected only when a profile is explicitly selected."""

import os


def pytest_ignore_collect(collection_path):
    if collection_path.name.startswith("test_") and not os.getenv("HS2_TEST_PROFILE"):
        return True
    return None
