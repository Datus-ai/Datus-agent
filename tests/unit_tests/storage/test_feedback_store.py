# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pytest

from tests.helpers.storage_acceptance import assert_feedback_storage_crud
from tests.unit_tests.storage._backend_discovery import BackendTestConfig


@pytest.fixture(scope="session")
def _storage_backend_environment() -> BackendTestConfig:
    """Keep the fast acceptance case local; PostgreSQL has a required P0 case."""
    return BackendTestConfig(rdb_type="sqlite", vector_type="lance")


@pytest.mark.acceptance
def test_feedback_storage_write_read_update_and_delete(real_agent_config):
    assert assert_feedback_storage_crud(real_agent_config.project_name) == {"feedback": None, "task": None}
