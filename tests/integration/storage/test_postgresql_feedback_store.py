# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Required PostgreSQL acceptance for the Agent's two feedback tables."""

import pytest
from datus_storage_base.backend_config import RdbBackendConfig, StorageBackendConfig, VectorBackendConfig

from datus.storage.backend_holder import init_backends, reset_backends
from tests.helpers.storage_acceptance import assert_feedback_storage_crud
from tests.integration.storage.test_postgresql_semantic_kb import (
    postgresql_agent_config as postgresql_agent_config,  # noqa: F401 - register the shared project fixture.
)


@pytest.mark.nightly
def test_postgresql_feedback_storage_write_read_update_and_delete(
    postgresql_agent_config, required_postgresql_storage, tmp_path
):
    """Run real CRUD against the same service providers as semantic KB P0."""
    reset_backends()
    try:
        init_backends(
            StorageBackendConfig(
                rdb=RdbBackendConfig(type="postgresql", params=required_postgresql_storage.rdb_config.params),
                vector=VectorBackendConfig(type="postgresql", params=required_postgresql_storage.vector_config.params),
            ),
            data_dir=str(tmp_path),
        )
        assert_feedback_storage_crud(postgresql_agent_config.project_name)
    finally:
        reset_backends()
