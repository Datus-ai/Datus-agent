import pytest

from datus.storage.backend_holder import get_vector_backend
from datus.storage.fts import FtsField, FtsIndexStatus, FtsSpec, normalize_fts_spec


def test_agent_uses_shared_fts_contract_when_available():
    shared = pytest.importorskip("datus_storage_base.vector.fts")

    assert FtsField is shared.FtsField
    assert FtsIndexStatus is shared.FtsIndexStatus
    assert FtsSpec is shared.FtsSpec
    assert normalize_fts_spec is shared.normalize_fts_spec


def test_real_agent_config_preserves_postgresql_storage_backend(real_agent_config, _init_storage_backends):
    if _init_storage_backends.vector_type != "postgresql":
        pytest.skip("Requires the PostgreSQL vector backend")

    from datus_storage_postgresql.vector import PgvectorBackend

    assert real_agent_config._backend_config.rdb.type == "postgresql"
    assert real_agent_config._backend_config.vector.type == "postgresql"
    assert isinstance(get_vector_backend(), PgvectorBackend)
