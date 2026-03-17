import tempfile

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

# Note: configure_logging is not called here to avoid creating logs/ directory
# in tests/unit_tests/. Tests use tmp_path for isolated environments.
logger = get_logger(__name__)


@pytest.fixture
def agent_config(tmp_path) -> AgentConfig:
    # Use tmp_path for isolated test environment
    config = load_acceptance_config(namespace="snowflake", home=str(tmp_path))

    # Benchmark paths will auto-derive from tmp_path/benchmark/
    config.benchmark_configs["spider2"].benchmark_path = str(tmp_path / "benchmark/spider2/spider2-snow")
    config.benchmark_configs["bird_dev"].benchmark_path = str(tmp_path / "benchmark/bird/dev_20240627")
    return config


class TestSnowflake:
    @pytest.fixture
    def rag_storage(self, agent_config: AgentConfig) -> SchemaWithValueRAG:
        rag_storage = SchemaWithValueRAG(agent_config)
        return rag_storage

    def test_search_all(self, rag_storage: SchemaWithValueRAG):
        all_schemas = rag_storage.search_all_schemas()
        all_values = rag_storage.search_all_value()
        print(len(all_schemas), all_schemas.num_rows)
        print(len(all_values), all_values.num_rows)


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for testing storage operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_save_batch(temp_db_path: str):
    store = SchemaStorage(embedding_model=get_db_embedding_model())
    store.store(
        [
            {
                "identifier": "1",
                "catalog_name": "c1",
                "database_name": "d1",
                "schema_name": "s1",
                "table_name": "table1",
                "table_type": "table",
                "definition": "create table table1(id int)",
            },
            {
                "identifier": "2",
                "catalog_name": "c1",
                "database_name": "d1",
                "schema_name": "s1",
                "table_name": "table2",
                "table_type": "table",
                "definition": "create table table2(id int)",
            },
        ]
    )

    result = store.search_all(catalog_name="c1")
    assert result.num_rows == 2
