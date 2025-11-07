import os
import tempfile
from datetime import datetime

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.benchmark_utils import load_benchmark_tasks
from datus.utils.loggings import configure_logging, get_logger
from tests.conftest import load_acceptance_config

configure_logging(debug=True)
logger = get_logger(__name__)

# Test instance IDs for snowflake schema initialization
SPIDER_INSTANCE_IDS = ["sf014", "sf002", "sf012", "sf044", "sf011", "sf040"]

# Test database names for bird schema initialization
BIRD_DATABASE_NAMES = ["california_schools", "card_games"]


@pytest.fixture
def agent_config() -> AgentConfig:
    config = load_acceptance_config(namespace="snowflake")
    from pathlib import Path

    config.benchmark_configs["spider2"].benchmark_path = str(
        Path("~/.datus/benchmark/spider2/spider2-snow").expanduser()
    )
    config.benchmark_configs["bird_dev"].benchmark_path = str(Path("~/.datus/benchmark/bird/dev_20240627").expanduser())
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

    @pytest.mark.parametrize("task_id", SPIDER_INSTANCE_IDS)
    def test_query_schema_by_spider2_task(
        self, agent_config: AgentConfig, task_id: str, rag_storage: SchemaWithValueRAG, top_n: int = 10
    ):
        for task in load_benchmark_tasks(agent_config, "spider2"):
            if task["instance_id"] != task_id:
                continue
            query_txt = task["instruction"]
            db_name = task["db_id"]
            print(query_txt)
            result, value_result = rag_storage.search_similar(query_text=query_txt, top_n=top_n, database_name=db_name)
            logger.info(f"TASK-{task_id} schema_len:{len(result)} value_len:{len(value_result)}")
            break


class TestBird:
    @pytest.fixture
    def rag_storage(self, agent_config: AgentConfig) -> SchemaWithValueRAG:
        agent_config.current_namespace = "bird_sqlite"
        return SchemaWithValueRAG(agent_config)

    @pytest.mark.parametrize("database_name", BIRD_DATABASE_NAMES)
    # @pytest.mark.acceptance
    def test_tables(self, agent_config, rag_storage: SchemaWithValueRAG, database_name: str):
        tables = rag_storage.search_all_schemas(database_name=database_name)
        assert tables.num_rows > 0, "tables should be greater than 0"

    @pytest.mark.parametrize("task_id", ["233"])
    # @pytest.mark.acceptance
    def test_task(self, rag_storage: SchemaWithValueRAG, agent_config: AgentConfig, task_id: str):
        data = load_benchmark_tasks(agent_config, "bird_dev")
        for task in data:
            if str(task["question_id"]) != task_id:
                continue
            query_txt = task["question"]
            db_name = task["db_id"]
            print(query_txt)
            all_tables = rag_storage.search_all_schemas(database_name=db_name)
            # all_values = rag_storage.value_store.search_all(db_name)
            result, value_result = rag_storage.search_similar(query_txt, top_n=10, database_name=db_name)

            print(
                f"TASK-{task['question_id']} schema_len:{len(result)} "
                f"value_len:{len(value_result)}, total_table: {len(all_tables)}"
            )

    @pytest.mark.parametrize("top_n", [5, 10, 20])
    def test_time_spends(self, top_n: int, rag_storage: SchemaWithValueRAG, agent_config: AgentConfig):
        spend_times = {}
        max_time = 0
        max_id = ""
        min_time = 100000
        min_id = ""
        data = load_benchmark_tasks(agent_config, "bird_dev")
        for task in data:
            query_txt = task["question"]
            db_name = task["db_id"]
            start = datetime.now()
            rag_storage.search_similar(query_txt, top_n=top_n, database_name=db_name)
            current_spends = (datetime.now() - start).total_seconds() * 1000
            spend_times[task["question_id"]] = current_spends
            if current_spends > max_time:
                max_time = current_spends
                max_id = task["question_id"]
            if current_spends < min_time:
                min_time = current_spends
                min_id = task["question_id"]

        # Calculate statistics
        times = list(spend_times.values())
        avg_time = sum(times) / len(times)

        # Calculate median
        sorted_times = sorted(times)
        n = len(sorted_times)
        if n % 2 == 0:
            median_time = (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        else:
            median_time = sorted_times[n // 2]

        output_dir = "tests/output/bird_dev"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/search_spends_top_{top_n}.txt", "w") as f:
            f.write(
                f"Time statistics for top_n={top_n} (in milliseconds):\n"
                f"  Maximum: {max_time:.2f}ms, id: {max_id}\n"
                f"  Minimum: {min_time:.2f}ms, id: {min_id}\n"
                f"  Median: {median_time:.2f}ms\n"
                f"  Average: {avg_time:.2f}ms\n"
            )

            f.write("Tasks that spends more than 500ms:\n")
            for task_id, spend_time in spend_times.items():
                if spend_time > 500:
                    f.write(f"  id: {task_id}, spends {spend_time}\n")


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for testing storage operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_save_batch(temp_db_path: str):
    store = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
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
