import json
import os
import tempfile
from datetime import datetime
from typing import Tuple

import pyarrow as pa
import pytest
from conftest import PROJECT_ROOT

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.store import SchemaWithValueRAG, rag_by_configuration
from datus.utils.benchmark_utils import load_bird_dev_tasks
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import configure_logging, get_logger

configure_logging(debug=True)
logger = get_logger(__name__)

# Test instance IDs for snowflake schema initialization
SPIDER_INSTANCE_IDS = ["sf014", "sf002", "sf012", "sf044", "sf011", "sf040"]

# Test database names for bird schema initialization
BIRD_DATABASE_NAMES = ["california_schools", "card_games"]


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config(config="tests/conf/agent.yml", namespace="snowflake")


@pytest.fixture
def rag_storage(agent_config: AgentConfig) -> SchemaWithValueRAG:
    rag_storage = rag_by_configuration(agent_config)
    return rag_storage


def do_query_schema(
    rag_storage: SchemaWithValueRAG, query_txt, top_n: int = 10, db_name: str = ""
) -> Tuple[pa.Table, pa.Table]:
    return rag_storage.search_similar(query_txt, top_n=top_n, database_name=db_name)


def test_search_all(rag_storage: SchemaWithValueRAG):
    all_schemas = rag_storage.search_all_schemas()
    all_values = rag_storage.search_all_value()
    print(len(all_schemas), all_schemas.num_rows)
    print(len(all_values), all_values.num_rows)


@pytest.mark.parametrize("task_id", SPIDER_INSTANCE_IDS)
def test_query_schema_by_spider2_task(task_id: str, rag_storage: SchemaWithValueRAG, top_n: int = 10):
    with open(os.path.join(PROJECT_ROOT, "benchmark/spider2/spider2-snow/spider2-snow.jsonl"), "r") as f:
        for line in f:
            json_data = json.loads(line)
            if json_data["instance_id"] != task_id:
                continue
            query_txt = json_data["instruction"]
            db_name = json_data["db_id"]
            print(query_txt)
            result, value_result = do_query_schema(rag_storage, query_txt, top_n=top_n, db_name=db_name)
            logger.info(f"TASK-{task_id} schema_len:{len(result)} value_len:{len(value_result)}")
            break


@pytest.fixture
def bird_agent_config() -> AgentConfig:
    return load_agent_config(config="tests/conf/agent.yml", namespace="bird_sqlite")


@pytest.fixture
def bird_rag_storage(bird_agent_config: AgentConfig) -> SchemaWithValueRAG:
    return rag_by_configuration(bird_agent_config)


@pytest.mark.parametrize("database_name", BIRD_DATABASE_NAMES)
# @pytest.mark.acceptance
def test_bird_tables(bird_rag_storage: SchemaWithValueRAG, database_name: str):
    tables = bird_rag_storage.search_all_schemas(database_name=database_name)
    assert len(tables) > 0, "tables should be greater than 0"


@pytest.mark.parametrize("task_id", ["233"])
# @pytest.mark.acceptance
def test_bird_task(bird_rag_storage: SchemaWithValueRAG, bird_agent_config: AgentConfig, task_id: str):
    bird_path = bird_agent_config.benchmark_path("bird_dev")
    data = load_bird_dev_tasks(bird_path)
    for task in data:
        if str(task["question_id"]) != task_id:
            continue
        query_txt = task["question"]
        db_name = task["db_id"]
        print(query_txt)
        all_tables = bird_rag_storage.search_all_schemas(database_name=db_name)
        # all_values = bird_rag_storage.value_store.search_all(db_name)
        result, value_result = do_query_schema(bird_rag_storage, query_txt, top_n=5, db_name=db_name)

        print(
            f"TASK-{task['question_id']} schema_len:{len(result)} "
            f"value_len:{len(value_result)}, total_table: {len(all_tables)}"
        )


@pytest.mark.parametrize("top_n", [5, 10, 20])
def test_time_spends_bird(top_n: int, bird_rag_storage: SchemaWithValueRAG, bird_agent_config: AgentConfig):
    spend_times = {}
    max_time = 0
    max_id = ""
    min_time = 100000
    min_id = ""
    data = load_bird_dev_tasks(bird_agent_config.benchmark_path("bird_dev"))
    for task in data:
        query_txt = task["question"]
        db_name = task["db_id"]
        start = datetime.now()
        bird_rag_storage.search_similar(query_txt, top_n=top_n, database_name=db_name)
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
    storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
    storage.store(
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

    result = storage.search_all(catalog_name="c1")
    assert result.num_rows == 2


def test_store_exception(rag_storage: SchemaWithValueRAG):
    with pytest.raises(DatusException) as exc_info:
        rag_storage.store_batch(schemas=[{"id": "1", "title": "1"}], values=[])

    assert exc_info.value.code == ErrorCode.STORAGE_SAVE_FAILED
