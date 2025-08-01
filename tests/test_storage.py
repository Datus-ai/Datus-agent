import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pytest
from conftest import PROJECT_ROOT

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema
from datus.storage.schema_metadata.benchmark_init_bird import init_dev_schema
from datus.storage.schema_metadata.store import SchemaWithValueRAG, rag_by_configuration
from datus.tools.db_tools.db_manager import db_manager_instance
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
    rag_storage_path = agent_config.rag_storage_path()
    schema_metadata_path = os.path.join(rag_storage_path, "schema_metadata.lance")
    if os.path.exists(schema_metadata_path):
        shutil.rmtree(schema_metadata_path)
        logger.info(f"Deleted existing directory {schema_metadata_path}")
    schema_value_path = os.path.join(rag_storage_path, "schema_value.lance")
    if os.path.exists(schema_value_path):
        shutil.rmtree(schema_value_path)
        logger.info(f"Deleted existing directory {schema_value_path}")

    rag_storage = rag_by_configuration(agent_config)
    init_snowflake_schema(rag_storage, instance_ids=SPIDER_INSTANCE_IDS)
    return rag_storage


def do_query_schema(
    rag_storage: SchemaWithValueRAG, query_txt, top_n: int = 10, db_name: str = ""
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Dict]]]:
    return rag_storage.search_similar(query_txt, top_n=top_n, database_name=db_name)


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
    rag_storage_path = bird_agent_config.rag_storage_path()
    schema_metadata_path = os.path.join(rag_storage_path, "schema_metadata.lance")
    print(f"schema_metadata_path: {schema_metadata_path}")
    if os.path.exists(schema_metadata_path):
        shutil.rmtree(schema_metadata_path)
        logger.info(f"Deleted existing directory {schema_metadata_path}")
    schema_value_path = os.path.join(rag_storage_path, "schema_value.lance")
    if os.path.exists(schema_value_path):
        shutil.rmtree(schema_value_path)
        logger.info(f"Deleted existing directory {schema_value_path}")

    benchmark_path = bird_agent_config.benchmark_path("bird_dev")
    rag_storage = rag_by_configuration(bird_agent_config)
    db_manager = db_manager_instance(bird_agent_config.namespaces)
    init_dev_schema(
        rag_storage,
        db_manager,
        bird_agent_config.current_namespace,
        benchmark_path,
        "overwrite",
        pool_size=4,
        database_names=BIRD_DATABASE_NAMES,
    )
    return rag_storage


@pytest.mark.parametrize("database_name", BIRD_DATABASE_NAMES)
# @pytest.mark.acceptance
def test_bird_tables(bird_rag_storage: SchemaWithValueRAG, database_name: str):
    tables = bird_rag_storage.search_all_schemas(database_name=database_name)
    assert len(tables) > 0, "tables should be greater than 0"


@pytest.mark.parametrize("task_id", ["233"])
# @pytest.mark.acceptance
def test_bird_task(bird_rag_storage: SchemaWithValueRAG, bird_agent_config: AgentConfig, task_id: str):
    bird_path = bird_agent_config.benchmark_path("bird_dev")
    with open(os.path.join(bird_path, "dev.json"), "r") as f:
        data = json.load(f)
        for task in data:
            if str(task["question_id"]) != task_id:
                continue
            query_txt = task["question"]
            db_name = task["db_id"]
            print(query_txt)
            all_tables = bird_rag_storage.search_all_schemas(database_name=db_name)
            result, value_result = do_query_schema(bird_rag_storage, query_txt, top_n=5, db_name=db_name)

            print(
                f"TASK-{task['question_id']} schema_len:{len(result)} "
                f"value_len:{len(value_result)}, total_table: {len(all_tables)}"
            )


@pytest.mark.parametrize("top_n", [5, 10, 20])
def test_time_spends_bird(top_n: int, bird_rag_storage: SchemaWithValueRAG, bird_agent_config: AgentConfig):
    bird_path = bird_agent_config.benchmark_path("bird_dev")
    spend_times = {}
    max_time = 0
    max_id = ""
    min_time = 100000
    min_id = ""
    with open(os.path.join(bird_path, "dev.json"), "r") as f:
        data = json.load(f)
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
