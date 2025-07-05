import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Tuple

import lancedb
import pandas as pd
import pytest
from conftest import PROJECT_ROOT
from lancedb.table import Table

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema
from datus.storage.schema_metadata.store import SchemaWithValueRAG, rag_by_configuration
from datus.utils.loggings import configure_logging, get_logger

configure_logging(debug=True)
logger = get_logger("test_storage")


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config(**{"namespace": "snowflake"})


@pytest.fixture
def rag_storage(agent_config: AgentConfig) -> SchemaWithValueRAG:
    return rag_by_configuration(agent_config)


def test_init_snowflake_schema(rag_storage: SchemaWithValueRAG):
    init_snowflake_schema(rag_storage)


def do_query_schema(
    rag_storage: SchemaWithValueRAG, query_txt, top_n: int = 10, db_name: str = ""
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Dict]]]:
    return rag_storage.search_similar(query_txt, top_n=top_n, database_name=db_name)


@pytest.mark.parametrize("task_id", ["sf014", "sf002", "sf012", "sf044", "sf011", "sf040"])
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


# you can use this function to move the table from one path to another
def move_path(
    executor: ThreadPoolExecutor,
    table_name: str,
    path: str,
    new_path: str,
    schema_claz,
    columns: List[str],
):
    old_db = lancedb.connect(path)
    new_db = lancedb.connect(new_path)
    old_table = old_db.open_table(table_name)
    data = old_table.search().limit(1000000).to_list()
    print("total", len(data))
    new_table = new_db.create_table(table_name, schema=schema_claz)
    batch_data = []
    for i in data:
        new_data = {}
        for c in columns:
            new_data[c] = i[c]
        batch_data.append(new_data)
        if len(batch_data) == 100:
            new_table.add(pd.DataFrame(batch_data))
            executor.submit(do_add_data, new_table, batch_data)
            batch_data = []
    if len(batch_data) > 0:
        executor.submit(do_add_data, new_table, batch_data)


def do_add_data(table: Table, data: List[Dict[str, Any]]):
    table.add(pd.DataFrame(data))


@pytest.fixture
def bird_agent_config() -> AgentConfig:
    # Modify namespace according to your config file
    return load_agent_config(**{"namespace": "bird_sqlite"})


@pytest.fixture
def bird_rag_storage(bird_agent_config: AgentConfig) -> SchemaWithValueRAG:
    return rag_by_configuration(bird_agent_config)


@pytest.mark.parametrize("database_name", ["california_schools", "card_games", "debit_card_specializing"])
@pytest.mark.acceptance
def test_bird_tables(bird_rag_storage: SchemaWithValueRAG, database_name: str):
    tables = bird_rag_storage.search_all_schemas(database_name=database_name)
    print(f"tables: {tables}")
    assert len(tables) > 0, "tables should be greater than 0"


@pytest.mark.parametrize("task_id", ["233"])
@pytest.mark.acceptance
def test_bird_task(bird_rag_storage: SchemaWithValueRAG, bird_agent_config: AgentConfig, task_id: str):
    bird_path = bird_agent_config.benchamrk_path("bird_dev")
    with open(os.path.join(bird_path, "dev.json"), "r") as f:
        data = json.load(f)
        for task in data:
            if str(task["question_id"]) != task_id:
                continue
            query_txt = task["question"]
            db_name = task["db_id"]
            print(query_txt)
            all_tables = bird_rag_storage.search_all_schemas(database_name=db_name)
            # all_values = bird_rag_storage.value_store.search_all(db_name)
            result, value_result = do_query_schema(bird_rag_storage, query_txt, top_n=5, db_name=db_name)
            # print(result)

            print(
                f"TASK-{task['question_id']} schema_len:{len(result)} "
                f"value_len:{len(value_result)}, total_table: {len(all_tables)}"
            )

            print(value_result)


@pytest.mark.acceptance
@pytest.mark.parametrize("top_n", [5, 10, 20])
def test_time_spends_bird(top_n: int, bird_rag_storage: SchemaWithValueRAG, bird_agent_config: AgentConfig):
    bird_path = bird_agent_config.benchamrk_path("bird_dev")
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

    with open(f"tests/output/bird_dev/search_spends_top_{top_n}.txt", "w") as f:
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
