import os
from datetime import datetime
from typing import Any, List

import pytest
from pandas import DataFrame

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.benchmark_utils import load_benchmark_tasks
from datus.utils.constants import DBType
from datus.utils.sql_utils import extract_table_names
from tests.conftest import PROJECT_ROOT


def match_result(target_schema: set[str], full_name_set: set[str]) -> dict[str, Any]:
    """Match the target schema with the schema tables"""

    matched_tables = []
    for table in full_name_set:
        for target_table in target_schema:
            if target_table.lower() == table.lower():
                matched_tables.append(table)
                break
    return (
        "\n".join(full_name_set),
        "\n".join(matched_tables),
        len(matched_tables),
        round(len(matched_tables) / len(target_schema), 2),
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    # FIXME Modify namespace according to your configuration
    config = load_agent_config(**{"benchmark": "bird_dev", "namespace": "bird_sqlite"})
    return config


@pytest.fixture
def rag(agent_config: AgentConfig) -> SchemaWithValueRAG:
    return SchemaWithValueRAG(agent_config=agent_config)


@pytest.mark.parametrize("task_ids", [[0, 1, 2, 3, 4, 5, 6]])
def test_recall(task_ids: List[str], rag: SchemaWithValueRAG, agent_config: AgentConfig):
    for task in load_benchmark_tasks(agent_config, "bird_dev"):
        question_id = task["question_id"]
        if question_id not in task_ids:
            continue
        result = _do_recall(rag, task, 5)
        if result:
            print(f"Task ID: {task['question_id']}")
            print(f"Actual Tables: {result['actual_tables']}")
            print(f"Matched Tables: {result['matched_tables']}")
            print(f"Match Scores: {result['match_tables_score']}")
            print(f"Matched Values: {result['matched_values']}")
            print(f"Match Values Score: {result['match_values_score']}")
            print(f"Union Match Tables: {result['union_match_tables']}")
            print(f"Union Match Tables Count: {result['union_match_tables_count']}")
            print(f"Union Match Tables Score: {result['union_match_tables_score']}")
            print(f"Table Count: {result['table_count']}")
            print(f"Recall Rate: {result['recall_rate']}")
            print("---")
        else:
            print(f"Task ID: {task['question_id']} result not found")
            print("TOTAL FOR ", task["db_id"], len(rag.search_all_schemas(database_name=task["db_id"])))


def _do_recall(rag: SchemaWithValueRAG, item: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    task_id = item["question_id"]
    db_id = item["db_id"]
    target_schema = set(extract_table_names(item["SQL"], dialect=DBType.SQLITE))
    schema_tables, schema_values = rag.search_similar(query_text=item["question"], database_name=db_id, top_n=top_n)
    table_count = len(rag.search_all_schemas(database_name=db_id))
    if schema_tables.num_rows == 0:
        print(f"No schema tables found for task {task_id} from {db_id}")
        return {}

    schema_full_tables = {item for item in schema_tables["table_name"].to_pandas().unique()}
    value_full_tables = {item for item in schema_values["table_name"].to_pandas().unique()}
    union_tables = schema_full_tables.union(value_full_tables)
    # target_schema = set(target_schema)

    query_schema_tables, match_tables, match_tables_count, match_tables_score = match_result(
        target_schema, schema_full_tables
    )
    query_values_tables, match_values, match_values_count, match_values_score = match_result(
        target_schema, value_full_tables
    )
    _, match_union_tables, match_union_tables_count, match_union_tables_score = match_result(
        target_schema, union_tables
    )
    return {
        "task_id": task_id,
        "actual_tables": "\n".join(target_schema),
        "actual_tables_count": len(target_schema),
        "query_schema_tables": query_schema_tables,
        "matched_tables": match_tables,
        "matched_tables_count": match_tables_count,
        "match_tables_score": match_tables_score,
        "query_values_tables": query_values_tables,
        "matched_values": match_values,
        "matched_values_count": match_values_count,
        "match_values_score": match_values_score,
        "union_match_tables": match_union_tables,
        "union_match_tables_count": match_union_tables_count,
        "union_match_tables_score": match_union_tables_score,
        "table_count": table_count,
        "recall_rate": round(len(union_tables) / table_count, 2),
    }


@pytest.mark.parametrize("top_n", [5, 10, 20])
def test_full_recall(top_n: int, rag: SchemaWithValueRAG, agent_config: AgentConfig):
    """Test the RAG SQL callback
    # Test matching tables from spider2-snow.jsonl tasks with actual schema
    # Output format: Task ID, Actual Tables, Schema Matched Tables, Schema Match Score, Value Matched Tables, V
    # alue Match Score

    Args:
        gold_schema_dir: The directory containing the gold schema files
        top_n: Number of top results to return
    """

    output_dir = os.path.join(PROJECT_ROOT, "tests/output/bird_dev/recall/")
    os.makedirs(output_dir, exist_ok=True)
    start_time = datetime.now()
    match_results = []
    total = 0

    for task in load_benchmark_tasks(agent_config, "bird_dev"):
        result = _do_recall(rag, task, top_n)
        if result:
            match_results.append(result)
            if result["union_match_tables_score"] == 1:
                total += 1
    df = DataFrame(match_results)
    df.to_excel(os.path.join(output_dir, f"match_results_{top_n}.xlsx"), engine="xlsxwriter", index=False)
    print(
        f"The number of top{top_n} exact matches is: {total}, total_result:{len(match_results)}, "
        f"spends {(datetime.now() - start_time).total_seconds()}"
    )
