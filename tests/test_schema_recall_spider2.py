import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Set

import pytest
from pandas import DataFrame

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.json_utils import load_jsonl
from tests.conftest import PROJECT_ROOT


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config(**{"namespace": "snowflake"})


@pytest.fixture
def rag(agent_config: AgentConfig) -> SchemaWithValueRAG:
    # setup schema store
    rag = SchemaWithValueRAG(agent_config)
    return rag


# Init gold schema dir. If you are running the use case for this file for the first time,
# run utils/extracting_sql_metadata.py first to generate the schema files.


def load_gold_tables() -> dict[str, Set[str]]:
    gold_schema_file: str = os.path.join(
        PROJECT_ROOT, "benchmark/spider2/methods/gold-tables/spider2-snow-gold-tables.jsonl"
    )

    gold_tables = {}
    for gold_table in load_jsonl(gold_schema_file):
        gold_tables[gold_table["instance_id"]] = set(gold_table["gold_tables"])
    return gold_tables


def clean_query_text(text: str) -> str:
    """clean the query text"""
    # Remove or replace characters that may cause syntax errors
    cleaned = re.sub(r"[^a-zA-Z0-9\s.,?!-]", " ", text)
    # Replace multiple spaces with single space
    cleaned = " ".join(cleaned.split())
    return cleaned


def match_result(target_schema: Set[str], full_name_set: Set[str]):
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


@pytest.mark.parametrize(
    "task_ids,use_rerank", [({"sf_ga011", "sf_ga019", "sf_ga030", "sf_ga005", "sf_ga028", "sf_ga022"}, False)]
)
def test_recall(agent_config: AgentConfig, rag: SchemaWithValueRAG, task_ids: Set[str], use_rerank: bool):
    gold_tables = load_gold_tables()
    with open(os.path.join(agent_config.benchmark_path("spider2"), "spider2-snow.jsonl")) as f:
        for line in f:
            task = json.loads(line)
            if task["instance_id"] not in task_ids:
                continue
            result = do_recall(rag, task, 5, use_rerank, gold_tables)
            if result:
                print(f"Task ID: {task['instance_id']}")
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
                print(f"Task ID: {task['instance_id']} result not found")
                print("TOTAL FOR ", task["db_id"], len(rag.search_all_schemas(database_name=task["db_id"])))


@pytest.mark.parametrize("top_n,use_rerank", [(5, False), (10, False), (20, False)])
def test_full_recall(agent_config: AgentConfig, rag: SchemaWithValueRAG, top_n: int, use_rerank: bool):
    """Test the RAG SQL callback
    # Test matching tables from spider2-snow.jsonl tasks with actual schema
    # Output format: Task ID, Actual Tables, Schema Matched Tables, Schema Match Score, Value Matched Tables,
    # Value Match Score

    Args:
        gold_schema_dir: The directory containing the gold schema files
        top_n: Number of top results to return
    """

    tasks_file = os.path.join(agent_config.benchmark_path("spider2"), "spider2-snow.jsonl")
    start_time = datetime.now()
    with open(tasks_file) as f:
        tasks = [json.loads(line) for line in f]

    match_results = []
    total = 0
    gold_tables = load_gold_tables()
    for task in tasks:
        result = do_recall(rag, task, top_n, use_rerank, gold_tables)
        if result:
            match_results.append(result)
            if result["union_match_tables_score"] == 1:
                total += 1
    output_dir = os.path.join(PROJECT_ROOT, "tests/output/spider2/recall")
    os.makedirs(output_dir, exist_ok=True)
    df = DataFrame(match_results)
    if use_rerank:
        df.to_excel(
            os.path.join(output_dir, f"match_results_{top_n}_{use_rerank}.xlsx"), engine="xlsxwriter", index=False
        )
    else:
        df.to_excel(os.path.join(output_dir, f"match_results_{top_n}.xlsx"), engine="xlsxwriter", index=False)
    print(
        f"The number of top{top_n} exact matches is: {total}, total_result:{len(match_results)}, "
        f"spends {(datetime.now() - start_time).total_seconds()}"
    )


def do_recall(
    rag: SchemaWithValueRAG, task: Dict[str, Any], top_n: int, use_rerank: bool, gold_tables: dict[str, Set[str]]
) -> Dict[str, Any]:
    task_id = task["instance_id"]
    db_id = task["db_id"]
    task = clean_query_text(task["instruction"])
    if task_id not in gold_tables:
        print(f"{task_id}  gold tables not found")
        return {}
    target_schema = gold_tables[task_id]

    # print(task_id, schema)

    (schema_tables, schema_values) = rag.search_similar(
        query_text=task, top_n=top_n, database_name=db_id, use_rerank=use_rerank
    )
    table_count = len(rag.search_all_schemas(database_name=db_id))
    if len(schema_tables) == 0:
        print(f"No schema tables found for task {task_id} from {db_id}")
        return {}

    union_tables = set()
    schema_full_tables = set()
    value_full_tables = set()
    for t in schema_tables:
        full_name = f"{t['database_name']}.{t['schema_name']}.{t['table_name']}"
        union_tables.add(full_name)
        schema_full_tables.add(full_name)
    for t in schema_values:
        full_name = f"{t['database_name']}.{t['schema_name']}.{t['table_name']}"
        value_full_tables.add(full_name)
        union_tables.add(full_name)

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


def test_unique():
    all_values = rag.search_all_value()
    all_schemas = rag.search_all_schemas()

    schema_size = rag.get_schema_size()
    value_size = rag.get_value_size()

    def parse_unique(tables: list[dict[str, Any]]) -> set[str]:
        unique_tables = set()
        for t in tables:
            unique_tables.add(f"{t['database_name']}.{t['schema_name']}.{t['table_name']}")
        return unique_tables

    assert schema_size == len(parse_unique(all_schemas))
    assert value_size == len(parse_unique(all_values))

    # Assume the embedding column name is "embedding"
    vectors = [(v) for v in rag.schema_store.table.to_pandas()["vector"].values]
    import numpy as np

    unique_vectors = np.unique(vectors, axis=0)
    print(f"总向量数: {len(vectors)}, 唯一向量数: {len(unique_vectors)}")


# print(rag.schema_store.table.list_indices())

# rag.schema_store.table.drop_index("vector_idx")
# rag.value_store.table.drop_index("vector_idx")
