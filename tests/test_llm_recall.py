import json
import os
import traceback
from typing import Any, Dict, List, Set, Union

import pytest
from pandas import DataFrame

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.llms_tools.match_schema import MatchSchemaTool, gen_all_table_dict
from datus.utils.constants import DBType
from datus.utils.json_utils import load_jsonl_iterator
from datus.utils.loggings import configure_logging, get_logger
from tests.conftest import PROJECT_ROOT
from tests.test_schema_recall_spider2 import load_gold_tables

configure_logging(debug=True)
logger = get_logger(__name__)


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config(namespace="snowflake")


json_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmark/spider2/spider2-snow/spider2-snow.jsonl",
)


not_matched_databases = {}


@pytest.fixture
def rag(agent_config: AgentConfig) -> SchemaWithValueRAG:
    rag = SchemaWithValueRAG(agent_config)
    return rag


@pytest.mark.parametrize("target_task_id, top_n", [("sf_bq236", 10)])
def test_match_schema(
    agent_config: AgentConfig, rag: SchemaWithValueRAG, target_task_id: str, top_n: int, check_exists=False
):
    """Match the schema by llm
    Args:
        task_id: the task id
        check_exists: whether to check if the result exists, if True, skip the task
    """
    # Init gold sql tables
    gold_sql_tables = load_gold_tables()
    need_matched_data = []
    for json_line in load_jsonl_iterator(json_path):
        task_id = json_line["instance_id"]
        if task_id != target_task_id or task_id not in gold_sql_tables:
            continue
        need_matched_data.append(json_line)

    do_gen_match_schema(agent_config, rag, need_matched_data, top_n, check_exists, gold_sql_tables=gold_sql_tables)


def test_full_match_schema(agent_config: AgentConfig, rag: SchemaWithValueRAG, top_n: int = 10, check_exists=True):
    """Match the schema by llm for all tasks"""
    need_matched_data = []
    # Init gold sql tables
    gold_sql_tables = load_gold_tables()
    with open(json_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            json_line = json.loads(line)
            task_id = json_line["instance_id"]
            database_name = json_line["db_id"]
            if task_id not in gold_sql_tables:
                continue

            target_db_tables = gold_sql_tables[task_id]
            if database_name != target_db_tables["database_name"]:
                # logger.info(f'{task_id},{database_name},{target_db_tables["database_name"]}')
                not_matched_databases[task_id] = {
                    "src_database_name": database_name,
                    "target_database_name": target_db_tables["database_name"],
                }
            task_id = json_line["instance_id"]
            database_name = json_line["db_id"]
            if task_id not in gold_sql_tables:
                continue
            if task_id in not_matched_databases:
                database_name = not_matched_databases[task_id]["target_database_name"]
            else:
                database_name = json_line["db_id"]
            user_question = json_line["instruction"]
            result, value_result = rag.search_similar(user_question, top_n=10, database_name=database_name)
            gold_tables = set(gold_sql_tables[task_id]["tables"])
            # Match the target schema with the schema tables"""
            full_name_set = set()
            for table in result:
                full_name = f"{table['database_name']}.{table['schema_name']}.{table['table_name']}"
                full_name_set.add(full_name)
            for table in value_result:
                full_name = f"{table['database_name']}.{table['schema_name']}.{table['table_name']}"
                full_name_set.add(full_name)

            match_result = do_match_result(gold_tables, full_name_set)

            if match_result["matched_score"] != 1:
                need_matched_data.append(json_line)

        recall_result = do_gen_match_schema(agent_config, rag, need_matched_data, top_n, check_exists, gold_sql_tables)
        logger.info(f"Full-Match-Schema done {recall_result}")

        if recall_result:
            df = DataFrame(recall_result)
            df.to_excel(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tests/gen_sql_by_full.xlsx",
                ),
                index=False,
            )
        logger.info("Full-Match-Schema result2 write done")


def do_gen_match_schema(
    agent_config: AgentConfig,
    rag: SchemaWithValueRAG,
    lines: List[Dict[str, Any]],
    top_n: int,
    check_exists: bool,
    gold_sql_tables: dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    """Match the schema by llm
    Args:
        lines: the lines to match
        check_exists: whether to check if the result exists, if True, skip the task
    Returns:
        Recall of generated results
    """
    output_dir = os.path.join(PROJECT_ROOT, "tests/llm_find_tables")
    os.makedirs(output_dir, exist_ok=True)
    model = LLMBaseModel.create_model(agent_config=agent_config)

    llm_tool = MatchSchemaTool(
        model,
        storage=rag.schema_store,
    )
    result = []
    logger.info(f"should match {len(lines)} tasks")
    for json_line in lines:
        task_id = json_line["instance_id"]

        if task_id in not_matched_databases:
            database_name = not_matched_databases[task_id]["target_database_name"]
        else:
            database_name = json_line["db_id"]

        user_question = json_line["instruction"]
        all_tables = rag.search_all_schemas(database_name=database_name)

        json_file = os.path.join(PROJECT_ROOT, f"tests/llm_find_tables/{task_id}.json")

        if check_exists and os.path.exists(json_file):
            logger.info(f"{task_id} already exists, skip")
            continue

        logger.info(f"{task_id} not found, should generate")

        try:
            response = llm_tool.match_schema(
                SchemaLinkingInput(
                    database_type=DBType.SNOWFLAKE,
                    database_name=database_name,
                    input_text=user_question,
                    top_n=top_n,
                ),
                all_tables,
                gen_all_table_dict(all_tables),
            )
            logger.info(f"{task_id} result: {response}")
            result.append(do_match_recall(task_id, database_name, response, gold_sql_tables))
            logger.info(f"{task_id} recall: {result[-1]}")
        except Exception:
            logger.info(f"Failed to generate for {task_id}, error: {traceback.format_exc()}")
            continue

        with open(os.path.join(PROJECT_ROOT, f"tests/llm_find_tables/{task_id}.json"), "w") as f:
            try:
                json.dump(response, f)
            except Exception as e:
                logger.info(f"Failed to dump json for {task_id}, result={response}, error: {e}")

        logger.info("-" * 50, task_id, "done", "-" * 50)
    return result


def do_match_recall(
    task_id: str,
    database_name: str,
    gen_result: Union[List[Dict[str, Any]], Dict[str, Any]],
    gold_sql_tables: dict[str, Set[str]],
) -> Dict[str, Any]:
    if not gen_result:
        return {}
    target_tables = gold_sql_tables[task_id]

    try:
        gen_tables = set()
        if isinstance(gen_result, dict):
            gen_table_name = gen_result["table"]
            if isinstance(gen_table_name, str):
                gen_table_name = [gen_table_name]
            for table in gen_table_name:
                if len(table.split(".")) == 2:
                    gen_tables.add(f"{database_name}.{table}")
                else:
                    gen_tables.add(table)
        else:
            for gen_table in gen_result:
                print("gen tables", gen_table)
                gen_table_name = gen_table["table"]
                if isinstance(gen_table_name, str):
                    gen_table_name = [gen_table_name]
                for table in gen_table_name:
                    if len(table.split(".")) == 2:
                        gen_tables.add(f"{database_name}.{table}")
                    else:
                        gen_tables.add(table)

        match_result = do_match_result(target_tables, gen_tables)
        match_result["task_id"] = task_id
        logger.info(f"{task_id} match recall: {match_result['matched_score']}")
        return match_result
    except Exception:
        logger.info(f"Failed to match for {task_id}, error: {traceback.format_exc()}")


def test_llm_match_recall():
    """calculate the matched score of the llm find table result"""
    target_dir = os.path.join(PROJECT_ROOT, "tests/llm_find_tables")
    os.makedirs(target_dir, exist_ok=True)

    gold_sql_tables = load_gold_tables()
    task_dbs = {}
    for json_line in load_jsonl_iterator(json_path):
        task_dbs[json_line["instance_id"]] = json_line["db_id"]

    match_results = []
    for json_file_name in os.listdir(target_dir):
        if not json_file_name.endswith(".json"):
            continue
        task_id = json_file_name.split(".")[0]
        with open(os.path.join(target_dir, json_file_name), "r") as f:
            try:
                gen_result = json.load(f)
                match_results.append(do_match_recall(task_id, task_dbs[task_id], gen_result, gold_sql_tables))
            except Exception as e:
                logger.info(f"Failed to load json for {task_id}, error: {e}")

    if not match_results:
        return
    df = DataFrame(match_results)
    df.to_excel(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests/gen_sql_by_full.xlsx",
        ),
        index=False,
    )


def check_find_table_result():
    """check if the llm find table result is correct"""
    target_dir = os.path.join(PROJECT_ROOT, "tests/llm_find_tables")
    for file in os.listdir(target_dir):
        if not file.endswith(".md"):
            continue

        name = file.split(".")[0]
        if not os.path.exists(os.path.join(target_dir, f"{name}.json")):
            logger.info(f"{name}.json not found")


def do_match_result(
    target_schema: Union[List[str], Set[str]], query_schemas: Union[List[str], Set[str]]
) -> Dict[str, Any]:
    """Return the intersection of target_schema and schema_tables"""
    # Find intersection between target_schema and schema_tables
    target_schema = set(target_schema)
    query_schemas = set(query_schemas)
    matched_tables = []
    for table in query_schemas:
        for target_table in target_schema:
            if target_table.lower() == table.lower():
                matched_tables.append(table)
                break
    return {
        "target_tables": "\n".join(target_schema),
        "target_tables_count": len(target_schema),
        "query_tables": "\n".join(query_schemas),
        "matched_tables": "\n".join(matched_tables),
        "matched_tables_count": len(matched_tables),
        "matched_score": round(len(matched_tables) / len(target_schema), 2),
    }
