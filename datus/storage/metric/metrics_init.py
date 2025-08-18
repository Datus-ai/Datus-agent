import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Set

import pandas as pd
import yaml

from datus.agent.node.generate_metrics_node import GenerateMetricsNode
from datus.agent.node.generate_semantic_model_node import GenerateSemanticModelNode
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput
from datus.schemas.node_models import Metrics, SqlTask
from datus.storage.metric.init_utils import exists_semantic_metrics, gen_metric_id, gen_semantic_model_id
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names

from .store import SemanticMetricsRAG

logger = get_logger(__name__)


def init_success_story_metrics(
    storage: SemanticMetricsRAG,
    args: argparse.Namespace,
    agent_config: AgentConfig,
    build_mode: str = "overwrite",
    pool_size: int = 1,
):
    all_semantic_models, all_metrics = exists_semantic_metrics(storage, build_mode)
    logger.debug(f"all_semantic_models: {all_semantic_models}")
    logger.debug(f"all_metrics: {all_metrics}")

    df = pd.read_csv(args.success_story)
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [
            executor.submit(
                process_line, storage, row.to_dict(), index, args, agent_config, all_semantic_models, all_metrics
            )
            for index, row in df.iterrows()
        ]
        for future in as_completed(futures):
            future.result()

    storage.after_init()


def process_line(
    storage: SemanticMetricsRAG,
    row: dict,
    index: int,
    args: argparse.Namespace,
    agent_config: AgentConfig,
    all_semantic_models: Set[str],
    all_metrics: Set[str],
):
    logger.info(f"processing line: {row}")

    current_db_config = agent_config.current_db_config()
    sql_task = SqlTask(
        id=f"sql_task_{index}",
        database_type=agent_config.db_type,
        task=row["question"],
        catalog_name=current_db_config.catalog,
        database_name=current_db_config.database,
        schema_name=current_db_config.schema,
    )
    logger.debug(f"sql task: {sql_task}")
    # Extract table name from SQL query
    table_names = extract_table_names(row["sql"], agent_config.db_type)
    table_name = table_names[0] if table_names else ""

    semantic_model_input = GenerateSemanticModelInput(sql_task=sql_task, table_name=table_name, prompt_version="1.0")
    logger.debug(f"semantic model input data: {semantic_model_input}")
    semantic_model_node = GenerateSemanticModelNode(
        node_id=f"semantic_model_node_{index}",
        description=f"Generate semantic model for {row['question']}",
        node_type=NodeType.TYPE_GENERATE_SEMANTIC_MODEL,
        input_data=semantic_model_input,
        agent_config=agent_config,
    )
    semantic_model_result = semantic_model_node.run()
    logger.info(f"semantic model result: {semantic_model_result}")
    if not semantic_model_result.success:
        logger.error(f"Failed to generate semantic model for {row['question']}: {semantic_model_result.error}")
        return
    semantic_model = gen_semantic_model(
        semantic_model_result.semantic_model_file,
        sql_task.database_name,
        semantic_model_result.table_name,
        sql_task.schema_name,
        sql_task.catalog_name,
        args.domain,
    )
    logger.debug(f"semantic model: {semantic_model}")
    if not semantic_model:
        logger.error(f"Failed to generate semantic model for {row['question']}")
        return
    if semantic_model.get("id", "") not in all_semantic_models:
        storage.semantic_model_storage.store([semantic_model])
        all_semantic_models.add(semantic_model.get("id", ""))
    else:
        logger.info(f"semantic model {semantic_model['id']} already exists")

    metric_input_data = GenerateMetricsInput(sql_task=sql_task, sql_query=row["sql"], prompt_version="1.0")
    logger.debug(f"metric input data: {metric_input_data}")
    metric_node = GenerateMetricsNode(
        node_id=f"metric_node_{index}",
        description=f"Generate metrics for {row['question']}",
        node_type=NodeType.TYPE_GENERATE_METRICS,
        input_data=metric_input_data,
        agent_config=agent_config,
    )
    metric_result = metric_node.run()
    logger.info(f"metric node result: {metric_result}")
    if not metric_result.success:
        logger.error(f"Failed to generate metrics for {row['question']}: {metric_result.error}")
        return

    current_metric_meta = agent_config.current_metric_meta(args.metric_meta)
    metrics = gen_metrics(
        semantic_model.get("semantic_model_name", ""),
        metric_result.sql_queries,
        metric_result.metrics,
        current_metric_meta.domain,
        current_metric_meta.layer1,
        current_metric_meta.layer2,
    )
    logger.debug(f"metrics: {metrics}")
    for metric in metrics:
        if metric.get("id", "") not in all_metrics:
            storage.metric_storage.store([metric])
            all_metrics.add(metric.get("id", ""))
        else:
            logger.info(f"metric {metric.get('id', '')} already exists")


def gen_semantic_model(
    semantic_model_file: str,
    database_name: str,
    table_name: str,
    schema_name: str,
    catalog_name: str,
    domain: str,
):
    semantic_model = {}
    if not os.path.exists(semantic_model_file):
        logger.error(f"semantic model file {semantic_model_file} not found")
        return semantic_model
    with open(semantic_model_file, "r") as f:
        docs = yaml.safe_load_all(f)
        for doc in docs:
            content = doc.get("data_source", {}) or doc.get("semantic_model", {})
            if not content:
                continue
            semantic_model["id"] = gen_semantic_model_id(catalog_name, database_name, schema_name, table_name)
            semantic_model["catalog_name"] = catalog_name
            semantic_model["database_name"] = database_name
            semantic_model["schema_name"] = schema_name
            semantic_model["table_name"] = table_name
            semantic_model["catalog_database_schema"] = f"{catalog_name}_{database_name}_{schema_name}"
            semantic_model["domain"] = domain
            semantic_model["semantic_file_path"] = semantic_model_file
            semantic_model["semantic_model_name"] = content.get("name", "")
            semantic_model["semantic_model_desc"] = content.get("description", "")
            semantic_model["identifiers"] = json.dumps(content.get("identifiers", []))
            semantic_model["dimensions"] = json.dumps(content.get("dimensions", []))
            semantic_model["measures"] = json.dumps(content.get("measures", []))
            semantic_model["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return semantic_model


def gen_metrics(
    semantic_model_name: str,
    sql_queries: List[str],
    metrics: List[Metrics],
    domain: str,
    layer1: str,
    layer2: str,
):
    metric_list = []
    for metric, sql_query in zip(metrics, sql_queries):
        metric_dict = {}
        metric_dict["id"] = gen_metric_id(domain, layer1, layer2, semantic_model_name, metric.metric_name)
        metric_dict["semantic_model_name"] = semantic_model_name
        metric_dict["domain"] = domain
        metric_dict["layer1"] = layer1
        metric_dict["layer2"] = layer2
        metric_dict["domain_layer1_layer2"] = f"{domain}_{layer1}_{layer2}"
        metric_dict["metric_name"] = metric.metric_name
        metric_dict["metric_value"] = metric.metric_value
        metric_dict["metric_type"] = metric.metric_type
        metric_dict["metric_sql_query"] = sql_query
        metric_dict["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metric_list.append(metric_dict)
    return metric_list
