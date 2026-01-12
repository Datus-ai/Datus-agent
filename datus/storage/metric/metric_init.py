# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import os
from typing import Any, Optional

import pandas as pd

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

BIZ_NAME = "metric_init"


def _action_status_value(action: Any) -> Optional[str]:
    status = getattr(action, "status", None)
    if status is None:
        return None
    return status.value if hasattr(status, "value") else str(status)


def init_success_story_metrics(
    args: argparse.Namespace,
    agent_config: AgentConfig,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """
    Initialize metrics from success story CSV by batch processing.

    This reads all SQL queries from the CSV and processes them as a batch
    to extract core unique metrics (deduplicating aggregation patterns).

    Args:
        args: Command line arguments
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
    """
    event_helper = BatchEventHelper(BIZ_NAME, emit)
    df = pd.read_csv(args.success_story)

    # Emit task started
    event_helper.task_started(total_items=len(df), success_story=args.success_story)

    # Build batch message with all SQL queries
    sql_queries = []
    for idx, row in df.iterrows():
        sql = row["sql"]
        question = row["question"]
        sql_queries.append(f"Query {idx + 1}:\nQuestion: {question}\nSQL:\n{sql}")

    batch_message = "Analyze the following SQL queries and extract core metrics:\n\n" + "\n\n---\n\n".join(sql_queries)

    logger.info(f"Processing {len(df)} SQL queries as batch for core metrics extraction")

    # Get database context
    current_db_config = agent_config.current_db_config()

    metrics_input = SemanticNodeInput(
        user_message=batch_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
    )

    metrics_node = GenMetricsAgenticNode(
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()
    metrics_node.input = metrics_input

    # Emit task processing
    event_helper.task_processing(total_items=1)

    async def process_batch() -> dict:
        try:
            final_result = None
            async for action in metrics_node.execute_stream(action_history_manager):
                if event_helper:
                    event_helper.item_processing(
                        item_id="batch",
                        action_name="gen_metrics",
                        status=_action_status_value(action),
                        messages=action.messages,
                        output=action.output,
                    )
                if action.status == ActionStatus.SUCCESS and action.output:
                    final_result = action.output
                    logger.debug(f"Metrics generation action: {action.messages}")
            logger.info("Batch metrics extraction completed successfully")
            return {"successful": True, "error": "", "result": final_result}
        except Exception as e:
            logger.error(f"Error in batch metrics extraction: {e}")
            return {"successful": False, "error": str(e)}

    result = asyncio.run(process_batch())

    # Emit task completed (single batch)
    if result.get("successful"):
        event_helper.task_completed(
            total_items=1,
            completed_items=1,
            failed_items=0,
        )
        return True, "", result.get("result")
    else:
        error = result.get("error", "Unknown error")
        event_helper.task_failed(
            error=error,
        )
        return False, error, None


def init_semantic_yaml_metrics(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Initialize ONLY metrics from semantic YAML file, skip semantic model objects.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
    """
    if not os.path.exists(yaml_file_path):
        logger.error(f"Semantic YAML file {yaml_file_path} not found")
        return False, f"Semantic YAML file {yaml_file_path} not found"

    # Import from semantic_model package to avoid circular dependency
    from datus.storage.semantic_model.semantic_model_init import process_semantic_yaml_file

    return process_semantic_yaml_file(yaml_file_path, agent_config, include_semantic_objects=False)
