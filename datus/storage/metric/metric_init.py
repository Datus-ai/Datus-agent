# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager
from datus.utils.sql_utils import extract_table_names

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
    pool_size: int = 1,
) -> tuple[bool, str]:
    """
    Initialize ONLY metrics from success story CSV.

    This processes each row individually to generate one metric per SQL query.
    Assumes semantic models are already generated and available in YAML files.

    Args:
        args: Command line arguments
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        pool_size: Number of concurrent tasks (default: 1 for sequential processing)
    """
    event_helper = BatchEventHelper(BIZ_NAME, emit)
    df = pd.read_csv(args.success_story)

    # Emit task started
    event_helper.task_started(total_items=len(df), success_story=args.success_story)

    async def process_all() -> tuple[bool, List[str]]:
        semaphore = asyncio.Semaphore(pool_size)
        errors: List[str] = []

        async def process_with_semaphore(position, row):
            async with semaphore:
                row_idx = position + 1  # Use position (0-based) instead of DataFrame index
                logger.info(f"Processing row {row_idx}/{len(df)} - generating metrics only")
                try:
                    result = await process_line_metrics_only(
                        row.to_dict(), agent_config, subject_tree, row_idx=row_idx, event_helper=event_helper
                    )
                    return row_idx, result
                except Exception as e:
                    logger.error(f"Error processing row {row_idx}: {e}")
                    return row_idx, {"successful": False, "error": str(e)}

        # Emit task processing
        event_helper.task_processing(total_items=len(df))

        # Process rows with controlled concurrency
        tasks = [
            asyncio.create_task(process_with_semaphore(position, row))
            for position, (_, row) in enumerate(df.iterrows())
        ]

        for task in asyncio.as_completed(tasks):
            row_idx, result = await task
            if not result.get("successful"):
                errors.append(f"Error processing row {row_idx}: {result.get('error')}")

        return (len(df) - len(errors)) > 0, errors

    successful, errors = asyncio.run(process_all())

    # Emit task completed
    event_helper.task_completed(
        total_items=len(df),
        completed_items=len(df) - len(errors),
        failed_items=len(errors),
    )

    error_message = "\n    ".join(errors) if errors else ""
    return successful, error_message


async def process_line_metrics_only(
    row: dict,
    agent_config: AgentConfig,
    subject_tree: Optional[list] = None,
    row_idx: Optional[int] = None,
    event_helper: Optional[BatchEventHelper] = None,
) -> Dict[str, Any]:
    """
    Process a single row to generate ONLY metrics (Step 2).
    Assumes semantic model YAML already exists.

    Args:
        row: CSV row data containing question and sql
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories
        row_idx: Optional row index for progress events
        event_helper: Optional BatchEventHelper to stream progress events
    """
    logger.info(f"Generating metrics for: {row}")

    current_db_config = agent_config.current_db_config()
    sql = row["sql"]
    question = row["question"]
    item_id = str(row_idx) if row_idx is not None else "unknown"

    table_names = extract_table_names(sql, agent_config.db_type)
    full_table_name = table_names[0] if table_names else ""

    # Extract the pure table name (last part of fully qualified name)
    table_name = full_table_name.split(".")[-1] if full_table_name else ""

    if event_helper:
        event_helper.item_started(
            item_id=item_id,
            row_idx=row_idx,
            question=question,
            table_name=table_name,
        )

    if not full_table_name:
        if event_helper:
            event_helper.item_failed(
                item_id=item_id,
                error="No table name found in SQL query",
                row_idx=row_idx,
                question=question,
                table_name=table_name,
            )
        return {"successful": False, "error": "No table name found in SQL query"}

    # Look for existing semantic_model YAML file
    path_manager = get_path_manager()
    semantic_model_dir = str(path_manager.semantic_model_path(agent_config.current_namespace))
    semantic_model_file = os.path.join(semantic_model_dir, f"{table_name}.yml")

    if not os.path.exists(semantic_model_file):
        logger.warning(f"Semantic model file not found: {semantic_model_file}, proceeding anyway")
        semantic_model_file = ""

    # Generate metrics using gen_metrics node
    metrics_user_message = (
        f"Generate metrics for the following SQL query:\n\nSQL:\n{sql}\n\n"
        f"Question: {question}\n\nTable: {table_name}"
    )
    if semantic_model_file:
        metrics_user_message += f"\n\nUse the following semantic model: {semantic_model_file}"

    metrics_input = SemanticNodeInput(
        user_message=metrics_user_message,
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

    try:
        async for action in metrics_node.execute_stream(action_history_manager):
            if event_helper:
                event_helper.item_processing(
                    item_id=item_id,
                    action_name="gen_metrics",
                    status=_action_status_value(action),
                    row_idx=row_idx,
                    messages=action.messages,
                    output=action.output,
                    question=question,
                    table_name=table_name,
                )
            if action.status == ActionStatus.SUCCESS and action.output:
                logger.debug(f"Metrics generation action: {action.messages}")

        logger.info(f"Generated metrics for {question}")
        if event_helper:
            event_helper.item_completed(
                item_id=item_id,
                row_idx=row_idx,
                question=question,
                table_name=table_name,
            )
        return {"successful": True, "error": ""}
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        if event_helper:
            event_helper.item_failed(
                item_id=item_id,
                error=f"Error generating metrics: {str(e)}",
                exception_type=type(e).__name__,
                row_idx=row_idx,
                question=question,
                table_name=table_name,
            )
        return {"successful": False, "error": str(e)}


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
