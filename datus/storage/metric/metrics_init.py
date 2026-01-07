# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from datus.agent.node.semantic_agentic_node import SemanticAgenticNode
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names

logger = get_logger(__name__)

BIZ_NAME = "metrics_init"


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
):
    """
    Initialize metrics from success story CSV file using SemanticAgenticNode in workflow mode.

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

        async def process_with_semaphore(position, idx, row):
            async with semaphore:
                row_idx = position + 1  # Use position (0-based) instead of DataFrame index
                logger.info(f"Processing row {row_idx}/{len(df)}")
                try:
                    result = await process_line(
                        row.to_dict(), agent_config, subject_tree, row_idx=row_idx, event_helper=event_helper
                    )
                    return row_idx, result
                except Exception as e:
                    logger.error(f"Error processing row {row_idx}: {e}")
                    return row_idx, {"successful": False, "error": str(e)}

        # Emit task processing
        event_helper.task_processing(total_items=len(df))

        # Process rows with controlled concurrency
        # Use enumerate to get position (0-based) independent of DataFrame index
        tasks = [
            asyncio.create_task(process_with_semaphore(position, idx, row))
            for position, (idx, row) in enumerate(df.iterrows())
        ]

        for task in asyncio.as_completed(tasks):
            row_idx, result = await task
            if not result.get("successful"):
                errors.append(f"Error processing row {row_idx}: {result.get('error')}")

        return (len(df) - len(errors)) > 0, errors

    # Run the async function
    successful, errors = asyncio.run(process_all())

    # Emit task completed
    event_helper.task_completed(
        total_items=len(df),
        completed_items=len(df) - len(errors),
        failed_items=len(errors),
    )

    if errors:
        error_message = "\n    ".join(errors)
    else:
        error_message = ""
    return successful, error_message


async def process_line(
    row: dict,
    agent_config: AgentConfig,
    subject_tree: Optional[list] = None,
    row_idx: Optional[int] = None,
    event_helper: Optional[BatchEventHelper] = None,
) -> Dict[str, Any]:
    """
    Process a single line from the CSV using SemanticAgenticNode in workflow mode.

    Args:
        row: CSV row data containing question and sql
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories
        row_idx: Optional row index for progress events
        event_helper: Optional BatchEventHelper to stream progress events
    """
    logger.info(f"processing line: {row}")

    current_db_config = agent_config.current_db_config()
    sql = row["sql"]
    question = row["question"]
    item_id = str(row_idx) if row_idx is not None else "unknown"

    # Extract table name from SQL query (as requested by user)
    table_names = extract_table_names(sql, agent_config.db_type)
    table_name = table_names[0] if table_names else ""

    if event_helper:
        event_helper.item_started(
            item_id=item_id,
            row_idx=row_idx,
            question=question,
            table_name=table_name,
        )

    if not table_name:
        logger.error(f"No table name found in SQL query: {row['sql']}")
        if event_helper:
            event_helper.item_failed(
                item_id=item_id,
                error="No table name found in SQL query",
                row_idx=row_idx,
                question=question,
                table_name=table_name,
            )
        return {
            "successful": False,
            "error": "No table name found in SQL query",
        }

    # Step 1: Generate semantic model using SemanticAgenticNode
    semantic_user_message = f"Generate semantic model for table: {table_name}\nQuestion context: {question}"
    semantic_input = SemanticNodeInput(
        user_message=semantic_user_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
    )

    semantic_node = SemanticAgenticNode(
        node_name="gen_semantic_model",
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()
    semantic_model_file = None

    try:
        semantic_node.input = semantic_input
        async for action in semantic_node.execute_stream(action_history_manager):
            if event_helper:
                event_helper.item_processing(
                    item_id=item_id,
                    action_name="gen_semantic_model",
                    status=_action_status_value(action),
                    row_idx=row_idx,
                    messages=action.messages,
                    output=action.output,
                    question=question,
                    table_name=table_name,
                )
            if action.status == ActionStatus.SUCCESS and action.output:
                output = action.output
                if isinstance(output, dict):
                    semantic_model_file = output.get("semantic_model")

        if not semantic_model_file:
            logger.error(f"Failed to generate semantic model for {row['question']}")
            if event_helper:
                event_helper.item_failed(
                    item_id=item_id,
                    error="Failed to generate semantic model",
                    row_idx=row_idx,
                    question=question,
                    table_name=table_name,
                )
            return {
                "successful": False,
                "error": "Failed to generate semantic model",
            }

        logger.info(f"Generated semantic model: {semantic_model_file}")

    except Exception as e:
        logger.error(f"Error generating semantic model for {row['question']}: {e}")
        if event_helper:
            event_helper.item_failed(
                item_id=item_id,
                error=f"Error generating semantic model for this question, reason: {str(e)}",
                exception_type=type(e).__name__,
                row_idx=row_idx,
                question=question,
                table_name=table_name,
            )
        return {
            "successful": False,
            "error": f"Error generating semantic model for this question, reason: {str(e)}",
        }

    # Step 2: Generate metrics using SemanticAgenticNode
    metrics_user_message = (
        f"Generate metrics for the following SQL query:\n\nSQL:\n{sql}\n\n"
        f"Question: {question}\n\nTable: {table_name}"
        f"\n\nUse the following semantic model: {semantic_model_file}"
    )
    metrics_input = SemanticNodeInput(
        user_message=metrics_user_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
    )

    metrics_node = SemanticAgenticNode(
        node_name="gen_metrics",
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()

    try:
        metrics_node.input = metrics_input
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

        logger.info(f"Generated metrics for {row['question']}")
        if event_helper:
            event_helper.item_completed(
                item_id=item_id,
                row_idx=row_idx,
                question=question,
                table_name=table_name,
            )
        return {
            "successful": True,
            "error": "",
        }
    except Exception as e:
        logger.error(f"Error generating metrics for {row['question']}: {e}")
        if event_helper:
            event_helper.item_failed(
                item_id=item_id,
                error=f"Error generating metrics for this question, reason: {str(e)}",
                exception_type=type(e).__name__,
                row_idx=row_idx,
                question=question,
                table_name=table_name,
            )
        return {
            "successful": False,
            "error": f"Error generating metrics for this question, reason: {str(e)}",
        }


def init_semantic_yaml_metrics(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Initialize metrics from semantic YAML file by syncing directly to LanceDB.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
        emit: Optional callback to stream progress events
    """
    if not os.path.exists(yaml_file_path):
        logger.error(f"Semantic YAML file {yaml_file_path} not found")
        return False, f"Semantic YAML file {yaml_file_path} not found"

    return process_semantic_yaml_file(yaml_file_path, agent_config)


def process_semantic_yaml_file(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Process semantic YAML file by directly syncing to LanceDB using GenerationHooks.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
        emit: Optional callback to stream progress events
    Returns:
        - Whether the execution was successful
        - Failed reason

    """
    logger.info(f"Processing semantic YAML file: {yaml_file_path}")

    # Use GenerationHooks static method to sync to DB
    result = GenerationHooks._sync_semantic_to_db(yaml_file_path, agent_config)

    if result.get("success"):
        logger.info(f"Successfully synced semantic YAML to LanceDB: {result.get('message')}")
        return True, ""
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to sync semantic YAML to LanceDB: {error}")
        return False, error
