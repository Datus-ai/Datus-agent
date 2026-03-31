# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import os
from typing import Any, Optional

import pandas as pd

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.storage.semantic_model.auto_create import ensure_semantic_models_exist, extract_tables_from_sql_list
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

logger = get_logger(__name__)

BIZ_NAME = "metric_init"


def _action_status_value(action: Any) -> Optional[str]:
    status = getattr(action, "status", None)
    if status is None:
        return None
    return status.value if hasattr(status, "value") else str(status)


async def init_success_story_metrics_async(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """
    Async version: Initialize metrics from success story CSV by batch processing.

    This reads all SQL queries from the CSV and processes them as a batch
    to extract core unique metrics (deduplicating aggregation patterns).

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM
    """
    event_helper = BatchEventHelper(BIZ_NAME, emit)
    df = pd.read_csv(success_story)

    # Emit task started
    event_helper.task_started(total_items=len(df), success_story=success_story)

    # Step 0: Check and create missing semantic models
    sql_list = [row["sql"] for _, row in df.iterrows() if row.get("sql")]
    all_tables = extract_tables_from_sql_list(sql_list, agent_config)

    if all_tables:
        logger.info(f"Found {len(all_tables)} tables in success story SQL: {all_tables}")

        # Check and create missing semantic models
        success, error, created_tables = await ensure_semantic_models_exist(all_tables, agent_config, emit=None)

        if not success:
            error_msg = f"Failed to create semantic models: {error}"
            logger.error(error_msg)
            event_helper.task_failed(error=error_msg)
            return False, error_msg, None

        if created_tables:
            logger.info(f"Created semantic models for tables: {created_tables}")

    # Build batch message with all SQL queries
    sql_queries = []
    for idx, row in df.iterrows():
        sql = row["sql"]
        question = row["question"]
        sql_queries.append(f"Query {idx + 1}:\nQuestion: {question}\nSQL:\n{sql}")

    batch_message = "Analyze the following SQL queries and extract core metrics:\n\n" + "\n\n---\n\n".join(sql_queries)

    # Append extra instructions if provided
    if extra_instructions:
        batch_message = f"{batch_message}\n\n## Additional Instructions\n{extra_instructions}"

    logger.info(f"Processing {len(df)} SQL queries as batch for core metrics extraction")

    # Get database context
    current_db_config = agent_config.current_db_config()

    metrics_input = SemanticNodeInput(
        user_message=batch_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
        prompt_version="1.1",
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
        if final_result is None:
            error_msg = "Metrics extraction completed but produced no output"
            logger.warning(error_msg)
            event_helper.task_failed(error=error_msg)
            return False, error_msg, None
        logger.info("Batch metrics extraction completed successfully")
        event_helper.task_completed(
            total_items=1,
            completed_items=1,
            failed_items=0,
        )
        return True, "", final_result
    except Exception as e:
        logger.error(f"Error in batch metrics extraction: {e}")
        error = str(e)
        event_helper.task_failed(error=error)
        return False, error, None


def init_success_story_metrics(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """
    Sync wrapper: Initialize metrics from success story CSV by batch processing.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM
    """
    with suppress_keyboard_input():
        return asyncio.run(
            init_success_story_metrics_async(agent_config, success_story, subject_tree, emit, extra_instructions)
        )


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
