# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import os
from typing import Any, Callable, Optional

import pandas as pd

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.prompts.prompt_manager import get_prompt_manager
from datus.schemas.action_history import (
    ActionHistory,  # noqa: F401  (forward-ref for action_callback)
    ActionHistoryManager,
    ActionStatus,
)
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


DEFAULT_METRICS_BATCH_SIZE = 1


async def _generate_metrics_batch(
    batch_queries: list[str],
    batch_idx: int,
    agent_config: AgentConfig,
    subject_tree: Optional[list],
    extra_instructions: Optional[str],
    event_helper: BatchEventHelper,
    action_callback: Optional[Callable[["ActionHistory"], None]],
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Process a single batch of SQL queries for metrics extraction."""
    batch_message = "Analyze the following SQL queries and extract core metrics:\n\n" + "\n\n---\n\n".join(
        batch_queries
    )

    if extra_instructions:
        batch_message = f"{batch_message}\n\n## Additional Instructions\n{extra_instructions}"

    current_db_config = agent_config.current_db_config()
    latest_prompt_version = get_prompt_manager(agent_config=agent_config).get_latest_version("gen_metrics_system")

    metrics_input = SemanticNodeInput(
        user_message=batch_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
        prompt_version=latest_prompt_version,
    )

    metrics_node = GenMetricsAgenticNode(
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()
    metrics_node.input = metrics_input

    batch_id = f"batch-{batch_idx}"

    try:
        final_result = None
        terminal_error = None
        async for action in metrics_node.execute_stream(action_history_manager):
            if action_callback is not None:
                try:
                    action_callback(action)
                except Exception as cb_exc:  # pragma: no cover - defensive
                    logger.debug("metric action_callback raised: %s", cb_exc)
            if event_helper:
                event_helper.item_processing(
                    item_id=batch_id,
                    action_name="gen_metrics",
                    status=_action_status_value(action),
                    messages=action.messages,
                    output=action.output,
                )
            action_type = getattr(action, "action_type", "")
            if action.status == ActionStatus.FAILED and action_type == "error":
                terminal_error = action.messages or "Metrics extraction failed"
                logger.error(terminal_error)
                continue
            if action.status == ActionStatus.SUCCESS and action_type == "metrics_response" and action.output:
                final_result = action.output
                logger.debug(f"Metrics generation action (batch {batch_idx}): {action.messages}")
        if terminal_error:
            return False, terminal_error, None
        if final_result is None:
            return False, "Metrics extraction completed but produced no output", None
        return True, "", final_result
    except Exception as e:
        logger.error(f"Error in metrics extraction (batch {batch_idx}): {e}")
        return False, str(e), None


async def init_success_story_metrics_async(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
    *,
    build_mode: str = "overwrite",
    action_callback: Optional[Callable[["ActionHistory"], None]] = None,
    batch_size: int = DEFAULT_METRICS_BATCH_SIZE,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """
    Async version: Initialize metrics from success story CSV by batch processing.

    This reads all SQL queries from the CSV and processes them in batches
    to extract core unique metrics (deduplicating aggregation patterns).
    Each batch is processed independently so that one failure does not
    block the rest.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM
        build_mode: ``"overwrite"`` (default) regenerates unconditionally;
            ``"incremental"`` skips the LLM call when the metric store
            already contains entries.
        batch_size: Number of SQL queries per batch (default 1).
    """
    if batch_size <= 0:
        from datus.utils.exceptions import DatusException, ErrorCode

        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT, error_message=f"batch_size must be > 0, got {batch_size}"
        )

    event_helper = BatchEventHelper(BIZ_NAME, emit)

    if build_mode == "overwrite":
        from datus.storage.metric.store import MetricRAG

        logger.info(
            "[overwrite] Wiping metrics store for project '%s' before re-population",
            agent_config.project_name,
        )
        MetricRAG(agent_config).truncate()
    elif build_mode == "incremental":
        from datus.storage.metric.init_utils import exists_metrics
        from datus.storage.metric.store import MetricRAG

        existing = exists_metrics(MetricRAG(agent_config), build_mode)
        if existing:
            logger.info(
                "Metrics incremental skip: %d existing metric(s) found, no LLM call.",
                len(existing),
            )
            event_helper.task_completed(
                total_items=len(existing),
                completed_items=len(existing),
                failed_items=0,
            )
            return True, "", {"skipped": True, "existing": len(existing)}

    df = pd.read_csv(success_story)

    # Emit task started
    event_helper.task_started(total_items=len(df), success_story=success_story)

    # Step 0: Check and create missing semantic models
    sql_list = [row["sql"] for _, row in df.iterrows() if row.get("sql")]
    all_tables = extract_tables_from_sql_list(sql_list, agent_config)

    if all_tables:
        logger.info(f"Found {len(all_tables)} tables in success story SQL: {all_tables}")

        # Check and create missing semantic models (per-table, partial failures tolerated)
        success, error, created_tables = await ensure_semantic_models_exist(all_tables, agent_config, emit=None)

        if not success:
            error_msg = f"Failed to create semantic models: {error}"
            logger.error(error_msg)
            event_helper.task_failed(error=error_msg)
            return False, error_msg, None

        if created_tables:
            logger.info(f"Created semantic models for tables: {created_tables}")
        if error:
            logger.warning(f"Semantic model generation had partial failures: {error}")

    # Build query strings for all rows
    all_query_strings = []
    for idx, row in df.iterrows():
        sql = row["sql"]
        question = row["question"]
        all_query_strings.append(f"Query {idx + 1}:\nQuestion: {question}\nSQL:\n{sql}")

    # Split into batches
    batches = [all_query_strings[i : i + batch_size] for i in range(0, len(all_query_strings), batch_size)]
    total_batches = len(batches)

    logger.info(
        f"Processing {len(df)} SQL queries in {total_batches} batch(es) (batch_size={batch_size}) for metrics extraction"
    )

    event_helper.task_processing(total_items=total_batches)

    completed_batches = 0
    failed_batches: list[tuple[int, str]] = []
    merged_result: Optional[dict[str, Any]] = None

    for batch_idx, batch_queries in enumerate(batches):
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_queries)} queries)")

        success, error, batch_result = await _generate_metrics_batch(
            batch_queries,
            batch_idx,
            agent_config,
            subject_tree,
            extra_instructions,
            event_helper,
            action_callback,
        )

        if success and batch_result is not None:
            completed_batches += 1
            if merged_result is None:
                merged_result = batch_result
            elif isinstance(merged_result, dict) and isinstance(batch_result, dict):
                for key, value in batch_result.items():
                    if key in merged_result and isinstance(merged_result[key], list) and isinstance(value, list):
                        merged_result[key].extend(value)
                    elif key not in merged_result:
                        merged_result[key] = value
            logger.info(f"Batch {batch_idx + 1}/{total_batches} completed successfully")
        else:
            failed_batches.append((batch_idx, error))
            logger.warning(f"Batch {batch_idx + 1}/{total_batches} failed: {error}, continuing with remaining batches")

    if completed_batches == 0:
        error_summary = "; ".join(f"batch {i + 1}: {e}" for i, e in failed_batches)
        error_msg = f"All {total_batches} batch(es) failed: {error_summary}"
        logger.error(error_msg)
        event_helper.task_failed(error=error_msg)
        return False, error_msg, None

    partial_error = ""
    if failed_batches:
        partial_error = "; ".join(f"batch {i + 1}: {e}" for i, e in failed_batches)
        logger.warning(f"Metrics extraction partially succeeded: {partial_error}")

    logger.info(f"Metrics extraction completed: {completed_batches}/{total_batches} batch(es) succeeded")
    event_helper.task_completed(
        total_items=total_batches,
        completed_items=completed_batches,
        failed_items=len(failed_batches),
    )
    return True, partial_error, merged_result


def init_success_story_metrics(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
    *,
    build_mode: str = "overwrite",
    batch_size: int = DEFAULT_METRICS_BATCH_SIZE,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """
    Sync wrapper: Initialize metrics from success story CSV by batch processing.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM
        build_mode: Forwarded to :func:`init_success_story_metrics_async`.
        batch_size: Number of SQL queries per batch (default 1).
    """
    with suppress_keyboard_input():
        return asyncio.run(
            init_success_story_metrics_async(
                agent_config,
                success_story,
                subject_tree,
                emit,
                extra_instructions,
                build_mode=build_mode,
                batch_size=batch_size,
            )
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
