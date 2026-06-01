# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import json
import os
from pathlib import Path
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
from datus.storage.knowledge_provenance import (
    METRIC_ARTIFACT_TYPE,
    KnowledgeProvenanceStore,
    build_metric_provenance_rows,
    is_knowledge_provenance_enabled,
)
from datus.storage.semantic_model.auto_create import (
    ensure_semantic_models_exist,
    extract_table_sql_evidence,
    extract_tables_from_sql_list,
)
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

logger = get_logger(__name__)

BIZ_NAME = "metric_init"


def _action_status_value(action: Any) -> Optional[str]:
    status = getattr(action, "status", None)
    if status is None:
        return None
    return status.value if hasattr(status, "value") else str(status)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _parse_context_ids(value: Any) -> list[str]:
    text = _clean_cell(value)
    if not text:
        return []
    parts = [part.strip() for part in text.replace(",", ";").split(";")]
    return [part for part in parts if part]


def _parse_source_metadata(value: Any) -> dict[str, Any]:
    text = _clean_cell(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"raw": text}


def _source_provenance_from_row(row: Any, row_index: int, success_story: str) -> Optional[dict[str, Any]]:
    context_ids: list[str] = []
    for column in ("source_context_ids", "source_context_id", "context_ids", "context_id"):
        context_ids.extend(_parse_context_ids(row.get(column)))
    context_ids = list(dict.fromkeys(context_ids))
    if not context_ids:
        return None

    metadata = _parse_source_metadata(row.get("source_metadata"))
    source_id = _clean_cell(row.get("source_id")) or f"{Path(success_story).name}:{row_index}"
    source_type = _clean_cell(row.get("source_type")) or "success_story"
    metadata.setdefault("source_id", source_id)
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("row_index", row_index)
    question = _clean_cell(row.get("question"))
    if question:
        metadata.setdefault("question", question)
    task_id = _clean_cell(row.get("task_id"))
    if task_id:
        metadata.setdefault("task_id", task_id)

    return {
        "source_id": source_id,
        "source_type": source_type,
        "source_context_ids": context_ids,
        "source_metadata": metadata,
    }


def _extract_metric_artifact_ids(payload: Any) -> list[str]:
    ids: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            text = str(item).strip()
            if text and text not in ids:
                ids.append(text)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            add(value.get("metric_artifact_ids"))
            add(value.get("_synced_metric_artifact_ids"))
            for nested_key in ("result", "sync", "execution_stats", "metric_sync"):
                if nested_key in value:
                    visit(value[nested_key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return ids


def _metric_ids_in_storage(agent_config: AgentConfig) -> set[str]:
    try:
        from datus.storage.metric.store import MetricRAG

        return {
            str(row["id"]) for row in MetricRAG(agent_config).search_all_metrics(select_fields=["id"]) if row.get("id")
        }
    except Exception as exc:  # pragma: no cover - defensive fallback for storage readiness issues
        logger.debug("Failed to snapshot metric IDs for provenance fallback: %s", exc)
        return set()


def _clear_metric_provenance(agent_config: AgentConfig) -> int:
    if not is_knowledge_provenance_enabled(agent_config):
        return 0
    try:
        return KnowledgeProvenanceStore(agent_config).delete_for_artifact_type(METRIC_ARTIFACT_TYPE)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to clear metric provenance sidecar: %s", exc)
        return 0


def _sync_metric_provenance(
    agent_config: AgentConfig,
    metric_artifact_ids: list[str],
    source_entries: list[dict[str, Any]],
) -> int:
    if not metric_artifact_ids or not source_entries or not is_knowledge_provenance_enabled(agent_config):
        return 0
    if len(source_entries) != 1:
        logger.warning(
            "Skipping metric provenance sync because source-to-metric attribution is ambiguous for %d source row(s)",
            len(source_entries),
        )
        return 0

    source = source_entries[0]
    items: list[dict[str, Any]] = []
    for artifact_id in metric_artifact_ids:
        items.append({"id": artifact_id, **source})
    rows = build_metric_provenance_rows(items)
    if not rows:
        return 0

    try:
        written = KnowledgeProvenanceStore(agent_config).upsert_many(rows)
        logger.info(
            "Synced %d metric provenance row(s) for %d metric artifact(s)",
            written,
            len(metric_artifact_ids),
        )
        return written
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to sync metric provenance sidecar: %s", exc)
        return 0


DEFAULT_METRICS_BATCH_SIZE = 1
METRICS_RESPONSE_ACTION_TYPE = f"{GenMetricsAgenticNode.NODE_NAME}_response"


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
        synced_metric_artifact_ids: list[str] = []
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
            for artifact_id in _extract_metric_artifact_ids(getattr(action, "output", None)):
                if artifact_id not in synced_metric_artifact_ids:
                    synced_metric_artifact_ids.append(artifact_id)
            if action.status == ActionStatus.FAILED and action_type == "error":
                terminal_error = action.messages or "Metrics extraction failed"
                logger.error(terminal_error)
                continue
            if action.status == ActionStatus.FAILED and action_type == METRICS_RESPONSE_ACTION_TYPE:
                terminal_error = action.messages or "Metrics extraction failed"
                logger.error(terminal_error)
                continue
            if action.status == ActionStatus.SUCCESS and action_type == METRICS_RESPONSE_ACTION_TYPE and action.output:
                final_result = action.output
                logger.debug(f"Metrics generation action (batch {batch_idx}): {action.messages}")
        if terminal_error:
            return False, terminal_error, None
        if final_result is None:
            return False, "Metrics extraction completed but produced no output", None
        if isinstance(final_result, dict) and synced_metric_artifact_ids:
            final_result["_synced_metric_artifact_ids"] = synced_metric_artifact_ids
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
        cleared_provenance = _clear_metric_provenance(agent_config)
        if cleared_provenance:
            logger.info("Cleared %d stale metric provenance row(s)", cleared_provenance)
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
    success_story_records = [
        {"sql": row["sql"], "question": row.get("question")} for _, row in df.iterrows() if row.get("sql")
    ]
    sql_list = [record["sql"] for record in success_story_records]
    all_tables = extract_tables_from_sql_list(sql_list, agent_config)

    if all_tables:
        logger.info(f"Found {len(all_tables)} tables in success story SQL: {all_tables}")
        sql_evidence_by_table = extract_table_sql_evidence(success_story_records, agent_config)

        # Check and create missing semantic models (per-table, partial failures tolerated)
        success, error, created_tables = await ensure_semantic_models_exist(
            all_tables,
            agent_config,
            emit=None,
            sql_evidence_by_table=sql_evidence_by_table,
        )

        if not success:
            error_msg = f"Failed to create semantic models: {error}"
            logger.error(error_msg)
            event_helper.task_failed(error=error_msg)
            return False, error_msg, None

        if created_tables:
            logger.info(f"Created semantic models for tables: {created_tables}")
        if error:
            logger.warning(f"Semantic model generation had partial failures: {error}")

    # Build query records for all rows. Optional source-context columns are only
    # used by benchmark provenance mode and do not affect normal bootstrap data.
    all_query_records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        sql = row["sql"]
        question = row["question"]
        all_query_records.append(
            {
                "query": f"Query {idx + 1}:\nQuestion: {question}\nSQL:\n{sql}",
                "source": _source_provenance_from_row(row, idx, success_story),
            }
        )

    # Split into batches
    batches = [all_query_records[i : i + batch_size] for i in range(0, len(all_query_records), batch_size)]
    total_batches = len(batches)

    logger.info(
        f"Processing {len(df)} SQL queries in {total_batches} batch(es) (batch_size={batch_size}) for metrics extraction"
    )

    event_helper.task_processing(total_items=total_batches)

    completed_batches = 0
    failed_batches: list[tuple[int, str]] = []
    merged_result: Optional[dict[str, Any]] = None
    provenance_entries = 0

    for batch_idx, batch_records in enumerate(batches):
        batch_queries = [record["query"] for record in batch_records]
        source_entries = [record["source"] for record in batch_records if record.get("source")]
        metric_ids_before = _metric_ids_in_storage(agent_config) if source_entries else set()

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
            metric_artifact_ids = _extract_metric_artifact_ids(batch_result)
            if source_entries and not metric_artifact_ids:
                metric_artifact_ids = sorted(_metric_ids_in_storage(agent_config) - metric_ids_before)
            batch_provenance_entries = _sync_metric_provenance(agent_config, metric_artifact_ids, source_entries)
            provenance_entries += batch_provenance_entries
            if isinstance(batch_result, dict) and batch_provenance_entries:
                batch_result["provenance_entries"] = (
                    batch_result.get("provenance_entries", 0) + batch_provenance_entries
                )

            if merged_result is None:
                merged_result = batch_result
            elif isinstance(merged_result, dict) and isinstance(batch_result, dict):
                for key, value in batch_result.items():
                    if key in merged_result and isinstance(merged_result[key], list) and isinstance(value, list):
                        merged_result[key].extend(value)
                    elif key in merged_result and isinstance(merged_result[key], int) and isinstance(value, int):
                        merged_result[key] += value
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

    if isinstance(merged_result, dict) and provenance_entries:
        merged_result["provenance_entries"] = provenance_entries

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
