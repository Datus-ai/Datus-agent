# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.agent.node.semantic_authoring import (
    AUTHORING_FORMAT_OSI,
    discover_osi_semantic_models,
    resolve_authoring_format,
)
from datus.configuration.agent_config import AgentConfig
from datus.prompts.prompt_manager import get_prompt_manager
from datus.schemas.action_history import (
    ActionHistory,  # noqa: F401  (forward-ref for action_callback)
    ActionHistoryManager,
    ActionStatus,
)
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SourceQueryEvidence
from datus.storage.knowledge_provenance import (
    METRIC_ARTIFACT_TYPE,
    KnowledgeProvenanceStore,
    build_metric_provenance_rows,
    is_knowledge_provenance_enabled,
)
from datus.tools.func_tool.sql_modeling_planner import (
    source_provenance_from_success_story_row,
    source_query_from_success_story_row,
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


def _source_provenance_from_row(row: Any, row_index: int, success_story: str) -> Optional[dict[str, Any]]:
    return source_provenance_from_success_story_row(row, row_index, success_story)


def _source_query_from_row(row: Any, row_index: int, success_story: str) -> Optional[SourceQueryEvidence]:
    return source_query_from_success_story_row(row, row_index, success_story)


def _source_provenance_from_query(source_query: SourceQueryEvidence) -> Optional[dict[str, Any]]:
    if not source_query.source_context_ids:
        return None
    return {
        "source_id": source_query.source_id,
        "source_type": source_query.source_type,
        "source_context_ids": list(source_query.source_context_ids),
        "source_metadata": dict(source_query.source_metadata),
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


DEFAULT_METRICS_BATCH_SIZE = 5
METRICS_NODE_NAME = GenMetricsAgenticNode.NODE_NAME
METRICS_RESPONSE_ACTION_TYPE = f"{METRICS_NODE_NAME}_response"


def _metrics_authoring_format(agent_config: AgentConfig) -> str:
    return resolve_authoring_format(agent_config)


async def _ensure_semantic_models_for_metrics(
    agent_config: AgentConfig,
    success_story: str,
    action_callback: Optional[Callable[["ActionHistory"], None]] = None,
) -> tuple[bool, str, Optional[dict[str, str]]]:
    from datus.storage.semantic_model.semantic_model_init import (
        SEMANTIC_MODEL_RESPONSE_ACTION_TYPE,
        init_success_story_semantic_model_async,
    )

    selected_files: list[str] = []

    def capture_semantic_result(action: "ActionHistory") -> None:
        if (
            _action_status_value(action) == ActionStatus.SUCCESS.value
            and getattr(action, "action_type", "") == SEMANTIC_MODEL_RESPONSE_ACTION_TYPE
            and isinstance(getattr(action, "output", None), dict)
        ):
            files = action.output.get("semantic_models")
            if isinstance(files, str):
                files = [files]
            if isinstance(files, list):
                selected_files.extend(str(path).strip() for path in files if str(path).strip())
        if action_callback is not None:
            action_callback(action)

    logger.info("Preparing semantic models through the shared gen_semantic_model workflow")
    success, error = await init_success_story_semantic_model_async(
        agent_config,
        success_story,
        emit=None,
        build_mode="incremental",
        action_callback=capture_semantic_result,
        require_exact_osi_target=_metrics_authoring_format(agent_config) == AUTHORING_FORMAT_OSI,
    )
    if not success or _metrics_authoring_format(agent_config) != AUTHORING_FORMAT_OSI:
        return success, error, None

    live_models = discover_osi_semantic_models(agent_config)
    if not live_models:
        return False, "OSI semantic model generation did not produce a valid model file", None
    selected_paths = {path.replace("\\", "/").removeprefix("./") for path in selected_files}
    selected_models = [
        model
        for model in live_models
        if model["semantic_model_file"] in selected_paths
        or str(Path(model["absolute_path"]).resolve(strict=False)).replace("\\", "/").removeprefix("./")
        in selected_paths
    ]
    if len(selected_models) != 1:
        return False, "gen_semantic_model did not report one exact OSI semantic-model target", None
    selected_model = selected_models[0]
    return (
        True,
        "",
        {
            "semantic_model_name": selected_model["semantic_model_name"],
            "semantic_model_file": selected_model["semantic_model_file"],
        },
    )


def _final_metric_count(agent_config: AgentConfig) -> int:
    """Return the persisted metric count after all Node batches finish."""
    from datus.storage.metric.store import MetricRAG

    try:
        count = MetricRAG(agent_config).get_metrics_size()
        return count if isinstance(count, int) else 0
    except Exception as exc:  # pragma: no cover - defensive reporting fallback
        logger.warning("Failed to load final metric count: %s", exc)
        return 0


async def _generate_metrics_batch(
    batch_sources: list[SourceQueryEvidence],
    batch_idx: int,
    agent_config: AgentConfig,
    subject_tree: Optional[list],
    extra_instructions: Optional[str],
    event_helper: BatchEventHelper,
    action_callback: Optional[Callable[["ActionHistory"], None]],
    semantic_model_target: Optional[dict[str, str]] = None,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Process a single batch of SQL queries for metrics extraction."""
    rendered_queries = []
    for source_index, source in enumerate(batch_sources, 1):
        rendered = f"Query {source_index}:\nQuestion: {source.question}\nSQL:\n{source.sql}"
        rendered_queries.append(rendered)
    batch_message = "Analyze the following SQL queries and extract core metrics:\n\n" + "\n\n---\n\n".join(
        rendered_queries
    )

    if extra_instructions:
        batch_message = f"{batch_message}\n\n## Additional Instructions\n{extra_instructions}"

    current_db_config = agent_config.current_db_config()
    runtime_db_context_getter = getattr(agent_config, "runtime_db_context", None)
    runtime_db_context = runtime_db_context_getter() if callable(runtime_db_context_getter) else {}
    runtime_db_context = runtime_db_context if isinstance(runtime_db_context, dict) else {}
    latest_prompt_version = get_prompt_manager(agent_config=agent_config).get_latest_version("gen_metrics_system")

    metrics_input = SemanticNodeInput(
        user_message=batch_message,
        semantic_model_name=(semantic_model_target or {}).get("semantic_model_name"),
        semantic_model_file=(semantic_model_target or {}).get("semantic_model_file"),
        catalog=runtime_db_context.get("catalog")
        or runtime_db_context.get("catalog_name")
        or current_db_config.catalog,
        database=runtime_db_context.get("database")
        or runtime_db_context.get("database_name")
        or current_db_config.database,
        db_schema=runtime_db_context.get("schema")
        or runtime_db_context.get("db_schema")
        or runtime_db_context.get("schema_name")
        or current_db_config.schema,
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

    This is a batch wrapper around the same ``gen_semantic_model`` and
    ``gen_metrics`` Nodes used by direct CLI/subagent calls. Each Node owns
    SQL planning and authoring; this function only loads rows, prepares the
    semantic prerequisite, batches metric calls, and aggregates progress.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM
        build_mode: ``"overwrite"`` (default) clears existing metric storage
            before invoking the Nodes; ``"incremental"`` preserves it and
            lets the Node reconcile existing definitions.
        batch_size: Number of SQL queries per batch (default 5).
    """
    if batch_size <= 0:
        from datus.utils.exceptions import DatusException, ErrorCode

        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT, error_message=f"batch_size must be > 0, got {batch_size}"
        )

    event_helper = BatchEventHelper(BIZ_NAME, emit)

    if build_mode == "check":
        from datus.storage.metric.store import MetricRAG

        metric_rag = MetricRAG(agent_config)
        logger.info(
            "[check] metrics rows=%d; generation skipped",
            metric_rag.get_metrics_size(),
        )
        return True, "", {"checked": True, "metrics_count": metric_rag.get_metrics_size()}

    df = pd.read_csv(success_story)

    # Emit task started
    event_helper.task_started(total_items=len(df), success_story=success_story)

    missing_columns = [column for column in ("question", "sql") if column not in df.columns]
    if missing_columns:
        error_msg = f"Success story CSV is missing required columns: {missing_columns}"
        logger.error(error_msg)
        event_helper.task_failed(error=error_msg)
        return False, error_msg, None

    source_queries: list[SourceQueryEvidence] = []
    for idx, row in df.iterrows():
        source_query = _source_query_from_row(row, idx, success_story)
        if source_query is not None:
            source_queries.append(source_query)
    if not source_queries:
        error_msg = "Success story CSV contains no SQL rows"
        logger.error(error_msg)
        event_helper.task_failed(error=error_msg)
        return False, error_msg, None

    # Step 0: Resolve one semantic model for the entire CSV before metric batching.
    success, error, semantic_model_target = await _ensure_semantic_models_for_metrics(
        agent_config,
        success_story,
        action_callback=action_callback,
    )

    if not success:
        error_msg = f"Failed to create semantic models: {error}"
        logger.error(error_msg)
        event_helper.task_failed(error=error_msg)
        return False, error_msg, None

    if build_mode == "overwrite":
        from datus.storage.metric.store import MetricRAG

        metric_rag = MetricRAG(agent_config)
        logger.info(
            "[overwrite] Wiping metrics rows for datasource '%s' before re-population",
            metric_rag.datasource_id,
        )
        metric_rag.truncate()
        cleared_provenance = _clear_metric_provenance(agent_config)
        if cleared_provenance:
            logger.info("Cleared %d stale metric provenance row(s)", cleared_provenance)

    # Split into batches
    batches = [source_queries[i : i + batch_size] for i in range(0, len(source_queries), batch_size)]
    total_batches = len(batches)

    logger.info(
        f"Processing {len(source_queries)} SQL queries in {total_batches} batch(es) "
        f"(batch_size={batch_size}) for metrics extraction"
    )

    event_helper.task_processing(total_items=total_batches)

    completed_batches = 0
    failed_batches: list[tuple[int, str]] = []
    merged_result: Optional[dict[str, Any]] = None
    provenance_entries = 0
    for batch_idx, batch_records in enumerate(batches):
        source_entries = [
            source for record in batch_records if (source := _source_provenance_from_query(record)) is not None
        ]

        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_records)} queries)")

        metric_ids_before = _metric_ids_in_storage(agent_config) if source_entries else set()

        success, error, batch_result = await _generate_metrics_batch(
            batch_records,
            batch_idx,
            agent_config,
            subject_tree,
            extra_instructions,
            event_helper,
            action_callback,
            semantic_model_target=semantic_model_target,
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

    if isinstance(merged_result, dict):
        if provenance_entries:
            merged_result["provenance_entries"] = provenance_entries
        final_metrics_count = _final_metric_count(agent_config)
        merged_result["metrics_count"] = final_metrics_count
        merged_result["final_metrics_count"] = final_metrics_count
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
        batch_size: Number of SQL queries per batch (default 5).
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

    if _metrics_authoring_format(agent_config) == AUTHORING_FORMAT_OSI:
        from datus.tools.func_tool.generation_tools import GenerationTools

        result = GenerationTools(agent_config=agent_config, authoring_format=AUTHORING_FORMAT_OSI).sync_osi_to_db(
            yaml_file_path,
            include_semantic_objects=False,
            include_metrics=True,
        )
        if result.get("success"):
            return True, result.get("message", "")
        return False, result.get("error", "Unknown error")

    # Import from semantic_model package to avoid circular dependency
    from datus.storage.semantic_model.semantic_model_init import process_semantic_yaml_file

    return process_semantic_yaml_file(yaml_file_path, agent_config, include_semantic_objects=False)
