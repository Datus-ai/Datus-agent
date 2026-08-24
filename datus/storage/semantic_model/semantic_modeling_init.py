# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Bootstrap orchestration for unified Dosi semantic modeling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import pandas as pd

from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type
from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.semantic_agentic_node_models import (
    SemanticNodeInput,
    SourceQueryEvidence,
    source_query_from_success_story_row,
)
from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_dataset.store import SemanticDatasetRAG
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig
    from datus.schemas.action_history import ActionHistory

logger = get_logger(__name__)

BIZ_NAME = "semantic_modeling_init"
DEFAULT_SEMANTIC_MODELING_BATCH_SIZE = 5
SEMANTIC_MODELING_RESPONSE_ACTION_TYPE = f"{SemanticModelingAgenticNode.NODE_NAME}_response"
SemanticAuthoringScope = Literal["datasets", "full"]


def _action_status_value(action: Any) -> str:
    status = getattr(action, "status", None)
    return str(getattr(status, "value", status) or "").lower()


def _build_batch_prompt(
    source_queries: list[SourceQueryEvidence],
    *,
    subject_tree: Optional[list[str]] = None,
    authoring_scope: SemanticAuthoringScope = "full",
) -> str:
    if authoring_scope == "datasets":
        intro = (
            "Use the unified Dosi semantic-modeling workflow for this bootstrap batch. "
            "Author only reusable datasets, fields, relationships, and model metadata supported by one semantic "
            "model. Do not create, update, or delete metrics, and preserve all existing metric definitions. "
            "Use the SQL as evidence rather than as a required result shape."
        )
    else:
        intro = (
            "Use the unified Dosi semantic-modeling workflow for this bootstrap batch. "
            "Author the reusable datasets, relationships, and native metrics supported by one semantic model, "
            "using the SQL as evidence rather than as a required result shape."
        )
    parts = [intro]
    if subject_tree:
        parts.append(f"Preferred subject tree: {', '.join(subject_tree)}")
    parts.append("SQL evidence:")
    for index, source in enumerate(source_queries, 1):
        query = [
            f"Query {index}:",
            f"Question: {source.question}",
            "SQL:",
            source.sql,
        ]
        parts.append("\n".join(query))
    return "\n\n".join(parts)


def _load_source_queries(success_story: str) -> tuple[list[SourceQueryEvidence], str]:
    try:
        frame = pd.read_csv(success_story)
    except FileNotFoundError:
        return [], f"Success story CSV file not found: {success_story}"
    except pd.errors.EmptyDataError:
        return [], f"Success story CSV file is empty: {success_story}"
    except Exception as exc:
        return [], f"Failed to read success story CSV file '{success_story}': {exc}"

    missing_columns = [column for column in ("question", "sql") if column not in frame.columns]
    if missing_columns:
        return [], f"Success story CSV is missing required columns: {missing_columns}"

    source_queries = [
        source
        for index, row in frame.iterrows()
        if (source := source_query_from_success_story_row(row, index, success_story)) is not None
    ]
    if not source_queries:
        return [], "Success story CSV contains no SQL rows"
    return source_queries, ""


def _prepare_storage(agent_config: "AgentConfig") -> None:
    """Initialize stores without deleting projections owned by sibling YAML artifacts."""
    agent_config.check_init_storage_config("semantic_model")
    agent_config.check_init_storage_config("metric")


def _runtime_database_context(agent_config: "AgentConfig") -> tuple[str, str, str]:
    current_db = agent_config.current_db_config()
    runtime_getter = getattr(agent_config, "runtime_db_context", None)
    runtime = runtime_getter() if callable(runtime_getter) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    catalog = runtime.get("catalog") or runtime.get("catalog_name") or current_db.catalog
    database = runtime.get("database") or runtime.get("database_name") or current_db.database
    schema = runtime.get("schema") or runtime.get("db_schema") or runtime.get("schema_name") or current_db.schema
    return str(catalog or ""), str(database or ""), str(schema or "")


def _normalized_model_path(path: str) -> str:
    return str(Path(path)).replace("\\", "/").removeprefix("./")


async def _run_semantic_modeling_batch(
    agent_config: "AgentConfig",
    source_queries: list[SourceQueryEvidence],
    *,
    subject_tree: Optional[list[str]],
    target_hint: str,
    event_helper: BatchEventHelper,
    batch_index: int,
    action_callback: Optional[Callable[["ActionHistory"], None]],
    authoring_scope: SemanticAuthoringScope = "full",
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    catalog, database, schema = _runtime_database_context(agent_config)
    node = SemanticModelingAgenticNode(
        agent_config=agent_config,
        execution_mode="workflow",
        authoring_scope=authoring_scope,
        is_subagent=True,
    )
    node.input = SemanticNodeInput(
        user_message=_build_batch_prompt(
            source_queries,
            subject_tree=subject_tree,
            authoring_scope=authoring_scope,
        ),
        authoring_scope=authoring_scope,
        catalog=catalog or None,
        database=database or None,
        db_schema=schema or None,
        semantic_model_file=target_hint or None,
        source_queries=source_queries,
    )

    batch_id = f"batch-{batch_index + 1}"
    event_helper.item_started(batch_id, sql_count=len(source_queries))
    final_result: Optional[dict[str, Any]] = None
    terminal_error = ""
    try:
        async for action in node.execute_stream(ActionHistoryManager()):
            if action_callback is not None:
                action_callback(action)
            action_type = str(getattr(action, "action_type", "") or "")
            status = _action_status_value(action)
            event_helper.item_processing(
                item_id=batch_id,
                action_name=action_type or SemanticModelingAgenticNode.NODE_NAME,
                status=status or None,
                messages=getattr(action, "messages", None),
                output=getattr(action, "output", None),
            )
            if status == ActionStatus.FAILED.value and action_type in {
                "error",
                SEMANTIC_MODELING_RESPONSE_ACTION_TYPE,
            }:
                terminal_error = str(getattr(action, "messages", None) or "semantic_modeling failed")
            if action_type == SEMANTIC_MODELING_RESPONSE_ACTION_TYPE:
                output = getattr(action, "output", None)
                if isinstance(output, dict):
                    final_result = dict(output)
    except Exception as exc:
        terminal_error = str(exc)

    if terminal_error:
        event_helper.item_failed(batch_id, terminal_error)
        return False, terminal_error, None
    if final_result is None:
        error = "semantic_modeling completed without a structured final result"
        event_helper.item_failed(batch_id, error)
        return False, error, None
    result_status = str(final_result.get("status") or "").strip().lower()
    completed = result_status == "generated" or (
        result_status == "skipped" and final_result.get("skip_reason") == "no_semantic_change"
    )
    if not final_result.get("success") or not completed:
        error = str(final_result.get("error") or "semantic_modeling did not complete the batch")
        event_helper.item_failed(batch_id, error)
        return False, error, final_result

    event_helper.item_completed(batch_id, sql_count=len(source_queries))
    return True, "", final_result


async def init_success_story_semantic_modeling_async(
    agent_config: "AgentConfig",
    success_story: str,
    subject_tree: Optional[list[str]] = None,
    emit: Optional[BatchEventEmitter] = None,
    *,
    build_mode: str = "overwrite",
    action_callback: Optional[Callable[["ActionHistory"], None]] = None,
    batch_size: int = DEFAULT_SEMANTIC_MODELING_BATCH_SIZE,
    authoring_scope: SemanticAuthoringScope = "full",
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Build Dosi semantic assets in bounded batches.

    ``overwrite`` and ``incremental`` are compatibility aliases for the same
    artifact-reconcile behavior. ``check`` remains the only non-authoring mode.
    """
    if resolve_semantic_adapter_type(agent_config) != "dosi":
        return False, "semantic_modeling bootstrap is available only when semantic_adapter=dosi", None
    if build_mode not in {"check", "overwrite", "incremental"}:
        return False, f"Unsupported semantic_modeling build mode: {build_mode}", None
    if authoring_scope not in {"datasets", "full"}:
        return False, f"Unsupported semantic_modeling authoring scope: {authoring_scope}", None
    if batch_size <= 0:
        return False, f"batch_size must be > 0, got {batch_size}", None

    if build_mode == "check":
        semantic_count = SemanticDatasetRAG(agent_config).get_size()
        metric_count = MetricRAG(agent_config).get_metrics_size()
        return (
            True,
            "",
            {
                "checked": True,
                "semantic_dataset_count": semantic_count,
                "metrics_count": metric_count,
                "authoring_scope": authoring_scope,
            },
        )

    source_queries, error = _load_source_queries(success_story)
    if error:
        return False, error, None
    _prepare_storage(agent_config)

    batches = [source_queries[index : index + batch_size] for index in range(0, len(source_queries), batch_size)]
    event_helper = BatchEventHelper(BIZ_NAME, emit)
    event_helper.task_started(total_items=len(source_queries), success_story=success_story)
    event_helper.task_processing(total_items=len(batches), batch_size=batch_size)

    completed = 0
    target_hint = ""
    semantic_models: list[str] = []
    for batch_index, batch in enumerate(batches):
        success, batch_error, batch_result = await _run_semantic_modeling_batch(
            agent_config,
            batch,
            subject_tree=subject_tree,
            target_hint=target_hint,
            event_helper=event_helper,
            batch_index=batch_index,
            action_callback=action_callback,
            authoring_scope=authoring_scope,
        )
        if not success:
            error = f"semantic_modeling batch {batch_index + 1}/{len(batches)} failed: {batch_error}"
            event_helper.task_failed(
                error,
                completed_batches=completed,
                failed_batch=batch_index + 1,
            )
            return (
                False,
                error,
                {
                    "batches_total": len(batches),
                    "batches_completed": completed,
                    "sql_entries_total": len(source_queries),
                    "sql_entries_covered": sum(len(item) for item in batches[:completed]),
                    "semantic_models": semantic_models,
                    "failed_result": batch_result,
                },
            )

        completed += 1
        batch_models = batch_result.get("semantic_models") if isinstance(batch_result, dict) else []
        if isinstance(batch_models, str):
            batch_models = [batch_models]
        for model_path in batch_models or []:
            normalized = _normalized_model_path(str(model_path))
            if normalized and normalized not in semantic_models:
                semantic_models.append(normalized)
        if batch_models:
            target_hint = str(batch_models[0])

    semantic_count = SemanticDatasetRAG(agent_config).get_size()
    metric_count = MetricRAG(agent_config).get_metrics_size()
    result = {
        "batches_total": len(batches),
        "batches_completed": completed,
        "sql_entries_total": len(source_queries),
        "sql_entries_covered": len(source_queries),
        "semantic_models": semantic_models,
        "semantic_dataset_count": semantic_count,
        "metrics_count": metric_count,
        "authoring_scope": authoring_scope,
    }
    event_helper.task_completed(total_items=len(batches), completed_items=completed)
    logger.info(
        "semantic_modeling bootstrap completed: batches=%d, sql_entries=%d, semantic_objects=%d, metrics=%d",
        completed,
        len(source_queries),
        semantic_count,
        metric_count,
    )
    return True, "", result


def init_success_story_semantic_modeling(
    agent_config: "AgentConfig",
    success_story: str,
    subject_tree: Optional[list[str]] = None,
    emit: Optional[BatchEventEmitter] = None,
    *,
    build_mode: str = "overwrite",
    batch_size: int = DEFAULT_SEMANTIC_MODELING_BATCH_SIZE,
    authoring_scope: SemanticAuthoringScope = "full",
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Synchronous ``bootstrap-kb`` wrapper."""
    with suppress_keyboard_input():
        return asyncio.run(
            init_success_story_semantic_modeling_async(
                agent_config,
                success_story,
                subject_tree,
                emit,
                build_mode=build_mode,
                batch_size=batch_size,
                authoring_scope=authoring_scope,
            )
        )


__all__ = [
    "DEFAULT_SEMANTIC_MODELING_BATCH_SIZE",
    "SemanticAuthoringScope",
    "init_success_story_semantic_modeling",
    "init_success_story_semantic_modeling_async",
]
