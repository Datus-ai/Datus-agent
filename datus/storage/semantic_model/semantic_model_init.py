# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import json
import os
import tempfile
from typing import Any, Callable, Optional

import pandas as pd
import yaml

from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory
from datus.schemas.batch_events import BatchEvent, BatchStage
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

logger = get_logger(__name__)

METRICFLOW_YAML_UNSUPPORTED_MESSAGE = (
    "MetricFlow semantic YAML (`data_source:` / `metric:` documents) can no longer be imported. "
    "Semantic authoring is Dosi-only: migrate the project to Dosi and re-author the model with semantic_modeling."
)


def reject_non_dosi_semantic_yaml(yaml_file_path: str, agent_config: Optional[AgentConfig]) -> Optional[str]:
    """Return the actionable error when semantic YAML import is unavailable.

    Import is Dosi-only: non-Dosi projects get the query-only migration
    message, and MetricFlow-format documents are rejected explicitly before
    any sync is attempted. Returns ``None`` when the import may proceed.
    """
    from datus.agent.node.semantic_authoring import (
        AUTHORING_FORMAT_OSI,
        QUERY_ONLY_MIGRATION_MESSAGE,
        resolve_authoring_format,
    )

    if resolve_authoring_format(agent_config) != AUTHORING_FORMAT_OSI:
        return QUERY_ONLY_MIGRATION_MESSAGE
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
    except Exception as exc:
        return f"Failed to read semantic YAML file '{yaml_file_path}': {exc}"
    if any(isinstance(doc, dict) and ("data_source" in doc or "metric" in doc) for doc in docs):
        return METRICFLOW_YAML_UNSUPPORTED_MESSAGE
    return None


def _resolve_semantic_yaml_refresh_path(yaml_file_path: str, agent_config: Optional[AgentConfig]) -> Optional[str]:
    raw_path = str(yaml_file_path or "")
    if not raw_path:
        return None

    path_manager = getattr(agent_config, "path_manager", None) if agent_config is not None else None
    subject_dir = getattr(path_manager, "subject_dir", None) if path_manager is not None else None
    if isinstance(subject_dir, (str, os.PathLike)):
        from datus.storage.artifact_path import resolve_kb_sandbox_path

        return resolve_kb_sandbox_path(raw_path, "semantic", os.fspath(subject_dir))
    return os.path.realpath(raw_path)


def _atomic_dump_semantic_yaml(yaml_file_path: str, docs: list) -> None:
    target_dir = os.path.dirname(os.path.realpath(yaml_file_path)) or "."
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(yaml_file_path)}.",
        suffix=".tmp",
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)
        os.replace(temp_path, yaml_file_path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise


async def init_success_story_semantic_model_async(
    agent_config: AgentConfig,
    success_story: str,
    emit: Optional[Callable[[BatchEvent], None]] = None,
    *,
    build_mode: str = "overwrite",
    action_callback: Optional[Callable[["ActionHistory"], None]] = None,
    require_exact_osi_target: bool = False,
) -> tuple[bool, str]:
    """Route the legacy semantic-model bootstrap entry point to datasets-only authoring."""
    del require_exact_osi_target
    from datus.storage.semantic_model.semantic_modeling_init import init_success_story_semantic_modeling_async

    successful, error_message, _ = await init_success_story_semantic_modeling_async(
        agent_config,
        success_story,
        emit=emit,
        build_mode=build_mode,
        action_callback=action_callback,
        authoring_scope="datasets",
    )
    return successful, error_message


def init_success_story_semantic_model(
    agent_config: AgentConfig,
    success_story: str,
    emit: Optional[Callable[[BatchEvent], None]] = None,
    *,
    build_mode: str = "overwrite",
) -> tuple[bool, str]:
    """Sync wrapper for datasets-only semantic modeling bootstrap."""
    with suppress_keyboard_input():
        return asyncio.run(
            init_success_story_semantic_model_async(agent_config, success_story, emit, build_mode=build_mode)
        )


def refresh_success_story_semantic_model_profile(
    agent_config: AgentConfig,
    yaml_file_path: str,
    success_story: str,
    emit: Optional[Callable[[BatchEvent], None]] = None,
    *,
    authoring_format: str = "",
    profile_mode: str = "deep",
    max_tables: int = 8,
    max_columns_per_table: int = 40,
    top_n: int = 8,
    max_profile_seconds: int = 120,
) -> tuple[bool, str, int]:
    """Refresh profile-derived descriptions for an existing semantic YAML file.

    This is the CLI/API orchestration path for ``kb_update_strategy=refresh-profile``:
    it mines historical SQL from the success-story CSV, samples bounded live-data
    profiles through read-only DB tools, updates only generated description
    suffixes in the YAML, and syncs that YAML back to semantic storage.
    """
    if not yaml_file_path:
        return False, "--semantic_yaml is required for semantic_model refresh-profile", 0
    if not success_story:
        return False, "--success_story is required for semantic_model refresh-profile", 0

    resolved_yaml_path = _resolve_semantic_yaml_refresh_path(yaml_file_path, agent_config)
    if not resolved_yaml_path:
        return False, f"Semantic YAML file rejected by sandbox check: {yaml_file_path}", 0
    if not os.path.exists(resolved_yaml_path):
        return False, f"Semantic YAML file not found: {resolved_yaml_path}", 0

    entries, error = _load_success_story_profile_entries(success_story)
    if error:
        return False, error, 0

    try:
        with open(resolved_yaml_path, "r", encoding="utf-8") as f:
            docs = [doc for doc in yaml.safe_load_all(f) if doc is not None]
    except Exception as exc:
        return False, f"Failed to read semantic YAML file '{resolved_yaml_path}': {exc}", 0

    fmt = _infer_semantic_yaml_authoring_format(docs, authoring_format)
    if fmt != "osi":
        return False, METRICFLOW_YAML_UNSUPPORTED_MESSAGE, 0
    tables = _semantic_yaml_profile_tables(docs, fmt)
    if not tables:
        return False, f"No table targets found in semantic YAML file: {resolved_yaml_path}", 0

    current_db_config = agent_config.current_db_config()
    runtime_db_context_getter = getattr(agent_config, "runtime_db_context", None)
    runtime_db_context = runtime_db_context_getter() if callable(runtime_db_context_getter) else {}
    runtime_db_context = runtime_db_context if isinstance(runtime_db_context, dict) else {}
    catalog = runtime_db_context.get("catalog") or runtime_db_context.get("catalog_name") or current_db_config.catalog
    database = (
        runtime_db_context.get("database") or runtime_db_context.get("database_name") or current_db_config.database
    )
    schema_name = (
        runtime_db_context.get("schema")
        or runtime_db_context.get("db_schema")
        or runtime_db_context.get("schema_name")
        or current_db_config.schema
    )

    if emit:
        emit(
            BatchEvent(
                biz_name="semantic_model_profile_refresh",
                stage=BatchStage.TASK_STARTED,
                total_items=len(tables),
                payload={"semantic_yaml": resolved_yaml_path, "tables": tables},
            )
        )

    try:
        from datus.tools.func_tool.database import DBFuncTool
        from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools

        db_tool = DBFuncTool(agent_config=agent_config, sub_agent_name="semantic_modeling", read_only=True)
        discovery_tools = SemanticDiscoveryTools(db_tool=db_tool, enable_semantic_model_profiler=True)
        profile_result = discovery_tools.profile_semantic_model_evidence(
            sql_entries_json=json.dumps(entries, ensure_ascii=False),
            tables=tables,
            catalog=str(catalog or ""),
            database=str(database or ""),
            schema_name=str(schema_name or ""),
            profile_mode=profile_mode,
            max_tables=max_tables,
            max_columns_per_table=max_columns_per_table,
            top_n=top_n,
            max_profile_seconds=max_profile_seconds,
        )
    except Exception as exc:
        error = f"Failed to profile semantic YAML '{resolved_yaml_path}': {exc}"
        logger.exception(error)
        if emit:
            emit(BatchEvent(biz_name="semantic_model_profile_refresh", stage=BatchStage.TASK_FAILED, error=error))
        return False, error, 0

    if not profile_result.success:
        error = profile_result.error or "profile_semantic_model_evidence failed"
        if emit:
            emit(BatchEvent(biz_name="semantic_model_profile_refresh", stage=BatchStage.TASK_FAILED, error=error))
        return False, error, 0

    success, error, changed = refresh_semantic_yaml_profile_descriptions(
        resolved_yaml_path,
        profile_result.result or {},
        authoring_format=fmt,
        agent_config=agent_config,
        sync_to_storage=True,
    )

    if emit:
        emit(
            BatchEvent(
                biz_name="semantic_model_profile_refresh",
                stage=BatchStage.TASK_COMPLETED if success else BatchStage.TASK_FAILED,
                completed_items=len(tables) if success else 0,
                failed_items=0 if success else len(tables),
                error=error or None,
                payload={"semantic_yaml": resolved_yaml_path, "changed_description_count": changed},
            )
        )
    return success, error, changed


def _load_success_story_profile_entries(success_story: str) -> tuple[list[dict[str, str]], str]:
    try:
        df = pd.read_csv(success_story)
    except FileNotFoundError:
        return [], f"Success story CSV file not found: {success_story}"
    except pd.errors.EmptyDataError:
        return [], f"Success story CSV file is empty: {success_story}"
    except Exception as exc:
        return [], f"Failed to read success story CSV file '{success_story}': {exc}"

    if "sql" not in df.columns:
        return [], f"Success story CSV '{success_story}' is missing required column: sql"

    entries = []
    for idx, row in df.iterrows():
        sql = _success_story_cell(row, "sql")
        if not sql:
            continue
        entries.append(
            {
                "name": _success_story_cell(row, "source_context_id")
                or _success_story_cell(row, "name")
                or f"success_story_{idx + 1}",
                "question": _success_story_cell(row, "question"),
                "sql": sql,
            }
        )
    if not entries:
        return [], f"Success story CSV '{success_story}' contains no SQL rows"
    return entries, ""


def _success_story_cell(row: Any, key: str) -> str:
    value = row.get(key)
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _infer_semantic_yaml_authoring_format(docs: list[dict], authoring_format: str = "") -> str:
    fmt = (authoring_format or "").strip().lower()
    if fmt:
        return fmt
    return "metricflow" if any(isinstance(doc, dict) and doc.get("data_source") for doc in docs) else "osi"


def _semantic_yaml_profile_tables(docs: list[dict], authoring_format: str) -> list[str]:
    del authoring_format
    return _dedupe_semantic_yaml_values(
        _osi_dataset_table(dataset) for doc in docs for dataset in _iter_osi_yaml_datasets(doc)
    )


def _iter_osi_yaml_datasets(doc: Any) -> list[dict]:
    if not isinstance(doc, dict):
        return []
    datasets = [dataset for dataset in doc.get("datasets") or [] if isinstance(dataset, dict)]
    semantic_models = doc.get("semantic_model")
    if isinstance(semantic_models, list):
        for semantic_model in semantic_models:
            if isinstance(semantic_model, dict):
                datasets.extend(
                    dataset for dataset in semantic_model.get("datasets") or [] if isinstance(dataset, dict)
                )
    return datasets


def _osi_dataset_table(dataset: dict) -> str:
    source = dataset.get("source")
    if isinstance(source, dict) and source.get("table"):
        return str(source["table"]).strip()
    if isinstance(source, str) and source.strip():
        return source.strip()
    return str(dataset.get("table") or dataset.get("name") or "").strip()


def _dedupe_semantic_yaml_values(values) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return result


def init_semantic_yaml_semantic_model(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Initialize ONLY semantic model (dataset/relationship) projections from a
    Dosi/OSI YAML file, skip metrics.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
    """
    if not os.path.exists(yaml_file_path):
        logger.error(f"Semantic YAML file {yaml_file_path} not found")
        return False, f"Semantic YAML file {yaml_file_path} not found"

    error = reject_non_dosi_semantic_yaml(yaml_file_path, agent_config)
    if error:
        return False, error

    from datus.tools.func_tool.generation_tools import GenerationTools

    try:
        result = GenerationTools(agent_config=agent_config, authoring_format="osi").sync_osi_to_db(
            yaml_file_path,
            include_semantic_objects=True,
            include_metrics=False,
        )
    except Exception as e:
        error_msg = f"Failed to sync semantic YAML file '{yaml_file_path}' to vector store: {e}"
        logger.exception(error_msg)
        return False, error_msg

    if result.get("success"):
        logger.info(f"Successfully synced to vector store: {result.get('message')}")
        return True, ""
    return False, result.get("error", "Unknown error")


def refresh_semantic_yaml_profile_descriptions(
    yaml_file_path: str,
    profile_evidence: dict,
    *,
    authoring_format: str = "",
    agent_config: Optional[AgentConfig] = None,
    sync_to_storage: bool = False,
) -> tuple[bool, str, int]:
    """Refresh generated `Observed profile` description suffixes in a semantic YAML file.

    The caller is responsible for producing ``profile_evidence`` via the read-only
    profiler. This function preserves semantic structure and only updates
    description fields. When ``sync_to_storage`` is true, the updated YAML is
    projected back into the semantic stores after it is written.
    """
    if sync_to_storage and agent_config is None:
        return False, "agent_config is required when sync_to_storage=True", 0

    resolved_yaml_path = _resolve_semantic_yaml_refresh_path(yaml_file_path, agent_config)
    if not resolved_yaml_path:
        return False, f"Semantic YAML file rejected by sandbox check: {yaml_file_path}", 0

    if not os.path.exists(resolved_yaml_path):
        return False, f"Semantic YAML file not found: {resolved_yaml_path}", 0

    try:
        with open(resolved_yaml_path, "r", encoding="utf-8") as f:
            docs = [doc for doc in yaml.safe_load_all(f) if doc is not None]
    except Exception as exc:
        return False, f"Failed to read semantic YAML file '{resolved_yaml_path}': {exc}", 0

    try:
        from datus.storage.semantic_model.profile_description import refresh_osi_yaml_descriptions

        fmt = _infer_semantic_yaml_authoring_format(docs, authoring_format)
        if fmt != "osi":
            return False, METRICFLOW_YAML_UNSUPPORTED_MESSAGE, 0
        changed = refresh_osi_yaml_descriptions(docs, profile_evidence)
    except Exception as exc:
        return False, f"Failed to refresh semantic YAML descriptions: {exc}", 0

    if changed <= 0 and not sync_to_storage:
        return True, "", 0

    if changed > 0:
        try:
            _atomic_dump_semantic_yaml(resolved_yaml_path, docs)
        except Exception as exc:
            return False, f"Failed to write semantic YAML file '{resolved_yaml_path}': {exc}", 0

    if sync_to_storage:
        try:
            from datus.tools.func_tool.generation_tools import GenerationTools

            result = GenerationTools(agent_config=agent_config, authoring_format="osi").sync_osi_to_db(
                resolved_yaml_path,
                include_semantic_objects=True,
                include_metrics=False,
            )
        except Exception as exc:
            sync_error = f"Failed to sync OSI semantic YAML file '{resolved_yaml_path}' to vector store: {exc}"
            logger.exception(sync_error)
            return False, sync_error, changed
        if not result.get("success"):
            return False, result.get("error", ""), changed

    return True, "", changed
