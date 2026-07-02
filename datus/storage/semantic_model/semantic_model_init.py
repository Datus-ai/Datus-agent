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

from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEvent, BatchStage
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

logger = get_logger(__name__)

SEMANTIC_MODEL_RESPONSE_ACTION_TYPE = f"{GenSemanticModelAgenticNode.NODE_NAME}_response"
_VALID_BUILD_MODES = {"check", "overwrite", "incremental"}


def _resolve_semantic_yaml_refresh_path(yaml_file_path: str, agent_config: Optional[AgentConfig]) -> Optional[str]:
    raw_path = str(yaml_file_path or "")
    if not raw_path:
        return None

    path_manager = getattr(agent_config, "path_manager", None) if agent_config is not None else None
    subject_dir = getattr(path_manager, "subject_dir", None) if path_manager is not None else None
    if isinstance(subject_dir, (str, os.PathLike)):
        from datus.cli.generation_hooks import resolve_kb_sandbox_path

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
) -> tuple[bool, str]:
    """
    Async version: Initialize ONLY semantic model from success story CSV using ALL SQL queries.

    IMPORTANT: This function processes the ENTIRE success_story CSV in one go,
    NOT line-by-line. It uses execution_mode="workflow" (not plan mode).

    The gen_semantic_model node will receive all SQL queries from the CSV
    and generate semantic models for all tables found in those queries.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        emit: Optional callback to stream BatchEvent progress events
        build_mode: ``"overwrite"`` (default) wipes the semantic model store
            for the current project before regenerating. ``"incremental"``
            skips generation when all referenced tables already have semantic
            model rows; otherwise it generates and upserts missing coverage.
    """
    # Load and validate CSV file
    csv_path = success_story
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        error_msg = f"Success story CSV file not found: {csv_path}"
        logger.error(error_msg)
        return False, error_msg
    except pd.errors.EmptyDataError:
        error_msg = f"Success story CSV file is empty: {csv_path}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Failed to read success story CSV file '{csv_path}': {e}"
        logger.exception(error_msg)
        return False, error_msg

    # Validate required columns
    required_columns = ["sql", "question"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = (
            f"Success story CSV '{csv_path}' is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
        logger.error(error_msg)
        return False, error_msg

    # Collect all SQL queries and questions
    all_sqls = df["sql"].tolist()
    all_questions = df["question"].tolist()

    # Validate data alignment
    if len(all_sqls) != len(all_questions):
        error_msg = (
            f"Success story CSV '{csv_path}' has mismatched column lengths: "
            f"sql={len(all_sqls)}, question={len(all_questions)}"
        )
        logger.error(error_msg)
        return False, error_msg

    if len(all_sqls) == 0:
        error_msg = f"Success story CSV '{csv_path}' contains no data rows"
        logger.error(error_msg)
        return False, error_msg

    if build_mode not in _VALID_BUILD_MODES:
        error_msg = (
            f"Unsupported semantic model build_mode: {build_mode!r}. "
            f"Expected one of: {', '.join(sorted(_VALID_BUILD_MODES))}"
        )
        logger.error(error_msg)
        return False, error_msg

    semantic_model_rag = None
    table_profile_rag = None
    if build_mode in {"check", "overwrite"}:
        try:
            from datus.storage.semantic_model.store import SemanticModelRAG
            from datus.storage.table_semantic_profile.store import TableSemanticProfileRAG

            semantic_model_rag = SemanticModelRAG(agent_config)
            table_profile_rag = TableSemanticProfileRAG(agent_config)
        except Exception as exc:
            error_msg = f"Failed to initialize semantic model storage for build_mode='{build_mode}': {exc}"
            logger.exception(error_msg)
            return False, error_msg

    if build_mode == "check":
        logger.info(
            "[check] semantic_model rows=%d table_semantic_profile rows=%d; generation skipped",
            semantic_model_rag.get_size(),
            table_profile_rag.get_size(),
        )
        return True, ""

    if build_mode == "overwrite":
        logger.info(
            "[overwrite] Wiping semantic_model rows for datasource '%s' before re-population",
            semantic_model_rag.datasource_id,
        )
        try:
            semantic_model_rag.truncate()
            table_profile_rag.truncate()
        except Exception as exc:
            error_msg = f"Failed to wipe semantic model storage for build_mode='overwrite': {exc}"
            logger.exception(error_msg)
            return False, error_msg

    elif build_mode == "incremental":
        try:
            from datus.storage.semantic_model.auto_create import (
                extract_tables_from_sql_list,
                find_missing_semantic_models,
            )

            referenced_tables = extract_tables_from_sql_list(
                [str(sql) for sql in all_sqls if str(sql).strip()], agent_config
            )
            if referenced_tables:
                missing_tables = find_missing_semantic_models(referenced_tables, agent_config)
                if not missing_tables:
                    logger.info(
                        "[incremental] semantic models already exist for %d referenced table(s); generation skipped",
                        len(referenced_tables),
                    )
                    return True, ""
                logger.info(
                    "[incremental] %d/%d referenced table(s) need semantic model refresh: %s",
                    len(missing_tables),
                    len(referenced_tables),
                    missing_tables,
                )
        except Exception as exc:
            logger.warning("Failed to compute incremental semantic-model coverage; generation will continue: %s", exc)

    # Build comprehensive context from all rows
    context_message = "Generate semantic models for the following SQL queries:\n\n"
    for idx, (sql, question) in enumerate(zip(all_sqls, all_questions), 1):
        context_message += f"Query {idx}:\n"
        context_message += f"Question: {question}\n"
        context_message += f"SQL:\n{sql}\n\n"

    current_db_config = agent_config.current_db_config()
    runtime_db_context_getter = getattr(agent_config, "runtime_db_context", None)
    runtime_db_context = runtime_db_context_getter() if callable(runtime_db_context_getter) else {}
    runtime_db_context = runtime_db_context if isinstance(runtime_db_context, dict) else {}

    # Emit task started event
    if emit:
        emit(BatchEvent(biz_name="semantic_model_init", stage=BatchStage.TASK_STARTED))

    # Create semantic model generation node (workflow mode, NOT plan mode)
    semantic_node = GenSemanticModelAgenticNode(
        agent_config=agent_config,
        execution_mode="workflow",  # CRITICAL: workflow mode only
    )

    semantic_input = SemanticNodeInput(
        user_message=context_message,
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
    )

    action_history_manager = ActionHistoryManager()
    semantic_node.input = semantic_input

    try:
        generated_files = []
        terminal_error = None
        async for action in semantic_node.execute_stream(action_history_manager):
            if action_callback is not None:
                try:
                    action_callback(action)
                except Exception as cb_exc:  # pragma: no cover - defensive
                    logger.debug("semantic_model action_callback raised: %s", cb_exc)
            # Emit streaming messages
            if emit and action.messages:
                emit(
                    BatchEvent(
                        biz_name="semantic_model_init",
                        stage=BatchStage.ITEM_PROCESSING,
                        payload={"messages": action.messages, "output": action.output},
                    )
                )

            action_type = getattr(action, "action_type", "")
            if (
                action.status == ActionStatus.SUCCESS
                and action_type == SEMANTIC_MODEL_RESPONSE_ACTION_TYPE
                and action.output
            ):
                if isinstance(action.output, dict):
                    # Check for semantic_models field (from SemanticNodeResult)
                    if "semantic_models" in action.output:
                        models = action.output["semantic_models"]
                        if isinstance(models, list):
                            generated_files.extend(models)
                        elif models:  # Single file as string
                            generated_files.append(models)
            elif action.status == ActionStatus.FAILED and action_type in {"error", SEMANTIC_MODEL_RESPONSE_ACTION_TYPE}:
                terminal_error = action.messages or "Semantic model generation failed"
                logger.error(terminal_error)
                continue

        if terminal_error:
            if emit:
                emit(BatchEvent(biz_name="semantic_model_init", stage=BatchStage.TASK_FAILED, error=terminal_error))
            return False, terminal_error

        if not generated_files:
            error_msg = f"Failed to generate any semantic models from {len(all_sqls)} SQL queries in '{csv_path}'"
            logger.error(error_msg)
            if emit:
                emit(BatchEvent(biz_name="semantic_model_init", stage=BatchStage.TASK_FAILED, error=error_msg))
            return False, error_msg

        logger.info(f"Generated {len(generated_files)} semantic model files: {generated_files}")
        if emit:
            emit(BatchEvent(biz_name="semantic_model_init", stage=BatchStage.TASK_COMPLETED))
        return True, ""

    except Exception as e:
        error_msg = f"Error generating semantic models from '{csv_path}': {e}"
        logger.exception(error_msg)
        if emit:
            emit(BatchEvent(biz_name="semantic_model_init", stage=BatchStage.TASK_FAILED, error=error_msg))
        return False, error_msg


def init_success_story_semantic_model(
    agent_config: AgentConfig,
    success_story: str,
    emit: Optional[Callable[[BatchEvent], None]] = None,
    *,
    build_mode: str = "overwrite",
) -> tuple[bool, str]:
    """
    Sync wrapper: Initialize ONLY semantic model from success story CSV using ALL SQL queries.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        emit: Optional callback to stream BatchEvent progress events
        build_mode: Forwarded to :func:`init_success_story_semantic_model_async`.
    """
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
    if fmt not in {"metricflow", "osi"}:
        return False, f"Unsupported semantic YAML authoring format: {authoring_format}", 0
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

        db_tool = DBFuncTool(agent_config=agent_config, sub_agent_name="gen_semantic_model", read_only=True)
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
    if authoring_format == "metricflow":
        return _dedupe_semantic_yaml_values(
            _metricflow_data_source_table(doc.get("data_source")) for doc in docs if isinstance(doc, dict)
        )
    return _dedupe_semantic_yaml_values(
        _osi_dataset_table(dataset) for doc in docs for dataset in _iter_osi_yaml_datasets(doc)
    )


def _metricflow_data_source_table(data_source: Any) -> str:
    if not isinstance(data_source, dict):
        return ""
    return str(data_source.get("sql_table") or data_source.get("name") or "").strip()


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
    Initialize ONLY semantic model (table/column/entity) from YAML, skip metrics.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
    """
    if not os.path.exists(yaml_file_path):
        logger.error(f"Semantic YAML file {yaml_file_path} not found")
        return False, f"Semantic YAML file {yaml_file_path} not found"

    return process_semantic_yaml_file(yaml_file_path, agent_config, include_metrics=False)


def process_semantic_yaml_file(
    yaml_file_path: str,
    agent_config: AgentConfig,
    include_semantic_objects: bool = True,
    include_metrics: bool = True,
) -> tuple[bool, str]:
    """
    Process semantic YAML file by directly syncing to vector store using GenerationHooks.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
        include_semantic_objects: Whether to sync tables/columns/entities
        include_metrics: Whether to sync metrics
    Returns:
        - Whether the execution was successful
        - Failed reason

    """
    logger.info(
        f"Processing semantic YAML file: {yaml_file_path} "
        f"(semantic_objects={include_semantic_objects}, metrics={include_metrics})"
    )

    # Validate file exists
    if not os.path.exists(yaml_file_path):
        error_msg = f"Semantic YAML file not found: {yaml_file_path}"
        logger.error(error_msg)
        return False, error_msg

    # Use GenerationHooks static method to sync to DB
    try:
        result = GenerationHooks._sync_semantic_to_db(
            yaml_file_path,
            agent_config,
            include_semantic_objects=include_semantic_objects,
            include_metrics=include_metrics,
        )
    except Exception as e:
        error_msg = f"Failed to sync semantic YAML file '{yaml_file_path}' to vector store: {e}"
        logger.exception(error_msg)
        return False, error_msg

    if result.get("success"):
        logger.info(f"Successfully synced to vector store: {result.get('message')}")
        return True, ""
    else:
        error = result.get("error", "Unknown error")
        error_msg = f"Failed to sync '{yaml_file_path}' to vector store: {error}"
        logger.error(error_msg)
        return False, error


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
        from datus.storage.semantic_model.profile_description import (
            refresh_metricflow_yaml_descriptions,
            refresh_osi_yaml_descriptions,
        )

        fmt = (authoring_format or "").strip().lower()
        if not fmt:
            fmt = "metricflow" if any(isinstance(doc, dict) and doc.get("data_source") for doc in docs) else "osi"
        if fmt == "metricflow":
            changed = refresh_metricflow_yaml_descriptions(docs, profile_evidence)
        elif fmt == "osi":
            changed = refresh_osi_yaml_descriptions(docs, profile_evidence)
        else:
            return False, f"Unsupported semantic YAML authoring format: {authoring_format}", 0
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
        if fmt == "metricflow":
            sync_success, sync_error = process_semantic_yaml_file(
                resolved_yaml_path,
                agent_config,
                include_semantic_objects=True,
                include_metrics=True,
            )
        else:
            try:
                from datus.tools.func_tool.generation_tools import GenerationTools

                result = GenerationTools(agent_config=agent_config, authoring_format="osi").sync_osi_semantic_to_db(
                    resolved_yaml_path
                )
            except Exception as exc:
                sync_error = f"Failed to sync OSI semantic YAML file '{resolved_yaml_path}' to vector store: {exc}"
                logger.exception(sync_error)
                return False, sync_error, changed
            sync_success = bool(result.get("success"))
            sync_error = result.get("error", "")
        if not sync_success:
            return False, sync_error, changed

    return True, "", changed
