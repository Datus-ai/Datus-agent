# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
from typing import Any, Dict, Optional

from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from datus.storage.reference_sql.init_utils import exists_reference_sql, gen_reference_sql_id
from datus.storage.reference_sql.sql_file_processor import process_sql_files
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import normalize_sql

logger = get_logger(__name__)

BIZ_NAME = "reference_sql_init"


def _action_status_value(action: Any) -> Optional[str]:
    status = getattr(action, "status", None)
    if status is None:
        return None
    return status.value if hasattr(status, "value") else str(status)


async def process_sql_item(
    item: dict,
    agent_config: AgentConfig,
    build_mode: str = "incremental",
    subject_tree: Optional[list] = None,
    event_helper: Optional[BatchEventHelper] = None,
    sql_id: Optional[str] = None,
) -> Optional[str]:
    """
    Process a single SQL item using SqlSummaryAgenticNode in workflow mode.

    Args:
        item: Dict containing sql, comment, summary, filepath fields
        agent_config: Agent configuration
        build_mode: "overwrite" or "incremental" - controls whether to skip existing entries
        subject_tree: Optional predefined subject tree categories
        event_helper: Optional BatchEventHelper to stream progress events
        sql_id: Optional precomputed SQL identifier

    Returns:
        SQL summary file path if successful, None otherwise
    """
    logger.debug(f"Processing SQL item: {item.get('filepath', '')}, {item.get('sql', '')}, {item.get('comment', '')}")
    sql_id = sql_id or gen_reference_sql_id(item.get("sql", ""), item.get("comment", ""))
    filepath = item.get("filepath")

    try:
        # Create input for SqlSummaryAgenticNode
        sql_input = SqlSummaryNodeInput(
            user_message="Analyze and summarize this SQL query",
            sql_query=item.get("sql"),
            comment=item.get("comment", ""),
        )

        # Create SqlSummaryAgenticNode in workflow mode (no user interaction)
        node = SqlSummaryAgenticNode(
            node_name="gen_sql_summary",
            agent_config=agent_config,
            execution_mode="workflow",
            build_mode=build_mode,
            subject_tree=subject_tree,
        )

        action_history_manager = ActionHistoryManager()
        sql_summary_file = None

        # Execute and collect results
        node.input = sql_input
        async for action in node.execute_stream(action_history_manager):
            if event_helper:
                event_helper.item_processing(
                    item_id=sql_id,
                    action_name="gen_sql_summary",
                    status=_action_status_value(action),
                    group_id=filepath,
                    messages=action.messages,
                    output=action.output,
                )
            if action.status == ActionStatus.SUCCESS and action.output:
                output = action.output
                if isinstance(output, dict):
                    sql_summary_file = output.get("sql_summary_file")

        if not sql_summary_file:
            logger.error(
                f"Failed to generate SQL summary for {item.get('filepath', '')},"
                f"sql: {item.get('sql', '')}, comment: {item.get('comment', '')}"
            )
            return None

        logger.info(f"Generated SQL summary: {sql_summary_file}")

        from datus.utils.path_manager import get_path_manager

        file_path = (
            get_path_manager(agent_config.home).sql_summary_path(agent_config.current_namespace) / sql_summary_file
        )
        import yaml

        try:
            # Load YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
                if not item.get("comment"):
                    item["comment"] = doc.get("comment", "")
                if not item.get("name"):
                    item["name"] = doc.get("name", "")
                item["subject_tree"] = doc.get("subject_tree")
        except Exception as e:
            logger.warning(f"Failed to open summary file for {file_path}: {e}")
        return sql_summary_file

    except Exception as e:
        logger.error(f"Error processing SQL item {item.get('filepath', '')}: {e}")
        return None


def init_reference_sql(
    storage: ReferenceSqlRAG,
    global_config: AgentConfig,
    sql_dir: str,
    validate_only: bool = False,
    build_mode: str = "overwrite",
    pool_size: int = 1,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
) -> Dict[str, Any]:
    """Initialize reference SQL from SQL files directory.

    Args:
        storage: ReferenceSqlRAG instance
        sql_dir: The path to the SQL files directory
        validate_only: If true, only validate SQL queries.
        global_config: Global agent configuration for LLM model creation
        build_mode: "overwrite" to replace all data, "incremental" to add new entries
        pool_size: Number of threads for parallel processing
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events

    Returns:
        Dict containing initialization results and statistics
    """
    event_helper = BatchEventHelper(BIZ_NAME, emit)

    if not sql_dir:
        logger.warning("No --sql_dir provided, reference SQL storage initialized but empty")
        return {
            "status": "success",
            "message": "reference_sql storage initialized (empty - no --sql_dir provided)",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": 0,
            "total_stored_entries": storage.get_reference_sql_size(),
        }

    logger.info(f"Processing SQL files from directory: {sql_dir}")

    # Emit task started
    event_helper.task_started(sql_dir=sql_dir)

    # Process and validate SQL files
    valid_items, invalid_items = process_sql_files(sql_dir)
    validate_errors = (
        []
        if not invalid_items
        else [
            f"Failed to validate SQL item at {i['filepath']}:{i['line_number']} with error `{i['error']}`. "
            "Please check that the SQL meets expectations (e.g., valid syntax, tables exist, comment field non-empty)."
            for i in invalid_items
        ]
    )

    # Emit task validated
    event_helper.task_validated(
        total_items=len(valid_items) + len(invalid_items) if invalid_items else len(valid_items),
        valid_items=len(valid_items) if valid_items else 0,
        invalid_items=len(invalid_items) if invalid_items else 0,
    )

    # If validate-only mode, exit after processing files
    if validate_only:
        logger.info(
            f"Validate-only mode: Processed {len(valid_items)} valid items and "
            f"{len(invalid_items) if invalid_items else 0} invalid items"
        )

        return {
            "status": "success",
            "message": "SQL files processing completed (validate-only mode)",
            "valid_entries": len(valid_items) if valid_items else 0,
            "processed_entries": 0,
            "invalid_entries": len(invalid_items) if invalid_items else 0,
            "total_stored_entries": 0,
            "validation_errors": "\n".join(validate_errors),
        }

    if not valid_items:
        logger.info("No valid SQL items found to process")
        return {
            "status": "success",
            "message": f"No valid SQL items found in directory: {sql_dir}. Please ensure the directory is correct.",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": len(invalid_items) if invalid_items else 0,
            "total_stored_entries": storage.get_reference_sql_size(),
            "validation_errors": "\n".join(validate_errors),
        }

    # Filter out existing items in incremental mode
    if build_mode == "incremental":
        # Check for existing entries
        existing_ids = exists_reference_sql(storage, build_mode)

        new_items = []
        for item_dict in valid_items:
            item_id = gen_reference_sql_id(item_dict["sql"], item_dict["comment"])
            if item_id not in existing_ids:
                new_items.append(item_dict)

        logger.info(f"Incremental mode: found {len(valid_items)} items, " f"{len(new_items)} new items to process")
        items_to_process = new_items
    else:
        items_to_process = valid_items

    processed_count = 0
    process_errors = []
    if items_to_process:
        # Emit task processing
        event_helper.task_processing(total_items=len(items_to_process))

        # Use SqlSummaryAgenticNode with parallel processing (unified approach)
        async def process_all():
            semaphore = asyncio.Semaphore(pool_size)
            logger.info(f"Processing {len(items_to_process)} SQL items with concurrency={pool_size}")
            file_counts: Dict[str, int] = {}
            for item in items_to_process:
                file_key = str(item.get("filepath") or "unknown_file")
                file_counts[file_key] = file_counts.get(file_key, 0) + 1
            file_remaining = dict(file_counts)
            file_started: set[str] = set()
            file_lock = asyncio.Lock()

            async def emit_group_started_if_needed(file_key: str, filepath: Optional[str]) -> None:
                async with file_lock:
                    if file_key in file_started:
                        return
                    file_started.add(file_key)
                event_helper.group_started(
                    group_id=file_key,
                    total_items=file_counts.get(file_key),
                    filepath=filepath,
                )

            async def emit_group_completed_if_done(file_key: str, filepath: Optional[str]) -> None:
                async with file_lock:
                    remaining = file_remaining.get(file_key)
                    if remaining is None:
                        return
                    remaining -= 1
                    file_remaining[file_key] = remaining
                    if remaining != 0:
                        return
                event_helper.group_completed(
                    group_id=file_key,
                    total_items=file_counts.get(file_key),
                    filepath=filepath,
                )

            async def process_with_semaphore(item):
                sql_id = gen_reference_sql_id(item.get("sql", ""), item.get("comment", ""))
                filepath = item.get("filepath")
                file_key = str(filepath or "unknown_file")
                async with semaphore:
                    await emit_group_started_if_needed(file_key, filepath)
                    event_helper.item_started(
                        item_id=sql_id,
                        group_id=filepath,
                        sql=item.get("sql"),
                    )
                    error = None
                    result = None
                    try:
                        result = await process_sql_item(
                            item,
                            global_config,
                            build_mode,
                            subject_tree,
                            event_helper=event_helper,
                            sql_id=sql_id,
                        )
                    except Exception as exc:
                        error = exc
                        event_helper.item_failed(
                            item_id=sql_id,
                            error=str(exc),
                            group_id=filepath,
                            exception_type=type(exc).__name__,
                        )
                    if result:
                        event_helper.item_completed(
                            item_id=sql_id,
                            group_id=filepath,
                            sql_summary_file=result,
                        )
                    elif error is None:
                        event_helper.item_failed(
                            item_id=sql_id,
                            error="Failed to generate SQL summary",
                            group_id=filepath,
                        )
                    await emit_group_completed_if_done(file_key, filepath)
                    return item, sql_id, result, error

            # Process all items in parallel
            tasks = [asyncio.create_task(process_with_semaphore(item)) for item in items_to_process]
            _errors = []
            # Count successful results
            success_count = 0
            success_items = []
            for task in asyncio.as_completed(tasks):
                item, _sql_id, result, error = await task
                sql = normalize_sql(item["sql"])
                if error:
                    logger.error(f"SQL processing failed with exception `{error}`. SQL: {sql};")
                    _errors.append(f"SQL processing failed with exception `{str(error)}`. SQL: {sql};")
                elif result:
                    success_items.append(item)
                    success_count += 1

            logger.info(f"Completed processing: {success_count}/{len(items_to_process)} successful")
            return success_count, success_items, _errors

        # Run the async function
        processed_count, process_items, errors = asyncio.run(process_all())
        if errors:
            process_errors.extend(errors)
        logger.info(f"Processed {processed_count} reference SQL entries")
    else:
        logger.info("No new items to process in incremental mode")
        process_items = []

    # Emit task completed
    event_helper.task_completed(
        total_items=len(items_to_process) if items_to_process else 0,
        completed_items=processed_count,
        failed_items=len(process_errors),
    )

    # Initialize indices
    storage.after_init()

    return {
        "status": "success",
        "message": f"reference_sql bootstrap completed ({build_mode} mode)",
        "valid_entries": len(valid_items) if valid_items else 0,
        "processed_entries": processed_count,
        "processed_items": process_items,
        "invalid_entries": len(invalid_items) if invalid_items else 0,
        "total_stored_entries": storage.get_reference_sql_size(),
        "validation_errors": "\n".join(validate_errors) if validate_errors else None,
        "process_errors": "\n".join(process_errors) if process_errors else None,
    }
