# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
from typing import Any, Dict, Optional

from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from datus.storage.reference_sql.init_utils import exists_reference_sql, gen_reference_sql_id
from datus.storage.reference_sql.sql_file_processor import process_sql_files
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def process_sql_item(
    item: dict,
    agent_config: AgentConfig,
    build_mode: str = "incremental",
    subject_tree: Optional[list] = None,
) -> Optional[str]:
    """
    Process a single SQL item using SqlSummaryAgenticNode in workflow mode.

    Args:
        item: Dict containing sql, comment, summary, filepath fields
        agent_config: Agent configuration
        build_mode: "overwrite" or "incremental" - controls whether to skip existing entries
        subject_tree: Optional predefined subject tree categories

    Returns:
        SQL summary file path if successful, None otherwise
    """
    logger.debug(f"Processing SQL item: {item.get('filepath', '')}, {item.get('sql', '')}, {item.get('comment', '')}")

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
        return sql_summary_file

    except Exception as e:
        logger.error(f"Error processing SQL item {item.get('filepath', '')}: {e}")
        return None


def init_reference_sql(
    storage: ReferenceSqlRAG,
    args: Any,
    global_config: AgentConfig,
    build_mode: str = "overwrite",
    pool_size: int = 1,
    subject_tree: Optional[list] = None,
) -> Dict[str, Any]:
    """Initialize reference SQL from SQL files directory.

    Args:
        storage: ReferenceSqlRAG instance
        args: Command line arguments containing sql_dir path
        global_config: Global agent configuration for LLM model creation
        build_mode: "overwrite" to replace all data, "incremental" to add new entries
        pool_size: Number of threads for parallel processing
        subject_tree: Optional predefined subject tree categories

    Returns:
        Dict containing initialization results and statistics
    """
    if not hasattr(args, "sql_dir") or not args.sql_dir:
        logger.warning("No --sql_dir provided, reference SQL storage initialized but empty")
        return {
            "status": "success",
            "message": "reference_sql storage initialized (empty - no --sql_dir provided)",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": 0,
            "total_stored_entries": storage.get_reference_sql_size(),
        }

    logger.info(f"Processing SQL files from directory: {args.sql_dir}")

    # Process and validate SQL files
    valid_items, invalid_items = process_sql_files(args.sql_dir)

    # If validate-only mode, exit after processing files
    if hasattr(args, "validate_only") and args.validate_only:
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
        }

    if not valid_items:
        logger.info("No valid SQL items found to process")
        return {
            "status": "success",
            "message": f"reference_sql bootstrap completed ({build_mode} mode) - no valid items",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": len(invalid_items) if invalid_items else 0,
            "total_stored_entries": storage.get_reference_sql_size(),
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
    if items_to_process:
        # Use SqlSummaryAgenticNode with parallel processing (unified approach)
        async def process_all():
            semaphore = asyncio.Semaphore(pool_size)
            logger.info(f"Processing {len(items_to_process)} SQL items with concurrency={pool_size}")

            async def process_with_semaphore(item):
                async with semaphore:
                    return await process_sql_item(item, global_config, build_mode, subject_tree)

            # Process all items in parallel
            results = await asyncio.gather(
                *[process_with_semaphore(item) for item in items_to_process], return_exceptions=True
            )

            # Count successful results
            success_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Item {i+1} failed with exception: {result}")
                elif result:
                    success_count += 1

            logger.info(f"Completed processing: {success_count}/{len(items_to_process)} successful")
            return success_count

        # Run the async function
        processed_count = asyncio.run(process_all())
        logger.info(f"Processed {processed_count} reference SQL entries")
    else:
        logger.info("No new items to process in incremental mode")

    # Initialize indices
    storage.after_init()

    return {
        "status": "success",
        "message": f"reference_sql bootstrap completed ({build_mode} mode)",
        "valid_entries": len(valid_items) if valid_items else 0,
        "processed_entries": processed_count,
        "invalid_entries": len(invalid_items) if invalid_items else 0,
        "total_stored_entries": storage.get_reference_sql_size(),
    }
