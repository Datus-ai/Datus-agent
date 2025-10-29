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


def parse_subject_tree(subject_tree_str: str) -> Dict[str, Any]:
    """
    Parse subject_tree string into taxonomy structure.

    Args:
        subject_tree_str: Comma-separated string of domain/layer1/layer2 paths

    Returns:
        Dict containing taxonomy with domains, layer1_categories, layer2_categories
    """
    if not subject_tree_str:
        return {}

    domains_set = set()
    layer1_set = set()
    layer2_dict = {}

    paths = [p.strip() for p in subject_tree_str.split(",")]

    for path in paths:
        parts = path.split("/")
        if len(parts) != 3:
            logger.warning(f"Skipping invalid subject_tree path (expected domain/layer1/layer2): {path}")
            continue

        domain, layer1, layer2 = parts
        domains_set.add(domain)
        layer1_set.add((layer1, domain))
        layer2_dict[layer2] = layer1

    taxonomy = {
        "domains": [
            {"name": domain, "description": f"{domain} domain", "examples": []} for domain in sorted(domains_set)
        ],
        "layer1_categories": [
            {"name": layer1, "description": f"{layer1} category", "domain": domain, "examples": []}
            for layer1, domain in sorted(layer1_set)
        ],
        "layer2_categories": [
            {"name": layer2, "description": f"{layer2} subcategory", "layer1": layer1, "examples": []}
            for layer2, layer1 in sorted(layer2_dict.items())
        ],
        "common_tags": [],
    }

    logger.info(
        f"Parsed subject_tree into taxonomy: {len(taxonomy['domains'])} domains, "
        f"{len(taxonomy['layer1_categories'])} layer1, {len(taxonomy['layer2_categories'])} layer2"
    )

    return taxonomy


async def process_sql_item(
    item: dict,
    agent_config: AgentConfig,
    build_mode: str = "incremental",
) -> Optional[str]:
    """
    Process a single SQL item using SqlSummaryAgenticNode in workflow mode.

    Args:
        item: Dict containing sql, comment, summary, filepath fields
        agent_config: Agent configuration
        build_mode: "overwrite" or "incremental" - controls whether to skip existing entries

    Returns:
        SQL summary file path if successful, None otherwise
    """
    logger.debug(f"Processing SQL item: {item.get('filepath', '')}")

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
        )

        action_history_manager = ActionHistoryManager()
        sql_summary_file = None

        # Execute and collect results
        async for action in node.execute_stream(sql_input, action_history_manager):
            if action.status == ActionStatus.SUCCESS and action.output:
                output = action.output
                if isinstance(output, dict):
                    sql_summary_file = output.get("sql_summary_file")

        if not sql_summary_file:
            logger.error(f"Failed to generate SQL summary for {item.get('filepath', '')}")
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
) -> Dict[str, Any]:
    """Initialize reference SQL from SQL files directory.

    Args:
        storage: ReferenceSqlRAG instance
        args: Command line arguments containing sql_dir path
        global_config: Global agent configuration for LLM model creation
        build_mode: "overwrite" to replace all data, "incremental" to add new entries
        pool_size: Number of threads for parallel processing

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
                    return await process_sql_item(item, global_config, build_mode)

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
