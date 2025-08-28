from typing import Any, Dict, List

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.storage.sql_history.init_utils import exists_sql_history, gen_sql_history_id
from datus.storage.sql_history.sql_file_processor import process_sql_files
from datus.storage.sql_history.store import SqlHistoryRAG
from datus.tools.llms_tools import LLMTool
from datus.tools.llms_tools.analyze_sql_history import (
    classify_items_batch,
    extract_summaries_batch,
    generate_classification_taxonomy,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def analyze_sql_history(
    llm_tool: LLMTool, items: List[Dict[str, Any]], pool_size: int = 4, existing_taxonomy: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Analyze SQL history items using LLM interaction process.

    Args:
        llm_tool: Initialized LLM tool
        items: List of dict objects containing sql, comment, filepath fields
        pool_size: Number of threads for parallel processing
        existing_taxonomy: Optional existing taxonomy for incremental updates

    Returns:
        List of enriched dict objects with additional summary, domain, layer1, layer2, tags, id fields
    """
    logger.info(f"Starting analysis for {len(items)} SQL items")

    # Step 1: Extract summaries in parallel
    logger.info("Step 1: Extracting summaries...")
    items_with_summaries = extract_summaries_batch(llm_tool, items, pool_size)
    for item in items_with_summaries:
        logger.debug(f"Item with id: {item['id']}")
        logger.debug(f"Item with comment: {item['comment']}")
        logger.debug(f"Item with summary: {item['summary']}")

    # Step 2: Generate classification taxonomy
    logger.info("Step 2: Generating classification taxonomy...")
    taxonomy = generate_classification_taxonomy(llm_tool, items_with_summaries, existing_taxonomy)

    for domain in taxonomy.get("domains", []):
        logger.debug(f"Domain: {domain}")
    for layer1 in taxonomy.get("layer1_categories", []):
        logger.debug(f"Layer1: {layer1}")
    for layer2 in taxonomy.get("layer2_categories", []):
        logger.debug(f"Layer2: {layer2}")
    for tag in taxonomy.get("common_tags", []):
        logger.debug(f"Tag: {tag}")

    # Step 3: Classify each item based on taxonomy
    logger.info("Step 3: Classifying SQL items...")
    classified_items = classify_items_batch(llm_tool, items_with_summaries, taxonomy, pool_size)

    logger.info(f"Analysis completed for {len(classified_items)} items")
    return classified_items


def init_sql_history(
    storage: SqlHistoryRAG,
    args: Any,
    global_config: AgentConfig,
    build_mode: str = "overwrite",
    pool_size: int = 1,
) -> Dict[str, Any]:
    """Initialize SQL history from SQL files directory.

    Args:
        storage: SqlHistoryRAG instance
        args: Command line arguments containing sql_dir path
        global_config: Global agent configuration for LLM model creation
        build_mode: "overwrite" to replace all data, "incremental" to add new entries
        pool_size: Number of threads for parallel processing

    Returns:
        Dict containing initialization results and statistics
    """
    if not hasattr(args, "sql_dir") or not args.sql_dir:
        logger.warning("No --sql_dir provided, SQL history storage initialized but empty")
        return {
            "status": "success",
            "message": "sql_history storage initialized (empty - no --sql_dir provided)",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": 0,
            "total_stored_entries": storage.get_sql_history_size(),
        }

    logger.info(f"Processing SQL files from directory: {args.sql_dir}")

    # Process and validate SQL files
    valid_items, invalid_items = process_sql_files(args.sql_dir)

    if not valid_items:
        logger.info("No valid SQL items found to process")
        return {
            "status": "success",
            "message": f"sql_history bootstrap completed ({build_mode} mode) - no valid items",
            "valid_entries": 0,
            "processed_entries": 0,
            "invalid_entries": len(invalid_items) if invalid_items else 0,
            "total_stored_entries": storage.get_sql_history_size(),
        }

    # Filter out existing items in incremental mode
    if build_mode == "incremental":
        # Check for existing entries
        existing_ids = exists_sql_history(storage, build_mode)

        new_items = []
        for item_dict in valid_items:
            item_id = gen_sql_history_id(item_dict["sql"], item_dict["comment"])
            if item_id not in existing_ids:
                new_items.append(item_dict)

        logger.info(f"Incremental mode: found {len(valid_items)} items, " f"{len(new_items)} new items to process")
        items_to_process = new_items
    else:
        items_to_process = valid_items

    processed_count = 0
    if items_to_process:
        # Analyze with LLM using parallel processing
        model = LLMBaseModel.create_model(global_config)
        llm_tool = LLMTool(model=model)

        # Get existing taxonomy for incremental updates
        existing_taxonomy = None
        if build_mode == "incremental":
            existing_taxonomy = storage.sql_history_storage.get_existing_taxonomy()

        enriched_items = analyze_sql_history(llm_tool, items_to_process, pool_size, existing_taxonomy)

        # enriched_items are already dict format, can store directly
        storage.store_batch(enriched_items)

        processed_count = len(enriched_items)
        logger.info(f"Stored {processed_count} SQL history entries")
    else:
        logger.info("No new items to process in incremental mode")

    # Initialize indices
    storage.after_init()

    return {
        "status": "success",
        "message": f"sql_history bootstrap completed ({build_mode} mode)",
        "valid_entries": len(valid_items) if valid_items else 0,
        "processed_entries": processed_count,
        "invalid_entries": len(invalid_items) if invalid_items else 0,
        "total_stored_entries": storage.get_sql_history_size(),
    }
