from typing import Any, Dict

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.storage.sql_history.init_utils import exists_sql_history, gen_sql_history_id
from datus.storage.sql_history.sql_file_processor import process_sql_files
from datus.storage.sql_history.store import SqlHistoryRAG
from datus.tools.llms_tools import LLMTool
from datus.tools.llms_tools.analyze_sql_history import analyze_sql_history_batch
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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
        enriched_items = analyze_sql_history_batch(llm_tool, items_to_process, pool_size)

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
