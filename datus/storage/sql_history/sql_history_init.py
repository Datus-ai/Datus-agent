from typing import Any, Dict, List

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
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


def ensure_unique_names(items: List[Dict[str, Any]], llm_tool: LLMTool = None) -> List[Dict[str, Any]]:
    """
    Ensure all SQL items have unique names using a hybrid approach:
    1. First detect duplicates
    2. For duplicates, try LLM regeneration (if llm_tool provided)
    3. Fallback to suffix approach for any remaining duplicates

    Args:
        items: List of dict objects with name field
        llm_tool: Optional LLM tool for intelligent name regeneration

    Returns:
        List of dict objects with unique names
    """
    # Step 1: Find duplicates
    name_to_items = {}
    for item in items:
        name = item.get("name", "")
        if not name:
            continue
        if name not in name_to_items:
            name_to_items[name] = []
        name_to_items[name].append(item)

    # Step 2: Handle duplicates
    duplicates_found = 0
    regeneration_success = 0

    for name, item_list in name_to_items.items():
        if len(item_list) > 1:
            duplicates_found += len(item_list) - 1
            logger.info(f"Found {len(item_list)} items with duplicate name: '{name}'")

            # Keep first item with original name, regenerate others
            for i, item in enumerate(item_list[1:], 1):
                new_name = None

                # Try LLM regeneration first
                if llm_tool:
                    try:
                        # Collect all existing names to avoid
                        existing_names = [other_name for other_name in name_to_items.keys()]
                        existing_names.extend(
                            [
                                other_item.get("name", "")
                                for other_items in name_to_items.values()
                                for other_item in other_items
                                if other_item.get("name")
                            ]
                        )
                        conflicting_names = list(set(existing_names))

                        new_name = regenerate_unique_name(llm_tool, item, conflicting_names)
                        if new_name and new_name not in conflicting_names:
                            item["name"] = new_name
                            regeneration_success += 1
                            logger.debug(f"Regenerated name: '{name}' -> '{new_name}'")
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to regenerate name for duplicate '{name}': {e}")

                # Fallback: append suffix
                item["name"] = f"{name}_{i+1}"
                logger.debug(f"Used suffix approach: '{name}' -> '{item['name']}'")

    logger.info(
        f"Handled {duplicates_found} duplicate names - {regeneration_success} regenerated, "
        f"{duplicates_found - regeneration_success} used suffixes"
    )
    return items


def regenerate_unique_name(llm_tool: LLMTool, item: Dict[str, Any], conflicting_names: List[str]) -> str:
    """
    Use LLM to regenerate a unique name for an SQL item.

    Args:
        llm_tool: Initialized LLM tool
        item: Dict object with sql, comment fields
        conflicting_names: List of names to avoid

    Returns:
        New unique name string
    """
    try:
        prompt = prompt_manager.render_template(
            "regenerate_sql_name",
            version="1.0",
            comment=item.get("comment", ""),
            sql=item.get("sql", ""),
            current_name=item.get("name", ""),
            conflicting_names=conflicting_names,
        )
        logger.debug(f"Regeneration prompt: {prompt}")

        parsed_data = llm_tool.model.generate_with_json_output(prompt)
        new_name = parsed_data.get("name", "")

        # Validate the new name
        if new_name and len(new_name) <= 20 and new_name not in conflicting_names:
            return new_name
        else:
            logger.warning(f"Invalid regenerated name: '{new_name}' (length: {len(new_name) if new_name else 0})")
            return ""

    except Exception as e:
        logger.error(f"Error regenerating name: {e}")
        return ""


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

    # Step 1.5: Ensure unique names
    logger.info("Step 1.5: Ensuring unique SQL names...")
    items_with_summaries = ensure_unique_names(items_with_summaries, llm_tool)

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
