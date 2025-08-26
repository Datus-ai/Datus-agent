from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from datus.prompts.prompt_manager import prompt_manager
from datus.storage.sql_history.init_utils import gen_sql_history_id
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def analyze_sql_history_batch(
    llm_tool: LLMTool, items: List[Dict[str, Any]], pool_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Analyze a batch of SQL history items using parallel processing.

    Args:
        llm_tool: Initialized LLM tool
        items: List of dict objects containing sql, comment, filepath fields
        pool_size: Number of threads for parallel processing

    Returns:
        List of enriched dict objects with additional summary, domain, layer1, layer2, tags, id fields
    """
    logger.info(f"Analyzing {len(items)} SQL items with LLM using {pool_size} threads")

    # Process items in parallel
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [executor.submit(analyze_single_item, llm_tool, item, index) for index, item in enumerate(items, 1)]

        enriched_items = []
        processed_count = 0
        error_count = 0

        for future in as_completed(futures):
            try:
                result_item = future.result()
                enriched_items.append(result_item)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing SQL item: {str(e)}")
                error_count += 1

    logger.info(f"Completed analysis - Processed: {processed_count}, Errors: {error_count}")
    return enriched_items


def analyze_single_item(llm_tool: LLMTool, item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Analyze a single SQL history item.

    Args:
        llm_tool: Initialized LLM tool
        item: Dict object containing sql, comment, filepath fields
        index: Item index for logging

    Returns:
        Enriched dict object with additional summary, domain, layer1, layer2, tags, id fields
    """
    logger.debug(f"Analyzing item {index}: {item.get('filepath', '')}")

    try:
        # Create prompt for LLM analysis using template
        prompt = prompt_manager.render_template(
            "analyze_sql_history",
            version="1.0",
            comment=item.get("comment", ""),
            sql=item.get("sql", ""),
        )

        # Get LLM response with JSON output
        parsed_data = llm_tool.model.generate_with_json_output(prompt)

        # Enrich item with parsed data
        item["summary"] = parsed_data.get("summary", item.get("comment", ""))
        item["domain"] = parsed_data.get("domain", "")
        item["layer1"] = parsed_data.get("layer1", "")
        item["layer2"] = parsed_data.get("layer2", "")
        item["tags"] = parsed_data.get("tags", "")

        # Generate ID from sql and comment
        item["id"] = gen_sql_history_id(item.get("sql", ""), item.get("comment", ""))

        logger.debug(f"Item {index}: Successfully analyzed SQL from {item.get('filepath', '')}")
        return item

    except Exception as e:
        logger.error(f"Item {index}: Failed to analyze SQL item: {str(e)}")
        # Return item with default values to avoid losing data
        item["summary"] = item.get("comment", "") if item.get("comment") else "Unable to analyze SQL"
        item["id"] = gen_sql_history_id(item.get("sql", ""), item.get("comment", ""))
        item["domain"] = ""
        item["layer1"] = ""
        item["layer2"] = ""
        item["tags"] = ""
        return item
