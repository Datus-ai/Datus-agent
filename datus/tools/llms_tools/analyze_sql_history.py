from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from datus.prompts.prompt_manager import prompt_manager
from datus.storage.sql_history.init_utils import gen_sql_history_id
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def analyze_sql_history_three_step(
    llm_tool: LLMTool, items: List[Dict[str, Any]], pool_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Analyze SQL history items using three-step LLM interaction process.

    Args:
        llm_tool: Initialized LLM tool
        items: List of dict objects containing sql, comment, filepath fields
        pool_size: Number of threads for parallel processing

    Returns:
        List of enriched dict objects with additional summary, domain, layer1, layer2, tags, id fields
    """
    logger.info(f"Starting three-step analysis for {len(items)} SQL items")

    # Step 1: Extract summaries in parallel
    logger.info("Step 1: Extracting summaries...")
    items_with_summaries = extract_summaries_batch(llm_tool, items, pool_size)

    # Step 2: Generate classification taxonomy
    logger.info("Step 2: Generating classification taxonomy...")
    taxonomy = generate_classification_taxonomy(llm_tool, items_with_summaries)

    # Step 3: Classify each item based on taxonomy
    logger.info("Step 3: Classifying SQL items...")
    classified_items = classify_items_batch(llm_tool, items_with_summaries, taxonomy, pool_size)

    logger.info(f"Three-step analysis completed for {len(classified_items)} items")
    return classified_items


def extract_summaries_batch(llm_tool: LLMTool, items: List[Dict[str, Any]], pool_size: int = 4) -> List[Dict[str, Any]]:
    """
    Extract summaries for SQL items using parallel processing.

    Args:
        llm_tool: Initialized LLM tool
        items: List of dict objects containing sql, comment, filepath fields
        pool_size: Number of threads for parallel processing

    Returns:
        List of dict objects with additional summary and id fields
    """
    logger.info(f"Extracting summaries for {len(items)} SQL items using {pool_size} threads")

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [
            executor.submit(extract_single_summary, llm_tool, item, index) for index, item in enumerate(items, 1)
        ]

        items_with_summaries = []
        processed_count = 0
        error_count = 0

        for future in as_completed(futures):
            try:
                result_item = future.result()
                items_with_summaries.append(result_item)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error extracting summary: {str(e)}")
                error_count += 1

    logger.info(f"Summary extraction completed - Processed: {processed_count}, Errors: {error_count}")
    return items_with_summaries


def extract_single_summary(llm_tool: LLMTool, item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Extract summary for a single SQL history item.

    Args:
        llm_tool: Initialized LLM tool
        item: Dict object containing sql, comment, filepath fields
        index: Item index for logging

    Returns:
        Dict object with additional summary and id fields
    """
    logger.debug(f"Extracting summary for item {index}: {item.get('filepath', '')}")

    try:
        prompt = prompt_manager.render_template(
            "extract_sql_summary",
            version="1.0",
            comment=item.get("comment", ""),
            sql=item.get("sql", ""),
        )
        logger.info(f"Prompt of extract_single_summary: {prompt}")

        parsed_data = llm_tool.model.generate_with_json_output(prompt)
        logger.info(f"Parsed data of extract_single_summary: {parsed_data}")
        item["summary"] = parsed_data.get("summary", item.get("comment", ""))
        item["id"] = gen_sql_history_id(item.get("sql", ""), item.get("comment", ""))

        logger.debug(f"Item {index}: Successfully extracted summary")
        return item

    except Exception as e:
        logger.error(f"Item {index}: Failed to extract summary: {str(e)}")
        item["summary"] = item.get("comment", "") if item.get("comment") else "Unable to analyze SQL"
        item["id"] = gen_sql_history_id(item.get("sql", ""), item.get("comment", ""))
        return item


def generate_classification_taxonomy(llm_tool: LLMTool, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate classification taxonomy based on all SQL summaries and comments.

    Args:
        llm_tool: Initialized LLM tool
        items: List of dict objects with summary and comment fields

    Returns:
        Dict containing the generated taxonomy
    """
    logger.info("Generating classification taxonomy based on all SQL items")

    try:
        prompt = prompt_manager.render_template(
            "generate_sql_taxonomy",
            version="1.0",
            sql_items=items,
        )
        logger.info(f"Prompt of generate_classification_taxonomy: {prompt}")

        taxonomy = llm_tool.model.generate_with_json_output(prompt)
        logger.info(f"Parsed data of generate_classification_taxonomy: {taxonomy}")

        logger.info("Generated taxonomy:")

        # Display domains with their layer1 and layer2 categories
        for domain in taxonomy.get("domains", []):
            logger.info(f"  {domain.get('name', '')}: {domain.get('description', '')}")

            # Display layer1 categories under this domain
            for layer1 in taxonomy.get("layer1_categories", []):
                if layer1.get("domain", "") == domain.get("name", ""):
                    logger.info(f"    {layer1.get('name', '')}: {layer1.get('description', '')}")

                    # Display layer2 categories under this layer1
                    for layer2 in taxonomy.get("layer2_categories", []):
                        if layer2.get("layer1", "") == layer1.get("name", ""):
                            logger.info(f"        {layer2.get('name', '')}: {layer2.get('description', '')}")

        # Display tags
        tags = [f"{t.get('tag', '')}: {t.get('description', '')}" for t in taxonomy.get("common_tags", [])]
        logger.info(f"  Tags ({len(tags)}):")
        for tag in tags:
            logger.info(f"    {tag}")

        return taxonomy

    except Exception as e:
        logger.error(f"Failed to generate taxonomy: {str(e)}")
        return {
            "domains": [{"name": "general", "description": "General queries", "examples": []}],
            "layer1_categories": [
                {"name": "data_query", "description": "Data queries", "domain": "general", "examples": []}
            ],
            "layer2_categories": [
                {"name": "basic_select", "description": "Basic select queries", "layer1": "data_query", "examples": []}
            ],
            "common_tags": [{"tag": "query", "description": "General query tag", "examples": []}],
        }


def classify_items_batch(
    llm_tool: LLMTool, items: List[Dict[str, Any]], taxonomy: Dict[str, Any], pool_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Classify SQL items based on generated taxonomy using parallel processing.

    Args:
        llm_tool: Initialized LLM tool
        items: List of dict objects with summary and comment fields
        taxonomy: Generated classification taxonomy
        pool_size: Number of threads for parallel processing

    Returns:
        List of fully classified dict objects
    """
    logger.info(f"Classifying {len(items)} SQL items using {pool_size} threads")

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [
            executor.submit(classify_single_item, llm_tool, item, taxonomy, index)
            for index, item in enumerate(items, 1)
        ]

        classified_items = []
        processed_count = 0
        error_count = 0

        for future in as_completed(futures):
            try:
                result_item = future.result()
                classified_items.append(result_item)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error classifying SQL item: {str(e)}")
                error_count += 1

    logger.info(f"Classification completed - Processed: {processed_count}, Errors: {error_count}")
    return classified_items


def classify_single_item(
    llm_tool: LLMTool, item: Dict[str, Any], taxonomy: Dict[str, Any], index: int
) -> Dict[str, Any]:
    """
    Classify a single SQL item based on taxonomy.

    Args:
        llm_tool: Initialized LLM tool
        item: Dict object with summary and comment fields
        taxonomy: Generated classification taxonomy
        index: Item index for logging

    Returns:
        Fully classified dict object
    """
    logger.debug(f"Classifying item {index}: {item.get('filepath', '')}")

    try:
        prompt = prompt_manager.render_template(
            "classify_sql_item",
            version="1.0",
            comment=item.get("comment", ""),
            summary=item.get("summary", ""),
            taxonomy=taxonomy,
        )
        logger.info(f"Prompt of classify_single_item: {prompt}")

        parsed_data = llm_tool.model.generate_with_json_output(prompt)
        logger.info(f"Parsed data of classify_single_item: {parsed_data}")

        item["domain"] = parsed_data.get("domain", "")
        item["layer1"] = parsed_data.get("layer1", "")
        item["layer2"] = parsed_data.get("layer2", "")
        item["tags"] = parsed_data.get("tags", "")

        logger.debug(f"Item {index}: Successfully classified")
        return item

    except Exception as e:
        logger.error(f"Item {index}: Failed to classify: {str(e)}")
        item["domain"] = ""
        item["layer1"] = ""
        item["layer2"] = ""
        item["tags"] = ""
        return item


def analyze_sql_history_batch(
    llm_tool: LLMTool, items: List[Dict[str, Any]], pool_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Legacy function - now redirects to three-step analysis.

    Args:
        llm_tool: Initialized LLM tool
        items: List of dict objects containing sql, comment, filepath fields
        pool_size: Number of threads for parallel processing

    Returns:
        List of enriched dict objects with additional summary, domain, layer1, layer2, tags, id fields
    """
    logger.info("Using legacy interface - redirecting to three-step analysis")
    return analyze_sql_history_three_step(llm_tool, items, pool_size)


def analyze_single_item(llm_tool: LLMTool, item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Legacy function - analyze a single SQL history item using old method.

    Args:
        llm_tool: Initialized LLM tool
        item: Dict object containing sql, comment, filepath fields
        index: Item index for logging

    Returns:
        Enriched dict object with additional summary, domain, layer1, layer2, tags, id fields
    """
    logger.debug(f"Analyzing item {index}: {item.get('filepath', '')}")

    try:
        prompt = prompt_manager.render_template(
            "analyze_sql_history",
            version="1.0",
            comment=item.get("comment", ""),
            sql=item.get("sql", ""),
        )

        parsed_data = llm_tool.model.generate_with_json_output(prompt)

        item["summary"] = parsed_data.get("summary", item.get("comment", ""))
        item["domain"] = parsed_data.get("domain", "")
        item["layer1"] = parsed_data.get("layer1", "")
        item["layer2"] = parsed_data.get("layer2", "")
        item["tags"] = parsed_data.get("tags", "")

        item["id"] = gen_sql_history_id(item.get("sql", ""), item.get("comment", ""))

        logger.debug(f"Item {index}: Successfully analyzed SQL from {item.get('filepath', '')}")
        return item

    except Exception as e:
        logger.error(f"Item {index}: Failed to analyze SQL item: {str(e)}")
        item["summary"] = item.get("comment", "") if item.get("comment") else "Unable to analyze SQL"
        item["id"] = gen_sql_history_id(item.get("sql", ""), item.get("comment", ""))
        item["domain"] = ""
        item["layer1"] = ""
        item["layer2"] = ""
        item["tags"] = ""
        return item
