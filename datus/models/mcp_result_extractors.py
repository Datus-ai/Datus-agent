# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List, Set

from agents.result import RunResultBase

from datus.schemas.node_models import SQLContext
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Mapping of mcp server types to their supported query function names
DB_QUERY_FUNCTIONS: Dict[str, Set[str]] = {
    DBType.SNOWFLAKE: {"read_query", "list_tables", "describe_table"},
    DBType.SQLITE: {"read_query", "write_query", "list_tables", "describe_table"},
    DBType.STARROCKS: {"read_query", "write_query", "table_overview", "db_overview"},
    DBType.DUCKDB: {"query"},
}


def get_function_call_names(db_type: str) -> Set[str]:
    """
    Get the set of query function names for a specific database type.
    Returns an empty set if the database type is not supported.
    """
    return DB_QUERY_FUNCTIONS.get(db_type, set())


def extract_sql_contexts(result: RunResultBase, db_type: str = DBType.SNOWFLAKE) -> List[SQLContext]:
    """
    Extract SQL contexts from the result

    Args:
        result: The run result to extract SQL contexts from
        db_type: The type of database being used (default: "snowflake")
    """
    sql_contexts = []
    valid_query_functions = get_function_call_names(db_type)

    if hasattr(result, "to_input_list"):
        input_list = result.to_input_list()

        # Iterate through the input list to find query function calls and their outputs
        for i, item in enumerate(input_list):
            logger.debug(f"Processing item {i}: type={item.get('type')}, name={item.get('name')}")

            if item.get("type") == "function_call" and item.get("name") in valid_query_functions:
                logger.debug(f"Found valid function call: {item.get('name')} with call_id: {item.get('call_id')}")

                # Get the SQL query from arguments
                call_id = item.get("call_id")
                try:
                    # import json
                    # query = json.loads(item.get("arguments", "{}")).get("query", "")
                    function_call_name = item.get("name", "")
                    arguments = item.get("arguments", "{}")
                    logger.debug(f"Function call arguments: {arguments}")

                    # Find the corresponding output
                    output = None
                    reflection = None
                    for j in range(i + 1, len(input_list)):
                        logger.debug(
                            f"Looking for output at position {j}: type={input_list[j].get('type')},"
                            f" call_id={input_list[j].get('call_id')}"
                        )

                        if (
                            input_list[j].get("type") == "function_call_output"
                            and input_list[j].get("call_id") == call_id
                        ):
                            output = input_list[j].get("output", "")
                            logger.debug(f"Found function call output: {output[:100]}...")  # Log first 100 chars

                            # Check if there's a reflection message after the output
                            if j + 1 < len(input_list):
                                next_item = input_list[j + 1]
                                logger.debug(
                                    f"Next item after output: type={next_item.get('type')},"
                                    f" role={next_item.get('role')}"
                                )

                                # if met another fc, it means AI just do function calling without thinking
                                if next_item.get("type") == "function_call":
                                    logger.debug("Found another function call, breaking reflection search")
                                    break
                                # find the reflection text in the assistant message from the continues message
                                if (
                                    next_item.get("type") == "message"
                                    and next_item.get("role") == "assistant"
                                    and next_item.get("content")
                                ):
                                    logger.debug("Found assistant message, extracting reflection")
                                    # Extract reflection text from content
                                    content = next_item.get("content", [])
                                    logger.debug(f"Content type: {type(content)}, content: {content}")

                                    if isinstance(content, list):
                                        for content_item in content:
                                            if content_item.get("text"):
                                                reflection = content_item.get("text")
                                                logger.debug(
                                                    f"Extracted reflection: {reflection[:100]}..."
                                                )  # Log first 100 chars
                                                break
                                    break
                            break

                    # Create SQLContext and add to list
                    logger.debug(
                        f"Creating SQLContext with arguments: {arguments}, output: {output}, reflection: {reflection}"
                    )
                    sql_context = SQLContext(
                        sql_query=f"{function_call_name}:{arguments}",
                        sql_return=output,
                        row_count=None,
                        reflection_explanation=reflection,
                    )
                    sql_contexts.append(sql_context)
                except Exception as e:
                    logger.error(f"Error parsing SQL interaction: {str(e)}")

    return sql_contexts
