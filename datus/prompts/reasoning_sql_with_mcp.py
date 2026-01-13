# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List

from pydantic import BaseModel

from datus.schemas.node_models import TableSchema, TableValue
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from .prompt_manager import prompt_manager

logger = get_logger(__name__)


# class for the output of the reasoning sql with mcp
class ReasoningSQLResponse(BaseModel):
    sql: str
    result: str
    explanation: str


def get_reasoning_system_prompt() -> str:
    """
    Return the system-level instructions for the MCP-based SQL reasoning agent.
    """
    return """
You are a SQL reasoning assistant that iteratively generates, executes,
and refines SQL queries based on database feedback.

1. Understand the user's question about data analysis
2. Generate appropriate SQL queries with provided schemas, details,
   metrics and previous attempts in the context
3. Execute the queries using the provided tools
4. Evaluate the results and decide to continue or return the results properly

Inputs Provided:
Database Type: Ensure that you use SQL dialect and functions specific
database type. And strictly follow the specific rules provided.

Table Schemas: Only use the tables provided in the context.

Data Details: Sample data and descriptions of key columns. Use related
dimension data as guidance for constructing your query.

Metrics (Semantic Layer): Use this if you already got some knowledge
about the data. Predefined SQL fragments representing common metrics
or aggregations.

Natural Language Question: A user's query written in plain language.

Context: Related context includes results from previous attempts.

Output format: Return a JSON object with the following structure, *only JSON*:
{
  "sql": "final sql you generate",
  "explanation": "final explanation of the task"
}
"""


def get_reasoning_prompt(
    database_type: str,
    table_schemas: List[TableSchema],
    data_details: List[TableValue],
    metrics: str,
    question: str,
    context: List[str],
    prompt_version: str = "",
    max_table_schemas_length: int = 4000,
    max_data_details_length: int = 2000,
    max_context_length: int = 8000,
    max_value_length: int = 500,
    max_text_mark_length: int = 16,
    knowledge_content: str = "",
) -> List[Dict[str, str]]:
    if isinstance(table_schemas, str):
        processed_schemas = table_schemas
    else:
        processed_schemas = "\n".join(schema.to_prompt(database_type) for schema in table_schemas)

    if data_details:
        processed_details = "\n---\n".join(
            detail.to_prompt(database_type, max_value_length, max_text_mark_length, processed_schemas)
            for detail in data_details
        )
    else:
        processed_details = ""

    processed_context = str(context)

    # Truncate if exceeds max length
    if len(processed_schemas) > max_table_schemas_length:
        logger.warning("Table schemas is too long, truncating to %s characters" % max_table_schemas_length)
        processed_schemas = processed_schemas[:max_table_schemas_length] + "\n... (truncated)"

    if len(processed_details) > max_data_details_length:
        logger.warning("Data details is too long, truncating to %s characters" % max_data_details_length)
        processed_details = processed_details[:max_data_details_length] + "\n... (truncated)"

    if len(processed_context) > max_context_length:
        logger.warning("Context is too long, truncating to %s characters" % max_context_length)
        processed_context = processed_context[:max_context_length] + "\n... (truncated)"

    # Add Snowflake specific notes
    database_notes = ""
    if database_type.lower() == DBType.SNOWFLAKE.value.lower():
        database_notes = (
            "\nEnclose all column names in double quotes to comply with Snowflake syntax requirements and avoid errors."
            "When referencing table names in Snowflake SQL, you must include both the database_name and schema_name."
        )
    elif database_type.lower() == DBType.STARROCKS.value.lower():
        database_notes = ""

    user_content = prompt_manager.render_template(
        "reasoning_user",
        database_type=database_type,
        database_notes=database_notes,
        question=question,
        processed_schemas=processed_schemas,
        processed_details=processed_details,
        metrics=metrics,
        processed_context=processed_context,
        version=prompt_version,
        knowledge_content=knowledge_content,
    )

    return [
        {"role": "user", "content": user_content},
    ]
