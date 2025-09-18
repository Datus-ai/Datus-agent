#!/usr/bin/env python3
"""
Simplified SQL reasoning implementation that doesn't require MCP/tool calling
"""
import json
from typing import Any, Dict

from datus.models.base import LLMBaseModel
from datus.schemas.node_models import SQLContext
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.utils.json_utils import strip_json_str
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


@optional_traceable()
def reasoning_sql_simple(
    model: LLMBaseModel, input_data: ReasoningInput, tool_config: Dict[str, Any] = None
) -> ReasoningResult:
    """
    Generate SQL reasoning without MCP - uses basic text generation only.
    This is a fallback for models that don't support tool calling.
    """
    if not isinstance(input_data, ReasoningInput):
        raise ValueError("Input must be a ReasoningInput instance")

    if tool_config is None:
        tool_config = {}

    logger.info("Using simplified reasoning without MCP (no tool calling)")

    try:
        # Create a simplified reasoning prompt
        prompt_text = f"""
You are a SQL expert analyzing a database query request. Based on the provided information,
generate an appropriate SQL query.

Database Type: {input_data.get('database_type', 'sqlite')}
Question: {input_data.sql_task.task}

Table Schemas:
{_format_table_schemas(input_data.table_schemas, input_data.get('database_type', 'sqlite'))}

Data Details:
{_format_data_details(input_data.data_details, input_data.get('database_type', 'sqlite'))}

Context from previous attempts:
{_format_contexts(input_data.contexts)}

Please generate a SQL query that answers the question. Respond with a JSON object containing:
{{
  "sql": "Your SQL query here",
  "explanation": "Explanation of your approach and reasoning"
}}

Only return valid JSON, no additional text.
"""

        # Generate response using the model
        response = model.generate(prompt_text, max_tokens=2000, temperature=0.3)

        try:
            # Parse the JSON response
            response_clean = strip_json_str(response)
            content_dict = json.loads(response_clean)

            sql_query = content_dict.get("sql", "")
            explanation = content_dict.get("explanation", "")

            # Create a simple SQL context (without execution since we don't have MCP)
            sql_context = SQLContext(
                sql_query=sql_query,
                explanation=explanation,
                sql_return="[Not executed - simplified reasoning mode]",
                row_count=0,
            )

            return ReasoningResult(
                success=True,
                sql_query=sql_query,
                sql_return="[Not executed - simplified reasoning mode]",
                sql_contexts=[sql_context],
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            logger.debug(f"Raw response: {response}")

            # Fallback: extract SQL from response text
            sql_query = _extract_sql_from_text(response)

            return ReasoningResult(
                success=bool(sql_query),
                error="Failed to parse JSON response" if not sql_query else None,
                sql_query=sql_query,
                sql_return="[Not executed - simplified reasoning mode]",
                sql_contexts=[],
            )

    except Exception as e:
        logger.error(f"Simplified reasoning failed: {e}")
        return ReasoningResult(success=False, error=str(e), sql_query="", row_count=0, sql_return="", sql_contexts=[])


def _format_table_schemas(table_schemas, database_type: str) -> str:
    """Format table schemas for the prompt"""
    if not table_schemas:
        return "No table schemas provided"

    if isinstance(table_schemas, str):
        return table_schemas

    try:
        return "\n".join([schema.to_prompt(database_type) for schema in table_schemas])
    except Exception:
        return str(table_schemas)


def _format_data_details(data_details, database_type: str) -> str:
    """Format data details for the prompt"""
    if not data_details:
        return "No data details provided"

    try:
        return "\n".join([detail.to_prompt(database_type, 500) for detail in data_details])
    except Exception:
        return str(data_details)


def _format_contexts(contexts) -> str:
    """Format context information for the prompt"""
    if not contexts:
        return "No previous context"

    try:
        return "\n".join([context.to_str() for context in contexts])
    except Exception:
        return str(contexts)


def _extract_sql_from_text(text: str) -> str:
    """Extract SQL query from text response as fallback"""
    lines = text.split("\n")
    sql_lines = []
    in_sql = False

    for line in lines:
        line_lower = line.lower().strip()

        # Look for SQL keywords
        if any(keyword in line_lower for keyword in ["select", "with", "insert", "update", "delete"]):
            in_sql = True
            sql_lines.append(line.strip())
        elif in_sql and line.strip():
            if line.strip().endswith(";"):
                sql_lines.append(line.strip())
                break
            else:
                sql_lines.append(line.strip())
        elif in_sql and not line.strip():
            break

    return " ".join(sql_lines) if sql_lines else ""
