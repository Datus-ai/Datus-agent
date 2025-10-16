# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json

from datus.models.base import LLMBaseModel
from datus.prompts.fix_sql import fix_sql_prompt
from datus.schemas.fix_node_models import FixInput, FixResult
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


@optional_traceable()
def autofix_sql(model: LLMBaseModel, input_data: FixInput, docs: list[str]) -> FixResult:
    """Generate SQL query using the provided model."""
    if not isinstance(input_data, FixInput):
        raise ValueError("Input must be a FixInput instance")

    try:
        sql_query = input_data.sql_context.sql_query
        sql_explanation = input_data.sql_context.explanation
        sql_result = input_data.sql_context.sql_return
        sql_error = input_data.sql_context.sql_error if hasattr(input_data.sql_context, "sql_error") else None
        # Constructing the SQL string with additional context fields
        sql_text = f"Query: {sql_query}\n Explanation: {sql_explanation}\n Result: {sql_result}\n Error: {sql_error}"
        # Format the prompt with schema list
        prompt = fix_sql_prompt(
            sql_task=input_data.sql_task.to_str(),
            prompt_version=input_data.prompt_version,
            sql_context=sql_text,
            schemas=input_data.schemas,
            docs=docs,
        )

        logger.debug(f"Fix SQL prompt:  {type(model)}, {prompt}")
        # Generate SQL using the provided model
        sql_query = model.generate_with_json_output(prompt)
        logger.debug(f"Fixed SQL: {sql_query}")

        # Clean and parse the response
        if isinstance(sql_query, str):
            # Remove markdown code blocks if present
            sql_query = sql_query.strip().replace("```json\n", "").replace("\n```", "")
            # Remove SQL comments
            cleaned_lines = []
            for line in sql_query.split("\n"):
                line = line.strip()
                if line and not line.startswith("--"):
                    cleaned_lines.append(line)
            cleaned_sql = " ".join(cleaned_lines)
            try:
                sql_query_dict = json.loads(cleaned_sql)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse cleaned SQL: {cleaned_sql}")
                return FixResult(success=False, error="Invalid JSON format", sql_query=sql_query)
        else:
            sql_query_dict = sql_query

        # Return result as FixResult
        if sql_query_dict and isinstance(sql_query_dict, dict):
            return FixResult(
                success=True,
                error=None,
                sql_query=sql_query_dict.get("sql", ""),
                explanation=sql_query_dict.get("explanation"),
            )
        else:
            return FixResult(success=False, error="sql generation failed, no result", sql_query=sql_query)
    except json.JSONDecodeError as e:
        logger.error(f"SQL json decode failed: {str(e)} SQL: {sql_query}")
        return FixResult(success=False, error=str(e), sql_query={sql_query}, explanation="")
    except Exception as e:
        logger.error(f"SQL fix failed: {str(e)}")
        return FixResult(success=False, error=str(e), sql_query="", explanation="")
