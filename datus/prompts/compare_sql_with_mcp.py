# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from datus.utils.loggings import get_logger

from .prompt_manager import get_prompt_manager

logger = get_logger(__name__)


def get_compare_prompt(
    sql_task,
    sql_query: str = "",
    sql_explanation: str = "",
    sql_result: str = "",
    sql_error: str = "",
    expectation: str = "",
    prompt_version: str = "",
    agent_config: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """Generate comparison prompt for MCP streaming."""

    pm = get_prompt_manager(agent_config=agent_config)
    system_content = pm.get_raw_template("compare_sql_system_mcp", version=prompt_version)
    user_content = pm.render_template(
        "compare_sql_user",
        database_type=sql_task.database_type,
        database_name=sql_task.database_name,
        sql_task=sql_task.task,
        external_knowledge=sql_task.external_knowledge,
        sql_query=sql_query,
        sql_explanation=sql_explanation,
        sql_result=sql_result,
        sql_error=sql_error,
        expectation=expectation,
        version=prompt_version,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def get_compare_system_prompt() -> str:
    """
    Return the system-level instructions for the MCP-based SQL comparison agent.
    """
    return """
You are a SQL comparison assistant that analyzes differences between SQL queries and expectations.

You have access to database tools to:
1. Execute the current SQL query to understand its behavior
2. Test the expected SQL to validate its correctness
3. Explore table schemas and data to understand structure differences
4. Provide detailed analysis and actionable suggestions

Your task is to:
1. Analyze the provided SQL query and understand its intent
2. Compare it against the expectation (whether it's another SQL query or expected data format)
3. Execute queries to validate behavioral differences
4. Identify differences in logic, structure, or expected outcomes
5. Provide specific, actionable suggestions for improving the SQL query

Instructions:
- Compare query logic, structure, and expected results
- Identify schema differences, join patterns, and filtering logic
- Execute queries to validate behavioral differences
- Provide specific, actionable improvement suggestions
- Use database tools to explore and validate your analysis

Output format: Return a JSON object with the following structure, *only JSON*:
{
  "explanation": "detailed analysis of differences between the SQL and expectation",
  "suggest": "concrete suggestions for modifying the SQL query to better align with the expectation"
}
"""
