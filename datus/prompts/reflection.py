# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
from typing import Optional

from .prompt_manager import prompt_manager


def get_evaluation_prompt(
    task_description: str,
    sql_generation_result: str,
    sql_execution_result: str,
    prompt_version: Optional[str] = None,
) -> str:
    """
    Generate a prompt for evaluating SQL execution results.

    Args:
        task_description: The description of the task
        sql_generation_result: The result from SQL generation
        sql_execution_result: The result from SQL execution

    Returns:
        A formatted prompt string
    """
    return prompt_manager.render_template(
        "evaluation",
        task_description=task_description,
        sql_generation_result=sql_generation_result,
        sql_execution_result=sql_execution_result,
        version=prompt_version,
    )
