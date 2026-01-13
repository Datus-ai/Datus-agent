# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List

from datus.utils.loggings import get_logger

from .prompt_manager import prompt_manager

logger = get_logger(__name__)


def compare_sql_prompt(
    sql_task,
    prompt_version: str = "",
    sql_query: str = "",
    sql_explanation: str = "",
    sql_result: str = "",
    sql_error: str = "",
    expectation: str = "",
) -> List[Dict[str, str]]:
    system_content = prompt_manager.get_raw_template("compare_sql_system_mcp", version=prompt_version)
    user_content = prompt_manager.render_template(
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
