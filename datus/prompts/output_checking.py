# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List, Union

from datus.schemas.node_models import Metric, TableSchema
from datus.utils.constants import DBType

from .prompt_manager import prompt_manager


def gen_prompt(
    user_question: str,
    table_schemas: Union[List[TableSchema], str],
    sql_query: str,
    sql_execution_result: str,
    metrics: List[Metric] = None,
    dialect: str = DBType.SQLITE,
    external_knowledge: str = "",
    prompt_version: str = "",
) -> List[Dict[str, str]]:
    """Generate a prompt for checking the output of a SQL query.

    Args:
        user_question (str): The user's question to be answered by the SQL query
        table_schemas (str): The schemas of the tables that the SQL query will be
            executed on
        sql_query (str): The SQL query to be checked
        sql_execution_result (str): The result of the SQL query
        dialect (str, optional): The dialect of the SQL query. Defaults to "sqlite".
        external_knowledge (str, optional): The external knowledge that the SQL
            query will be executed on.

    Returns:
        List[Dict[str, str]]: The prompt for checking the output of a SQL query
    """
    if metrics is None:
        metrics = []

    if external_knowledge:
        external_knowledge = f"### External Knowledge:\n{external_knowledge}"

    if not metrics:
        metrics_str = ""
    else:
        metrics_str = "\n".join([m.to_prompt() for m in metrics]).strip()
        if metrics_str:
            metrics_str = f"### Metrics:\n{metrics_str}"

    if len(sql_execution_result) > 500:
        sql_execution_result = sql_execution_result[:500] + " ... (truncated)"

    if isinstance(table_schemas, str):
        table_schemas_str = table_schemas
    else:
        table_schemas_str = "\n".join([schema.to_prompt(dialect) for schema in table_schemas])

    # Render template
    content = prompt_manager.render_template(
        "output_checking",
        dialect=dialect,
        user_question=user_question,
        table_schemas=table_schemas_str,
        external_knowledge=external_knowledge,
        metrics=metrics_str,
        sql_query=sql_query,
        sql_execution_result=sql_execution_result,
        version=prompt_version,
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]
