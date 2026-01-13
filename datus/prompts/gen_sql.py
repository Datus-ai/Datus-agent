# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List, Union

from datus.schemas.node_models import Metric, TableSchema, TableValue
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from ..utils.json_utils import to_pretty_str
from .prompt_manager import prompt_manager

logger = get_logger(__name__)


def get_sql_prompt(
    database_type: str,
    table_schemas: Union[List[TableSchema], str],
    data_details: List[TableValue],
    metrics: List[Metric],
    question: str,
    external_knowledge: str = "",
    prompt_version: str = "",
    context=None,
    max_table_schemas_length: int = 4000,
    max_data_details_length: int = 2000,
    max_context_length: int = 8000,
    max_value_length: int = 500,
    max_text_mark_length: int = 16,
    database_docs: str = "",
    current_date: str = None,
    date_ranges: str = "",
) -> List[Dict[str, str]]:
    if context is None:
        context = []

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
    knowledge_content = "" if not external_knowledge else f"External Knowledge:\n{external_knowledge}"
    if database_type.lower() == DBType.SNOWFLAKE.value.lower():
        database_notes = (
            "\nEnclose all column names in double quotes to comply with Snowflake syntax requirements and avoid erros. "
            "When referencing table names in Snowflake SQL, you must include both the database_name and schema_name."
        )
    elif database_type.lower() == DBType.STARROCKS.value.lower():
        database_notes = ""

    processed_metrics = ""
    if metrics:
        processed_metrics = to_pretty_str([m.__dict__ for m in metrics])

    system_content = prompt_manager.get_raw_template("gen_sql_system", version=prompt_version)
    user_content = prompt_manager.render_template(
        "gen_sql_user",
        database_type=database_type,
        database_notes=database_notes,
        processed_schemas=processed_schemas,
        processed_details=processed_details,
        metrics=processed_metrics,
        knowledge_content=knowledge_content,
        question=question,
        version=prompt_version,
        processed_context=processed_context,
        database_docs=database_docs,
        current_date=current_date,
        date_ranges=date_ranges,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
