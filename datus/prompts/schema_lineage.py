# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import datetime
from typing import Dict, List, Optional

from datus.utils.json_utils import json2csv

from .prompt_manager import prompt_manager


def gen_prompt(
    dialect: str,
    database_name: str,
    user_question: str,
    table_metadata: List[Dict[str, str]],
    prompt_version: Optional[str] = None,
    top_n: int = 5,
) -> List[Dict[str, str]]:
    if len(table_metadata) == 0:
        return []

    table_metadata_csv = json2csv(table_metadata, ["identifier", "definition"])
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    system_content = prompt_manager.render_template(
        "schema_lineage_system",
        dialect=dialect,
        current_date=current_date,
        version=prompt_version,
    )

    user_content = prompt_manager.render_template(
        "schema_lineage_user",
        database_name=database_name,
        user_question=user_question,
        table_metadata=table_metadata_csv,
        version=prompt_version,
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages


def gen_summary_prompt(
    dialect: str,
    database_name: str,
    user_question: str,
    table_metadata: List[Dict[str, str]],
    prompt_version: Optional[str] = None,
    top_n: int = 5,
) -> List[Dict[str, str]]:
    table_metadata_csv = json2csv(table_metadata, ["schema_name", "table_name", "schema_text", "score", "reasons"])
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    content = prompt_manager.render_template(
        "schema_lineage_summary",
        dialect=dialect,
        current_date=current_date,
        user_question=user_question,
        database_name=database_name,
        table_metadata=table_metadata_csv,
        version=prompt_version,
    )

    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]
    return messages
