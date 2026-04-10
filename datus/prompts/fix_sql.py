# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from datus.schemas.node_models import TableSchema
from datus.utils.loggings import get_logger

from .prompt_manager import get_prompt_manager

logger = get_logger(__name__)


def fix_sql_prompt(
    sql_task: str,
    prompt_version: str = "",
    sql_context: str = "",
    schemas: list[TableSchema] = None,
    docs: list[str] = None,
    agent_config: Optional[Any] = None,
) -> List[Dict[str, str]]:
    if schemas is None:
        schemas = []
    if docs is None:
        docs = []

    pm = get_prompt_manager(agent_config=agent_config)
    system_content = pm.get_raw_template("fix_sql_system", version=prompt_version)
    user_content = pm.render_template(
        "fix_sql_user",
        sql_task=sql_task,
        sql_context=sql_context,
        docs="\n".join(docs),
        schemas="\n".join([schema.to_prompt() for schema in schemas]),
        version=prompt_version,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
