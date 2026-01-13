# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import List, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SQLContext, SqlTask, TableSchema


class FixInput(BaseInput):
    """
    Input model for fixing node.
    Validates the input for fixing.
    """

    sql_task: SqlTask = Field(..., description="The SQL task of this request")
    schemas: List[TableSchema] = Field(default_factory=list, description="The schema of the tables")
    sql_context: SQLContext = Field(..., description="The SQL context want to fix")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt")


class FixResult(BaseResult):
    """
    Result model for fixing node.
    Contains the fixing result.
    """

    sql_query: str = Field(..., description="The fixed SQL query")
    explanation: str = Field(..., description="The explanation of the fixing")
