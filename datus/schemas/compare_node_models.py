# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SQLContext, SqlTask


class CompareInput(BaseInput):
    """
    Input model for compare node.
    Validates the input for comparison analysis.
    """

    sql_task: SqlTask = Field(..., description="The SQL task of this request")
    sql_context: SQLContext = Field(..., description="The SQL context to compare")
    expectation: str = Field(..., description="Ground truth expectation (SQL query or data text)")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt")


class CompareResult(BaseResult):
    """
    Result model for compare node.
    Contains the comparison analysis result.
    """

    explanation: str = Field(..., description="Detailed comparison analysis")
    suggest: str = Field(..., description="Suggestions for the SQL query")
    tokens_used: int = Field(default=0, description="Total tokens consumed during comparison")
