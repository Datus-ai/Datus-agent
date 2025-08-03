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
    prompt_version: str = Field(default="1.0", description="Version for prompt")


class CompareResult(BaseResult):
    """
    Result model for compare node.
    Contains the comparison analysis result.
    """

    explanation: str = Field(..., description="Detailed comparison analysis")
    suggest: str = Field(..., description="Suggestions for the SQL query")
