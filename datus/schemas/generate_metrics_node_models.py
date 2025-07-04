from typing import List

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import Metrics, SqlTask


class GenerateMetricsInput(BaseInput):
    """
    Input model for generating metrics node.
    Validates the input for generating metrics.
    """

    sql_task: SqlTask = Field(..., description="The SQL task of this request")
    sql_query: str = Field(..., description="The SQL query to generate metrics from")
    prompt_version: str = Field(default="1.0", description="Version for prompt")


class GenerateMetricsResult(BaseResult):
    """
    Result model for generating metrics node.
    Contains the generated metric file path.
    """

    sql_query: str = Field(..., description="The SQL query generated from metrics")
    metrics: List[Metrics] = Field(
        default_factory=list, description="The metrics you added to the semantic model YAML file"
    )
