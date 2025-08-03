from typing import List, Literal, Optional

from pydantic import Field, field_validator

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.generate_semantic_model_node_models import SemanticModelMeta
from datus.schemas.node_models import Metrics, SQLContext
from datus.utils.constants import DBType


class SearchMetricsInput(BaseInput):
    input_text: str = Field(..., description="The query text to analyze for schema linking")
    semantic_model_meta: SemanticModelMeta = Field(..., description="The semantic model meta of this request")
    database_type: str = Field(DBType.SQLITE, description="Database type: sqlite, duckdb snowflake, etc ")
    sql_contexts: Optional[List[SQLContext]] = Field(default=[], description="The SQL context")
    top_n: int = Field(default=5, description="Number of top tables to return")
    matching_rate: Literal["fast", "medium", "slow"] = Field(
        "fast",
        description="Match rates of the search metrics, allowed values: fast, medium, slow",
    )

    def top_n_by_rate(self) -> int:
        # give hard code of the limit according to the scale of medium-sized company
        if self.matching_rate == "fast":
            return 5
        elif self.matching_rate == "medium":
            return 10
        return 20

    @field_validator("matching_rate")
    def validate_matching_rate(cls, v):
        if v not in ["fast", "medium", "slow"]:
            raise ValueError("'matching_rate' must be one of: fast, medium, slow")
        return v


class SearchMetricsResult(BaseResult):
    semantic_model_meta: SemanticModelMeta = Field(..., description="The semantic model meta of this request")
    metrics: List[Metrics] = Field(..., description="The search metric")
    metrics_count: int = Field(..., description="Number of metrics values found")
