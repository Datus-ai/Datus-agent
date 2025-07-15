from typing import List, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.generate_semantic_model_node_models import SemanticModelMeta
from datus.schemas.node_models import Metrics, SQLContext
from datus.utils.constants import DBType


class SearchMetricsInput(BaseInput):
    input_text: str = Field(..., description="The query text to analyze for schema linking")
    semantic_model_meta: SemanticModelMeta = Field(..., description="The semantic model meta of this request")
    database_type: str = Field(DBType.SQLITE, description="Database type: sqlite, duckdb snowflake, etc ")
    sql_context: Optional[SQLContext] = Field(None, description="The SQL context")


class SearchMetricsResult(BaseResult):
    semantic_model_meta: SemanticModelMeta = Field(..., description="The semantic model meta of this request")
    metrics: List[Metrics] = Field(..., description="The search metric")
    metrics_count: int = Field(..., description="Number of metrics values found")
