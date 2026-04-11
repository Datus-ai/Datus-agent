from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult

DataLike = Union[pd.DataFrame, List[Dict[str, Any]], pa.Table]


class VisualizationInput(BaseInput):
    data: DataLike = Field(description="Data to be visualized (DataFrame, list, or PyArrow Table)")

    class Config:
        arbitrary_types_allowed = True


class VisualizationOutput(BaseResult):
    chart_type: str = Field(description="Type of chart visualization")
    x_col: str = Field(description="Column name for the X-axis")
    y_cols: list[str] = Field(description="List of column names for the Y-axis")
    reason: str = Field(description="A short explanation")


class VisualizationWithContextOutput(VisualizationOutput):
    """Extended output that includes data context metadata."""

    period: Optional[str] = Field(None, description="Time range from SQL")
    filters: Optional[List[str]] = Field(None, description="Human-readable filters")
    insight: Optional[str] = Field(None, description="Analytical summary")
