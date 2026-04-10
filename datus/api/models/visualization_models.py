"""Pydantic models for the data-visualization API."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ChartType = Literal["Bar", "Line", "Pie", "Scatter", "Unknown"]


class CsvData(BaseModel):
    """Tabular data sent by the frontend."""

    columns: List[str] = Field(..., description="Column names")
    data: List[Dict[str, Any]] = Field(..., description="Row records (list of dicts)")


class DataVisualizationRequest(BaseModel):
    """POST body for /api/v1/data_visualization."""

    csv_data: CsvData
    chart_type: Optional[ChartType] = Field(None, description="Desired chart type; omit for auto-recommendation")


# ── Response payloads ────────────────────────────────────────────────


class ChartData(BaseModel):
    """Chart recommendation payload returned by the visualization API."""

    chart_type: ChartType = Field(..., description="Recommended chart type")
    columns: List[str] = Field(..., description="All column names in the dataset")
    numeric_columns: List[str] = Field(..., description="Numeric column names (eligible for Y-axis)")
    x_col: Optional[str] = Field(None, description="X-axis column (absent when chart_type is Unknown)")
    y_cols: Optional[List[str]] = Field(None, description="Y-axis column(s) (absent when chart_type is Unknown)")
    reason: str = Field("", description="Explanation for the recommendation")


class DataVisualizationData(BaseModel):
    """Wrapper returned in ``Result.data``."""

    data: ChartData = Field(..., description="Chart recommendation payload")
