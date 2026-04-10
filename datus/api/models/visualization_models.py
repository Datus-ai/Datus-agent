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


class LineChartData(BaseModel):
    chart_type: Literal["Line"] = "Line"
    x_col: str = Field(..., description="X-axis (dimension) column")
    y_cols: List[str] = Field(..., description="Y-axis (metric) columns")
    reason: str = ""


class BarChartData(BaseModel):
    chart_type: Literal["Bar"] = "Bar"
    x_col: str = Field(..., description="X-axis (dimension) column")
    y_cols: List[str] = Field(..., description="Y-axis (metric) columns")
    reason: str = ""


class PieChartData(BaseModel):
    chart_type: Literal["Pie"] = "Pie"
    x_col: str = Field(..., description="Category column")
    y_cols: List[str] = Field(..., description="Value column(s)")
    reason: str = ""


class ScatterChartData(BaseModel):
    chart_type: Literal["Scatter"] = "Scatter"
    x_col: str = Field(..., description="X-axis column")
    y_cols: List[str] = Field(..., description="Y-axis column(s)")
    reason: str = ""


class UnknownChartData(BaseModel):
    chart_type: Literal["Unknown"] = "Unknown"
    reason: str = Field(..., description="Why no chart type could be determined")


class DataVisualizationData(BaseModel):
    """Wrapper returned in ``Result.data``."""

    data: Dict[str, Any] = Field(..., description="Chart-type-specific payload")
