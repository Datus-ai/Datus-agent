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
    total_rows: Optional[int] = Field(
        None,
        description=(
            "Row count of the full result set when ``csv_data`` carries only a sample of it. "
            "Omit when ``csv_data`` is the whole set."
        ),
    )
    chart_type: Optional[ChartType] = Field(None, description="Desired chart type; omit for auto-recommendation")
    sql: Optional[str] = Field(None, description="SQL query that produced the data (enables metadata extraction)")
    user_question: Optional[str] = Field(None, description="User's original question (improves insight quality)")
    language: Optional[str] = Field(
        None,
        description=(
            "Response language override for the generated text (reason / filters / insight), "
            "e.g. 'en', 'zh'. Falls back to agent.language from yaml when unset."
        ),
    )


# ── Response payloads ────────────────────────────────────────────────


class ChartData(BaseModel):
    """Chart configuration for rendering."""

    chart_type: ChartType = Field(..., description="Recommended chart type")
    columns: List[str] = Field(..., description="All column names in the dataset")
    numeric_columns: List[str] = Field(..., description="Numeric column names (eligible for Y-axis)")
    x_col: Optional[str] = Field(None, description="X-axis column (absent when chart_type is Unknown)")
    y_cols: Optional[List[str]] = Field(None, description="Y-axis column(s) (absent when chart_type is Unknown)")
    reason: str = Field("", description="Explanation for the recommendation")


class DataInsight(BaseModel):
    """Context metadata extracted from SQL and data analysis."""

    period: Optional[str] = Field(None, description="Time range extracted from SQL")
    filters: Optional[List[str]] = Field(None, description="Human-readable filter descriptions")
    insight: Optional[str] = Field(None, description="AI-generated analytical summary")


class DataVisualizationData(BaseModel):
    """Wrapper returned in ``Result.data``."""

    chart: ChartData = Field(..., description="Chart configuration for rendering")
    data_insight: Optional[DataInsight] = Field(None, description="Context metadata (present when sql is provided)")
