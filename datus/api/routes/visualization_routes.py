"""
API routes for data visualization.

Provides an endpoint that accepts tabular data and returns a chart
configuration recommendation (chart type, axes, reason).
"""

import pandas as pd
from fastapi import APIRouter

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.visualization_models import (
    DataVisualizationData,
    DataVisualizationRequest,
)
from datus.models.base import LLMBaseModel
from datus.schemas.visualization import VisualizationInput
from datus.tools.llms_tools.visualization_tool import VisualizationTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["visualization"])

# Map VisualizationTool output chart_type → short frontend key
_CHART_TYPE_MAP = {
    "Bar Chart": "Bar",
    "Line Chart": "Line",
    "Pie Chart": "Pie",
    "Scatter Plot": "Scatter",
    "Unknown": "Unknown",
}


def _build_visualization_tool(svc: ServiceDep) -> VisualizationTool:
    """Create a VisualizationTool backed by the project's LLM config."""
    agent_config = svc.agent_config
    model = None
    try:
        model = LLMBaseModel.create_model(agent_config=agent_config)
    except Exception as exc:
        logger.warning(f"Unable to initialize visualization model, using heuristics: {exc}")
    return VisualizationTool(agent_config=agent_config, model=model)


@router.post(
    "/data_visualization",
    response_model=Result[DataVisualizationData],
    summary="Generate Data Visualization",
    description="Recommend a chart configuration for the provided tabular data.",
)
async def data_visualization(
    request: DataVisualizationRequest,
    svc: ServiceDep,
) -> Result[DataVisualizationData]:
    """Return a chart recommendation for the uploaded CSV-style data."""
    csv_data = request.csv_data

    # ── Build DataFrame from frontend payload ─────────────────────
    try:
        df = pd.DataFrame(csv_data.data, columns=csv_data.columns)
    except Exception as exc:
        return Result(
            success=False,
            errorCode="INVALID_DATA",
            errorMessage=f"Failed to parse csv_data into DataFrame: {exc}",
        )

    if df.empty or df.shape[1] == 0:
        return Result(
            success=False,
            errorCode="EMPTY_DATA",
            errorMessage="Provided dataset is empty or has no columns.",
        )

    # ── Run VisualizationTool ─────────────────────────────────────
    tool = _build_visualization_tool(svc)
    try:
        viz_input = VisualizationInput(data=df)
        result = tool.execute(viz_input)
    except Exception as exc:
        logger.error(f"Visualization tool execution failed: {exc}")
        return Result(
            success=False,
            errorCode="VISUALIZATION_FAILED",
            errorMessage=f"Visualization analysis failed: {exc}",
        )

    if not result.success:
        return Result(
            success=False,
            errorCode="VISUALIZATION_FAILED",
            errorMessage=result.error or "Visualization analysis failed.",
        )

    # ── Map to frontend chart type key ────────────────────────────
    chart_type = _CHART_TYPE_MAP.get(result.chart_type, "Unknown")

    # If the caller requested a specific chart_type, honour it (the tool's
    # x_col / y_cols recommendation is still useful even when overridden).
    if request.chart_type is not None:
        chart_type = request.chart_type

    chart_data: dict = {
        "chart_type": chart_type,
    }

    if chart_type == "Unknown":
        chart_data["reason"] = result.reason
    else:
        chart_data["x_col"] = result.x_col
        chart_data["y_cols"] = result.y_cols
        chart_data["reason"] = result.reason

    return Result(
        success=True,
        data=DataVisualizationData(data=chart_data),
    )
