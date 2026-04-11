"""
API routes for data visualization.

Provides an endpoint that accepts tabular data and returns a chart
configuration recommendation (chart type, axes, reason).
"""

import asyncio

from fastapi import APIRouter

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.visualization_models import (
    ChartData,
    DataInsight,
    DataVisualizationData,
    DataVisualizationRequest,
)

router = APIRouter(prefix="/api/v1", tags=["visualization"])


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
    result = await asyncio.to_thread(
        svc.visualization.generate,
        csv_data=request.csv_data,
        chart_type=request.chart_type,
        sql=request.sql,
        user_question=request.user_question,
    )

    if not result["success"]:
        return Result(
            success=False,
            errorCode=result["errorCode"],
            errorMessage=result["errorMessage"],
        )

    data = result["data"]
    data_insight = DataInsight(**data["data_insight"]) if data.get("data_insight") else None

    return Result(
        success=True,
        data=DataVisualizationData(
            chart=ChartData(**data["chart"]),
            data_insight=data_insight,
        ),
    )
