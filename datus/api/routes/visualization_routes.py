"""
API routes for data visualization.

Provides an endpoint that accepts tabular data and returns a chart
configuration recommendation (chart type, axes, reason).
"""

from fastapi import APIRouter

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.visualization_models import (
    ChartData,
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
    result = svc.visualization.generate(
        csv_data=request.csv_data,
        chart_type=request.chart_type,
    )

    if not result["success"]:
        return Result(
            success=False,
            errorCode=result["errorCode"],
            errorMessage=result["errorMessage"],
        )

    return Result(
        success=True,
        data=DataVisualizationData(data=ChartData(**result["data"]["data"])),
    )
