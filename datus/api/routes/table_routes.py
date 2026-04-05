"""
API routes for Table and SemanticModel endpoints.
"""

from fastapi import APIRouter, Depends, Query

from datus.api.deps import get_datus_service
from datus.api.models.base_models import Result
from datus.api.models.table_models import (
    GetSemanticModelData,
    GetTableDetailData,
    SemanticModelInput,
    ValidateSemanticModelData,
)
from datus.api.services.datus_service import DatusService

router = APIRouter(prefix="/api/v1", tags=["table"])


# ========== Table Endpoints ==========


@router.get(
    "/table/detail",
    response_model=Result[GetTableDetailData],
    summary="Get Table Detail",
    description="Get detailed information about a table including columns, indexes, and row count",
)
async def get_table_detail(
    table: str = Query(
        ...,
        description="Full table name e.g. 'production_db.public.frpm' or 'db.schema.table'",
    ),
    svc: DatusService = Depends(get_datus_service),
) -> Result[GetTableDetailData]:
    """Get table detail."""
    return svc.database.get_table_schema(table)


# ========== SemanticModel Endpoints ==========


@router.get(
    "/semantic_model",
    response_model=Result[GetSemanticModelData],
    summary="Get Semantic Model",
    description="Get SemanticModel YAML configuration for a specific table",
)
async def get_semantic_model(
    table: str = Query(
        ...,
        description="Full table name e.g. 'production_db.public.frpm' or 'db.schema.table'",
    ),
    svc: DatusService = Depends(get_datus_service),
) -> Result[GetSemanticModelData]:
    """Get SemanticModel YAML."""
    return svc.database.get_semantic_model(table)


@router.post(
    "/semantic_model",
    response_model=Result[dict],
    summary="Save Semantic Model",
    description="Save or update SemanticModel YAML configuration for a table",
)
async def save_semantic_model(
    request: SemanticModelInput,
    svc: DatusService = Depends(get_datus_service),
) -> Result[dict]:
    """Save SemanticModel YAML."""
    return await svc.database.save_semantic_model(request)


@router.post(
    "/semantic_model/validate",
    response_model=Result[ValidateSemanticModelData],
    summary="Validate Semantic Model",
    description="Validate SemanticModel YAML structure and syntax",
)
async def validate_semantic_model(
    request: SemanticModelInput,
    svc: DatusService = Depends(get_datus_service),
) -> Result[ValidateSemanticModelData]:
    """Validate SemanticModel YAML."""
    return await svc.database.validate_semantic_model(request)
