"""
API routes for Table and SemanticModel endpoints.
"""

import asyncio

from fastapi import APIRouter, Query

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.table_models import (
    SEMANTIC_MODEL_FILE_DESCRIPTION,
    GetSemanticModelData,
    GetTableDetailData,
    GetTablesColumnsData,
    GetTablesColumnsInput,
    SaveSemanticModelData,
    SaveSemanticModelInput,
    ValidateSemanticModelData,
    ValidateSemanticModelInput,
)

router = APIRouter(prefix="/api/v1", tags=["table"])


# ========== Table Endpoints ==========


@router.get(
    "/table/detail",
    response_model=Result[GetTableDetailData],
    summary="Get Table Detail",
    description="Get detailed information about a table including columns, indexes, and row count",
)
async def get_table_detail(
    svc: ServiceDep,
    table: str = Query(
        ...,
        description="Full table name e.g. 'production_db.public.frpm' or 'db.schema.table'",
    ),
) -> Result[GetTableDetailData]:
    """Get table detail."""
    return await asyncio.to_thread(svc.datasource.get_table_schema, table)


@router.post(
    "/table/columns",
    response_model=Result[GetTablesColumnsData],
    summary="Get Columns For Multiple Tables",
    description="Batch-fetch column metadata for a list of tables (autocomplete prefetch). "
    "Results are cached in memory; tables that fail to resolve are omitted.",
)
async def get_tables_columns(
    request: GetTablesColumnsInput,
    svc: ServiceDep,
) -> Result[GetTablesColumnsData]:
    """Batch table columns."""
    return await asyncio.to_thread(svc.datasource.get_tables_columns, request.tables)


# ========== SemanticModel Endpoints ==========


@router.get(
    "/semantic_model",
    response_model=Result[GetSemanticModelData],
    summary="Get Semantic Model",
    description="Read one SemanticModel YAML artifact together with its stable identity and revision",
)
async def get_semantic_model(
    svc: ServiceDep,
    semantic_model_file: str = Query(..., description=SEMANTIC_MODEL_FILE_DESCRIPTION),
) -> Result[GetSemanticModelData]:
    """Get SemanticModel YAML."""
    return await asyncio.to_thread(svc.datasource.get_semantic_model, semantic_model_file)


@router.post(
    "/semantic_model",
    response_model=Result[SaveSemanticModelData],
    summary="Save Semantic Model",
    description="Atomically save and reconcile one SemanticModel YAML artifact",
)
async def save_semantic_model(
    request: SaveSemanticModelInput,
    svc: ServiceDep,
) -> Result[SaveSemanticModelData]:
    """Save SemanticModel YAML and reconcile its Knowledge Base rows."""
    return await svc.datasource.save_semantic_model(request)


@router.post(
    "/semantic_model/validate",
    response_model=Result[ValidateSemanticModelData],
    summary="Validate Semantic Model",
    description="Validate SemanticModel YAML structure and syntax without writing it",
)
async def validate_semantic_model(
    request: ValidateSemanticModelInput,
    svc: ServiceDep,
) -> Result[ValidateSemanticModelData]:
    """Validate SemanticModel YAML."""
    return await svc.datasource.validate_semantic_model(request)
