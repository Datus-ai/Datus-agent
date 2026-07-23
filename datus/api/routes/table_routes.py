"""
API routes for Table and SemanticModel endpoints.
"""

import asyncio

from fastapi import APIRouter, Query

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.table_models import (
    GetSemanticModelData,
    GetTableDetailData,
    GetTablesColumnsData,
    GetTablesColumnsInput,
    SemanticModelInput,
    ValidateSemanticModelData,
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
    description="Get SemanticModel YAML configuration for a specific table",
)
async def get_semantic_model(
    svc: ServiceDep,
    table: str = Query(
        ...,
        description="Full table name e.g. 'production_db.public.frpm' or 'db.schema.table'",
    ),
    catalog: str | None = Query(None, description="Current catalog context"),
    database: str | None = Query(None, description="Current database context"),
    db_schema: str | None = Query(None, description="Current schema context"),
    semantic_model_name: str | None = Query(None, description="Semantic model owning a shared physical table"),
) -> Result[GetSemanticModelData]:
    """Get SemanticModel YAML."""
    return await asyncio.to_thread(
        svc.datasource.get_semantic_model,
        table,
        catalog=catalog,
        database=database,
        db_schema=db_schema,
        semantic_model_name=semantic_model_name,
    )


@router.post(
    "/semantic_model",
    response_model=Result[dict],
    summary="Save Semantic Model",
    description="Save or update SemanticModel YAML configuration for a table",
)
async def save_semantic_model(
    request: SemanticModelInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Save SemanticModel YAML."""
    return await svc.datasource.save_semantic_model(request)


@router.post(
    "/semantic_model/validate",
    response_model=Result[ValidateSemanticModelData],
    summary="Validate Semantic Model",
    description="Validate SemanticModel YAML structure and syntax",
)
async def validate_semantic_model(
    request: SemanticModelInput,
    svc: ServiceDep,
) -> Result[ValidateSemanticModelData]:
    """Validate SemanticModel YAML."""
    return await svc.datasource.validate_semantic_model(request)
