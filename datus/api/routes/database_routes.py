"""
API routes for Database Management endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Query

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.database_models import (
    DatabasesData,
    ListDatabasesData,
    ListDatabasesInput,
)

router = APIRouter(prefix="/api/v1", tags=["databases"])

# Pre-configured parameters to avoid definition-time evaluation in defaults
DATASOURCE_QUERY = Query("", description="Datasource to list databases from")
DATABASE_NAME_QUERY = Query("", description="Database name")
SCHEMA_NAME_QUERY = Query("", description="Schema name")
CATALOG_NAME_QUERY = Query("", description="Catalog name")
INCLUDE_SYS_SCHEMAS_QUERY = Query(False, description="Include system schemas")


@router.get(
    "/catalog/list",
    response_model=Result[DatabasesData],
    summary="List Catalogs",
    description="List available catalogs",
)
async def list_catalogs(
    svc: ServiceDep,
    datasource_id: Optional[str] = DATASOURCE_QUERY,
    catalog_name: Optional[str] = CATALOG_NAME_QUERY,
    database_name: Optional[str] = DATABASE_NAME_QUERY,
    schema_name: Optional[str] = SCHEMA_NAME_QUERY,
    include_sys_schemas: bool = INCLUDE_SYS_SCHEMAS_QUERY,
) -> Result[DatabasesData]:
    """List available databases."""
    request = ListDatabasesInput(
        datasource_id=datasource_id or svc.database.current_datasource,
        catalog_name=catalog_name,
        database_name=database_name,
        schema_name=schema_name,
        include_sys_schemas=include_sys_schemas,
    )
    databases: Result[ListDatabasesData] = svc.database.list_databases(request)
    if not databases.success or databases.data is None:
        return Result(
            success=False,
            errorCode=databases.errorCode,
            errorMessage=databases.errorMessage,
        )
    return Result(success=True, data=DatabasesData(databases=databases.data.databases))
