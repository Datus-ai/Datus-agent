"""
API routes for CLI Command Type endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Path

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.cli_models import (
    ExecuteContextData,
    ExecuteContextInput,
    ExecuteSQLData,
    ExecuteSQLInput,
    ExecuteToolData,
    ExecuteToolInput,
    InternalCommandData,
    InternalCommandInput,
)

router = APIRouter(prefix="/api/v1", tags=["cli"])


@router.post(
    "/sql/execute",
    response_model=Result[ExecuteSQLData],
    summary="Execute SQL Query",
    description="Execute SQL query directly against the database",
)
async def execute_sql(
    request: ExecuteSQLInput,
    svc: ServiceDep,
) -> Result[ExecuteSQLData]:
    """Execute SQL query directly."""
    return svc.cli.execute_sql(request)


@router.post(
    "/tools/{tool_name}",
    response_model=Result[ExecuteToolData],
    summary="Execute Tool Command",
    description="Execute agent tool commands (! prefix commands)",
)
async def execute_tool(
    tool_name: Annotated[str, Path(description="Name of the tool to execute")],
    svc: ServiceDep,
    request: ExecuteToolInput = None,
) -> Result[ExecuteToolData]:
    """Execute tool command."""
    if request is None:
        request = ExecuteToolInput(tool_name="", args="")
    # Update the tool_name from path parameter
    request.tool_name = tool_name
    return svc.cli.execute_tool(tool_name, request)


@router.post(
    "/context/{context_type}",
    response_model=Result[ExecuteContextData],
    summary="Execute Context Command",
    description="Execute context-related commands (@ prefix commands)",
)
async def execute_context(
    context_type: Annotated[str, Path(description="Type of context command")],
    svc: ServiceDep,
    request: ExecuteContextInput = None,
) -> Result[ExecuteContextData]:
    """Execute context command."""
    if request is None:
        request = ExecuteContextInput(context_type="")
    # Update the context_type from path parameter
    request.context_type = context_type
    return svc.cli.execute_context(context_type, request)


@router.post(
    "/internal/{command}",
    response_model=Result[InternalCommandData],
    summary="Execute Internal Command",
    description="Execute internal management commands (. prefix commands)",
)
async def execute_internal_command(
    command: Annotated[str, Path(description="Internal command name")],
    svc: ServiceDep,
    request: InternalCommandInput = None,
) -> Result[InternalCommandData]:
    """Execute internal command."""
    if request is None:
        request = InternalCommandInput(command="", args="")
    # Update the command from path parameter
    request.command = command
    return svc.cli.execute_internal_command(command, request)
