"""API routes for direct tool dispatch."""

from typing import Annotated, Dict, Optional

from fastapi import APIRouter, Body, Path

from datus.api.deps import AppContextDep, ServiceDep
from datus.api.models.base_models import Result
from datus.tools.func_tool.base import FuncToolResult

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


@router.post(
    "/{tool_name}",
    response_model=Result[FuncToolResult],
    summary="Execute Tool",
    description="Execute a tool by name with parameters passed in the request body.",
)
def execute_tool(
    tool_name: Annotated[str, Path(description="Name of the tool to execute")],
    svc: ServiceDep,
    ctx: AppContextDep,
    params: Annotated[Optional[Dict], Body()] = None,
) -> Result[FuncToolResult]:
    """Execute a tool by name, scoped to the request's sub-agent if it has one.

    ``ServiceDep`` must stay declared before ``AppContextDep``: the context is
    published on ``request.state`` while the service dependency resolves, so the
    reverse order trips the assertion in ``get_app_context``.
    """
    if params is None:
        params = {}
    return svc.tool_for(ctx.sub_agent_name).execute(tool_name, params)
