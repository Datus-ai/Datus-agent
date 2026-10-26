"""API routes for direct tool dispatch."""

import asyncio
from typing import Annotated, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Path
from starlette.concurrency import run_in_threadpool

from datus.api.deps import AppContextDep, ServiceDep, SubAgentDep
from datus.api.models.base_models import Result
from datus.tools.func_tool.base import FuncToolResult

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


@router.post(
    "/{tool_name}",
    response_model=Result[FuncToolResult],
    summary="Execute Tool",
    description="Execute a tool by name with parameters passed in the request body.",
)
async def execute_tool(
    tool_name: Annotated[str, Path(description="Name of the tool to execute")],
    svc: ServiceDep,
    sub_agent: SubAgentDep,
    params: Annotated[Optional[Dict], Body()] = None,
    ctx: AppContextDep = None,
    session_id: Optional[str] = None,
    call_tool_id: Optional[str] = None,
) -> Result[FuncToolResult]:
    """Execute a tool by name, scoped to the request's sub-agent if it has one.

    ``SubAgentDep`` has already rejected a name that matches no configured
    sub-agent, so this only ever sees a real scope or ``None``.
    """
    if session_id is not None or call_tool_id is not None:
        if not session_id or not call_tool_id:
            raise HTTPException(status_code=400, detail="session_id and call_tool_id are required together")
        task = svc.task_manager.get_task(session_id)
        node = task.node if task else None
        if node is None or task.status != "running" or getattr(node, "scope", None) != ctx.user_id:
            raise HTTPException(status_code=404, detail="No pending tool call")
        # Tool-start SSE may arrive just before the proxy starts awaiting its result.
        for _ in range(100):
            invocation = node.tool_channel.invocations.get(call_tool_id)
            if task.status != "running":
                break
            if invocation is not None:
                if invocation.name != tool_name:
                    break
                return Result(success=True, data=FuncToolResult.model_validate(await invocation.execute()))
            await asyncio.sleep(0.05)
        raise HTTPException(status_code=404, detail="No pending tool call")
    if params is None:
        params = {}
    return await run_in_threadpool(svc.tool_for(sub_agent).execute, tool_name, params)
