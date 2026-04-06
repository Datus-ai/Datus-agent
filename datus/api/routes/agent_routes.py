"""
API routes for Agent endpoints.

- /agent: returns agent detail
- /agent/list: returns all sub-agents (builtin + custom from agent.yml)
- /agent/create: create a new custom sub-agent
- /agent/edit: update an existing custom sub-agent
- /agent/use_tools: get available tools for a given agent type
- /agent/tools: list all valid tool categories
"""

from fastapi import APIRouter, Query

from datus.api.deps import ServiceDep
from datus.api.models.agent_models import CreateAgentInput, EditAgentInput
from datus.api.models.base_models import Result
from datus.api.services.agent_service import VALID_TOOL_METHODS, AgentService

router = APIRouter(prefix="/api/v1", tags=["agent"])


# ========== Agent Use Tools ==========


@router.get(
    "/agent/use_tools",
    response_model=Result[dict],
    summary="Get Agent Available Tools",
    description="Get available tool types for a given sub-agent type",
)
async def get_agent_use_tools(
    agent_type: str = Query(..., description="Agent type: 'gen_sql' or 'gen_report'"),
) -> Result[dict]:
    """Return available tools for the specified agent type."""
    return AgentService.get_use_tools(agent_type)


# ========== Agent Detail ==========


@router.get(
    "/agent",
    response_model=Result[dict],
    summary="Get Agent Detail",
    description="Get configuration details for a specific agent by name",
)
async def get_agent(
    svc: ServiceDep,
    name: str = Query(..., description="Agent name"),
) -> Result[dict]:
    """Return agent configuration matching IAgentInfo."""
    agent_service = AgentService()
    return await agent_service.get_agent(
        name=name,
        agent_config=svc.agent_config,
    )


# ========== Agent List ==========


@router.get(
    "/agent/list",
    response_model=Result[dict],
    summary="List Agents",
    description="Get list of all available agents (builtin + custom sub-agents)",
)
async def list_agents(
    svc: ServiceDep,
) -> Result[dict]:
    """List all agents available for this project."""
    agent_service = AgentService()
    return await agent_service.list_agents(agent_config=svc.agent_config)


# ========== Agent Create ==========


@router.post(
    "/agent/create",
    response_model=Result[dict],
    summary="Create Agent",
    description="Create a new custom sub-agent",
)
async def create_agent(
    request: CreateAgentInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Create a new custom sub-agent."""
    agent_service = AgentService()
    return await agent_service.create_agent(
        request=request,
        agent_config=svc.agent_config,
    )


# ========== Agent Edit ==========


@router.post(
    "/agent/edit",
    response_model=Result[dict],
    summary="Edit Agent",
    description="Update an existing custom sub-agent configuration",
)
async def edit_agent(
    request: EditAgentInput,
    svc: ServiceDep,
) -> Result[dict]:
    """Edit an existing custom sub-agent."""
    agent_service = AgentService()
    return await agent_service.edit_agent(
        request=request,
        agent_config=svc.agent_config,
    )


# ========== Available Tools ==========


@router.get(
    "/agent/tools",
    response_model=Result[dict],
    summary="List Available Tools",
    description="Get all valid tool categories and their methods for agent configuration",
)
async def list_available_tools() -> Result[dict]:
    """Return all valid tool categories and their methods."""
    return Result(
        success=True,
        data={
            "tools": {category: sorted(methods) for category, methods in VALID_TOOL_METHODS.items()},
        },
    )
