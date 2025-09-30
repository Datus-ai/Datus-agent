from typing import List, Optional

from pydantic import BaseModel, Field


class ScopedContext(BaseModel):
    tables: Optional[str] = Field(default=None, init=True, description="Tables to be used by sub-agents")
    metrics: Optional[str] = Field(default=None, init=True, description="Metrics to be used by sub-agents")
    sqls: Optional[str] = Field(default=None, init=True, description="Historical SQL to be used by sub-agents")


class SubAgentConfig(BaseModel):
    system_prompt: str = Field(..., init=True, description="Name of sub agent")
    agent_description: str = Field(..., init=True, description="Description of sub agent")
    tools: str = Field(default="", init=True, description="Native tools to be used by sub-agents")
    mcp: str = Field(default="", init=True, description="MCP tools to be used by sub-agents")
    scoped_context: Optional[ScopedContext] = Field(
        default=None, init=True, description="Scoped context for sub-agents"
    )
    rules: List[str] = Field(default_factory=list, init=True, description="Rules to be used by sub-agents")
    prompt_version: str = Field(default="1.0", init=True, description="System Prompt version")
    prompt_language: str = Field(default="en", init=True, description="System Prompt language")

    class Config:
        populate_by_name = True
