"""Data models for Agent API endpoints."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ========== Agent List ==========


class AgentInfo(BaseModel):
    """Agent information."""

    name: str = Field(..., description="Agent name")
    type: str = Field(..., description="Agent type (builtin/customize)")
    config_yaml: Optional[str] = Field(None, description="Agent configuration YAML")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class AgentListData(BaseModel):
    """Agent list data."""

    agents: List[AgentInfo]


# ========== Create Agent ==========


class CreateAgentInput(BaseModel):
    """Input for creating a new sub-agent."""

    name: str = Field(..., min_length=1, description="Agent name (unique within workspace)")
    datasource_id: str = Field(default="", description="Datasource ID this agent is bound to")
    type: str = Field(default="gen_sql", description="Node class: gen_sql or gen_report")
    description: Optional[str] = Field(default=None, description="Agent description")
    prompt_template: Optional[str] = Field(default=None, description="System prompt content")
    prompt_version: Optional[str] = Field(default="1.0", description="Prompt version (None = latest)")
    prompt_language: str = Field(default="en", description="Prompt language")
    tools: Optional[List[str]] = Field(default=None, description="Tool names")
    mcp: Optional[List[str]] = Field(default_factory=list, description="MCP tool names")
    skills: Optional[List[str]] = Field(default_factory=list, description="Skills pattern filter")
    catalogs: Optional[List[str]] = Field(
        default_factory=list,
        description="Catalog access patterns (e.g., 'production_db.*', 'production_db.public.*')",
    )
    subjects: Optional[List[str]] = Field(
        default_factory=list, description="Subject access patterns (e.g., 'Finance.Revenue.*')"
    )
    permissions: Optional[dict] = Field(default_factory=dict, description="Permission overrides")
    hooks: Optional[dict] = Field(default_factory=dict, description="Hook configuration")
    rules: Optional[list[str]] = Field(default_factory=list, description="Instruction rules")
    max_turns: Optional[int] = Field(default=30, description="Max conversation turns")
    workspace_root: Optional[str] = Field(default=None, description="Workspace root path")
    adapter_type: Optional[str] = Field(default=None, description="Adapter type")
    sql_file_threshold: Optional[int] = Field(default=None, description="SQL file threshold")
    sql_preview_lines: Optional[int] = Field(default=None, description="SQL preview lines")


class CreateAgentData(BaseModel):
    """Create agent result data."""

    name: str = Field(..., description="Created agent name")


# ========== Get Agent ==========


class GetAgentInput(BaseModel):
    """Get agent input."""

    name: str = Field(..., description="Agent name")


class IAgentInfo(BaseModel):
    """Detailed agent information."""

    name: str = Field(..., description="Agent name")
    type: str = Field(..., description="Agent type (builtin/customize)")
    config_yaml: str = Field(..., description="Agent configuration YAML")
    system_prompt: str = Field(..., description="System prompt")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    catalogs: List[str] = Field(default_factory=list, description="Catalog access patterns")
    subjects: List[str] = Field(default_factory=list, description="Subject access patterns")
    rules: List[str] = Field(default_factory=list, description="Additional rules")
    created_at: str = Field(..., description="Creation timestamp")


class GetAgentData(BaseModel):
    """Get agent result data."""

    agent: IAgentInfo


# ========== Edit Agent ==========


class EditAgentInput(BaseModel):
    """Input for editing an existing sub-agent. Only provided fields are updated."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Agent name to edit")
    description: Optional[str] = None
    prompt_template: Optional[str] = Field(default=None, alias="system_prompt")
    prompt_version: Optional[str] = Field(default=None, description="Prompt version (None = latest)")
    prompt_language: Optional[str] = None
    tools: Optional[List[str]] = None
    mcp: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    scoped_context: Optional[dict] = None
    permissions: Optional[dict] = None
    hooks: Optional[dict] = None
    rules: Optional[list[str]] = None
    max_turns: Optional[int] = None
    workspace_root: Optional[str] = None
    adapter_type: Optional[str] = None
    sql_file_threshold: Optional[int] = None
    sql_preview_lines: Optional[int] = None
