"""Data models for Agent API endpoints."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    type: str = Field(
        default="gen_sql", description="Node class: gen_sql / gen_report / ask_report / ask_dashboard / ..."
    )
    artifact_slug: Optional[str] = Field(
        default=None,
        description=(
            "Slug of the visual report / dashboard this agent is bound to. "
            "Required when ``type`` is ``ask_report`` or ``ask_dashboard``; ignored "
            "for other types. Must match ``^[a-z0-9_]{1,80}$``. The matching "
            "``reports/<slug>`` or ``dashboards/<slug>`` directory must already exist."
        ),
    )
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

    @field_validator("artifact_slug", mode="before")
    @classmethod
    def _strip_artifact_slug(cls, value):
        """Trim incidental whitespace before validation / persistence.

        Done at the model layer so every downstream reader (the ask-binding
        validator, the agentic_nodes persistence write, the SaaS-side
        uniqueness check) sees the same normalised value — avoids
        whitespace-in-stored-slug mismatches with the regex-validated form.
        """
        if isinstance(value, str):
            return value.strip() or None
        return value


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


# ========== Channels ==========


class ChannelInput(BaseModel):
    """One IM gateway channel binding submitted with an agent edit.

    Each channel carries its own ``enabled`` switch (adapters toggle
    independently) and a generic ``secrets`` bag whose keys depend on the
    adapter ``type`` — e.g. ``app_token`` / ``bot_token`` for slack. Values are
    written verbatim into the ``agent.yml`` ``channels`` section (no
    encryption); ``${ENV_VAR}`` placeholders are resolved at load time.
    """

    enabled: bool = Field(True, description="Whether this channel's gateway adapter is active.")
    type: str = Field(..., description="Adapter type, e.g. 'slack'.")
    name: str = Field(..., min_length=1, description="Reference name, used as the channel config key.")
    secrets: dict = Field(
        default_factory=dict,
        description="Adapter-specific secret fields (e.g. app_token/bot_token). Written into `extra` verbatim.",
    )


class ChannelBinding(BaseModel):
    """The channel state for one sub-agent.

    Carries the full desired channel list for the agent being edited; the
    backend replaces only that agent's entries in the global ``channels`` map
    (matched by ``subagent_id``). An empty list clears this agent's channels.
    """

    channels: Optional[List[ChannelInput]] = Field(default=None, description="Channel bindings for this agent.")


# ========== Edit Agent ==========


class EditAgentInput(BaseModel):
    """Input for editing an existing sub-agent. Only provided fields are updated."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Agent id to edit")
    name: Optional[str] = Field(default=None, description="Agent name to edit")
    description: Optional[str] = None
    prompt_template: Optional[str] = Field(default=None, alias="system_prompt")
    prompt_version: Optional[str] = Field(default=None, description="Prompt version (None = latest)")
    prompt_language: Optional[str] = None
    tools: Optional[List[str]] = None
    mcp: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    scoped_context: Optional[dict] = None
    permissions: Optional[dict] = None
    catalogs: Optional[List[str]] = Field(
        default=None,
        description="Catalog access patterns (e.g., 'production_db.*', 'production_db.public.*')",
    )
    subjects: Optional[List[str]] = Field(
        default=None, description="Subject access patterns (e.g., 'Finance.Revenue.*')"
    )
    hooks: Optional[dict] = None
    rules: Optional[list[str]] = None
    max_turns: Optional[int] = None
    workspace_root: Optional[str] = None
    adapter_type: Optional[str] = None
    sql_file_threshold: Optional[int] = None
    sql_preview_lines: Optional[int] = None
    artifact_slug: Optional[str] = Field(
        default=None,
        description=(
            "Echoed by the SaaS UI when re-saving an ask_report / ask_dashboard "
            "agent. The backend treats this field as IMMUTABLE — if it differs "
            "from the persisted binding the request is rejected with "
            "``IMMUTABLE_FIELD``. Identical / omitted values are silently dropped."
        ),
    )
    channels: Optional[ChannelBinding] = Field(
        default=None,
        description="IM gateway channel bindings for this agent. Omit to leave channels untouched.",
    )
