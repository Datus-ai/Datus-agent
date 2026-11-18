# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Serializable definitions for config-driven agents, independent of node execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from datus.schemas.agent_models import ScopedContext


def _identifiers(value: list[str]) -> list[str]:
    """Reject empty identifiers and retain the declared order without duplicates."""
    result = []
    for item in value:
        item = item.strip()
        if not item:
            raise ValueError("Identifiers must not be empty")
        if any(separator in item for separator in (",", "\r", "\n")):
            raise ValueError("List entries must contain one identifier each")
        if item not in result:
            result.append(item)
    return result


class MCPToolSelection(BaseModel):
    """A project connection reference with an explicit tool selection, never credentials."""

    model_config = ConfigDict(extra="forbid")

    connection_ref: str = Field(min_length=1)
    tools: list[StrictStr] = Field(min_length=1)

    @field_validator("connection_ref")
    @classmethod
    def validate_connection_ref(cls, value: str) -> str:
        """Normalize references without interpreting them as addresses or commands."""
        value = value.strip()
        if not value:
            raise ValueError("An MCP connection reference is required")
        return value

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: list[str]) -> list[str]:
        """Retain only explicitly named tools, rather than enabling a whole connection."""
        return _identifiers(value)


def validate_scoped_context_config(value: Any) -> Any:
    """Reuse ScopedContext while rejecting unknown fields in the new public protocol."""
    if isinstance(value, dict):
        unknown = set(value) - set(ScopedContext.model_fields)
        if unknown:
            raise ValueError(f"Unknown scoped_context fields: {sorted(unknown)}")
    return value


class AgentOutput(BaseModel):
    """Output requirements supported by the common execution path."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["text", "markdown", "json"] = "text"
    instructions: str = ""


class AgentSpec(BaseModel):
    """One agent's behavior and explicit capabilities, with no Python factory references."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    instructions: str = ""
    skills: list[StrictStr] = Field(default_factory=list)
    tools: list[StrictStr] = Field(default_factory=list)
    mcp: list[MCPToolSelection] = Field(default_factory=list)
    scoped_context: ScopedContext | None = None
    output: AgentOutput = Field(default_factory=AgentOutput)

    @field_validator("scoped_context", mode="before")
    @classmethod
    def validate_scope(cls, value: Any) -> Any:
        return validate_scoped_context_config(value)

    @field_validator("tools", "skills")
    @classmethod
    def validate_selections(cls, value: list[str]) -> list[str]:
        """Configurable agents use lists; absence never selects default tools."""
        return _identifiers(value)

    @field_validator("mcp")
    @classmethod
    def validate_connections(cls, value: list[MCPToolSelection]) -> list[MCPToolSelection]:
        """Reject duplicate connections instead of silently replacing a tool filter."""
        names = [selection.connection_ref for selection in value]
        if len(names) != len(set(names)):
            raise ValueError("MCP connections must be listed once, with their complete tool selection")
        return value

    @classmethod
    def from_node_config(cls, name: str, config: dict[str, Any]) -> AgentSpec:
        """Read behavior fields without treating existing node routing settings as capabilities."""
        if "resources" in config:
            raise ValueError("Configure business access through scoped_context; resources is not supported")
        fields = {key: value for key, value in config.items() if key in cls.model_fields}
        declared_name = fields.setdefault("name", name)
        if declared_name != name:
            raise ValueError(f"Agent name {declared_name!r} does not match registration name {name!r}")
        return cls.model_validate(fields)
