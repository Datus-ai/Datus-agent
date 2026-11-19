# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Host-side validation of plain-data plugin agents and runtime preparations.

Plugins declare metadata in YAML and return ordinary dictionaries and callables.
These models belong to the host; plugins neither import them nor subclass nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import jsonschema
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from datus.schemas.agent_models import ScopedContext
from datus.schemas.agent_spec import AgentSpec, MCPToolSelection, _identifiers, validate_scoped_context_config
from datus.tools.registry.contracts import (
    Access,
    ToolCapability,
    ToolDefinition,
    plugin_tool_definition,
    validate_tool_id,
)


def _check_schema(schema: dict[str, Any]) -> None:
    if "$schema" in schema and not isinstance(schema["$schema"], str):
        raise ValueError("JSON schema dialect must be a string")
    validator = validator_for(schema, default=None) if "$schema" in schema else jsonschema.Draft202012Validator
    if validator is None:
        raise ValueError("JSON schema declares an unsupported dialect")
    validator.check_schema(schema)


class PluginToolDeclaration(BaseModel):
    """A declared tool contract, separated from the instance-bound implementation."""

    model_config = ConfigDict(extra="forbid")

    tool_id: str
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    access: Access
    direct_call: bool = False
    strict: bool = False

    @model_validator(mode="after")
    def validate_contract(self) -> PluginToolDeclaration:
        """Require valid identities, effect metadata and object-shaped JSON arguments."""
        validate_tool_id(self.tool_id)
        ToolCapability(self.access, self.direct_call)
        if self.input_schema.get("type") != "object":
            raise ValueError("Plugin tool input_schema must describe an object")
        _check_schema(self.input_schema)
        return self

    @property
    def definition(self) -> ToolDefinition:
        """Build the host's normal registration evidence without decorating plugin functions."""
        return plugin_tool_definition(self.tool_id, self.access, self.direct_call)


class PluginAgentDefinition(BaseModel):
    """One validated agent from an installed manifest; no plugin code has run yet."""

    model_config = ConfigDict(extra="forbid")

    plugin_name: str
    package_dir: Path
    spec: AgentSpec
    protocol_version: Literal[1] = 1
    runtime_factory: str | None = None
    instructions_file: str | None = None
    config_schema: dict[str, Any] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    tool_definitions: list[PluginToolDeclaration] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_definition(self) -> PluginAgentDefinition:
        """Validate factories and resources without importing the referenced package."""
        if self.runtime_factory is not None:
            from datus.plugins.base import parse_code_ref

            if parse_code_ref(self.runtime_factory) is None:
                raise ValueError("runtime_factory must be a module:function reference")
        if self.instructions_file is not None:
            root = self.package_dir.resolve()
            try:
                resource = (root / self.instructions_file).resolve()
            except (OSError, RuntimeError) as exc:
                raise ValueError("instructions_file could not be resolved inside the plugin package") from exc
            if not resource.is_relative_to(root) or not resource.is_file():
                raise ValueError("instructions_file must name a file inside the plugin package")
        if self.config_schema is not None:
            _check_schema(self.config_schema)
        identities = [tool.tool_id for tool in self.tool_definitions]
        definitions = [tool.definition for tool in self.tool_definitions]
        if self.plugin_name != "datus-semantic-plugin" and any(
            tool.category == "ontology_tools" for tool in definitions
        ):
            raise ValueError("The ontology_tools permission family is reserved for datus-semantic-plugin")
        names = [tool.name for tool in definitions]
        if len(identities) != len(set(identities)) or len(names) != len(set(names)):
            raise ValueError("Plugin agent tool identities and model-visible names must be unique")
        if self.tool_definitions and self.runtime_factory is None:
            raise ValueError("Plugin tool declarations require a runtime_factory to bind their implementations")
        return self

    @classmethod
    def from_manifest(
        cls, plugin_name: str, package_dir: Path, name: str, declaration: dict[str, Any]
    ) -> PluginAgentDefinition:
        """Split common configuration from publisher-only runtime metadata."""
        behavior = {key: value for key, value in declaration.items() if key in AgentSpec.model_fields}
        behavior.setdefault("name", name)
        if behavior["name"] != name:
            raise ValueError("Agent declaration name must match its manifest key")
        extension = {key: value for key, value in declaration.items() if key not in AgentSpec.model_fields}
        return cls.model_validate(
            {**extension, "plugin_name": plugin_name, "package_dir": package_dir, "spec": behavior}
        )


class RuntimePreparation(BaseModel):
    """Dynamic contributions returned by prepare_turn; never an authorization grant."""

    model_config = ConfigDict(extra="forbid")

    instructions: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    tool_bindings: dict[str, Callable[..., Any]] = Field(default_factory=dict)
    mcp: list[MCPToolSelection] | None = None
    skills: list[StrictStr] | None = None
    scoped_context: ScopedContext | None = None

    @field_validator("scoped_context", mode="before")
    @classmethod
    def validate_scope(cls, value: Any) -> Any:
        return validate_scoped_context_config(value)

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, value: list[str] | None) -> list[str] | None:
        """Retain explicit selections for the host's subset check."""
        return None if value is None else _identifiers(value)

    @model_validator(mode="after")
    def validate_connections(self) -> RuntimePreparation:
        """Apply the same unambiguous connection selection contract as AgentSpec."""
        if self.mcp is not None:
            names = [selection.connection_ref for selection in self.mcp]
            if len(names) != len(set(names)):
                raise ValueError("Prepared MCP connections must be unique")
        return self


def parse_agent_declarations(
    raw: Any, plugin_name: str, package_dir: Path
) -> tuple[dict[str, PluginAgentDefinition], dict[str, str]]:
    """Parse metadata without disabling unrelated CLI/skill features of an old plugin."""
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        return {}, {"*": "Plugin agents must be a mapping of names to definitions"}
    definitions = {}
    errors = {}
    for name, declaration in raw.items():
        try:
            if not isinstance(name, str) or not isinstance(declaration, dict):
                raise ValueError("Each plugin agent requires a string name and a definition mapping")
            definitions[name] = PluginAgentDefinition.from_manifest(plugin_name, package_dir, name, declaration)
        except (ValueError, TypeError, jsonschema.SchemaError, jsonschema.ValidationError) as exc:
            errors[str(name)] = str(exc)
    return definitions, errors
