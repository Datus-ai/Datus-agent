# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Resolve configured and project-enabled plugin agents through one definition path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jsonschema

from datus.plugins.agents import PluginAgentDefinition
from datus.plugins.registry import collect_plugin_agents
from datus.schemas.agent_spec import AgentSpec
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.exceptions import DatusException, ErrorCode

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig


@dataclass(frozen=True)
class ResolvedAgentDefinition:
    """Keep the publisher's declaration alongside the instance's selected configuration."""

    spec: AgentSpec
    plugin: PluginAgentDefinition | None = None
    runtime_config: dict[str, Any] | None = None

    def validate_runtime_config(self) -> None:
        """Check required instance parameters when executing, not while discovering metadata."""
        if self.plugin is None or self.plugin.config_schema is None:
            return
        try:
            jsonschema.validate(self.runtime_config or {}, self.plugin.config_schema)
        except jsonschema.ValidationError as exc:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"Plugin agent {self.spec.name!r} runtime configuration is invalid at {list(exc.path)}: "
                    f"failed {exc.validator} validation."
                ),
            ) from exc


def is_configurable_entry(config: Any) -> bool:
    """Keep existing typed nodes on their existing configuration and execution paths."""
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    return isinstance(config, dict) and (config.get("node_class") or config.get("type")) == "configurable"


def _node_config(name: str, agent_config: AgentConfig) -> dict[str, Any]:
    nodes = getattr(agent_config, "agentic_nodes", None) or {}
    config = nodes.get(name, {})
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    if not isinstance(config, dict):
        raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message=f"Agent {name!r} must be a configuration mapping.")
    return config


def resolve_configurable_agent(name: str, agent_config: AgentConfig) -> ResolvedAgentDefinition | None:
    """Resolve a definition, rejecting collisions and missing explicit plugin references."""
    config = _node_config(name, agent_config)
    configured = is_configurable_entry(config)
    if configured and "resources" in config:
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message="Configure business access through scoped_context; resources is not supported.",
        )
    plugins = collect_plugin_agents(agent_config)
    source = config.get("source") if configured else None
    plugin = None

    if configured and any(key in config for key in ("runtime_factory", "tool_definitions", "instructions_file")):
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Agent {name!r} executable code and package resources must come from a registered plugin source.",
        )

    if source is not None:
        if (
            not isinstance(source, dict)
            or set(source) != {"plugin", "agent"}
            or any(not isinstance(value, str) or not value.strip() for value in source.values())
        ):
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Agent {name!r} source requires exactly plugin and agent references.",
            )
        plugin = plugins.get(source["agent"])
        if plugin is None or plugin.plugin_name != source["plugin"]:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Agent {name!r} references a missing or disabled plugin agent.",
            )
    elif configured:
        if name in plugins:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Configured agent {name!r} conflicts with a plugin agent; declare its source explicitly.",
            )
    elif name in plugins:
        if config or name in SYS_SUB_AGENTS:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Plugin agent {name!r} conflicts with an existing node; configure an explicit source.",
            )
        plugin = plugins[name]
    else:
        return None

    behavior = plugin.spec.model_dump() if plugin else {}
    behavior.update({key: value for key, value in config.items() if key in AgentSpec.model_fields})
    behavior["name"] = name
    spec = AgentSpec.model_validate(behavior)
    if "name" in config and config["name"] != name:
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Agent {name!r} cannot replace its registered name with {config['name']!r}.",
        )
    runtime_config = dict(plugin.config) if plugin else {}
    overrides = config.get("runtime_config", {})
    if not isinstance(overrides, dict) or (overrides and plugin is None):
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Agent {name!r} runtime_config requires a registered plugin source and a mapping.",
        )
    runtime_config.update(overrides)
    return ResolvedAgentDefinition(spec, plugin, runtime_config)


def list_configurable_agents(agent_config: AgentConfig) -> dict[str, ResolvedAgentDefinition]:
    """Use the same resolution for discovery, direct calls and task delegation."""
    names = set(collect_plugin_agents(agent_config))
    nodes = getattr(agent_config, "agentic_nodes", None) or {}
    names.update(name for name, config in nodes.items() if is_configurable_entry(config))
    definitions = {}
    for name in sorted(names):
        definition = resolve_configurable_agent(name, agent_config)
        if definition is not None:
            definitions[name] = definition
    return definitions
