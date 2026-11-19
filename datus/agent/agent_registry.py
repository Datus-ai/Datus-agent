# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Resolve configured and project-enabled plugin agents through one definition path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jsonschema
from referencing import Registry
from referencing.exceptions import Unresolvable

from datus.plugins.agents import PluginAgentDefinition
from datus.plugins.registry import collect_plugin_agents
from datus.schemas.agent_spec import AgentSpec
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.node_utils import validate_node_class

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)

# Also include built-in dispatch names kept out of the public subagent menu.
_BUILTIN_DISPATCH_NAMES = SYS_SUB_AGENTS | {
    "analytics_builder",
    "analytics_explorer",
    "ask_dashboard",
    "ask_report",
    "chat",
    "explore",
}


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
            jsonschema.validate(self.runtime_config or {}, self.plugin.config_schema, registry=Registry())
        except jsonschema.ValidationError as exc:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"Plugin agent {self.spec.name!r} runtime configuration is invalid at {list(exc.path)}: "
                    f"failed {exc.validator} validation."
                ),
            ) from exc
        except Unresolvable as exc:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Plugin agent {self.spec.name!r} config_schema has an unresolvable reference.",
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


def resolve_configurable_agent(
    name: str, agent_config: AgentConfig, *, plugin_agents: dict[str, PluginAgentDefinition] | None = None
) -> ResolvedAgentDefinition | None:
    """Resolve a definition, rejecting collisions and missing explicit plugin references."""
    config = _node_config(name, agent_config)
    configured = is_configurable_entry(config)
    if not configured and name in _BUILTIN_DISPATCH_NAMES:
        return None
    if configured:
        validate_node_class("configurable", name)
    if configured and "resources" in config:
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message="Configure business access through scoped_context; resources is not supported.",
        )
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
    if plugin_agents is None:
        plugin_agents = collect_plugin_agents(
            agent_config, source["agent"] if source is not None else name, strict=source is not None or not configured
        )

    if source is not None:
        plugin = plugin_agents.get(source["agent"])
        if plugin is None or plugin.plugin_name != source["plugin"]:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Agent {name!r} references a missing or disabled plugin agent.",
            )
    elif configured:
        if name in plugin_agents:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Configured agent {name!r} conflicts with a plugin agent; declare its source explicitly.",
            )
    elif name in plugin_agents:
        if config:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Plugin agent {name!r} conflicts with an existing node; configure an explicit source.",
            )
        validate_node_class("configurable", name)
        plugin = plugin_agents[name]
    else:
        return None

    behavior = plugin.spec.model_dump() if plugin else {}
    behavior.update({key: value for key, value in config.items() if key in AgentSpec.model_fields})
    behavior["name"] = name
    spec = AgentSpec.model_validate(behavior)
    scope = spec.scoped_context
    datasources = getattr(getattr(agent_config, "services", None), "datasources", None)
    if scope is not None and isinstance(datasources, dict):
        requested = set(scope.tables_by_datasource or {}) | ({scope.datasource} if scope.datasource else set())
        if missing := requested - datasources.keys():
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Datasources are unavailable in this project: {sorted(missing)}",
            )
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
    """Discover available definitions while excluding individually invalid entries."""
    plugin_agents = collect_plugin_agents(agent_config, strict=False)
    names = set(plugin_agents)
    nodes = getattr(agent_config, "agentic_nodes", None) or {}
    names.update(name for name, config in nodes.items() if is_configurable_entry(config))
    definitions = {}
    for name in sorted(names):
        try:
            definition = resolve_configurable_agent(name, agent_config, plugin_agents=plugin_agents)
        except (DatusException, ValueError) as exc:
            logger.warning("Skipping unavailable configurable agent %r: %s", name, exc)
            continue
        if definition is not None:
            definitions[name] = definition
    return definitions
