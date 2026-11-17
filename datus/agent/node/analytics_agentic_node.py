# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Conversational analytics nodes with shared builtin capability policies.

Artifact delegation and isolated writable workspaces are intentionally deferred.
"""

from typing import Optional

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.tools.registry.contracts import tool_definition
from datus.tools.registry.subagent_policy import SUBAGENT_POLICIES, builtin_tools, resolve_tools


class AnalyticsExplorerAgenticNode(ChatAgenticNode):
    NODE_NAME = "analytics_explorer"
    DEFAULT_SUBAGENTS = ",".join(SUBAGENT_POLICIES[NODE_NAME].subagents)
    _db_read_only = SUBAGENT_POLICIES[NODE_NAME].db_read_only

    def __init__(self, *args, node_name: Optional[str] = None, **kwargs):
        self._analytics_node_name = node_name or self.NODE_NAME
        super().__init__(*args, **kwargs)
        self.configured_node_name = self._analytics_node_name
        self.max_turns = self.node_config.get("max_turns", 200)

    def get_node_name(self) -> str:
        return self._analytics_node_name

    def setup_tools(self):
        # Hosts may narrow the builtin policy (for example on publications).
        # Missing configuration gets defaults; explicit empty scope stays empty.
        self.node_config.setdefault("tools", ",".join(builtin_tools(self.NODE_NAME)))
        super().setup_tools()

    def _selected_tool_families(self) -> set:
        supported = {pattern.split(".", 1)[0] for pattern in SUBAGENT_POLICIES[self.NODE_NAME].tools}
        return ((super()._selected_tool_families() or set()) & supported) | {"skills"}

    def _setup_semantic_tools(self) -> None:
        super()._setup_semantic_tools()
        if self.semantic_tools is not None:
            # Chat defers runtime loading, so enable ontology before mounting or
            # the first call initializes the runtime without the advertised tools.
            self.semantic_tools.ontology_enabled = True

    def _available_context_search_tools(self) -> list:
        # Availability can change between turns. Filter at discovery as well
        # as rebuild so lazy prompt refresh cannot widen the host's selection.
        tools = super()._available_context_search_tools()
        definitions = [tool_definition(tool) for tool in tools]
        allowed = set(
            resolve_tools(
                self.NODE_NAME,
                (definition.tool_id for definition in definitions),
                configured=self.node_config.get("tools") or "",
            )
        )
        return [tool for tool, definition in zip(tools, definitions) if definition.tool_id in allowed]

    def _rebuild_tools(self):
        super()._rebuild_tools()
        definitions = [tool_definition(tool) for tool in self.tools]
        allowed = set(
            resolve_tools(
                self.NODE_NAME,
                (definition.tool_id for definition in definitions),
                configured=self.node_config.get("tools") or "",
            )
        )
        self.tools = [
            tool
            for tool, definition in zip(self.tools, definitions)
            if definition.tool_id in allowed or definition.access == "session" or definition.category == "skills"
        ]

    def _get_system_prompt(self, *args, **kwargs):
        # Custom instance names must use the type's builtin template.
        if not self.node_config.get("system_prompt"):
            self.node_config["system_prompt"] = self.NODE_NAME
        return super()._get_system_prompt(*args, **kwargs)


class AnalyticsBuilderAgenticNode(AnalyticsExplorerAgenticNode):
    NODE_NAME = "analytics_builder"
    DEFAULT_SUBAGENTS = ",".join(SUBAGENT_POLICIES[NODE_NAME].subagents)
    _db_read_only = SUBAGENT_POLICIES[NODE_NAME].db_read_only
