# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Conversational analytics nodes with shared builtin capability policies.

Hosts supply independent context, artifact output and session roots.
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
            if definition.tool_id in allowed
            or definition.access == "session"
            or definition.category == "skills"
            or (definition.tool_id == "sub_agent_tools.task" and self.NODE_NAME == "analytics_builder")
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

    def _selected_tool_families(self) -> set:
        return super()._selected_tool_families() | {"sub_agent_tools"}

    def _setup_sub_agent_task_tool(self):
        allowed = {"gen_visual_report", "gen_visual_dashboard"}
        configured = self.node_config.get("subagents")
        from datus.schemas.agent_models import SubAgentConfig

        normalized = SubAgentConfig(subagents=configured).subagent_list
        selected = allowed if configured is None or normalized == ["*"] else allowed & set(normalized)
        self.node_config["subagents"] = ",".join(sorted(selected))
        super()._setup_sub_agent_task_tool()

    def _resolve_workspace_root(self) -> str:
        output = getattr(self.agent_config, "artifact_output_root", None)
        return str(output) if output else super()._resolve_workspace_root()

    def _make_filesystem_tool(self, **kwargs):
        if getattr(self.agent_config, "artifact_output_root", None):
            from functools import partial

            from datus.utils.artifact_output import artifact_mutation_guard

            kwargs["mutation_guard"] = partial(artifact_mutation_guard, self.agent_config)
        return super()._make_filesystem_tool(**kwargs)
