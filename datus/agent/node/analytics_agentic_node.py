# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Conversational analytics nodes; deployment hosts supply the tool policy.

Artifact delegation and isolated writable workspaces are intentionally deferred.
"""

from typing import Optional

from datus.agent.node.chat_agentic_node import ChatAgenticNode


class AnalyticsExplorerAgenticNode(ChatAgenticNode):
    NODE_NAME = "analytics_explorer"
    DEFAULT_SUBAGENTS = ""
    _db_read_only = True

    def __init__(self, *args, node_name: Optional[str] = None, **kwargs):
        self._analytics_node_name = node_name or self.NODE_NAME
        super().__init__(*args, **kwargs)
        self.configured_node_name = self._analytics_node_name
        self.max_turns = self.node_config.get("max_turns", 200)

    def get_node_name(self) -> str:
        return self._analytics_node_name

    def _selected_tool_families(self) -> set:
        # No implicit full-chat surface when a host forgets the policy.
        return (super()._selected_tool_families() or set()) | {"skills"}

    def _rebuild_tools(self):
        super()._rebuild_tools()
        raw = self.node_config.get("tools") or ""
        entries = raw.split(",") if isinstance(raw, str) else raw if isinstance(raw, (list, tuple)) else [raw]
        patterns = {str(part).strip() for part in entries if str(part).strip()}
        groups = {}
        for category, attr in (
            ("db_tools", "db_func_tool"),
            ("semantic_tools", "semantic_tools"),
            ("filesystem_tools", "filesystem_func_tool"),
        ):
            instance = getattr(self, attr, None)
            if instance:
                groups.update({tool.name: category for tool in instance.available_tools()})
        groups["get_query_metrics_result"] = "semantic_tools"
        self.tools = [
            tool
            for tool in self.tools
            if tool.name not in groups
            or patterns & {groups[tool.name], f"{groups[tool.name]}.*", f"{groups[tool.name]}.{tool.name}"}
        ]

    def _get_system_prompt(self, *args, **kwargs):
        # Custom instance names must use the type's builtin template.
        if not self.node_config.get("system_prompt"):
            self.node_config["system_prompt"] = self.NODE_NAME
        return super()._get_system_prompt(*args, **kwargs)


class AnalyticsBuilderAgenticNode(AnalyticsExplorerAgenticNode):
    NODE_NAME = "analytics_builder"
