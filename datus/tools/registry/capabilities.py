# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Query the build-generated tool inventory without importing implementations.

Methods declare their contract beside their implementation. The catalog compiler
and packaging hook generate capabilities.json; this module has no manual method
list. Restricted families deny mutations and unknown methods, not declared reads
or session-local interaction channels.
"""

import json
from pathlib import Path

from datus.tools.registry.contracts import Access as Access
from datus.tools.registry.contracts import ToolCapability, ToolDefinitionError, validate_tool_id

DENIED_TOOL_FAMILIES = frozenset(
    {
        "bash_tools",
        "scheduler_tools",
        "skill_tools",
        "skills",
        "memory_tools",
        "visual_report_tools",
        "visual_dashboard_tools",
        "export_tools",
        "bi_tools",
        "artifact_tools",
        "sub_agent_tools",
        "web_tool",
        "mcp",
        "tools",
    }
)


def _load_catalog() -> dict[str, ToolCapability]:
    """Load dependency-free package data and reject invalid release artifacts."""
    document = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("tools"), dict):
        raise ToolDefinitionError("Unsupported tool catalog; rebuild the agent package")
    catalog = {}
    for tool_id, capability in document["tools"].items():
        validate_tool_id(tool_id)
        catalog[tool_id] = ToolCapability(**capability)
    return catalog


TOOL_CAPABILITIES = _load_catalog()


def tool_capability(tool_id: str) -> ToolCapability | None:
    """Resolve safe declarations before applying the family's mutation restriction."""
    capability = TOOL_CAPABILITIES.get(tool_id)
    if tool_id.split(".", 1)[0] in DENIED_TOOL_FAMILIES and (
        capability is None or capability.access not in {"read", "session"}
    ):
        return ToolCapability("restricted", direct_call=False)
    return capability
