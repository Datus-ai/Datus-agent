# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Builtin Subagent capabilities shared by runtime nodes and deployment hosts.

Contracts describe effects; type policies choose capabilities. Hosts may narrow
these policies further and decide whether persisted tool selections are honored.
MCP and Skill authorization remain independent of these native data-tool lists.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from datus.tools.registry.capabilities import tool_capability

GEN_SQL_DEFAULT_TOOLS = ("db_tools.*", "semantic_tools.*", "context_search_tools.*")
ASK_METRICS_DEFAULT_TOOLS = (
    "context_search_tools.search_metrics",
    "semantic_tools.list_metrics",
    "semantic_tools.get_metric",
    "semantic_tools.get_dataset",
    "semantic_tools.query_metrics",
    "semantic_tools.attribution_analyze",
    "context_search_tools.list_subject_tree",
    "filesystem_tools.read_file",
    "filesystem_tools.glob",
    "filesystem_tools.grep",
)
_DB_READ_TOOLS = tuple(
    f"db_tools.{name}"
    for name in ("list_databases", "list_schemas", "list_tables", "describe_table", "search_table", "execute_sql")
)
# Curated even though validate_semantic also declares access=read. A contract
# alone does not express which operations fit this conversational role.
_SEMANTIC_READ_TOOLS = tuple(
    f"semantic_tools.{name}"
    for name in (
        "list_metrics",
        "get_metric",
        "get_dataset",
        "query_metrics",
        "get_query_metrics_result",
        "attribution_analyze",
        "describe_concept",
        "find_concept_paths",
        "select_records",
    )
)
_FILE_READ_TOOLS = tuple(f"filesystem_tools.{name}" for name in ("read_file", "glob", "grep"))
_FILE_WRITE_TOOLS = ("filesystem_tools.write_file", "filesystem_tools.edit_file")


@dataclass(frozen=True)
class SubagentPolicy:
    tools: tuple[str, ...]
    subagents: tuple[str, ...] = ()
    db_read_only: bool | None = None
    filesystem_access: str | None = None


SUBAGENT_POLICIES = {
    "analytics_explorer": SubagentPolicy(
        (*_DB_READ_TOOLS, *_SEMANTIC_READ_TOOLS, *_FILE_READ_TOOLS),
        db_read_only=True,
        filesystem_access="read_only",
    ),
    "analytics_builder": SubagentPolicy(
        (*_DB_READ_TOOLS, *_SEMANTIC_READ_TOOLS, *_FILE_READ_TOOLS, *_FILE_WRITE_TOOLS),
        db_read_only=True,
        filesystem_access="isolated_read_write",
    ),
    "gen_sql": SubagentPolicy((*GEN_SQL_DEFAULT_TOOLS, "filesystem_tools.read_file", *_FILE_WRITE_TOOLS)),
    "ask_metrics": SubagentPolicy((*ASK_METRICS_DEFAULT_TOOLS, *_FILE_WRITE_TOOLS)),
}
# Capability discovery only; hosts decide which types override saved choices.
SUPPORTED_POLICY_TYPES = frozenset(SUBAGENT_POLICIES)


def tool_patterns(raw: Any) -> tuple[str, ...]:
    entries = raw.split(",") if isinstance(raw, str) else raw if isinstance(raw, (list, tuple)) else [raw]
    return tuple(str(entry).strip() for entry in entries if entry is not None and str(entry).strip())


def _matches(tool_id: str, patterns: Iterable[str]) -> bool:
    family = tool_id.split(".", 1)[0]
    return any(pattern in (tool_id, family, f"{family}.*") for pattern in patterns)


def resolve_tools(agent_type: str, available: Iterable[str], *, configured: Any = None) -> list[str]:
    """Intersect provider tools, the type policy, contracts and optional host scope.

    direct_call is deliberately irrelevant inside an agent turn. SQL needs the
    runtime DB read-only guard; an access=sql declaration is not a read guarantee.
    """
    policy = SUBAGENT_POLICIES[agent_type]
    scope = tool_patterns(configured) if configured is not None else policy.tools
    resolved = []
    for tool_id in dict.fromkeys(available):
        capability = tool_capability(tool_id)
        if capability is None or not _matches(tool_id, policy.tools) or not _matches(tool_id, scope):
            continue
        if policy.db_read_only:
            allowed = {"read"}
            if tool_id == "db_tools.execute_sql":
                allowed.add("sql")
            if policy.filesystem_access == "isolated_read_write" and tool_id in _FILE_WRITE_TOOLS:
                allowed.add("write")
            if capability.access not in allowed:
                continue
        resolved.append(tool_id)
    return resolved


def _provider_tools(agent_type: str) -> list[str]:
    # Only providers mounted by these node types; the global validation catalog
    # also contains authoring providers and must not be used for expansion.
    from datus.tools.func_tool.context_search import ContextSearchTools
    from datus.tools.func_tool.database import DBFuncTool
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
    from datus.tools.func_tool.semantic_tools import SemanticTools

    providers = {
        "db_tools": DBFuncTool,
        "semantic_tools": SemanticTools,
        "context_search_tools": ContextSearchTools,
        "filesystem_tools": FilesystemFuncTool,
    }
    candidates = [f"{family}.{name}" for family, provider in providers.items() for name in provider.all_tools_name()]
    if agent_type in {"analytics_explorer", "analytics_builder"}:
        candidates.append("semantic_tools.get_query_metrics_result")
    return candidates


def builtin_tools(agent_type: str) -> list[str]:
    """Return host configuration, retaining legacy wildcard mounting semantics."""
    policy = SUBAGENT_POLICIES[agent_type]
    resolved = set(resolve_tools(agent_type, _provider_tools(agent_type)))
    return [pattern for pattern in policy.tools if any(_matches(tool_id, (pattern,)) for tool_id in resolved)]


def tool_reference(agent_type: str) -> dict[str, Any]:
    """Describe the type's supported tools; runtime may lack optional providers."""
    groups: dict[str, dict[str, list[str]]] = {}
    for tool_id in sorted(resolve_tools(agent_type, _provider_tools(agent_type))):
        family, method = tool_id.split(".", 1)
        groups.setdefault(family, {"tools": []})["tools"].append(method)
    return {"default_tools": builtin_tools(agent_type), "tool_types": groups}


def builtin_node_config(agent_type: str) -> dict[str, Any]:
    policy = SUBAGENT_POLICIES[agent_type]
    result: dict[str, Any] = {"tools": ",".join(builtin_tools(agent_type)), "subagents": ",".join(policy.subagents)}
    if policy.db_read_only is not None:
        result["db_read_only"] = policy.db_read_only
    if policy.filesystem_access is not None:
        # Definition only. Directory isolation is implemented in a later PR.
        result["filesystem_access"] = policy.filesystem_access
    return result
