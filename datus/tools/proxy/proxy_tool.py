# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Proxy tool wrapper for print mode.

Replaces real tool invocations with a channel-based proxy that waits for
results from stdin, enabling external callers to provide tool results.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from agents import FunctionTool
from agents.tool_context import ToolContext

from datus.tools.proxy.tool_result_channel import ToolResultChannel
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)


def create_proxy_tool(original: FunctionTool, channel: ToolResultChannel) -> FunctionTool:
    """Wrap a FunctionTool so it awaits results from the channel instead of executing."""

    async def proxy_invoke(tool_ctx: ToolContext, args_str: str) -> dict:
        call_id = tool_ctx.tool_call_id
        logger.debug(f"Proxy tool '{original.name}' waiting for result, call_id={call_id}")
        try:
            return await channel.wait_for(call_id)
        except RuntimeError as e:
            logger.warning(f"Proxy tool '{original.name}' error: {e}, call_id={call_id}")
            return {"success": 0, "error": str(e), "result": None}

    return FunctionTool(
        name=original.name,
        description=original.description,
        params_json_schema=original.params_json_schema,
        on_invoke_tool=proxy_invoke,
    )


def apply_proxy_tools(node: AgenticNode, proxy_patterns: List[str]) -> None:
    """Replace matching tools on the node with proxy wrappers.

    Args:
        node: AgenticNode instance (must have .tools and .tool_channel)
        proxy_patterns: List of patterns like ``"filesystem_tools.*"`` or ``"read_file"``
    """
    parsed = _parse_patterns(proxy_patterns)
    registry = node.tool_registry.to_dict()

    new_tools = []
    for tool in node.tools:
        if isinstance(tool, FunctionTool) and _matches(tool.name, registry, parsed):
            logger.info(f"Proxying tool: {tool.name}")
            new_tools.append(create_proxy_tool(tool, node.tool_channel))
        else:
            new_tools.append(tool)
    node.tools = new_tools


# ── Internal helpers ─────────────────────────────────────────────────


def _parse_patterns(patterns: List[str]) -> List[Tuple[Optional[str], str]]:
    """Parse ``"category.method_glob"`` patterns into ``(category, method_glob)`` tuples.

    - ``"filesystem_tools.*"``  → ``("filesystem_tools", "*")``
    - ``"read_file"``           → ``(None, "read_file")``
    - ``"*"``                   → ``(None, "*")``
    """
    result: List[Tuple[Optional[str], str]] = []
    for p in patterns:
        if "." in p:
            cat, method = p.split(".", 1)
            result.append((cat, method))
        else:
            result.append((None, p))
    return result


def _matches(tool_name: str, registry: Dict[str, str], patterns: List[Tuple[Optional[str], str]]) -> bool:
    """Check if a tool name matches any of the parsed patterns."""
    category = registry.get(tool_name)

    for pat_cat, pat_method in patterns:
        if pat_cat is not None:
            # Category-qualified pattern: both category and method must match
            if category and fnmatch(category, pat_cat) and fnmatch(tool_name, pat_method):
                return True
        else:
            # Bare pattern: match against tool name directly
            if fnmatch(tool_name, pat_method):
                return True

    return False
