# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Proxy tool wrapper for print mode.

Replaces real tool invocations with a channel-based proxy that waits for
results from stdin, enabling external callers to provide tool results.
"""

from __future__ import annotations

import asyncio
import time
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from agents import FunctionTool
from agents.tool_context import ToolContext

from datus.tools.proxy.tool_result_channel import DEFAULT_RESULT_TIMEOUT, ToolResultChannel
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)

# Node types whose GenerationHooks depend on local filesystem tools (write_file, etc.)
# Their filesystem_tools must NOT be proxied; other tools are still proxied.
# gen_visual_report / gen_visual_dashboard author their render/*.jsx tree
# server-side and surface it via ``/api/v1/(report|dashboard)/detail``; proxying
# write_file / edit_file to the browser would round-trip every chunk for no
# benefit, so the same exclusion applies to them.
_FS_DEPENDENT_NODES: Set[str] = {
    "gen_semantic_model",
    "gen_metrics",
    "gen_sql_summary",
    "gen_visual_report",
    "gen_visual_dashboard",
}


def create_proxy_tool(original: FunctionTool, channel: ToolResultChannel) -> FunctionTool:
    """Wrap a FunctionTool so it awaits results from the channel instead of executing."""

    async def proxy_invoke(tool_ctx: ToolContext, args_str: str) -> dict:
        call_id = tool_ctx.tool_call_id
        started = time.monotonic()
        logger.info(
            f"Proxy tool '{original.name}' awaiting client result, call_id={call_id}, timeout={DEFAULT_RESULT_TIMEOUT}s"
        )
        try:
            result = await channel.wait_for(call_id, timeout=DEFAULT_RESULT_TIMEOUT)
            waited_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                f"Proxy tool '{original.name}' received client result, call_id={call_id}, waited_ms={waited_ms}"
            )
            return result
        except asyncio.TimeoutError:
            waited_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                f"Proxy tool '{original.name}' timed out after {DEFAULT_RESULT_TIMEOUT}s "
                f"waiting for client result, call_id={call_id}, waited_ms={waited_ms}"
            )
            return {
                "success": 0,
                "error": (
                    f"Timed out after {int(DEFAULT_RESULT_TIMEOUT)}s waiting for the client "
                    f"to report the result of '{original.name}'."
                ),
                "result": None,
            }
        except RuntimeError as e:
            logger.warning(f"Proxy tool '{original.name}' error: {e}, call_id={call_id}")
            return {"success": 0, "error": str(e), "result": None}

    return FunctionTool(
        name=original.name,
        description=original.description,
        params_json_schema=original.params_json_schema,
        on_invoke_tool=proxy_invoke,
        strict_json_schema=original.strict_json_schema,
    )


def apply_proxy_tools(
    node: AgenticNode, proxy_patterns: List[str], channel: Optional[ToolResultChannel] = None
) -> None:
    """Replace matching tools on the node with proxy wrappers.

    Args:
        node: AgenticNode instance (must have .tools and .tool_channel)
        proxy_patterns: List of patterns like ``"filesystem_tools.*"`` or ``"read_file"``
        channel: Explicit ToolResultChannel to use. When *None* (default),
                 falls back to ``node.tool_channel``.  Sub-agents pass the
                 parent's channel so that stdin dispatch can resolve their
                 proxy futures.
    """
    node.proxy_tool_patterns = proxy_patterns
    target_channel = channel or node.tool_channel
    # ``_FS_DEPENDENT_NODES`` exclusion below relies on ``tool_registry`` being
    # populated. In production that population is normally driven by
    # ``_ensure_permission_hooks``, which runs lazily on the first LLM turn —
    # but ``apply_proxy_tools`` is called during node setup, well before any
    # LLM call. Trigger registry population eagerly so the exclusion can fire.
    # Guarded with ``hasattr`` so the SimpleNamespace-style mocks in tests
    # still work without growing a no-op stub.
    if hasattr(node, "_populate_tool_registry"):
        node._populate_tool_registry()
    parsed = _parse_patterns(proxy_patterns)
    registry = node.tool_registry.to_dict()

    # Auto-detect nodes whose GenerationHooks depend on filesystem tools
    exclude_categories: Optional[Set[str]] = None
    node_name = getattr(node, "get_node_name", lambda: "")()
    if node_name in _FS_DEPENDENT_NODES:
        exclude_categories = {"filesystem_tools"}

    new_tools = []
    proxied_names: Set[str] = set()
    for tool in node.tools:
        if isinstance(tool, FunctionTool) and _matches(tool.name, registry, parsed):
            if exclude_categories and registry.get(tool.name) in exclude_categories:
                logger.info(f"Skipping proxy for tool '{tool.name}' (excluded category on node '{node_name}')")
                new_tools.append(tool)
            else:
                logger.info(f"Proxying tool: {tool.name}")
                new_tools.append(create_proxy_tool(tool, target_channel))
                proxied_names.add(tool.name)
        else:
            new_tools.append(tool)
    node.tools = new_tools
    # ``PermissionHooks`` consults this set in ``on_tool_start`` to skip the
    # ALLOW/ASK/DENY check for proxied tools — the external caller (e.g.
    # ``print_mode`` stdin protocol) owns secondary confirmation. Mutate in
    # place so existing PermissionHooks instances built earlier still see the
    # latest snapshot via the shared reference held on the node.
    existing = getattr(node, "proxied_tool_names", None)
    if isinstance(existing, set):
        existing.clear()
        existing.update(proxied_names)
    else:
        node.proxied_tool_names = proxied_names


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
