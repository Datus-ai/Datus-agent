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
import json
import time
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from agents import FunctionTool
from agents.tool_context import ToolContext

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.tools.proxy.tool_result_channel import DEFAULT_RESULT_TIMEOUT, ToolResultChannel
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)

# Node types whose success-path finalizers depend on local filesystem tools (write_file, etc.)
# CLI/VS Code keep these filesystem tools native. Web uses the existing IDE
# executor with server-side preparation and mutation bookkeeping.
_FS_DEPENDENT_NODES: Set[str] = {
    "semantic_modeling",
    "gen_sql_summary",
    "gen_visual_report",
    "gen_visual_dashboard",
}


def create_proxy_tool(
    original: FunctionTool, channel: ToolResultChannel, *, source: Optional[str] = None
) -> FunctionTool:
    """Wrap a FunctionTool so it awaits results from the channel instead of executing."""

    prepared = {}

    async def proxy_invoke(tool_ctx: ToolContext, args_str: str) -> dict:
        call_id = tool_ctx.tool_call_id
        mutation = None
        started = time.monotonic()
        logger.info(
            f"Proxy tool '{original.name}' awaiting client result, call_id={call_id}, timeout={DEFAULT_RESULT_TIMEOUT}s"
        )
        try:
            result = await channel.wait_for(call_id, timeout=DEFAULT_RESULT_TIMEOUT)
            mutation = prepared.pop(call_id, None)
            if mutation is not None and isinstance(result, dict) and result.get("success") == 1:
                try:
                    failure = await asyncio.to_thread(mutation.complete)
                except (ValueError, OSError) as exc:
                    return {"success": 0, "error": str(exc), "result": None}
                if failure is not None:
                    return failure.model_dump()
            waited_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                f"Proxy tool '{original.name}' received client result, call_id={call_id}, waited_ms={waited_ms}"
            )
            return result
        except asyncio.TimeoutError:
            tool_ctx._datus_tool_status = "proxy_timeout"
            waited_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                f"Proxy tool '{original.name}' timed out after {DEFAULT_RESULT_TIMEOUT}s "
                f"waiting for client result, call_id={call_id}, waited_ms={waited_ms}"
            )
            return {
                "success": 0,
                "error": (
                    f"'{original.name}' expired: no result came back from the client within "
                    f"{int(DEFAULT_RESULT_TIMEOUT)}s, and this call can no longer be completed — "
                    "a confirmation pressed on that card now is dropped. Whether the client "
                    "actually performed it is UNKNOWN: it may have run and reported too late, or "
                    "the user may accept the card after this and it runs without reaching us. "
                    "Check the real state (read the file, list the directory) before concluding "
                    "anything or repeating the work. If it is still needed, issue a fresh call, "
                    "and tell the user what they have to confirm."
                ),
                "result": None,
            }
        except RuntimeError as e:
            logger.warning(f"Proxy tool '{original.name}' error: {e}, call_id={call_id}")
            return {"success": 0, "error": str(e), "result": None}
        finally:
            remaining = prepared.pop(call_id, None) or mutation
            if remaining is not None and remaining.cancel is not None:
                await asyncio.to_thread(remaining.cancel)

    proxy = FunctionTool(
        name=original.name,
        description=original.description,
        params_json_schema=original.params_json_schema,
        on_invoke_tool=proxy_invoke,
        strict_json_schema=original.strict_json_schema,
    )

    prepare = getattr(original, "_datus_prepare_client_mutation", None)
    if source == "web" and prepare is not None:

        async def prepare_action(action: ActionHistory) -> None:
            arguments = action.input.get("arguments", "{}")
            mutation = await asyncio.to_thread(prepare, arguments)
            from datus.tools.func_tool.base import FuncToolResult

            if isinstance(mutation, FuncToolResult):
                action.input = {**action.input, "_proxied": False}
                await channel.publish(action.action_id, mutation.model_dump())
            else:
                prepared[action.action_id] = mutation
                action.input = {**action.input, "arguments": json.dumps(mutation.arguments)}

        proxy._datus_prepare_proxy_action = prepare_action
    return proxy


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
    # ChatTaskManager already records the resolved client source on the request's
    # cloned AgentConfig. Subagents inherit it together with the proxy patterns.
    source = getattr(getattr(node, "agent_config", None), "_client_source", None)
    target_channel = channel or node.tool_channel
    # Web descendants share the channel exposed by the parent session's API.
    if source == "web":
        node.tool_channel = target_channel
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

    # Auto-detect nodes whose finalizers depend on filesystem tools
    exclude_categories: Optional[Set[str]] = None
    node_name = getattr(node, "get_node_name", lambda: "")()
    if source != "web" and node_name in _FS_DEPENDENT_NODES:
        exclude_categories = {"filesystem_tools"}

    new_tools = []
    proxied_names: Set[str] = set()
    web_filesystem_names: Set[str] = set()
    preparations = {}
    for tool in node.tools:
        if isinstance(tool, FunctionTool) and _matches(tool.name, registry, parsed):
            if exclude_categories and registry.get(tool.name) in exclude_categories:
                logger.info(f"Skipping proxy for tool '{tool.name}' (excluded category on node '{node_name}')")
                new_tools.append(tool)
            else:
                logger.info(f"Proxying tool: {tool.name}")
                proxy = create_proxy_tool(tool, target_channel, source=source)
                if source == "web" and registry.get(tool.name) == "filesystem_tools":
                    web_filesystem_names.add(tool.name)
                    prepare = getattr(proxy, "_datus_prepare_proxy_action", None)
                    if prepare is not None:
                        preparations[tool.name] = prepare
                new_tools.append(proxy)
                proxied_names.add(tool.name)
        else:
            new_tools.append(tool)
    node.tools = new_tools
    if source == "web":
        node._web_file_preparations = preparations
    existing_web = getattr(node, "web_filesystem_proxy_names", None)
    if isinstance(existing_web, set):
        existing_web.clear()
        existing_web.update(web_filesystem_names)
    elif source == "web":
        node.web_filesystem_proxy_names = web_filesystem_names
    # ``PermissionHooks`` consults this set in ``on_tool_start`` to skip the
    # ALLOW/ASK/DENY check for generic external proxies — the caller (e.g.
    # ``print_mode`` stdin protocol) owns secondary confirmation. Mutate in
    # place so existing PermissionHooks instances built earlier still see the
    # latest snapshot via the shared reference held on the node.
    existing = getattr(node, "proxied_tool_names", None)
    if isinstance(existing, set):
        existing.clear()
        existing.update(proxied_names)
    else:
        node.proxied_tool_names = proxied_names


async def prepare_web_tool_action(action: ActionHistory, node: AgenticNode) -> None:
    """Prepare a web file call before the existing frontend executor can see it."""
    if (
        getattr(getattr(node, "agent_config", None), "_client_source", None) != "web"
        or action.role != ActionRole.TOOL
        or action.status != ActionStatus.PROCESSING
        or not isinstance(action.input, dict)
        or "_proxied" in action.input
    ):
        return
    name = action.input.get("function_name", "")
    if name not in getattr(node, "web_filesystem_proxy_names", set()):
        return
    from datus.tools.permission.permission_config import PermissionLevel

    manager = getattr(node, "permission_manager", None)
    if (
        manager is not None
        and manager.check_permission("filesystem_tools", name, node.get_node_name()) == PermissionLevel.DENY
    ):
        # Tool-start events can precede the SDK's permission hook. Never expose
        # an executable file card for an operation the hook will reject.
        action.input = {**action.input, "_proxied": False}
        return
    prepare = getattr(node, "_web_file_preparations", {}).get(name)
    if prepare is not None:
        await prepare(action)


def stamp_tool_execution(action: ActionHistory, node: AgenticNode) -> None:
    """Preserve the producing node's execution ownership through nested forwarding."""
    if (
        getattr(getattr(node, "agent_config", None), "_client_source", None) == "web"
        and action.role == ActionRole.TOOL
        and isinstance(action.input, dict)
        and "_proxied" not in action.input
    ):
        name = action.input.get("function_name", "")
        action.input = {**action.input, "_proxied": name in getattr(node, "proxied_tool_names", set())}


# ── Internal helpers ─────────────────────────────────────────────────


def parse_tool_patterns(patterns: List[str]) -> List[Tuple[Optional[str], str]]:
    """Parse ``"category.method_glob"`` patterns into ``(category, method_glob)`` tuples.

    - ``"filesystem_tools.*"``  → ``("filesystem_tools", "*")``
    - ``"read_file"``           → ``(None, "read_file")``
    - ``"*"``                   → ``(None, "*")``

    Public API shared with the tool middleware so both the proxy and the
    transformer layers speak one pattern syntax.
    """
    result: List[Tuple[Optional[str], str]] = []
    for p in patterns:
        if "." in p:
            cat, method = p.split(".", 1)
            result.append((cat, method))
        else:
            result.append((None, p))
    return result


def tool_name_matches(tool_name: str, registry: Dict[str, str], patterns: List[Tuple[Optional[str], str]]) -> bool:
    """Check if a tool name matches any of the parsed patterns.

    Public API shared with the tool middleware (see :func:`parse_tool_patterns`).
    """
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


# Backward-compatible private aliases: existing in-module call sites and the
# proxy unit tests reference these names. New callers should use the public
# ``parse_tool_patterns`` / ``tool_name_matches`` above.
_parse_patterns = parse_tool_patterns
_matches = tool_name_matches
