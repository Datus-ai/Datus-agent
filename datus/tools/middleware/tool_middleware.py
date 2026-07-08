# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tool argument middleware: rewrite or deny tool calls before execution.

``AgentHooks.on_tool_start`` cannot modify tool arguments — the SDK executes
``tool_call.arguments`` verbatim and ``ToolContext.tool_arguments`` is a copy.
This module provides the capability the hook layer cannot: it rebuilds a
:class:`FunctionTool` whose ``on_invoke_tool`` runs registered *transformers*
over the LLM-provided arguments first, then delegates to the original tool.

Both execution paths (the SDK Runner and the native Claude loop) converge on
``FunctionTool.on_invoke_tool``, so wrapping the tool covers every LLM-driven
call. Python callers that invoke tool methods directly (e.g. reference-template
execution) bypass ``FunctionTool`` entirely — enforcement that must also cover
those paths needs its own tool-layer check (see
``DBFuncTool._enforce_sql_policy``).

Transformer contract (duck-typed so plugin packages never import ``datus.*``):

    transformer(tool_name: str, args: dict, context: dict) -> dict

* May be sync or async.
* Returns the (possibly modified) argument dict; execution continues with it.
* Raises to DENY the call: the wrapper returns the standard tool failure
  payload ``{"success": 0, "error": <reason>, "result": None}`` to the model
  and the wrapped tool never runs (fail closed). Returning anything that is
  not a dict is treated the same way.
* ``context`` carries request-scoped data injected at wrap time (see
  :func:`apply_tool_transformers`): ``node_name``, ``principal``,
  ``project_root``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional

from agents import FunctionTool

# Pattern matching intentionally shares the proxy layer's semantics
# ("category.*", bare tool name, fnmatch globs) so operators learn one syntax.
from datus.tools.proxy.proxy_tool import parse_tool_patterns, tool_name_matches
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Type alias for readability; transformers are duck-typed callables.
ToolTransformer = Callable[[str, Dict[str, Any], Dict[str, Any]], Any]
ContextProvider = Callable[[], Dict[str, Any]]


def _denial_payload(tool_name: str, reason: str) -> dict:
    """Standard tool-failure payload returned to the model on deny/failure.

    Mirrors the shape of ``FuncToolResult`` (and the proxy layer's timeout
    payload) so the model sees a normal tool error instead of a crashed run.
    """
    return {
        "success": 0,
        "error": f"Tool call '{tool_name}' was blocked by policy: {reason}",
        "result": None,
    }


def wrap_tool_with_transformers(
    original: FunctionTool,
    transformers: List[ToolTransformer],
    context_provider: Optional[ContextProvider] = None,
) -> FunctionTool:
    """Rebuild ``original`` so ``transformers`` run over its arguments first.

    Transformers run in list order; each receives the previous one's output.
    Any transformer exception (or non-dict return) denies the call fail-closed:
    the original tool never executes and the model receives a standard failure
    payload carrying the reason.

    Arguments that do not parse as a JSON object are passed through to the
    original tool untouched — there is nothing for a transformer to rewrite,
    and the tool's own malformed-arguments error path stays authoritative.

    ``context_provider`` is called once per invocation so request-scoped values
    (e.g. a per-request principal set on the owning tool instance after wrap
    time) are read fresh, not frozen at wrap time.
    """

    async def transforming_invoke(tool_ctx: Any, args_str: str) -> Any:
        try:
            args = json.loads(args_str) if args_str else {}
        except (json.JSONDecodeError, TypeError):
            args = None
        if not isinstance(args, dict):
            logger.debug("Tool middleware: non-object arguments for '%s'; passing through to the tool", original.name)
            return await original.on_invoke_tool(tool_ctx, args_str)

        try:
            context = dict(context_provider()) if context_provider is not None else {}
        except Exception as e:
            logger.warning("Tool middleware context provider failed for '%s': %s", original.name, e)
            context = {}

        for transformer in transformers:
            transformer_name = getattr(transformer, "__name__", repr(transformer))
            try:
                result = transformer(original.name, args, context)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:
                logger.warning(
                    "Tool transformer %s denied '%s': %s",
                    transformer_name,
                    original.name,
                    e,
                )
                return _denial_payload(original.name, str(e) or type(e).__name__)
            if not isinstance(result, dict):
                logger.warning(
                    "Tool transformer %s returned %s for '%s'; denying fail-closed",
                    transformer_name,
                    type(result).__name__,
                    original.name,
                )
                return _denial_payload(
                    original.name,
                    f"transformer {transformer_name} returned {type(result).__name__} instead of an argument dict",
                )
            args = result

        return await original.on_invoke_tool(tool_ctx, json.dumps(args, ensure_ascii=False))

    # Carry over every behavioural field except ``on_invoke_tool``: rebuilding
    # the tool must not silently re-enable a tool that was disabled/gated, nor
    # drop its input/output guardrails. Use getattr defaults so the wrapper
    # keeps working across openai-agents SDK versions that add or rename fields.
    return FunctionTool(
        name=original.name,
        description=original.description,
        params_json_schema=original.params_json_schema,
        on_invoke_tool=transforming_invoke,
        strict_json_schema=original.strict_json_schema,
        is_enabled=getattr(original, "is_enabled", True),
        tool_input_guardrails=getattr(original, "tool_input_guardrails", None),
        tool_output_guardrails=getattr(original, "tool_output_guardrails", None),
    )


def apply_tool_transformers(node: Any, transformers_by_pattern: Dict[str, List[ToolTransformer]]) -> int:
    """Wrap matching tools on ``node`` with the given transformers, in place.

    Args:
        node: AgenticNode-like object exposing ``tools``, ``tool_registry``
            and (optionally) ``proxied_tool_names`` / ``db_func_tool``.
        transformers_by_pattern: Mapping of tool patterns (proxy syntax:
            ``"execute_sql"``, ``"db_tools.*"``) to transformer lists.

    Returns:
        The number of tools that were wrapped.

    Proxied tools are skipped: their execution is delegated to the external
    client via the stdin proxy channel, so rewriting server-side arguments
    would not affect what actually runs.
    """
    if not transformers_by_pattern:
        return 0

    # Populate the registry eagerly — matching "category.*" patterns needs it,
    # and lazy population normally only happens on the first permission check.
    if hasattr(node, "_populate_tool_registry"):
        node._populate_tool_registry()
    registry = node.tool_registry.to_dict() if getattr(node, "tool_registry", None) is not None else {}
    proxied = getattr(node, "proxied_tool_names", None) or set()
    node_name = getattr(node, "get_node_name", lambda: "")()

    def context_provider() -> Dict[str, Any]:
        # Read request-scoped values fresh on every tool call: the API layer
        # sets ``db_func_tool.principal`` per request, after tools are wrapped.
        principal = getattr(getattr(node, "db_func_tool", None), "principal", None)
        agent_config = getattr(node, "agent_config", None)
        return {
            "node_name": node_name,
            "principal": dict(principal) if isinstance(principal, dict) else {},
            "project_root": getattr(agent_config, "project_root", None),
            # Live AgentConfig reference so transformers can read their own
            # plugin profile (``get_plugin_profile``) and datasource metadata.
            # Duck-typed access only — transformers must not import datus.*.
            "agent_config": agent_config,
        }

    parsed_by_pattern = {pattern: parse_tool_patterns([pattern]) for pattern in transformers_by_pattern}

    wrapped_count = 0
    new_tools = []
    existing_tools = getattr(node, "tools", None)
    for tool in existing_tools or []:
        if not isinstance(tool, FunctionTool):
            new_tools.append(tool)
            continue
        if tool.name in proxied:
            logger.debug("Tool middleware: skipping proxied tool '%s'", tool.name)
            new_tools.append(tool)
            continue
        matched: List[ToolTransformer] = []
        for pattern, transformers in transformers_by_pattern.items():
            if tool_name_matches(tool.name, registry, parsed_by_pattern[pattern]):
                matched.extend(transformers)
        if matched:
            logger.info("Tool middleware: wrapping '%s' with %d transformer(s)", tool.name, len(matched))
            new_tools.append(wrap_tool_with_transformers(tool, matched, context_provider))
            wrapped_count += 1
        else:
            new_tools.append(tool)
    if existing_tools is None:
        node.tools = new_tools
    else:
        # Mutate in place: callers may have already captured the list object
        # (e.g. a ``tools=self.tools`` argument evaluated before hooks compose),
        # and rebinding would leave that captured reference unwrapped.
        existing_tools[:] = new_tools
    return wrapped_count
