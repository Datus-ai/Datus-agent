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
``DBFuncTool.execute_read_enforced``).

Transformer contract (duck-typed so plugin packages never import ``datus.*``):

    transformer(tool_name: str, args: dict, context: dict) -> dict

* May be sync or async.
* Returns the (possibly modified) argument dict; execution continues with it.
* Raises to DENY the call: the wrapper returns the standard tool failure
  payload ``{"success": 0, "error": <reason>, "result": None}`` to the model
  and the wrapped tool never runs (fail closed). Returning anything that is
  not a dict is treated the same way.
* ``context`` carries request-scoped data injected at wrap time (see
  :func:`apply_tool_transformers`): ``node_name``, ``policy_context``,
  ``project_root``.
* A rewrite changes what executes, not what the conversation records: the
  model's own arguments are restored onto the tool call afterwards.
"""

from __future__ import annotations

import dataclasses
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

# Marker set on a rebuilt tool's ``on_invoke_tool`` so :func:`apply_tool_transformers`
# can recognise already-wrapped tools and skip them. Makes wrapping idempotent
# per tool: callers may reset their "transformers applied" flag and re-run after
# a tool-list rebuild (e.g. a runtime ``/model`` switch remounts web tools)
# without double-wrapping tools that were already transformed on a prior turn.
_TRANSFORMED_MARKER = "_datus_tool_transformed"


def tool_is_transformed(tool: Any) -> bool:
    """Return True if ``tool`` was already wrapped by :func:`wrap_tool_with_transformers`."""
    return bool(getattr(getattr(tool, "on_invoke_tool", None), _TRANSFORMED_MARKER, False))


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


def _restore_tool_call_args(tool_ctx: Any, args_str: str) -> None:
    """Put the model's own arguments back on the SDK tool call.

    Tools canonicalise the arguments they received onto ``tool_ctx`` (see
    ``write_back_tool_args``), and that is what session persistence and the
    next turn's messages replay. A rewrite left there shows the model a call
    it never made: it "corrects" itself back to its original query, which is
    rewritten again — a loop that re-wraps the previous rewrite each round.

    ``tool_call`` is the one the SDK persists and replays, so it is restored
    on its own: a context that refuses ``tool_arguments`` still gets it back.
    """
    if tool_ctx is None:
        return
    try:
        tool_ctx.tool_arguments = args_str
    except Exception as e:  # noqa: BLE001 - a frozen or foreign context must not fail the call
        logger.debug("Tool middleware: could not restore tool_arguments: %s", e)
    try:
        tool_call = getattr(tool_ctx, "tool_call", None)
        if isinstance(tool_call, dict):
            tool_call["arguments"] = args_str
        elif tool_call is not None:
            tool_call.arguments = args_str
    except Exception as e:  # noqa: BLE001 - a frozen or foreign context must not fail the call
        logger.debug("Tool middleware: could not restore the tool call arguments: %s", e)


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
    (e.g. a per-request policy context set on the owning tool instance after wrap
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

        try:
            return await original.on_invoke_tool(tool_ctx, json.dumps(args, ensure_ascii=False))
        finally:
            _restore_tool_call_args(tool_ctx, args_str)

    transforming_invoke._datus_tool_transformed = True  # type: ignore[attr-defined]

    # Carry over every declared field except ``on_invoke_tool`` by forwarding
    # the FunctionTool dataclass's own init fields. Rebuilding the tool must not
    # silently re-enable a disabled/gated tool, drop its input/output guardrails,
    # or weaken approval/timeout settings that newer openai-agents SDK versions
    # add (e.g. ``needs_approval`` / ``timeout_behavior`` — absent in the pinned
    # 0.7.0). Forwarding by field name keeps the wrapper faithful across SDK
    # versions without this module tracking each new field by hand; ``getattr``
    # defaults tolerate SDK-added fields that a test double may not carry.
    carried = {
        field.name: getattr(original, field.name, None)
        for field in dataclasses.fields(FunctionTool)
        if field.init and field.name != "on_invoke_tool"
    }
    carried["on_invoke_tool"] = transforming_invoke
    return FunctionTool(**carried)


def _metric_catalog_providers(node: Any) -> List[Any]:
    """Semantic tool groups on ``node`` that can report a metric-to-datasets map.

    Found by what the object declares, not by the attribute holding it — the
    same rule ``_iter_tool_groups`` uses to classify tools, and the reason a
    node may hold its semantic tools under any attribute name. Aliases of one instance collapse to one
    provider.
    """
    iter_groups = getattr(node, "_iter_tool_groups", None)
    if callable(iter_groups):
        try:
            candidates = list(iter_groups())
        except Exception as e:  # noqa: BLE001 - tool discovery must not break tool calls
            logger.warning("Could not enumerate tool groups for transformer context: %s", e)
            return []
    else:
        # Same shape test ``_iter_tool_groups`` applies, for objects that do not
        # implement it (a type is a class, not a mounted group).
        candidates = [
            value
            for value in (getattr(node, name, None) for name in sorted(vars(node)))
            if not isinstance(value, type) and callable(getattr(value, "available_tools", None))
        ]

    providers: List[Any] = []
    seen: set = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        if getattr(candidate, "permission_category", None) != "semantic_tools":
            continue
        if not callable(getattr(candidate, "metric_datasets", None)):
            continue
        seen.add(id(candidate))
        providers.append(candidate)
    return providers


def _metric_datasets(node: Any) -> Optional[Dict[str, List[str]]]:
    """Metric-to-datasets map from the node's semantic tools.

    ``None`` when no single provider can be identified or the catalog cannot be
    read; ``{}`` when the adapter reports no metrics. Several providers means
    several catalogs, and picking one would scope a policy against the wrong
    metrics — the transformers reading this deny on ``None``.
    """
    providers = _metric_catalog_providers(node)
    if not providers:
        return None
    if len(providers) > 1:
        logger.warning(
            "Node exposes %d semantic tool groups; cannot pick one metric catalog for transformer context",
            len(providers),
        )
        return None
    try:
        mapping = providers[0].metric_datasets()
    except Exception as e:  # noqa: BLE001 - an unreadable catalog must not break tool calls
        logger.warning("Could not read metric datasets for transformer context: %s", e)
        return None
    if mapping is not None and not isinstance(mapping, dict):
        logger.warning("Ignoring metric datasets of unexpected type %s", type(mapping).__name__)
        return None
    return mapping


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
        # assigns ``policy_context`` onto a per-request clone of the config, which
        # is also where ``DBFuncTool`` reads its own copy from. Reading the
        # config directly covers nodes whose tool set excludes ``db_tools`` and
        # therefore never build a DBFuncTool at all (``ask_metrics`` is one).
        agent_config = getattr(node, "agent_config", None)
        policy_context = getattr(agent_config, "policy_context", None)
        return {
            "node_name": node_name,
            "policy_context": dict(policy_context) if isinstance(policy_context, dict) else {},
            "project_root": getattr(agent_config, "project_root", None),
            # Live AgentConfig reference so transformers can read their own
            # plugin profile (``get_plugin_profile``) and datasource metadata.
            # Duck-typed access only — transformers must not import datus.*.
            "agent_config": agent_config,
            # Metric name -> datasets it reads, so transformers enforcing read
            # policies over metric queries know which physical data is touched.
            "metric_datasets": _metric_datasets(node),
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
        if tool_is_transformed(tool):
            # Already wrapped on a prior pass: re-wrapping would run the
            # transformers twice. Callers reset their applied-flag and re-run
            # after a tool-list rebuild, so skipping keeps that idempotent.
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
