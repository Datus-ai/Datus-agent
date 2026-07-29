# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import asyncio
import inspect
import json
from typing import Any, Callable, Dict, Iterable, List, Optional

import json_repair
from agents import FunctionTool, function_tool
from pydantic import BaseModel, Field

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def normalize_null(value):
    """Convert string 'null', 'None', empty, or whitespace-only values to None for LLM compatibility.

    LLMs sometimes output the string 'null' / 'None' / '' instead of JSON null.
    This function normalizes such values to Python None.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("null", "none", ""):
        return None
    return value


def parse_tool_args(
    args_str: Any,
    *,
    tool_name: str = "unknown",
) -> tuple[dict, str, Optional[str]]:
    """Parse tool arguments, repairing malformed JSON objects when possible.

    Returns ``(arguments, canonical_json, error)``. ``canonical_json`` is always
    a valid JSON object so it can safely replace the raw tool-call arguments
    before the Agents SDK replays or persists the call. A repaired call returns
    an error so the model retries with valid arguments instead of executing
    parameters that permission hooks did not inspect.
    """
    if not args_str:
        return {}, "{}", None

    original_error: Optional[Exception] = None

    if isinstance(args_str, str):
        stripped = args_str.strip()
        if not stripped:
            return {}, "{}", None
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError) as exc:
            original_error = exc
            parsed = None
            # Function-tool arguments must be a JSON object. Limiting repair to
            # object-shaped input avoids turning arbitrary text into an empty
            # object and treating it as a successful repair.
            if stripped.startswith("{"):
                try:
                    parsed = json_repair.loads(stripped)
                except Exception:  # noqa: BLE001 - json-repair is best effort
                    parsed = None
                if isinstance(parsed, dict):
                    canonical_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                    logger.warning(
                        "Repaired malformed JSON arguments for tool '%s': %s",
                        tool_name,
                        original_error,
                    )
                    error = (
                        f"Malformed JSON arguments for tool '{tool_name}' were repaired for replay. "
                        "The tool was not executed. Retry the tool with one complete JSON object."
                    )
                    return parsed, canonical_json, error
    else:
        try:
            parsed = dict(args_str)
        except (TypeError, ValueError) as exc:
            original_error = exc
            parsed = None

    if not isinstance(parsed, dict):
        if original_error is None:
            original_error = TypeError(f"expected a JSON object, got {type(parsed).__name__}")
        args_len = len(args_str) if isinstance(args_str, str) else 0
        truncated_hint = ""
        if isinstance(args_str, str):
            stripped = args_str.rstrip()
            if stripped and not stripped.endswith("}"):
                truncated_hint = " Output appears truncated — likely hit model max_output_tokens limit."
        error = (
            f"Invalid JSON arguments for tool '{tool_name}' ({original_error}). "
            f"Args length: {args_len} chars.{truncated_hint}"
        )
        return {}, "{}", error

    canonical_json = (
        args_str if isinstance(args_str, str) else json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    )
    return parsed, canonical_json, None


def write_back_tool_args(tool_ctx: Any, canonical_json: str) -> None:
    """Update the current SDK tool call so replay and session persistence use valid JSON."""
    if tool_ctx is None:
        return

    tool_ctx.tool_arguments = canonical_json
    tool_call = getattr(tool_ctx, "tool_call", None)
    if tool_call is None:
        return
    if isinstance(tool_call, dict):
        tool_call["arguments"] = canonical_json
    else:
        tool_call.arguments = canonical_json


class FuncToolResult(BaseModel):
    success: int = Field(
        default=1, description="Whether the execution is successful or not, 1 is success, 0 is failure", init=True
    )
    error: Optional[str] = Field(
        default=None, description="Error message: field is not empty when success=0", init=True
    )
    result: Optional[Any] = Field(default=None, description="Result of the execution", init=True)


class FuncToolListResult(BaseModel):
    """Canonical envelope for list-shaped FuncTool results.

    Put ``FuncToolListResult(...).model_dump()`` inside ``FuncToolResult.result``
    whenever a tool method conceptually returns "a list of records" (BI
    ``list_dashboards``, scheduler ``list_scheduler_jobs``, semantic
    ``list_metrics``, ...). Separating row data (``items``) from pagination
    signals (``total`` / ``has_more``) and tool-specific metadata (``extra``)
    lets CLI / LLM / agent consumers share one shape instead of each inventing
    their own heuristic.

    Field rules:
      * ``items`` is the single source of truth for row data. Always
        ``List[Dict]``; empty is ``[]``, never ``None``. Never carries an
        alternative encoding (CSV blob, scalars).
      * ``total`` is the upstream full count when known. ``None`` means the
        source doesn't expose a total — consumers should fall back to
        ``has_more`` or ``len(items) < limit`` for pagination decisions.
        Do not set ``total = len(items)`` as a placeholder; it makes
        consumers wrongly conclude there is no next page.
      * ``has_more`` is the explicit "another page exists" hint. ``None``
        when the source gives no signal.
      * ``extra`` holds tool-specific side-channel data — most commonly
        ``{"next_offset": <int>}`` so the LLM can copy the value instead
        of computing the next offset itself. Never holds an alternative
        encoding of ``items``; never holds error state (that belongs in
        ``FuncToolResult.error``).
    """

    items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The rows. Always a list of dicts; never None; empty is [].",
    )
    total: Optional[int] = Field(
        default=None,
        description=(
            "Upstream full row count. May exceed len(items) when paginated. "
            "None when the source doesn't expose a total."
        ),
    )
    has_more: Optional[bool] = Field(
        default=None,
        description="Explicit 'next page exists' hint. None when unknown.",
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Tool-specific side channel. Typically contains 'next_offset' "
            "when has_more is True. Consumers ignore unknown keys."
        ),
    )


def trans_to_function_tool(
    bound_method: Callable,
    *,
    strict_mode: bool = True,
    excluded_params: Optional[Iterable[str]] = None,
) -> FunctionTool:
    """
    Transfer a bound method to a function tool.
    This method is to solve the problem that '@function_tool' can only be applied to static methods

    Args:
        bound_method: The instance method to wrap.
        strict_mode: When True (default), the OpenAI Agents SDK enforces a strict JSON schema
            (no extra properties, no free-form ``Dict[str, Any]`` parameters). Set to False
            for tools that genuinely need an open-ended object parameter — e.g. a
            ``sample_params``-style dict where the LLM provides arbitrary keys matching
            a declaration the tool itself validates.
        excluded_params: Optional parameter names to remove from the exposed JSON schema.
            Use this for dialect-specific parameters that the current tool instance does not support.
    """
    tool_template = function_tool(bound_method, strict_mode=strict_mode)
    excluded_param_set = set(excluded_params or [])

    corrected_schema = json.loads(json.dumps(tool_template.params_json_schema))
    params_to_remove = {"self", *excluded_param_set}
    for param_name in params_to_remove:
        if param_name in corrected_schema.get("properties", {}):
            del corrected_schema["properties"][param_name]
        if param_name in corrected_schema.get("required", []):
            corrected_schema["required"].remove(param_name)

    # The invoker MUST be an 'async' function.
    # We define a closure to correctly capture the 'bound_method' for each iteration.
    def create_async_invoker(method_to_call: Callable) -> Callable:
        async def final_invoker(tool_ctx, args_str) -> dict:
            """
            This is an async wrapper for tool methods.
            The agent framework will 'await' this coroutine.
            """
            args_dict, canonical_json, error = parse_tool_args(
                args_str,
                tool_name=method_to_call.__name__,
            )
            write_back_tool_args(tool_ctx, canonical_json)
            if error:
                return {"success": 0, "error": error, "result": None}

            # Call sync or async bound methods transparently
            if inspect.ismethod(method_to_call):
                tool = method_to_call.__self__
                if hasattr(tool, "set_tool_context"):
                    tool.set_tool_context(tool_ctx)

            # Filter out unexpected parameters that LLM may hallucinate
            sig = inspect.signature(method_to_call)
            valid_params = set(sig.parameters.keys()) - {"self"} - excluded_param_set
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            excluded_in_args = set(args_dict.keys()) & excluded_param_set
            if excluded_in_args:
                params = ", ".join(sorted(excluded_in_args))
                return {
                    "success": 0,
                    "error": f"Unsupported parameters for this tool: {params}",
                    "result": None,
                }
            if not has_var_keyword:
                extra_params = set(args_dict.keys()) - valid_params
                if extra_params:
                    logger.warning(
                        f"Tool '{method_to_call.__name__}' received unexpected parameters "
                        f"{extra_params}, filtering them out"
                    )
                    args_dict = {k: v for k, v in args_dict.items() if k in valid_params}

            # Reject missing required parameters symmetrically with the extra/
            # excluded-parameter handling above. Without this, an LLM tool call
            # that omits a required argument (e.g. ``edit_file`` without
            # ``path``) reaches ``method_to_call(**args_dict)`` and raises a raw
            # ``TypeError`` at bind time. That exception is uncaught here and
            # aborts the whole agent interaction/batch, silently dropping work.
            # Returning a recoverable error lets the model retry in-turn, the
            # same as it does for malformed JSON or unsupported parameters.
            missing_required = {
                name
                for name, param in sig.parameters.items()
                if name != "self"
                and name not in excluded_param_set
                and param.default is inspect.Parameter.empty
                and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                and name not in args_dict
            }
            if missing_required:
                params = ", ".join(sorted(missing_required))
                return {
                    "success": 0,
                    "error": f"Missing required parameter(s) for '{method_to_call.__name__}': {params}",
                    "result": None,
                }

            if inspect.iscoroutinefunction(method_to_call):
                result = await method_to_call(**args_dict)
            else:
                # Offload synchronous tool methods to a worker thread. Many tools
                # (DB metadata/queries, filesystem ops) make blocking I/O calls;
                # ``final_invoker`` is awaited directly on the asyncio event-loop
                # thread, so calling them inline freezes the loop — a single slow
                # ``list_tables`` against a large StarRocks cluster would hang the
                # whole server (all requests unresponsive). ``asyncio.to_thread``
                # keeps the loop free and copies the current contextvars (trace
                # context) into the worker. ``set_tool_context`` is applied above
                # on the per-session tool instance, so it is visible in the thread.
                result = await asyncio.to_thread(method_to_call, **args_dict)
            if isinstance(result, FuncToolResult):
                return result.model_dump(mode="json")
            return result

        return final_invoker

    async_invoker = create_async_invoker(bound_method)

    final_tool = FunctionTool(
        name=tool_template.name,
        description=tool_template.description,
        params_json_schema=corrected_schema,
        on_invoke_tool=async_invoker,  # <--- Assign the async function
        strict_json_schema=strict_mode,
    )
    return final_tool
