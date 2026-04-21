# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import inspect
import json
from typing import Any, Callable, Dict, List, Optional

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


def trans_to_function_tool(bound_method: Callable) -> FunctionTool:
    """
    Transfer a bound method to a function tool.
    This method is to solve the problem that '@function_tool' can only be applied to static methods
    """
    tool_template = function_tool(bound_method)

    corrected_schema = json.loads(json.dumps(tool_template.params_json_schema))
    if "self" in corrected_schema.get("properties", {}):
        del corrected_schema["properties"]["self"]
    if "self" in corrected_schema.get("required", []):
        corrected_schema["required"].remove("self")

    # The invoker MUST be an 'async' function.
    # We define a closure to correctly capture the 'bound_method' for each iteration.
    def create_async_invoker(method_to_call: Callable) -> Callable:
        async def final_invoker(tool_ctx, args_str) -> dict:
            """
            This is an async wrapper for tool methods.
            The agent framework will 'await' this coroutine.
            """
            # The actual work (JSON parsing, method call)
            try:
                if args_str:
                    args_dict = json.loads(args_str) if isinstance(args_str, str) else dict(args_str or {})
                else:
                    args_dict = {}
            except (TypeError, json.JSONDecodeError) as e:
                args_len = len(args_str) if isinstance(args_str, str) else 0
                truncated_hint = ""
                if isinstance(args_str, str) and args_len > 0:
                    # Check if it looks like truncated output (no closing brace)
                    stripped = args_str.rstrip()
                    if not stripped.endswith("}") and not stripped.endswith("]"):
                        truncated_hint = " Output appears truncated — likely hit model max_output_tokens limit."
                return {
                    "success": 0,
                    "error": f"Invalid JSON arguments ({e}). Args length: {args_len} chars.{truncated_hint}",
                    "result": None,
                }

            # Call sync or async bound methods transparently
            if inspect.ismethod(method_to_call):
                tool = method_to_call.__self__
                if hasattr(tool, "set_tool_context"):
                    tool.set_tool_context(tool_ctx)

            # Filter out unexpected parameters that LLM may hallucinate
            sig = inspect.signature(method_to_call)
            valid_params = set(sig.parameters.keys()) - {"self"}
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_keyword:
                extra_params = set(args_dict.keys()) - valid_params
                if extra_params:
                    logger.warning(
                        f"Tool '{method_to_call.__name__}' received unexpected parameters "
                        f"{extra_params}, filtering them out"
                    )
                    args_dict = {k: v for k, v in args_dict.items() if k in valid_params}

            if inspect.iscoroutinefunction(method_to_call):
                result = await method_to_call(**args_dict)
            else:
                result = method_to_call(**args_dict)
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
    )
    return final_tool
