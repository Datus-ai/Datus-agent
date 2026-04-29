# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import inspect
import json
from typing import Any, Callable, Dict, List, Optional

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


def parse_tool_args(
    args_str: Any,
    required_fields: set[str] | None = None,
    tool_name: str = "unknown",
) -> tuple[dict, str | None]:
    """Parse JSON tool arguments with json_repair fallback and required-field validation.

    Returns (args_dict, error_message). error_message is None on success.
    """
    if not args_str:
        if required_fields:
            return {}, f"Empty arguments for tool '{tool_name}', missing required fields {required_fields}."
        return {}, None
    if not isinstance(args_str, str):
        try:
            result = dict(args_str)
        except (TypeError, ValueError) as e:
            return {}, f"Invalid arguments for tool '{tool_name}' ({e})"
        else:
            if required_fields:
                missing = required_fields - set(result.keys())
                if missing:
                    return {}, f"Arguments for tool '{tool_name}' missing required fields {missing}."
            return result, None

    stripped = args_str.strip()
    if not stripped:
        if required_fields:
            return {}, f"Empty arguments for tool '{tool_name}', missing required fields {required_fields}."
        return {}, None

    args_dict = None
    original_error = None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            args_dict = parsed
        else:
            original_error = TypeError(f"Expected dict, got {type(parsed).__name__}")
    except (json.JSONDecodeError, TypeError) as e:
        original_error = e

    if args_dict is None and original_error is not None:
        try:
            repaired = json_repair.loads(stripped)
            if isinstance(repaired, dict):
                args_dict = repaired
                logger.warning(f"Repaired malformed JSON arguments for tool '{tool_name}': {original_error}")
        except Exception:
            pass

    if args_dict is None or not isinstance(args_dict, dict):
        args_len = len(args_str)
        truncated_hint = ""
        if not stripped.endswith("}") and not stripped.endswith("]"):
            truncated_hint = " Output appears truncated — likely hit model max_output_tokens limit."
        return {}, f"Invalid JSON arguments ({original_error}). Args length: {args_len} chars.{truncated_hint}"

    if required_fields:
        missing = required_fields - set(args_dict.keys())
        if missing:
            return {}, (
                f"Repaired JSON for tool '{tool_name}' is missing required fields {missing}. "
                f"Args length: {len(args_str)} chars."
            )

    return args_dict, None


def trans_to_function_tool(bound_method: Callable, *, strict_mode: bool = True) -> FunctionTool:
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
    """
    tool_template = function_tool(bound_method, strict_mode=strict_mode)

    corrected_schema = json.loads(json.dumps(tool_template.params_json_schema))
    if "self" in corrected_schema.get("properties", {}):
        del corrected_schema["properties"]["self"]
    if "self" in corrected_schema.get("required", []):
        corrected_schema["required"].remove("self")

    def create_async_invoker(method_to_call: Callable) -> Callable:
        sig = inspect.signature(method_to_call)
        required_params = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

        async def final_invoker(tool_ctx, args_str) -> dict:
            args_dict, error = parse_tool_args(
                args_str, required_fields=required_params, tool_name=method_to_call.__name__
            )
            if error:
                return {"success": 0, "error": error, "result": None}

            if inspect.ismethod(method_to_call):
                tool = method_to_call.__self__
                if hasattr(tool, "set_tool_context"):
                    tool.set_tool_context(tool_ctx)

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
        on_invoke_tool=async_invoker,
        strict_json_schema=strict_mode,
    )
    return final_tool
