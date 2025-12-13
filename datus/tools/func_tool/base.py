# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import json
from typing import Any, Callable, Optional

from agents import FunctionTool, function_tool
from pydantic import BaseModel, Field


class FuncToolResult(BaseModel):
    success: int = Field(
        default=1, description="Whether the execution is successful or not, 1 is success, 0 is failure", init=True
    )
    error: Optional[str] = Field(
        default=None, description="Error message: field is not empty when success=0", init=True
    )
    result: Optional[Any] = Field(default=None, description="Result of the execution", init=True)


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
            except (TypeError, json.JSONDecodeError):
                return {"success": 0, "error": "Invalid JSON arguments", "result": None}

            # Call sync or async bound methods transparently
            import inspect

            if inspect.ismethod(method_to_call):
                tool = method_to_call.__self__
                if hasattr(tool, "set_tool_context"):
                    tool.set_tool_context(tool_ctx)

            if inspect.iscoroutinefunction(method_to_call):
                result = await method_to_call(**args_dict)
            else:
                result = method_to_call(**args_dict)
            if isinstance(result, FuncToolResult):
                return result.model_dump()
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
