# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Wrap plain plugin callbacks with the host's ordinary tool registration contract."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

import jsonschema
from agents import FunctionTool
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable

from datus.plugins.agents import PluginToolDeclaration
from datus.tools.func_tool.base import parse_tool_args, write_back_tool_args
from datus.tools.registry.capabilities import TOOL_CAPABILITIES
from datus.tools.registry.contracts import plugin_function_tool
from datus.utils.exceptions import DatusException, ErrorCode


def bind_plugin_tools(
    declarations: list[PluginToolDeclaration], bindings: dict[str, Callable[..., Any]]
) -> list[FunctionTool]:
    """Bind only declared callbacks, serializing access to their shared runtime state."""
    declared = {item.tool_id: item for item in declarations}
    unknown = set(bindings) - set(declared)
    if unknown:
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Runtime returned undeclared plugin tool bindings: {sorted(unknown)}.",
        )
    lock = asyncio.Lock()
    tools = []
    for identity, handler in bindings.items():
        declaration = declared[identity]
        if identity in TOOL_CAPABILITIES:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Plugin binding {identity!r} cannot replace a native tool.",
            )
        if not callable(handler):
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Plugin binding {identity!r} is not callable.",
            )

        def make_invoker(item: PluginToolDeclaration, callback: Callable[..., Any]) -> Callable:
            validator = validator_for(item.input_schema)(item.input_schema, registry=Registry())

            async def invoke(context: Any, arguments: str) -> Any:
                args, canonical, error = parse_tool_args(arguments, tool_name=item.definition.name)
                write_back_tool_args(context, canonical)
                if error:
                    return {"success": 0, "error": error, "result": None}
                try:
                    validator.validate(args)
                    inspect.signature(callback).bind(**args)
                except jsonschema.ValidationError as exc:
                    return {
                        "success": 0,
                        "error": f"Invalid arguments at {list(exc.path)}: failed {exc.validator} validation.",
                        "result": None,
                    }
                except Unresolvable:
                    return {
                        "success": 0,
                        "error": "Plugin tool input schema has an unresolvable reference.",
                        "result": None,
                    }
                except (TypeError, ValueError) as exc:
                    return {"success": 0, "error": f"Invalid callback arguments: {exc}", "result": None}
                async with lock:
                    try:
                        # Synchronous callbacks are for nonblocking business logic;
                        # clients and other I/O should use asynchronous handlers.
                        result = callback(**args)
                        if inspect.isawaitable(result):
                            result = await result
                        json.dumps(result)
                        return result
                    except Exception as exc:
                        return {"success": 0, "error": str(exc), "result": None}

            return invoke

        tool = plugin_function_tool(
            definition=declaration.definition,
            name=declaration.definition.name,
            description=declaration.description,
            params_json_schema=declaration.input_schema,
            on_invoke_tool=make_invoker(declaration, handler),
            strict_json_schema=declaration.strict,
        )
        # A callback closes over this runtime instance and cannot be executed
        # by an external tool proxy holding a different instance's state.
        tool.__datus_tool_local_only__ = True
        tools.append(tool)
    return tools
