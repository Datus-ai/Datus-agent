# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""
MCP Tool Decorators and Registration Utilities

This module provides decorators and utilities for automatically registering
tool methods and classes as MCP (Model Context Protocol) tools.

Method-level decorator usage:
    from datus.utils.mcp_decorators import mcp_tool

    class MyToolClass:
        @mcp_tool()
        def my_tool(self, param: str) -> FuncToolResult:
            '''Tool description from docstring.'''
            ...

        @mcp_tool(availability_check="has_feature")
        def feature_tool(self, query: str) -> FuncToolResult:
            '''Only available when has_feature is True.'''
            ...

Class-level decorator usage:
    from datus.utils.mcp_decorators import mcp_tool_class

    @mcp_tool_class(
        name="db_tool",
        availability_property="has_db_tools",
    )
    class DBFuncTool:
        @classmethod
        def create_dynamic(cls, agent_config, sub_agent_name=None):
            '''Factory for dynamic mode (multi-connector)'''
            return db_function_tool_instance_multi(agent_config, sub_agent_name=sub_agent_name)

        @classmethod
        def create_static(cls, agent_config, sub_agent_name=None, database_name=None):
            '''Factory for static mode (single connector)'''
            return db_function_tool_instance(agent_config, database_name=database_name, sub_agent_name=sub_agent_name)

        @mcp_tool()
        def list_databases(self, catalog: str = "", include_sys: bool = False):
            ...
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datus.tools.func_tool.base import FuncToolResult

# ============================================================================
# Global Tool Registry
# ============================================================================

# Global registry for tool classes decorated with @mcp_tool_class
_GLOBAL_TOOL_REGISTRY: List["ToolClassConfig"] = []


@dataclass
class MCPToolConfig:
    """MCP tool configuration metadata (for method-level decorator)."""

    availability_check: Optional[str] = None  # Attribute name like "has_schema"


@dataclass
class ToolClassConfig:
    """Configuration for a tool class registered with @mcp_tool_class."""

    name: str  # Attribute name (e.g., "db_tool", "context_tool")
    tool_class: type  # Tool class (e.g., DBFuncTool, ContextSearchTools)
    availability_property: str  # Property name for availability check (e.g., "has_db_tools")


def mcp_tool(availability_check: Optional[str] = None):
    """
    Decorator to mark a method as an MCP-exportable tool.

    The decorated method will be automatically registered as an MCP tool when
    using register_dynamic_tools or register_static_tools.

    Args:
        availability_check: Optional attribute name to check before tool is available.
                           e.g., "has_schema" means tool is only available if
                           self.has_schema is True

    Returns:
        Decorated function with _mcp_config attribute

    Example:
        @mcp_tool()
        def list_databases(self, catalog: str = "", include_sys: bool = False):
            '''List all databases.'''
            ...

        @mcp_tool(availability_check="has_schema")
        def search_table(self, query_text: str, top_n: int = 5):
            '''Search tables using semantic similarity.'''
            ...
    """

    def decorator(func: Callable) -> Callable:
        func._mcp_config = MCPToolConfig(availability_check=availability_check)
        return func

    return decorator


def get_mcp_tools(cls: type) -> List[Tuple[str, Callable, MCPToolConfig]]:
    """
    Get all methods with @mcp_tool decorator from a class.

    Args:
        cls: The class to inspect

    Returns:
        List of tuples: (method_name, method, config)
    """
    tools = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if hasattr(method, "_mcp_config"):
            tools.append((name, method, method._mcp_config))
    return tools


def create_dynamic_tool_wrapper(
    method_name: str,
    method: Callable,
    config: MCPToolConfig,
    context_getter: Callable,
    instance_attr: str,
    availability_attr: str,
    format_result: Callable[[Union[FuncToolResult, Any]], Dict[str, Any]],
) -> Callable:
    """
    Create a wrapper function for dynamic mode (context-based tool lookup).

    The wrapper preserves the original method signature (minus 'self') for
    proper MCP schema generation.

    Args:
        method_name: Name of the original method
        method: The original method function
        config: MCP tool configuration
        context_getter: Function to get current request context
        instance_attr: Attribute name for tool instance in context (e.g., "db_tool")
        availability_attr: Attribute name for availability check (e.g., "has_db_tools")
        format_result: Function to format the result for MCP response

    Returns:
        Wrapper function with preserved signature
    """
    sig = inspect.signature(method)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    new_sig = sig.replace(parameters=params)

    def wrapper(**kwargs) -> Dict[str, Any]:
        ctx = context_getter()

        # Check tool availability
        if not getattr(ctx, availability_attr, False):
            return {"success": 0, "error": f"{instance_attr} not available", "result": None}

        instance = getattr(ctx, instance_attr)

        # Check feature availability
        if config.availability_check and not getattr(instance, config.availability_check, False):
            return {"success": 0, "error": "Feature not available", "result": None}

        # Call the actual method
        result = getattr(instance, method_name)(**kwargs)
        return format_result(result)

    # Preserve original function metadata for MCP schema generation
    wrapper.__name__ = method_name
    wrapper.__doc__ = method.__doc__
    wrapper.__signature__ = new_sig

    return wrapper


def create_static_tool_wrapper(
    method_name: str,
    bound_method: Callable,
    config: MCPToolConfig,
    format_result: Callable[[Union[FuncToolResult, Any]], Dict[str, Any]],
) -> Callable:
    """
    Create a wrapper function for static mode (fixed tool instance).

    Args:
        method_name: Name of the original method
        bound_method: The bound method from tool instance
        config: MCP tool configuration
        format_result: Function to format the result for MCP response

    Returns:
        Wrapper function with preserved signature
    """
    sig = inspect.signature(bound_method)

    def wrapper(**kwargs) -> Dict[str, Any]:
        instance = bound_method.__self__

        # Check feature availability
        if config.availability_check and not getattr(instance, config.availability_check, False):
            return {"success": 0, "error": "Feature not available", "result": None}

        # Call the actual method
        result = bound_method(**kwargs)
        return format_result(result)

    # Preserve original function metadata
    wrapper.__name__ = method_name
    wrapper.__doc__ = bound_method.__doc__
    wrapper.__signature__ = sig

    return wrapper


def register_dynamic_tools(
    mcp,
    tool_class: type,
    context_getter: Callable,
    instance_attr: str,
    availability_attr: str,
    format_result: Callable[[Union[FuncToolResult, Any]], Dict[str, Any]],
) -> None:
    """
    Register all @mcp_tool decorated methods from a class for dynamic mode.

    In dynamic mode, the tool instance is retrieved from context at runtime,
    allowing the same MCP server to serve multiple namespaces.

    Args:
        mcp: FastMCP server instance
        tool_class: The tool class (e.g., DBFuncTool, ContextSearchTools)
        context_getter: Function to get current request context
        instance_attr: Attribute name for tool instance in context
        availability_attr: Attribute name for availability check in context
        format_result: Function to format results for MCP response

    Example:
        register_dynamic_tools(
            mcp=self.mcp,
            tool_class=DBFuncTool,
            context_getter=self._get_context,
            instance_attr="db_tool",
            availability_attr="has_db_tools",
            format_result=self._format_result,
        )
    """
    for name, method, config in get_mcp_tools(tool_class):
        wrapper = create_dynamic_tool_wrapper(
            method_name=name,
            method=method,
            config=config,
            context_getter=context_getter,
            instance_attr=instance_attr,
            availability_attr=availability_attr,
            format_result=format_result,
        )
        mcp.tool()(wrapper)


def register_static_tools(
    mcp,
    tool_instance: Any,
    format_result: Callable[[Union[FuncToolResult, Any]], Dict[str, Any]],
    skip_unavailable: bool = True,
) -> None:
    """
    Register all @mcp_tool decorated methods from an instance for static mode.

    In static mode, the tool instance is fixed at registration time.

    Args:
        mcp: FastMCP server instance
        tool_instance: The tool instance (e.g., DBFuncTool instance)
        format_result: Function to format results for MCP response
        skip_unavailable: If True (default), skip registering tools whose
                         availability_check returns False. If False, register
                         all tools (unavailable ones will return errors at call time).

    Example:
        register_static_tools(
            mcp=self.mcp,
            tool_instance=self.db_tool,
            format_result=self._format_result,
        )
    """
    for name, _, config in get_mcp_tools(type(tool_instance)):
        # Check availability at registration time if requested
        if skip_unavailable and config.availability_check:
            if not getattr(tool_instance, config.availability_check, False):
                continue

        bound_method = getattr(tool_instance, name)
        wrapper = create_static_tool_wrapper(
            method_name=name,
            bound_method=bound_method,
            config=config,
            format_result=format_result,
        )
        mcp.tool()(wrapper)


# ============================================================================
# Class-level Decorator and Registry Access
# ============================================================================


def mcp_tool_class(
    name: str,
    availability_property: str,
):
    """
    Class decorator to register a tool class in the global MCP tool registry.

    The decorated class should implement:
    - create_dynamic(cls, agent_config, sub_agent_name=None): Factory for dynamic mode
    - create_static(cls, agent_config, sub_agent_name=None, database_name=None): Factory for static mode

    Args:
        name: Attribute name for this tool (e.g., "db_tool", "context_tool")
        availability_property: Property name for availability check (e.g., "has_db_tools")

    Returns:
        Decorated class (unchanged)

    Example:
        @mcp_tool_class(
            name="db_tool",
            availability_property="has_db_tools",
        )
        class DBFuncTool:
            @classmethod
            def create_dynamic(cls, agent_config, sub_agent_name=None):
                return db_function_tool_instance_multi(agent_config, sub_agent_name=sub_agent_name)

            @classmethod
            def create_static(cls, agent_config, sub_agent_name=None, database_name=None):
                return db_function_tool_instance(agent_config, database_name=database_name,
                    sub_agent_name=sub_agent_name)

            @mcp_tool()
            def list_databases(self, ...):
                ...
    """

    def decorator(cls: type) -> type:
        # Validate that the class has required factory methods
        if not hasattr(cls, "create_dynamic"):
            raise TypeError(f"{cls.__name__} must implement 'create_dynamic' classmethod")
        if not hasattr(cls, "create_static"):
            raise TypeError(f"{cls.__name__} must implement 'create_static' classmethod")

        # Register in global registry
        config = ToolClassConfig(
            name=name,
            tool_class=cls,
            availability_property=availability_property,
        )
        _GLOBAL_TOOL_REGISTRY.append(config)

        return cls

    return decorator


def get_tool_registry() -> List[ToolClassConfig]:
    """
    Get the global tool registry containing all @mcp_tool_class decorated classes.

    Returns:
        List of ToolClassConfig objects
    """
    return list(_GLOBAL_TOOL_REGISTRY)


def clear_tool_registry():
    """Clear the global tool registry (mainly for testing)."""
    _GLOBAL_TOOL_REGISTRY.clear()
