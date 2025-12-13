# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import functools
import inspect
from abc import ABC
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

from agents.tool_context import ToolContext


@dataclass
class BaseToolExecResult:
    result: Any = field(init=True, default=None)
    success: bool = field(init=True, default=True)
    message: str = field(init=True, default="")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolAction:
    """Decorator for marking tool action methods"""

    registry = {}

    def __init__(self, name: Optional[str] = None, description: str = ""):
        self.name = name
        self.description = description

    def __call__(self, func: Callable):
        action_name = self.name or func.__name__

        @functools.wraps(func)
        def wrapper(instance, *args, **kwargs):
            return func(instance, *args, **kwargs)

        # Store method metadata
        wrapper.is_tool_action = True
        wrapper.action_name = action_name
        wrapper.description = self.description
        wrapper.signature = inspect.signature(func)

        return wrapper


class BaseTool(ABC):
    """Abstract base class for all tools.
    This provides a common interface and shared functionality for different tools.
    """

    tool_name = "base_tool"
    tool_description = "Base tool class"

    def __init__(self, **kwargs):
        """Initialize the tool with common parameters.

        Args:
            **kwargs: Additional tool-specific parameters
        """
        self.tool_params = kwargs
        self._actions = {}
        self._register_actions()
        self.tool_ctx = ContextVar("tool_ctx")

    def set_tool_context(self, tool_context: ToolContext):
        self.tool_ctx.set(tool_context)

    def _register_actions(self):
        """Register all methods decorated with ToolAction"""
        for name, method in inspect.getmembers(self.__class__):
            if hasattr(method, "is_tool_action") and method.is_tool_action:
                self._actions[method.action_name] = {
                    "method": getattr(self, name),
                    "description": method.description,
                    "signature": method.signature,
                }

    def get_actions(self) -> Dict[str, Dict]:
        """Get all available actions"""
        return self._actions

    def call_action(self, action_name: str, *args, **kwargs) -> Any:
        """Call the specified action method"""
        if action_name not in self._actions:
            raise ValueError(f"Action '{action_name}' not found in tool {self.tool_name}")

        return self._actions[action_name]["method"](*args, **kwargs)

    # @ToolAction(description="Execute the main functionality of the tool")
    # def execute(self, input_params: BaseInput) -> BaseResult:
    #    """Execute the tool's main functionality.

    #    Args:
    #        input_params: Dictionary containing the input parameters for the tool

    #    Returns:
    #        Dictionary containing the execution results and status
    #    """
    #    pass

    @classmethod
    def get_tool_manifest(cls) -> Dict[str, Any]:
        """Get the tool manifest information for MCP server registration"""
        instance = cls()
        actions = {}

        for action_name, action_info in instance.get_actions().items():
            params = {}
            for name, param in action_info["signature"].parameters.items():
                if name == "self":
                    continue
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                params[name] = {
                    "type": str(param_type),
                    "required": param.default == inspect.Parameter.empty,
                }

            actions[action_name] = {
                "description": action_info["description"],
                "parameters": params,
            }

        return {
            "name": cls.tool_name,
            "description": cls.tool_description,
            "actions": actions,
        }
