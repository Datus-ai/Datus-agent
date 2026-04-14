"""Service for direct tool dispatch via tool name."""

import inspect
from typing import Any, Callable, Dict

from datus.api.models.base_models import Result
from datus.api.models.config_models import ErrorCode
from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ToolService:
    """Registry-based tool dispatch: tool_name -> bound method."""

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self._context_search_tools = ContextSearchTools(agent_config)
        self._registry: Dict[str, Callable[..., FuncToolResult]] = self._build_registry()

    def _build_registry(self) -> Dict[str, Callable[..., FuncToolResult]]:
        """Build a flat dict mapping tool_name -> bound method."""
        registry: Dict[str, Callable[..., FuncToolResult]] = {}

        # ContextSearchTools methods
        context_tool_names = [
            "list_subject_tree",
            "search_metrics",
            "get_metrics",
            "search_reference_sql",
            "get_reference_sql",
            "search_semantic_objects",
            "search_knowledge",
            "get_knowledge",
        ]
        for name in context_tool_names:
            method = getattr(self._context_search_tools, name, None)
            if method and callable(method):
                registry[name] = method

        return registry

    @property
    def registered_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return sorted(self._registry.keys())

    def execute(self, tool_name: str, params: Dict[str, Any]) -> Result[FuncToolResult]:
        """Dispatch tool_name with params and return wrapped Result."""
        method = self._registry.get(tool_name)
        if method is None:
            return Result(
                success=False,
                errorCode=ErrorCode.TOOL_NOT_FOUND,
                errorMessage=f"Tool '{tool_name}' not found. Available tools: {', '.join(self.registered_tools)}",
            )

        # Validate params against method signature
        sig = inspect.signature(method)
        try:
            sig.bind(**params)
        except TypeError as e:
            return Result(
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=f"Invalid parameters for tool '{tool_name}': {e}",
            )

        try:
            result = method(**params)
            return Result(success=True, data=result)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.TOOL_EXECUTION_ERROR,
                errorMessage=str(e),
            )
