"""Service for direct tool dispatch via tool name."""

import inspect
from typing import Any, Callable, Dict, Optional

from datus.api.models.base_models import Result
from datus.api.models.config_models import ErrorCode
from datus.configuration.agent_config import AgentConfig
from datus.storage.embedding_diagnostics import format_context_unavailable
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ToolService:
    """Registry-based tool dispatch: tool_name -> bound method."""

    CONTEXT_TOOL_NAMES = {
        "list_subject_tree",
        "search_metrics",
        "get_metrics",
        "search_reference_sql",
        "get_reference_sql",
        "search_semantic_objects",
    }

    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None) -> None:
        """Build the tool registry, optionally scoped to one sub-agent.

        ``sub_agent_name`` must be the ``agentic_nodes`` key, never a config
        entry's ``id``: ``AgentConfig.sub_agent_config`` is a plain lookup on
        that mapping, so an ``id`` misses, yields ``{}``, and the scope filter
        silently degrades to "no filter" — a 200 with unrestricted results
        rather than an error. Callers resolve id -> key before getting here.
        """
        self._agent_config = agent_config
        self._sub_agent_name = sub_agent_name
        self.context_warning = ""
        try:
            self._context_search_tools = ContextSearchTools(agent_config, sub_agent_name=sub_agent_name)
        except Exception as exc:
            self._context_search_tools = None
            self.context_warning = format_context_unavailable(exc)
            logger.warning("Context search tool registry disabled: %s", self.context_warning)
        self._registry: Dict[str, Callable[..., FuncToolResult]] = self._build_registry()

    def _build_registry(self) -> Dict[str, Callable[..., FuncToolResult]]:
        """Build a flat dict mapping tool_name -> bound method."""
        registry: Dict[str, Callable[..., FuncToolResult]] = {}

        # ContextSearchTools methods
        if self._context_search_tools is None:
            return registry

        for name in self.CONTEXT_TOOL_NAMES:
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
            if self.context_warning and tool_name in self.CONTEXT_TOOL_NAMES:
                return Result(
                    success=False,
                    errorCode=ErrorCode.TOOL_EXECUTION_ERROR,
                    errorMessage=self.context_warning,
                )
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
