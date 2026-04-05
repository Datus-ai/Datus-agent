"""API Services module.

Consolidated service layer for Datus Agent API.
"""

# Core services
from datus.api.services.datus_service_cache import DatusServiceCache

# Lazy imports - services are imported only when needed by routes
# This avoids circular dependencies and import errors

__all__ = [
    "DatusServiceCache",
]


def __getattr__(name):
    """Lazy import of services on demand."""
    if name == "DatusService":
        from datus.api.services.datus_service import DatusService

        return DatusService
    elif name == "ChatService":
        from datus.api.services.chat_service import ChatService

        return ChatService
    elif name == "ChatTaskManager":
        from datus.api.services.chat_task_manager import ChatTaskManager

        return ChatTaskManager
    elif name == "ChatTask":
        from datus.api.services.chat_task_manager import ChatTask

        return ChatTask
    elif name == "CLIService":
        from datus.api.services.cli_service import CLIService

        return CLIService
    elif name == "DatabaseService":
        from datus.api.services.database_service import DatabaseService

        return DatabaseService
    elif name == "ExplorerService":
        from datus.api.services.explorer_service import ExplorerService

        return ExplorerService
    elif name == "MCPService":
        from datus.api.services.mcp_service import MCPService

        return MCPService
    elif name == "KbService":
        from datus.api.services.kb_service import KbService

        return KbService
    elif name == "action_to_sse_event":
        from datus.api.services.action_sse_converter import action_to_sse_event

        return action_to_sse_event
    elif name == "AgentService":
        from datus.api.services.agent_service import AgentService

        return AgentService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
