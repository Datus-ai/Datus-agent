"""Application context — request authentication and configuration."""

from dataclasses import dataclass
from typing import Optional

from datus.configuration.agent_config import AgentConfig


@dataclass
class AppContext:
    """Request context with optional agent configuration."""

    user_id: str
    project_id: str = "default"
    config: Optional[AgentConfig] = None


@dataclass
class UserContext:
    """Lightweight context for agent-related routes without AgentConfig."""

    user_id: str
    workspace_id: str
    tenant_id: str
    project_id: Optional[str] = None
