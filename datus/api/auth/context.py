"""Application context — request authentication and configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from datus.configuration.agent_config import AgentConfig


@dataclass
class AppContext:
    """Request context with optional agent configuration.

    - ``user_id``: identifier from the auth provider; ``None`` means anonymous.
      Used as ``SessionManager.scope`` to isolate sessions per user.
    - ``project_id``: optional project identifier; ``None`` means the single
      (default) project.
    - ``config``: optional preloaded ``AgentConfig``; when ``None``,
      ``get_datus_service`` loads it on demand.
    - ``policy_context``: request-scoped inputs consumed by active policy
      plugins. Authentication and authorization happen before this boundary.
    - ``sub_agent_name``: which sub-agent's ``scoped_context`` bounds the
      knowledge-base reads on this request; ``None`` means unscoped, which is
      the single-tenant default.
    """

    user_id: Optional[str] = None
    project_id: Optional[str] = None
    config: Optional[AgentConfig] = None
    policy_context: Dict[str, Any] = field(default_factory=dict)
    # Per request, because a hosting deployment serves many sub-agents from one
    # cached DatusService. The AuthProvider fills it from whatever transport
    # convention that host uses, so Agent needs no knowledge of the header.
    #
    # Must be the ``agentic_nodes`` key, not an entry's ``id`` — only a key
    # resolves to a scope. A value that resolves to nothing is refused with 400
    # by ``deps.get_scoped_sub_agent`` before any scoped service is built, so a
    # provider that sets an ``id`` here gets an error rather than an unfiltered
    # read. Providers should still resolve id -> key so callers see a working
    # request rather than a rejected one.
    sub_agent_name: Optional[str] = None
