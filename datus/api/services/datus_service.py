"""Per-project agent service facade.

Consolidates all agent services (chat, cli, database, explorer, mcp, kb)
and a project-scoped ChatTaskManager into a single cached instance.
"""

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

from datus.configuration.agent_config import AgentConfig
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    # Import-time only: these modules import back into the service layer, and
    # the accessors below import them lazily inside the function body to keep
    # that cycle from forming at runtime.
    from datus.api.services.explorer_service import ExplorerService
    from datus.api.services.tool_service import ToolService

logger = get_logger(__name__)

# Constant marker folded into the fingerprint when the plugin snapshot itself
# fails. Constant (not an error string) so a failing snapshot keeps producing
# one stable fingerprint instead of evicting the cached service every request.
_PLUGIN_STATE_UNAVAILABLE = "unavailable"


def _plugin_state(agent_config: AgentConfig) -> Any:
    """Return the plugin-state snapshot to fold into the config fingerprint.

    Kept outside the ``dataclasses.asdict`` failure path: a config object that
    predates the accessor contributes ``None`` rather than degrading the whole
    fingerprint to the ``id:`` fallback (which would defeat content-based
    caching for hosts that build a fresh AgentConfig per request).
    """
    getter = getattr(agent_config, "plugin_state_signature", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception as e:
        logger.warning(f"Failed to snapshot plugin state for AgentConfig fingerprint: {e}")
        return _PLUGIN_STATE_UNAVAILABLE


class DatusService:
    """Per-project agent service facade.

    Heavy sub-services (database, cli, explorer, mcp, kb) are lazy-loaded
    via properties. Since the event loop is single-threaded, simple None
    checks are sufficient (no locking needed).
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        project_id: str,
        default_source: "str | None" = None,
        default_interactive: bool = True,
        stream_thinking: bool = False,
    ):
        self._agent_config = agent_config
        self._project_id = project_id
        self._config_fingerprint = self.compute_fingerprint(agent_config)

        # ChatTaskManager — project-scoped (not process-level singleton)
        from datus.api.services.chat_task_manager import ChatTaskManager

        self._task_manager = ChatTaskManager(
            default_source=default_source,
            default_interactive=default_interactive,
            stream_thinking=stream_thinking,
        )

        # Lazy service slots
        self._chat = None
        self._cli = None
        self._datasource = None
        # Keyed by sub-agent name, not a single slot: these two services carry a
        # scope filter, and DatusServiceCache keys only on project, so one slot
        # would serve the first caller's scope to every later sub-agent.
        self._explorers: Dict[Optional[str], "ExplorerService"] = {}
        self._mcp = None
        self._kb = None
        self._visualization = None
        self._tools: Dict[Optional[str], "ToolService"] = {}
        self._success_story = None
        self._dashboard = None
        self._report = None

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def agent_config(self) -> AgentConfig:
        return self._agent_config

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    @staticmethod
    def compute_fingerprint(agent_config: AgentConfig) -> str:
        """Compute a stable content-based fingerprint for an AgentConfig.

        Hashes the declared dataclass fields plus
        ``AgentConfig.plugin_state_signature()`` — the plugin master switch,
        activation whitelist, ``agent.plugins`` profiles and ``plugin_paths``.
        None of that is reachable through ``dataclasses.asdict``, so without it
        a plugin being toggled, re-profiled, re-pinned or remounted would keep
        serving the cached ``DatusService`` built from the previous state.

        Falls back to an id-based string if the config cannot be serialized.
        """
        try:
            payload = {
                "config": dataclasses.asdict(agent_config),
                "plugins": _plugin_state(agent_config),
            }
            serialized = json.dumps(payload, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute AgentConfig fingerprint, falling back to id(): {e}")
            return f"id:{id(agent_config)}"

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def task_manager(self):
        return self._task_manager

    def has_active_tasks(self) -> bool:
        """Return True if any chat task is still running."""
        return self._task_manager.has_active_tasks()

    # ------------------------------------------------------------------
    # Lazy service properties
    # ------------------------------------------------------------------

    @property
    def chat(self):
        if self._chat is None:
            from datus.api.services.chat_service import ChatService

            self._chat = ChatService(
                agent_config=self._agent_config,
                task_manager=self._task_manager,
                project_id=self._project_id,
            )
        return self._chat

    @property
    def cli(self):
        if self._cli is None:
            from datus.api.services.cli_service import CLIService

            self._cli = CLIService(agent_config=self._agent_config, chat_service=self.chat)
        return self._cli

    @property
    def datasource(self):
        if self._datasource is None:
            from datus.api.services.database_service import DatasourceService

            self._datasource = DatasourceService(agent_config=self._agent_config)
        return self._datasource

    def has_sub_agent(self, sub_agent_name: str) -> bool:
        """Whether ``sub_agent_name`` is a configured ``agentic_nodes`` key.

        The key, not an entry's ``id``. ``AgentConfig.sub_agent_config`` looks up
        that mapping directly, so only a key resolves to a scope.
        """
        nodes = getattr(self._agent_config, "agentic_nodes", None) or {}
        return sub_agent_name in nodes

    def _require_known_sub_agent(self, sub_agent_name: Optional[str]) -> None:
        """Refuse a name that resolves to no scope.

        ``None`` is not a name — it means "unscoped", the single-tenant default.
        Any other value must be a configured key, because an unrecognised one
        would build a service whose scope filter is ``None``: a successful,
        completely unfiltered read. That turns a caller's identifier mistake —
        passing an ``id`` instead of a key, a typo, a name left over from a
        rename — into a scope bypass, which is the exact failure this scoping
        exists to prevent. Fail closed instead.

        The message deliberately does not list the configured sub-agents: on a
        deployment that publishes one sub-agent to a consumer, enumerating the
        others is a disclosure in its own right.
        """
        if sub_agent_name is None or self.has_sub_agent(sub_agent_name):
            return
        raise DatusException(
            code=ErrorCode.COMMON_UNSUPPORTED,
            message_args={"your_value": sub_agent_name, "field_name": "sub_agent_name"},
        )

    def explorer_for(self, sub_agent_name: Optional[str] = None) -> "ExplorerService":
        """ExplorerService scoped to ``sub_agent_name`` (``None`` = unscoped).

        One instance per name. A single lazy slot would be worse than the
        unscoped status quo: the first request's sub-agent would be cached and
        every later request, whatever sub-agent it named, would silently read
        through that scope. Wrong results, no error.

        An unknown name raises and is never cached — see
        :meth:`_require_known_sub_agent`.
        """
        self._require_known_sub_agent(sub_agent_name)
        if sub_agent_name not in self._explorers:
            from datus.api.services.explorer_service import ExplorerService

            self._explorers[sub_agent_name] = ExplorerService(
                agent_config=self._agent_config, sub_agent_name=sub_agent_name
            )
        return self._explorers[sub_agent_name]

    @property
    def explorer(self) -> "ExplorerService":
        """The unscoped ExplorerService. Kept so existing callers are unchanged."""
        return self.explorer_for(None)

    @property
    def mcp(self):
        if self._mcp is None:
            from datus.api.services.mcp_service import MCPService

            self._mcp = MCPService(agent_config=self._agent_config)
        return self._mcp

    @property
    def kb(self):
        if self._kb is None:
            from datus.api.services.kb_service import KbService

            self._kb = KbService(agent_config=self._agent_config)
        return self._kb

    def tool_for(self, sub_agent_name: Optional[str] = None) -> "ToolService":
        """ToolService scoped to ``sub_agent_name`` (``None`` = unscoped).

        Per-name, and unknown names refused, for the same reasons as
        :meth:`explorer_for`.
        """
        self._require_known_sub_agent(sub_agent_name)
        if sub_agent_name not in self._tools:
            from datus.api.services.tool_service import ToolService

            self._tools[sub_agent_name] = ToolService(agent_config=self._agent_config, sub_agent_name=sub_agent_name)
        return self._tools[sub_agent_name]

    @property
    def tool(self) -> "ToolService":
        """The unscoped ToolService. Kept so existing callers are unchanged."""
        return self.tool_for(None)

    @property
    def success_story(self):
        if self._success_story is None:
            from datus.api.services.success_story_service import SuccessStoryService

            self._success_story = SuccessStoryService(agent_config=self._agent_config)
        return self._success_story

    @property
    def visualization(self):
        if self._visualization is None:
            from datus.api.services.visualization_service import DataVisualizationService

            self._visualization = DataVisualizationService(agent_config=self._agent_config)
        return self._visualization

    @property
    def dashboard(self):
        if self._dashboard is None:
            from datus.api.services.dashboard_service import DashboardService

            self._dashboard = DashboardService(agent_config=self._agent_config)
        return self._dashboard

    @property
    def report(self):
        if self._report is None:
            from datus.api.services.report_service import ReportService

            self._report = ReportService(agent_config=self._agent_config)
        return self._report

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self):
        """Shutdown all sub-services. Called when evicted from cache."""
        try:
            await self._task_manager.shutdown()
        except Exception:
            logger.exception(f"Error shutting down task_manager for project {self._project_id}")
