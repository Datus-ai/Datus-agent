"""Per-project agent service facade.

Consolidates all agent services (chat, cli, database, explorer, mcp, kb)
and a project-scoped ChatTaskManager into a single cached instance.
"""

from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DatusService:
    """Per-project agent service facade.

    Heavy sub-services (database, cli, explorer, mcp, kb) are lazy-loaded
    via properties. Since the event loop is single-threaded, simple None
    checks are sufficient (no locking needed).
    """

    def __init__(self, agent_config: AgentConfig, project_id: str):
        self._agent_config = agent_config
        self._project_id = project_id

        # ChatTaskManager — project-scoped (not process-level singleton)
        from datus.api.services.chat_task_manager import ChatTaskManager

        self._task_manager = ChatTaskManager()

        # Lazy service slots
        self._chat = None
        self._cli = None
        self._database = None
        self._explorer = None
        self._mcp = None
        self._kb = None

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def agent_config(self) -> AgentConfig:
        return self._agent_config

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
    def database(self):
        if self._database is None:
            from datus.api.services.database_service import DatabaseService

            self._database = DatabaseService(agent_config=self._agent_config)
        return self._database

    @property
    def explorer(self):
        if self._explorer is None:
            from datus.api.services.explorer_service import ExplorerService

            self._explorer = ExplorerService(agent_config=self._agent_config)
        return self._explorer

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self):
        """Shutdown all sub-services. Called when evicted from cache."""
        try:
            await self._task_manager.shutdown()
        except Exception:
            logger.exception(f"Error shutting down task_manager for project {self._project_id}")
