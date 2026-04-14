# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Centralized path management for all .datus related directories and files.

This module provides a unified interface for managing all paths related to the
.datus directory structure. The home directory is determined from agent.yml config.
"""

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


class DatusPathManager:
    """
    Centralized manager for all .datus related paths.

    The home directory can be customized via agent.yml config (agent.home).
    If not configured, defaults to ~/.datus

    Example:
        >>> from datus.utils.path_manager import get_path_manager
        >>> pm = get_path_manager()
        >>> config_path = pm.agent_config_path()
        >>> sessions_dir = pm.sessions_dir
    """

    def __init__(
        self,
        datus_home: Optional[PathLike] = None,
        knowledge_base_home: Optional[PathLike] = None,
    ):
        """
        Initialize the path manager.

        Args:
            datus_home: Custom .datus root directory. If None, defaults to ~/.datus
            knowledge_base_home: Custom root for knowledge-base directories
                (``semantic_models``, ``sql_summaries``, ``ext_knowledge``).
                If None, falls back to ``datus_home`` so behavior is unchanged.
        """
        self._datus_home = self.resolve_home(datus_home)
        self._knowledge_base_home = (
            Path(knowledge_base_home).expanduser().resolve() if knowledge_base_home else self._datus_home
        )

    @staticmethod
    def resolve_home(datus_home: Optional[PathLike] = None) -> Path:
        """Resolve a configured home path or fall back to ``~/.datus``."""
        if datus_home:
            return Path(datus_home).expanduser().resolve()
        return (Path.home() / ".datus").resolve()

    def update_home(self, new_home: PathLike) -> None:
        """
        Update the datus home directory.

        Deprecated compatibility helper.
        Prefer creating a new ``DatusPathManager(new_home)`` and passing it explicitly.

        Also resets ``knowledge_base_home`` to track ``new_home`` to avoid stale cross-tenant
        state in long-running processes. Callers that need a separate knowledge root
        must construct a new ``DatusPathManager(new_home, knowledge_base_home=...)`` instead.

        Args:
            new_home: New home directory path (can include ~)
        """
        import warnings

        warnings.warn(
            "DatusPathManager.update_home is deprecated; construct a new DatusPathManager instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._datus_home = self.resolve_home(new_home)
        # Reset knowledge_base_home to match so stale KB paths from a prior tenant don't leak.
        self._knowledge_base_home = self._datus_home

    @property
    def datus_home(self) -> Path:
        """Root .datus directory path"""
        return self._datus_home

    @property
    def knowledge_base_home(self) -> Path:
        """Root directory for knowledge-base data (semantic_models, sql_summaries, ext_knowledge)."""
        return self._knowledge_base_home

    @property
    def conf_dir(self) -> Path:
        """Configuration directory: ~/.datus/conf"""
        return self._datus_home / "conf"

    @property
    def data_dir(self) -> Path:
        """Data directory: ~/.datus/data"""
        return self._datus_home / "data"

    @property
    def logs_dir(self) -> Path:
        """Logs directory: ~/.datus/logs"""
        return self._datus_home / "logs"

    @property
    def sessions_dir(self) -> Path:
        """Sessions directory: ~/.datus/sessions"""
        return self._datus_home / "sessions"

    @property
    def template_dir(self) -> Path:
        """Template directory: ~/.datus/template"""
        return self._datus_home / "template"

    @property
    def sample_dir(self) -> Path:
        """Sample directory: ~/.datus/sample"""
        return self._datus_home / "sample"

    @property
    def run_dir(self) -> Path:
        """Runtime directory: ~/.datus/run"""
        return self._datus_home / "run"

    @property
    def benchmark_dir(self) -> Path:
        """Benchmark directory: ~/.datus/benchmark"""
        return self._datus_home / "benchmark"

    @property
    def save_dir(self) -> Path:
        """Save directory: ~/.datus/save"""
        return self._datus_home / "save"

    @property
    def workspace_dir(self) -> Path:
        """Workspace directory: ~/.datus/workspace"""
        return self._datus_home / "workspace"

    @property
    def trajectory_dir(self) -> Path:
        """Trajectory directory: ~/.datus/trajectory"""
        return self._datus_home / "trajectory"

    @staticmethod
    def resolve_run_dir(base: Path, namespace: str, run_id: Optional[str] = None) -> Path:
        """Resolve a namespaced run directory, creating it if needed.

        Args:
            base: Base directory (e.g. save_dir or trajectory_dir, may be overridden)
            namespace: Namespace name
            run_id: Optional run identifier (timestamp). If None, returns namespace dir only.

        Returns:
            Path: {base}/{namespace}/{run_id} or {base}/{namespace}
        """
        path = base / namespace
        if run_id:
            path = path / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def semantic_models_dir(self) -> Path:
        """Semantic models directory: {knowledge_base_home}/semantic_models"""
        return self._knowledge_base_home / "semantic_models"

    @property
    def sql_summaries_dir(self) -> Path:
        """SQL summaries directory: {knowledge_base_home}/sql_summaries"""
        return self._knowledge_base_home / "sql_summaries"

    @property
    def ext_knowledge_dir(self) -> Path:
        """ext knowledge directory: {knowledge_base_home}/ext_knowledge"""
        return self._knowledge_base_home / "ext_knowledge"

    # Valid directory names mapping
    _VALID_DIR_NAMES = {
        "conf": "conf_dir",
        "data": "data_dir",
        "logs": "logs_dir",
        "sessions": "sessions_dir",
        "template": "template_dir",
        "sample": "sample_dir",
        "run": "run_dir",
        "benchmark": "benchmark_dir",
        "save": "save_dir",
        "workspace": "workspace_dir",
        "trajectory": "trajectory_dir",
        "semantic_models": "semantic_models_dir",
        "sql_summaries": "sql_summaries_dir",
        "ext_knowledge": "ext_knowledge_dir",
    }

    # Configuration file paths

    def agent_config_path(self) -> Path:
        """Agent configuration file: ~/.datus/conf/agent.yml"""
        return self.conf_dir / "agent.yml"

    def mcp_config_path(self) -> Path:
        """MCP configuration file: ~/.datus/conf/.mcp.json"""
        return self.conf_dir / ".mcp.json"

    def auth_config_path(self) -> Path:
        """Authentication configuration file: ~/.datus/conf/auth_clients.yml"""
        return self.conf_dir / "auth_clients.yml"

    def history_file_path(self) -> Path:
        """Command history file: ~/.datus/history"""
        return self._datus_home / "history"

    def dashboard_path(self) -> Path:
        return self._datus_home / "dashboard"

    def pid_file_path(self, service_name: str = "datus-agent-api") -> Path:
        """
        PID file path for a service.

        Args:
            service_name: Service name for the PID file

        Returns:
            Path to PID file: ~/.datus/run/{service_name}.pid
        """
        return self.run_dir / f"{service_name}.pid"

    # Data paths

    def rag_storage_path(self) -> Path:
        """
        RAG storage path (unified for all namespaces).

        Returns:
            Path: ~/.datus/data/datus_db
        """
        path = self.data_dir / "datus_db"
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sub_agent_path(self, agent_name: str) -> Path:
        """
        Sub-agent storage path.

        Args:
            agent_name: Sub-agent name

        Returns:
            Path: ~/.datus/data/sub_agents/{agent_name}
        """
        path = self.data_dir / "sub_agents" / agent_name
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    def session_db_path(self, session_id: str) -> Path:
        """
        Session database file path.

        Args:
            session_id: Session identifier

        Returns:
            Path: ~/.datus/sessions/{session_id}.db
        """
        # Ensure the parent directory exists
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        return self.sessions_dir / f"{session_id}.db"

    def semantic_model_path(self, namespace: str) -> Path:
        """
        Semantic model path for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Path: ~/.datus/semantic_models/{namespace}
        """
        path = self.semantic_models_dir / namespace
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sql_summary_path(self, namespace: str) -> Path:
        """
        SQL summary path for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Path: ~/.datus/sql_summaries/{namespace}
        """
        path = self.sql_summaries_dir / namespace
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ext_knowledge_path(self, namespace: str) -> Path:
        """
        Ext knowledge path for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Path: ~/.datus/ext_knowledge/{namespace}
        """
        path = self.ext_knowledge_dir / namespace
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Utility methods

    def resolve_config_path(self, filename: str, local_path: Optional[str] = None) -> Path:
        """
        Resolve configuration file path with priority order.

        Priority:
        1. Explicit local_path if provided and exists
        2. Current directory conf/{filename}
        3. ~/.datus/conf/{filename}

        Args:
            filename: Configuration filename
            local_path: Optional explicit path to check first

        Returns:
            Resolved path (may not exist)
        """
        # 1. Check explicit path
        if local_path:
            explicit_path = Path(local_path).expanduser()
            if explicit_path.exists():
                return explicit_path

        # 2. Check current directory
        local_conf = Path("conf") / filename
        if local_conf.exists():
            return local_conf

        # 3. Default to ~/.datus/conf
        return self.conf_dir / filename

    def ensure_dirs(self, *dirs: str) -> None:
        """
        Ensure specified directories exist, creating them if necessary.

        Args:
            *dirs: Directory names to ensure. If empty, ensures all standard directories.
                   Valid names: conf, data, logs, sessions, template, sample, run,
                   benchmark, save, workspace, trajectory, semantic_models,
                   sql_summaries

        Raises:
            ValueError: If an invalid directory name is provided
        """
        if not dirs:
            # Ensure all standard directories
            for attr_name in self._VALID_DIR_NAMES.values():
                directory = getattr(self, attr_name)
                directory.mkdir(parents=True, exist_ok=True)
        else:
            # Ensure specified directories with validation
            for dir_name in dirs:
                if dir_name not in self._VALID_DIR_NAMES:
                    valid_names = ", ".join(sorted(self._VALID_DIR_NAMES.keys()))
                    raise ValueError(f"Invalid directory name '{dir_name}'. Valid names are: {valid_names}")
                attr_name = self._VALID_DIR_NAMES[dir_name]
                directory = getattr(self, attr_name)
                directory.mkdir(parents=True, exist_ok=True)

    def ensure_templates(self):
        """
        init prompt templates
        :return:
        """
        self.ensure_dirs("template")
        from datus.utils.resource_utils import copy_data_file

        # copy new version templates
        copy_data_file(resource_path="prompts/prompt_templates", target_dir=self.template_dir, replace=False)


# Context-local path manager for legacy helpers that do not receive ``AgentConfig`` directly.
# Unlike the previous process-wide fallback, this does not leak across threads/tasks.
# We store the full ``DatusPathManager`` instance (not just the home path string) so that
# ``knowledge_base_home`` and any other instance-level state survives the ContextVar round-trip.
_current_path_manager: ContextVar[Optional["DatusPathManager"]] = ContextVar("datus_current_path_manager", default=None)


def set_current_path_manager(
    path_manager: Optional[Union["DatusPathManager", PathLike]] = None,
    *,
    agent_config: Optional[Any] = None,
) -> Token:
    """Set the current context-local path manager used by ``get_path_manager()``.

    The full ``DatusPathManager`` instance is stored, so ``knowledge_base_home`` and
    any instance-level state is preserved for implicit callers.
    """
    if path_manager is None and agent_config is not None:
        path_manager = getattr(agent_config, "path_manager", None)

    if isinstance(path_manager, DatusPathManager):
        resolved: Optional["DatusPathManager"] = path_manager
    elif path_manager:
        resolved = DatusPathManager(path_manager)
    else:
        resolved = None

    return _current_path_manager.set(resolved)


def get_path_manager(
    datus_home: Optional[PathLike] = None,
    *,
    path_manager: Optional["DatusPathManager"] = None,
    agent_config: Optional[Any] = None,
) -> DatusPathManager:
    """
    Get a path manager instance.

    Resolution order:
    1. Explicit ``path_manager`` argument
    2. Explicit ``agent_config.path_manager``
    3. Explicit ``datus_home`` argument
    4. Context-local ``DatusPathManager`` set via ``set_current_path_manager()``
    5. ``~/.datus``

    Args:
        datus_home: Optional custom .datus root directory.
        path_manager: Optional explicit path manager instance to reuse.
        agent_config: Optional config object exposing ``path_manager``.

    Returns:
        DatusPathManager instance
    """
    if path_manager is not None:
        return path_manager

    if agent_config is not None:
        config_path_manager = getattr(agent_config, "path_manager", None)
        if config_path_manager is not None:
            return config_path_manager

    if datus_home is not None:
        return DatusPathManager(datus_home)

    current = _current_path_manager.get()
    if current is not None:
        return current
    return DatusPathManager()


def reset_path_manager(token: Optional[Token] = None) -> None:
    """
    Reset context-local path-manager defaults. Primarily for testing.
    """
    if token is not None:
        _current_path_manager.reset(token)
        return
    _current_path_manager.set(None)
