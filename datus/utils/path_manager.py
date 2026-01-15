# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Centralized path management for all .datus related directories and files.

This module provides a unified interface for managing all paths related to the
.datus directory structure. The home directory is determined from agent.yml config.
"""

import threading
from pathlib import Path
from typing import Optional


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

    def __init__(self, datus_home: Optional[str] = None):
        """
        Initialize the path manager.

        Args:
            datus_home: Custom .datus root directory. If None, defaults to ~/.datus
        """
        if datus_home:
            self._datus_home = Path(datus_home).expanduser().resolve()
        else:
            self._datus_home = Path.home() / ".datus"

    def update_home(self, new_home: str) -> None:
        """
        Update the datus home directory.

        This is called after loading agent config to apply the configured home path.

        Args:
            new_home: New home directory path (can include ~)
        """
        self._datus_home = Path(new_home).expanduser().resolve()

    @property
    def datus_home(self) -> Path:
        """Root .datus directory path"""
        return self._datus_home

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

    def trajectory_run_dir(self, namespace: str, run_id: Optional[str] = None) -> Path:
        """
        Get trajectory directory for a specific namespace and run.

        Args:
            namespace: Namespace name
            run_id: Optional run identifier (timestamp). If None, returns namespace dir only.

        Returns:
            Path: ~/.datus/trajectory/{namespace}/{run_id} or ~/.datus/trajectory/{namespace}
        """
        base = self.trajectory_dir / namespace
        if run_id:
            path = base / run_id
            path.mkdir(parents=True, exist_ok=True)
            return path
        base.mkdir(parents=True, exist_ok=True)
        return base

    def save_run_dir(self, namespace: str, run_id: Optional[str] = None) -> Path:
        """
        Get save directory for a specific namespace and run.

        Args:
            namespace: Namespace name
            run_id: Optional run identifier (timestamp). If None, returns namespace dir only.

        Returns:
            Path: ~/.datus/save/{namespace}/{run_id} or ~/.datus/save/{namespace}
        """
        base = self.save_dir / namespace
        if run_id:
            path = base / run_id
            path.mkdir(parents=True, exist_ok=True)
            return path
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def semantic_models_dir(self) -> Path:
        """Semantic models directory: ~/.datus/semantic_models"""
        return self._datus_home / "semantic_models"

    @property
    def sql_summaries_dir(self) -> Path:
        """SQL summaries directory: ~/.datus/sql_summaries"""
        return self._datus_home / "sql_summaries"

    @property
    def ext_knowledge_dir(self) -> Path:
        """ext knowledge directory: ~/.datus/ext_knowledge"""
        return self._datus_home / "ext_knowledge"

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

    def rag_storage_path(self, namespace: str) -> Path:
        """
        RAG storage path for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Path: ~/.datus/data/datus_db_{namespace}
        """
        path = self.data_dir / f"datus_db_{namespace}"
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
                    raise ValueError(f"Invalid directory name '{dir_name}'. " f"Valid names are: {valid_names}")
                attr_name = self._VALID_DIR_NAMES[dir_name]
                directory = getattr(self, attr_name)
                directory.mkdir(parents=True, exist_ok=True)


# Global singleton instance and lock for thread-safe initialization
_path_manager: Optional[DatusPathManager] = None
_path_manager_lock = threading.Lock()


def get_path_manager(datus_home: Optional[Path] = None) -> DatusPathManager:
    """
    Get the global path manager instance (thread-safe singleton).

    Uses double-checked locking to ensure thread-safe initialization
    without holding the lock on every access.

    Args:
        datus_home: Optional custom .datus root directory. Only used on first call.

    Returns:
        DatusPathManager instance
    """
    global _path_manager

    # First check (without lock) - fast path for already initialized instance
    if _path_manager is None:
        # Acquire lock for initialization
        with _path_manager_lock:
            # Second check (with lock) - ensure another thread didn't initialize
            if _path_manager is None:
                _path_manager = DatusPathManager(datus_home)

    return _path_manager


def reset_path_manager() -> None:
    """
    Reset the global path manager instance. Primarily for testing.

    Thread-safe: Acquires lock before resetting to prevent race conditions.
    """
    global _path_manager
    with _path_manager_lock:
        _path_manager = None
