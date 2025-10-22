# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Centralized path management for all .datus related directories and files.

This module provides a unified interface for managing all paths related to the
.datus directory structure. The home directory is determined from agent.yml config.
"""

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
    def output_dir(self) -> Path:
        """Output directory: ~/.datus/output"""
        return self._datus_home / "output"

    @property
    def metricflow_dir(self) -> Path:
        """MetricFlow directory: ~/.datus/metricflow"""
        return self._datus_home / "metricflow"

    @property
    def workspace_dir(self) -> Path:
        """Workspace directory: ~/.datus/workspace"""
        return self._datus_home / "workspace"

    @property
    def trajectory_dir(self) -> Path:
        """Trajectory directory: ~/.datus/trajectory"""
        return self._datus_home / "trajectory"

    @property
    def semantic_models_dir(self) -> Path:
        """Semantic models directory: ~/.datus/semantic_models"""
        return self._datus_home / "semantic_models"

    @property
    def sql_summaries_dir(self) -> Path:
        """SQL summaries directory: ~/.datus/sql_summaries"""
        return self._datus_home / "sql_summaries"

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

    def metricflow_config_path(self) -> Path:
        """MetricFlow environment settings: ~/.datus/metricflow/env_settings.yml"""
        return self.metricflow_dir / "env_settings.yml"

    def history_file_path(self) -> Path:
        """Command history file: ~/.datus/history"""
        return self._datus_home / "history"

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
        return self.data_dir / f"datus_db_{namespace}"

    def sub_agent_path(self, agent_name: str) -> Path:
        """
        Sub-agent storage path.

        Args:
            agent_name: Sub-agent name

        Returns:
            Path: ~/.datus/data/sub_agents/{agent_name}
        """
        return self.data_dir / "sub_agents" / agent_name

    def session_db_path(self, session_id: str) -> Path:
        """
        Session database file path.

        Args:
            session_id: Session identifier

        Returns:
            Path: ~/.datus/sessions/{session_id}.db
        """
        return self.sessions_dir / f"{session_id}.db"

    def semantic_model_path(self, namespace: str) -> Path:
        """
        Semantic model path for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Path: ~/.datus/semantic_models/{namespace}
        """
        return self.semantic_models_dir / namespace

    def sql_summary_path(self, namespace: str) -> Path:
        """
        SQL summary path for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Path: ~/.datus/sql_summaries/{namespace}
        """
        return self.sql_summaries_dir / namespace

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
        """
        if not dirs:
            # Ensure all standard directories
            standard_dirs = [
                self.conf_dir,
                self.data_dir,
                self.logs_dir,
                self.sessions_dir,
                self.template_dir,
                self.trajectory_dir,
                self.sample_dir,
                self.run_dir,
                self.benchmark_dir,
                self.output_dir,
                self.metricflow_dir,
                self.workspace_dir,
                self.semantic_models_dir,
                self.sql_summaries_dir,
            ]
            for directory in standard_dirs:
                directory.mkdir(parents=True, exist_ok=True)
        else:
            # Ensure specified directories
            for dir_name in dirs:
                directory = getattr(self, f"{dir_name}_dir", None)
                if directory:
                    directory.mkdir(parents=True, exist_ok=True)


# Global singleton instance
_path_manager: Optional[DatusPathManager] = None


def get_path_manager(datus_home: Optional[Path] = None) -> DatusPathManager:
    """
    Get the global path manager instance.

    Args:
        datus_home: Optional custom .datus root directory. Only used on first call.

    Returns:
        DatusPathManager instance
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = DatusPathManager(datus_home)
    return _path_manager


def reset_path_manager() -> None:
    """Reset the global path manager instance. Primarily for testing."""
    global _path_manager
    _path_manager = None
