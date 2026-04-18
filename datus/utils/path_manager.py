# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Centralized path management for all .datus related directories and files.

This module provides a unified interface for managing all paths related to the
.datus directory structure. The home directory is determined from agent.yml config.

Storage layout (refactored):

- ``{project_root}/subject/{semantic_models, sql_summaries, ext_knowledge}/``
  — knowledge-base content lives alongside the project so every CWD gets its
  own copy. There is no per-namespace subdirectory anymore.
- ``{project_root}/.datus/skills/`` — project-level skills.
- ``{datus_home}/sessions/{project_name}/`` — sessions sharded by project.
- ``{datus_home}/data/`` — storage-backend root.  Each backend owns its own
  project isolation strategy (e.g. a ``{project_name}/`` subdirectory for
  file-based backends, a schema/collection name for remote backends).
  Non-backend callers that still want a project-scoped on-disk location (e.g.
  document/ storage) should use :pyattr:`DatusPathManager.project_data_dir`.
- ``{datus_home}/{conf, logs, template, run, benchmark, workspace, skills, ...}``
  — global, shared across projects.
"""

import re
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Optional, Union

from datus.utils.exceptions import DatusException, ErrorCode

PathLike = Union[str, Path]

# Defense-in-depth guard for the project_name path segment.  AgentConfig already
# validates or normalizes project_name before it reaches the path manager, but
# this class is a public API and takes project_name directly in tests/SaaS
# callers, so we re-check to keep ``~/.datus/data/{project_name}`` and
# ``~/.datus/sessions/{project_name}`` safely sandboxed.
_PROJECT_NAME_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


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
        project_name: Optional[str] = None,
        project_root: Optional[PathLike] = None,
    ):
        """
        Initialize the path manager.

        Args:
            datus_home: Custom .datus root directory. If None, defaults to ~/.datus
            project_name: Logical project identifier used to shard ``sessions/`` and
                ``data/`` under ``datus_home``. Callers should pass a sanitized name
                (e.g. via ``datus.configuration.agent_config._normalize_project_name``).
                Non-project callers that only need global paths (``conf``/``logs``/...)
                may leave this empty; accessing ``project_data_dir`` or
                ``sessions_dir`` without a project_name raises ``DatusException``.
            project_root: Root directory for project-scoped KB content (the
                ``subject/`` tree and ``.datus/skills``). Defaults to ``Path.cwd()``.
        """
        self._datus_home = self.resolve_home(datus_home)
        self._project_name = self._validate_project_name_segment(project_name)
        self._project_root = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()

    @staticmethod
    def _validate_project_name_segment(project_name: Optional[str]) -> str:
        """Reject project names that would escape or reshape the shard directory."""
        if not project_name:
            return ""
        if not _PROJECT_NAME_SEGMENT_RE.match(project_name):
            raise ValueError(
                f"Invalid project_name {project_name!r}: must match "
                f"{_PROJECT_NAME_SEGMENT_RE.pattern} (no path separators or traversal "
                f"components). Sanitize via datus.configuration.agent_config."
                f"_normalize_project_name before constructing DatusPathManager."
            )
        return project_name

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
        Prefer creating a new ``DatusPathManager(new_home, ...)`` and passing it
        explicitly.
        """
        import warnings

        warnings.warn(
            "DatusPathManager.update_home is deprecated; construct a new DatusPathManager instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._datus_home = self.resolve_home(new_home)

    @property
    def datus_home(self) -> Path:
        """Root .datus directory path"""
        return self._datus_home

    @property
    def project_name(self) -> str:
        """Logical project name used to shard ``sessions/`` and ``data/``."""
        return self._project_name

    @property
    def project_root(self) -> Path:
        """Project root directory (typically the CWD at startup)."""
        return self._project_root

    @property
    def conf_dir(self) -> Path:
        """Configuration directory: ~/.datus/conf"""
        return self._datus_home / "conf"

    @property
    def data_dir(self) -> Path:
        """Storage-backend root: ``~/.datus/data``.

        This is intentionally project-agnostic so each storage backend can own
        its isolation strategy (e.g. a ``{project_name}/`` subdirectory, a
        schema name, a collection prefix).  Callers that need an on-disk
        project-scoped directory for non-backend use should use
        :pyattr:`project_data_dir` instead.
        """
        return self._datus_home / "data"

    @property
    def project_data_dir(self) -> Path:
        """Project-scoped local data directory: ``~/.datus/data/{project_name}``.

        Requires ``project_name`` to be set; raises ``DatusException`` when it
        is empty. Intended for non-backend callers (e.g. ``document/`` storage)
        that want a project-sharded on-disk location. Storage backends should
        use :pyattr:`data_dir` and apply their own isolation.
        """
        if not self._project_name:
            raise DatusException(
                ErrorCode.STORAGE_FAILED,
                message="project_data_dir requires a non-empty project_name.",
            )
        return self._datus_home / "data" / self._project_name

    @property
    def logs_dir(self) -> Path:
        """Logs directory: ~/.datus/logs"""
        return self._datus_home / "logs"

    @property
    def sessions_dir(self) -> Path:
        """Sessions directory: ``~/.datus/sessions/{project_name}``.

        Requires ``project_name`` to be set; raises ``DatusException`` when it
        is empty.
        """
        if not self._project_name:
            raise DatusException(
                ErrorCode.STORAGE_FAILED,
                message="sessions_dir requires a non-empty project_name.",
            )
        return self._datus_home / "sessions" / self._project_name

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
    def subject_dir(self) -> Path:
        """Project-scoped knowledge-base root: ``{project_root}/subject``."""
        return self._project_root / "subject"

    @property
    def semantic_models_dir(self) -> Path:
        """Semantic models directory: ``{project_root}/subject/semantic_models``."""
        return self.subject_dir / "semantic_models"

    @property
    def sql_summaries_dir(self) -> Path:
        """SQL summaries directory: ``{project_root}/subject/sql_summaries``."""
        return self.subject_dir / "sql_summaries"

    @property
    def ext_knowledge_dir(self) -> Path:
        """ext knowledge directory: ``{project_root}/subject/ext_knowledge``."""
        return self.subject_dir / "ext_knowledge"

    @property
    def project_skills_dir(self) -> Path:
        """Project-level skills directory: ``{project_root}/.datus/skills``."""
        return self._project_root / ".datus" / "skills"

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
        "subject": "subject_dir",
        "semantic_models": "semantic_models_dir",
        "sql_summaries": "sql_summaries_dir",
        "ext_knowledge": "ext_knowledge_dir",
        "project_skills": "project_skills_dir",
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
        RAG storage path (unified per project).

        Returns:
            Path: ``{project_data_dir}/datus_db`` (i.e. ``{home}/data/{project_name}/datus_db``)
        """
        path = self.project_data_dir / "datus_db"
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    def session_db_path(self, session_id: str) -> Path:
        """
        Session database file path.

        Args:
            session_id: Session identifier

        Returns:
            Path: ``{sessions_dir}/{session_id}.db``
        """
        # Ensure the parent directory exists
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        return self.sessions_dir / f"{session_id}.db"

    def semantic_model_path(self) -> Path:
        """
        Semantic model directory for the current project.

        Returns:
            Path: ``{project_root}/subject/semantic_models``
        """
        path = self.semantic_models_dir
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sql_summary_path(self) -> Path:
        """
        SQL summary directory for the current project.

        Returns:
            Path: ``{project_root}/subject/sql_summaries``
        """
        path = self.sql_summaries_dir
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ext_knowledge_path(self) -> Path:
        """
        External knowledge directory for the current project.

        Returns:
            Path: ``{project_root}/subject/ext_knowledge``
        """
        path = self.ext_knowledge_dir
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
                   benchmark, save, workspace, trajectory, subject, semantic_models,
                   sql_summaries, ext_knowledge, project_skills

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
# ``project_name`` / ``project_root`` and any other instance-level state survives the
# ContextVar round-trip.
_current_path_manager: ContextVar[Optional["DatusPathManager"]] = ContextVar("datus_current_path_manager", default=None)


def set_current_path_manager(
    path_manager: Optional[Union["DatusPathManager", PathLike]] = None,
    *,
    agent_config: Optional[Any] = None,
) -> Token:
    """Set the current context-local path manager used by ``get_path_manager()``.

    The full ``DatusPathManager`` instance is stored, so ``project_name`` /
    ``project_root`` and any instance-level state is preserved for implicit callers.
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
