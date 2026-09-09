# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import structlog
from rich.console import Console

from datus.configuration.logging_config import resolve_log_level, resolve_logging_arguments

fileno = False
_log_redact_config = None

_LITELLM_LOGGER_NAMES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")

# Global log manager
_log_manager = None

if TYPE_CHECKING:
    from datus.utils.path_manager import DatusPathManager


def _is_source_environment() -> bool:
    """Check if running from source code directory (development mode).

    Returns:
        True if running from source directory, False if packaged/installed
    """
    try:
        # Get the directory where this module is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to project root (from datus/utils/ to project root)
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # Check for source code markers: pyproject.toml and datus/ directory
        has_pyproject = os.path.exists(os.path.join(project_root, "pyproject.toml"))
        has_datus_dir = os.path.exists(os.path.join(project_root, "datus"))

        return has_pyproject and has_datus_dir
    except Exception:
        return False


class DynamicLogManager:
    """Dynamic log manager that supports switching log output targets at runtime"""

    def __init__(self, debug=False, log_dir=None, path_manager=None, agent_config=None, *, level=None):
        self.level = logging._nameToLevel[resolve_log_level(level=level, debug=debug)[0]]
        self.debug = self.level == logging.DEBUG
        # Default to ~/.datus/logs (via path manager) when log_dir is not specified.
        if log_dir is None:
            from datus.utils.path_manager import get_path_manager

            log_dir = str(get_path_manager(path_manager=path_manager, agent_config=agent_config).logs_dir)
        # Expand user directory and convert to absolute path
        self.log_dir = os.path.abspath(os.path.expanduser(log_dir))
        self.root_logger = logging.getLogger()
        self.file_handler = None
        self.console_handler = None
        self.original_handlers = []
        self._lock = threading.RLock()
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up file and console handlers"""
        os.makedirs(self.log_dir, exist_ok=True)

        # Create file handler
        from datetime import datetime

        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file_base = os.path.join(self.log_dir, f"agent.{current_date}")

        self.file_handler = TimedRotatingFileHandler(
            log_file_base + ".log", when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        self.file_handler.suffix = "%Y-%m-%d"

        self.file_handler.setFormatter(_log_formatter())
        self.file_handler.setLevel(self.level)

        # Create console handler with normal formatter
        self.console_handler = logging.StreamHandler(sys.stderr)
        self.console_handler.setFormatter(_log_formatter(colors=True))
        self.console_handler.setLevel(self.level)

        # Set up root logger
        self.root_logger.setLevel(self.level)
        self.original_handlers = self.root_logger.handlers.copy()

    def set_output_target(self, target: Literal["both", "file", "console", "none"]):
        """Set log output target

        Args:
            target: Output target
                - "both": Output to both file and console (default)
                - "file": Output to file only
                - "console": Output to console only
                - "none": No output
        """
        with self._lock:
            self.root_logger.handlers = []

            if target in ["both", "file"]:
                self.root_logger.addHandler(self.file_handler)

            if target in ["both", "console"]:
                self.root_logger.addHandler(self.console_handler)

    def restore_default(self):
        """Restore to default configuration (file + console)"""
        with self._lock:
            self.set_output_target("both")

    def restore_original(self):
        """Restore to original handler configuration"""
        with self._lock:
            self.root_logger.handlers = self.original_handlers.copy()

    @contextmanager
    def temporary_output(self, target: Literal["both", "file", "console", "none"]):
        """Context manager for temporarily setting output target

        Args:
            target: Temporary output target
        """
        with self._lock:
            original_handlers = self.root_logger.handlers.copy()
            try:
                self.set_output_target(target)
                yield
            finally:
                self.root_logger.handlers = original_handlers


def get_log_manager(
    *, path_manager: Optional["DatusPathManager"] = None, agent_config: Optional[Any] = None
) -> DynamicLogManager:
    """Get global log manager"""
    global _log_manager
    if _log_manager is None:
        _log_manager = DynamicLogManager(path_manager=path_manager, agent_config=agent_config)
    return _log_manager


def configure_litellm_logging(file_handler: Optional[logging.Handler] = None) -> None:
    """Route LiteLLM's verbose loggers away from the interactive console."""
    if file_handler is None and _log_manager is not None:
        file_handler = _log_manager.file_handler

    for logger_name in _LITELLM_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(max(logging.INFO, logging.getLogger().level))
        logger.addHandler(file_handler if file_handler is not None else logging.NullHandler())


def configure_logging(
    debug=False,
    log_dir=None,
    console_output=True,
    *,
    path_manager: Optional["DatusPathManager"] = None,
    agent_config: Optional[Any] = None,
    level: str | None = None,
    level_source: str | None = None,
    redact: dict[str, Any] | None = None,
) -> DynamicLogManager:
    """Configure logging with the specified debug level.
    Args:
        debug: If True, set log level to DEBUG
        log_dir: Directory for log files. If None, defaults to ``~/.datus/logs``
                 (resolved via the active ``DatusPathManager``).
        console_output: If False, disable logging to console
    """
    config = getattr(agent_config, "logging", None)
    effective_level, source = resolve_log_level(level=level, debug=debug, config=config)
    numeric_level = logging._nameToLevel[effective_level]
    from datus.observability.config import RedactConfig

    global _log_redact_config
    _log_redact_config = RedactConfig.from_dict(redact if redact is not None else getattr(config, "redact", None))
    _configure_structlog()
    # Retain targeted noise suppression without bypassing the selected threshold.
    for name in ("httpx", "httpcore", "openai.agents"):
        logging.getLogger(name).setLevel(max(logging.WARNING, numeric_level))
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        named_logger = logging.getLogger(name)
        named_logger.handlers.clear()
        named_logger.propagate = True
        named_logger.setLevel(numeric_level)
    for name in ("web_chatbot",):
        named_logger = logging.getLogger(name)
        named_logger.setLevel(numeric_level)
        for handler in named_logger.handlers:
            handler.setLevel(numeric_level)
    global fileno
    fileno = numeric_level == logging.DEBUG

    # Default to ~/.datus/logs (via path manager) when log_dir is not specified.
    if log_dir is None:
        from datus.utils.path_manager import get_path_manager

        log_dir = str(get_path_manager(path_manager=path_manager, agent_config=agent_config).logs_dir)

    # Create or get log manager with specified parameters
    global _log_manager
    previous_manager = _log_manager
    _log_manager = DynamicLogManager(
        level=effective_level,
        log_dir=log_dir,
        path_manager=path_manager,
        agent_config=agent_config,
    )
    _log_manager.configured = True

    try:
        import litellm  # noqa: F401
    except ModuleNotFoundError:
        pass
    configure_litellm_logging(_log_manager.file_handler)

    # Set output target based on console_output parameter
    if console_output:
        _log_manager.set_output_target("both")
    else:
        _log_manager.set_output_target("file")
    if previous_manager is not None:
        previous_manager.file_handler.close()
        previous_manager.console_handler.close()
    get_logger(__name__).info("logging.configured", log_level=effective_level, source=level_source or source)
    return _log_manager


def configure_entrypoint_logging(args: Any, *, if_unconfigured: bool = False, **kwargs: Any) -> DynamicLogManager:
    """Apply the shared configuration at an application/worker entry point."""
    config = resolve_logging_arguments(args, kwargs.get("agent_config"))
    if if_unconfigured and _log_manager is not None and getattr(_log_manager, "configured", False):
        # An in-process API app must retain its host's handler destinations.
        # Spawned workers have no configured manager and initialize normally.
        return _log_manager
    return configure_logging(
        level=config.level,
        level_source=args.log_level_source,
        redact=config.redact,
        **kwargs,
    )


def add_exc_info(logger, method_name, event_dict):
    """Compatibility processor: exception() and explicit exc_info own tracebacks."""
    return event_dict


def add_code_location(logger, method_name, event_dict):
    """Add the correct code location by inspecting the call stack."""
    if method_name == "debug" or fileno:
        try:
            frames = traceback.extract_stack()
            # Find the first frame that is not in structlog or logging modules
            for frame in reversed(frames[:-1]):  # Exclude the current frame
                if "structlog" not in frame.filename and "logging" not in frame.filename:
                    event_dict["fileno"] = f" {frame.filename}:{frame.lineno}"
                    break
        except Exception as e:
            print(str(e))
    return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


def setup_web_chatbot_logging(
    debug=False,
    log_dir=None,
    *,
    path_manager: Optional["DatusPathManager"] = None,
    agent_config: Optional[Any] = None,
    level: str | None = None,
):
    """Setup simplified logging for web chatbot using same format as agent.log

    Args:
        debug: Enable debug logging
        log_dir: Directory for log files. If None, defaults to ``~/.datus/logs``
                 (resolved via the active ``DatusPathManager``).

    Returns:
        structlog.BoundLogger: Configured logger for web chatbot
    """
    # Default to ~/.datus/logs (via path manager) when log_dir is not specified.
    if log_dir is None:
        from datus.utils.path_manager import get_path_manager

        log_dir = str(get_path_manager(path_manager=path_manager, agent_config=agent_config).logs_dir)

    # Expand user directory and convert to absolute path
    log_dir = os.path.abspath(os.path.expanduser(log_dir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create independent logger for web chatbot
    web_logger = logging.getLogger("web_chatbot")
    effective, _ = resolve_log_level(level=level, debug=debug, config=getattr(agent_config, "logging", None))
    web_logger.setLevel(effective)

    # Remove existing handlers to avoid duplicates
    web_logger.handlers.clear()

    # Create file handler with same naming pattern as agent.log
    from datetime import datetime

    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_base = os.path.join(log_dir, f"web_chatbot.{current_date}")

    file_handler = TimedRotatingFileHandler(
        log_file_base + ".log", when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"

    # Use same formatter as agent.log (simple message format)
    file_handler.setFormatter(_log_formatter())
    file_handler.setLevel(effective)

    web_logger.addHandler(file_handler)
    web_logger.propagate = False  # Prevent propagation to root logger

    return structlog.get_logger("web_chatbot")


@contextmanager
def log_context(target: Literal["both", "file", "console", "none"]):
    """Log output context manager

    Args:
        target: Output target

    Example:
        with log_context("console"):
            logger.info("This log will only output to console")
    """
    with get_log_manager().temporary_output(target):
        yield


class AdaptiveRenderer:
    """Adaptive renderer that uses colored output by default"""

    def __init__(self):
        self.colored_renderer = structlog.dev.ConsoleRenderer(
            colors=True, exception_formatter=structlog.dev.plain_traceback
        )

    def __call__(self, logger, name, event_dict):
        """Always use colored renderer - file handler will strip colors with its formatter"""
        return self.colored_renderer(logger, name, event_dict)


def _redact_log_event(logger, method_name, event_dict):
    # Independent of tracing being configured or enabled.
    from datus.observability.config import RedactConfig
    from datus.observability.privacy import redact_value

    return redact_value(event_dict, _log_redact_config or RedactConfig())


def _log_formatter(*, colors: bool = False) -> logging.Formatter:
    renderer = structlog.dev.ConsoleRenderer(colors=colors, exception_formatter=structlog.dev.plain_traceback)
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            _redact_log_event,
            renderer,
        ],
    )


def _configure_structlog():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_code_location,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


if not structlog.is_configured():
    _configure_structlog()


def _get_current_log_file() -> Path | None:
    """Try to locate the current agent log file.

    Checks the active log manager first and falls back to the latest
    agent log in the logs directory.
    """
    try:
        manager = get_log_manager()
        handler = getattr(manager, "file_handler", None)
        if handler and getattr(handler, "baseFilename", None):
            return Path(handler.baseFilename).expanduser().resolve()
    except Exception:
        # Fall through to the log-dir search
        pass

    try:
        from datus.utils.path_manager import get_path_manager

        # Utility helper exemption: called from deep exception-printing contexts
        # that have no access to agent_config; fall back to the context-local
        # path manager so at least the currently-active tenant's log dir is used.
        log_dir = get_path_manager().logs_dir
        if not log_dir.exists():
            return None
        log_files = sorted(log_dir.glob("agent.*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return log_files[0].resolve() if log_files else None
    except Exception:
        return None


def print_rich_exception(
    console: Console,
    ex: Exception,
    error_description: str = "Processed failed",
    file_logger: Optional[structlog.BoundLogger] = None,
) -> None:
    if not file_logger:
        file_logger = get_logger(__name__)
    """Print a concise, user-friendly error with a log file hint."""

    file_logger.error(f"{error_description}, Reason: {ex}", exc_info=(type(ex), ex, ex.__traceback__))
    log_file = _get_current_log_file()

    console.print(
        f" ❌ [bold][red]{error_description}[/], Reason: {str(ex)}. See error details in [cyan]{log_file}[/][/]"
    )
