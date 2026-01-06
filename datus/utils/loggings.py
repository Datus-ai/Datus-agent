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
from typing import Literal, Optional

import structlog
from rich.console import Console

fileno = False

# Global log manager
_log_manager = None


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

    def __init__(self, debug=False, log_dir=None):
        self.debug = debug
        # Auto-detect log directory if not specified
        if log_dir is None:
            if _is_source_environment():
                log_dir = "./logs"
            else:
                from datus.utils.path_manager import get_path_manager

                log_dir = str(get_path_manager().logs_dir)
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
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create file handler
        from datetime import datetime

        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file_base = os.path.join(self.log_dir, f"agent.{current_date}")

        self.file_handler = TimedRotatingFileHandler(
            log_file_base + ".log", when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        self.file_handler.suffix = "%Y-%m-%d"

        # Use a custom formatter for the file handler that removes color codes
        class PlainTextFormatter(logging.Formatter):
            def format(self, record):
                # Get the original message
                msg = super().format(record)
                # Remove ANSI color codes
                import re

                ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
                return ansi_escape.sub("", msg)

        file_formatter = PlainTextFormatter("%(message)s")
        self.file_handler.setFormatter(file_formatter)

        # Create console handler with normal formatter
        self.console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(message)s")
        self.console_handler.setFormatter(console_formatter)

        # Set up root logger
        self.root_logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
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


def get_log_manager() -> DynamicLogManager:
    """Get global log manager"""
    global _log_manager
    if _log_manager is None:
        _log_manager = DynamicLogManager()
    return _log_manager


def configure_logging(debug=False, log_dir=None, console_output=True) -> DynamicLogManager:
    """Configure logging with the specified debug level.
    Args:
        debug: If True, set log level to DEBUG
        log_dir: Directory for log files. If None, automatically determine:
                 - "./logs" for source code environment
                 - "~/.datus/logs" for packaged installation
        console_output: If False, disable logging to console
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    global fileno
    fileno = debug

    # Auto-detect log directory if not specified
    if log_dir is None:
        if _is_source_environment():
            log_dir = "./logs"
        else:
            from datus.utils.path_manager import get_path_manager

            log_dir = str(get_path_manager().logs_dir)

    # Create or get log manager with specified parameters
    global _log_manager
    _log_manager = DynamicLogManager(debug=debug, log_dir=log_dir)

    # Set output target based on console_output parameter
    if console_output:
        _log_manager.set_output_target("both")
    else:
        _log_manager.set_output_target("file")
    return _log_manager


def add_exc_info(logger, method_name, event_dict):
    """Add exception info to error logs."""
    if method_name == "error":
        event_dict["exc_info"] = True
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


def setup_web_chatbot_logging(debug=False, log_dir=None):
    """Setup simplified logging for web chatbot using same format as agent.log

    Args:
        debug: Enable debug logging
        log_dir: Directory for log files. If None, automatically determine:
                 - "./logs" for source code environment
                 - "~/.datus/logs" for packaged installation

    Returns:
        structlog.BoundLogger: Configured logger for web chatbot
    """
    # Auto-detect log directory if not specified
    if log_dir is None:
        if _is_source_environment():
            log_dir = "./logs"
        else:
            from datus.utils.path_manager import get_path_manager

            log_dir = str(get_path_manager().logs_dir)

    # Expand user directory and convert to absolute path
    log_dir = os.path.abspath(os.path.expanduser(log_dir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create independent logger for web chatbot
    web_logger = logging.getLogger("web_chatbot")
    web_logger.setLevel(logging.DEBUG if debug else logging.INFO)

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
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

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


if not structlog.is_configured():
    # Initialize event dict to avoid NoneType errors
    structlog.configure_once(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            add_code_location,
            add_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            AdaptiveRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


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

    file_logger.error(f"{error_description}, Reason: {ex}")
    log_file = _get_current_log_file()

    console.print(
        f" ❌ [bold][red]{error_description}[/], Reason: {str(ex)}. See error details in [cyan]{log_file}[/][/]"
    )
