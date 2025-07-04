import logging
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from logging.handlers import TimedRotatingFileHandler
from typing import Literal

import structlog

fileno = False

# Global log manager
_log_manager = None


class DynamicLogManager:
    """Dynamic log manager that supports switching log output targets at runtime"""

    def __init__(self, debug=False, log_dir="logs"):
        self.debug = debug
        self.log_dir = log_dir
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
        file_formatter = logging.Formatter("%(message)s")
        self.file_handler.setFormatter(file_formatter)

        # Create console handler
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


def configure_logging(debug=False, log_dir="logs", console_output=True) -> DynamicLogManager:
    """Configure logging with the specified debug level.
    Args:
        debug: If True, set log level to DEBUG
        log_dir: Directory for log files
        console_output: If False, disable logging to console
    """
    global fileno
    fileno = debug

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
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
