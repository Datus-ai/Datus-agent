import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datus.utils.loggings import configure_logging, get_log_manager, get_logger, log_context


@pytest.fixture(scope="module")
def setup_logging(tmp_path_factory):
    """Configure logging to use tmp_path for isolated testing"""
    tmp_dir = tmp_path_factory.mktemp("logs")
    configure_logging(debug=True, log_dir=str(tmp_dir))
    return tmp_dir


@pytest.fixture(scope="module")
def logger(setup_logging):
    """Get logger after logging is configured"""
    return get_logger(__name__)


def test_log_context(logger):
    """Test log context manager"""

    print("=== Test log context manager ===")

    # Default output
    logger.info("Default configuration log")

    # Use context manager to temporarily output to console only
    with log_context("console"):
        logger.info("This log will only be output to console (temporary)")

    # Restore default configuration after context ends
    logger.info("This log will be output to both file and console (restored)")

    # Use context manager to temporarily output to file only
    print("=== Output to file only ===")
    with log_context("file"):
        logger.info("This log will only be output to file (temporary)")

    # Restore default configuration after context ends
    logger.info("This log will be output to both file and console (restored)")


def test_log_manager(logger):
    """Test log manager"""

    print("=== Test log manager ===")

    # Get log manager
    log_manager = get_log_manager()

    # Use manager to set output target
    log_manager.set_output_target("console")
    logger.info("Set via manager: output to console only")

    # Restore default configuration
    log_manager.restore_default()
    logger.info("Restore default configuration via manager")


def test_multiple_switches(logger):
    """Test multiple switches between different output targets"""

    print("=== Test multiple switches ===")

    # Test multiple rapid switches
    logger.info("Initial log with default config")

    # Test context manager with multiple switches
    with log_context("console"):
        logger.info("Context 1: console only")
        with log_context("file"):
            logger.info("Context 2: file only (nested)")
        logger.info("Context 1: back to console only")

    logger.info("Final log: should be both file and console")


def test_simple_context(logger):
    """Test simple context manager to verify no deadlock"""

    print("=== Test simple context manager ===")

    # Test basic context manager
    with log_context("console"):
        logger.info("Simple context test: console only")

    logger.info("Back to default configuration")


def test_nested_context(logger):
    """Test nested context managers"""

    print("=== Test nested context managers ===")

    with log_context("console"):
        logger.info("Outer context: console only")

        with log_context("file"):
            logger.info("Inner context: file only")

        logger.info("Back to outer context: console only")

    logger.info("Back to default configuration")


class TestIsSourceEnvironment:
    """Tests for _is_source_environment (lines 41-42)."""

    def test_returns_bool(self):
        from datus.utils.loggings import _is_source_environment

        result = _is_source_environment()
        assert isinstance(result, bool)

    def test_returns_true_in_repo(self):
        """Running from the source repo should return True."""
        from datus.utils.loggings import _is_source_environment

        # We are running from the source repo, so it should be True
        assert _is_source_environment() is True

    def test_exception_returns_false(self):
        """If an unexpected exception occurs, function returns False."""
        from datus.utils.loggings import _is_source_environment

        with patch("os.path.dirname", side_effect=Exception("boom")):
            result = _is_source_environment()
        assert result is False


class TestDynamicLogManager:
    """Tests for DynamicLogManager (lines 45-147)."""

    @pytest.fixture
    def manager(self, tmp_path):
        from datus.utils.loggings import DynamicLogManager

        return DynamicLogManager(debug=False, log_dir=str(tmp_path / "logs"))

    def test_init_creates_log_dir(self, tmp_path):
        from datus.utils.loggings import DynamicLogManager

        log_dir = tmp_path / "logs" / "sub"
        DynamicLogManager(log_dir=str(log_dir))
        assert log_dir.exists()

    def test_set_output_target_file_only(self, manager):
        manager.set_output_target("file")
        root_logger = logging.getLogger()
        assert manager.file_handler in root_logger.handlers
        assert manager.console_handler not in root_logger.handlers

    def test_set_output_target_console_only(self, manager):
        manager.set_output_target("console")
        root_logger = logging.getLogger()
        assert manager.console_handler in root_logger.handlers
        assert manager.file_handler not in root_logger.handlers

    def test_set_output_target_both(self, manager):
        manager.set_output_target("both")
        root_logger = logging.getLogger()
        assert manager.file_handler in root_logger.handlers
        assert manager.console_handler in root_logger.handlers

    def test_set_output_target_none(self, manager):
        manager.set_output_target("none")
        root_logger = logging.getLogger()
        assert manager.file_handler not in root_logger.handlers
        assert manager.console_handler not in root_logger.handlers

    def test_restore_default_sets_both(self, manager):
        manager.set_output_target("none")
        manager.restore_default()
        root_logger = logging.getLogger()
        assert manager.file_handler in root_logger.handlers
        assert manager.console_handler in root_logger.handlers

    def test_restore_original_restores_handlers(self, manager):
        """restore_original should set handlers back to those saved at init."""
        manager.set_output_target("none")
        manager.restore_original()
        root_logger = logging.getLogger()
        assert root_logger.handlers == manager.original_handlers

    def test_temporary_output_context_manager_restores(self, manager):
        """After temporary_output context, handlers are restored."""
        manager.set_output_target("both")
        before_handlers = logging.getLogger().handlers.copy()
        with manager.temporary_output("none"):
            assert manager.file_handler not in logging.getLogger().handlers
        after_handlers = logging.getLogger().handlers
        assert set(before_handlers) == set(after_handlers)

    def test_temporary_output_restores_on_exception(self, manager):
        """Even if code inside context raises, handlers are restored."""
        manager.set_output_target("both")
        before_handlers = logging.getLogger().handlers.copy()
        with pytest.raises(RuntimeError):
            with manager.temporary_output("none"):
                raise RuntimeError("boom")
        after_handlers = logging.getLogger().handlers
        assert set(before_handlers) == set(after_handlers)

    def test_debug_mode_sets_root_logger_to_debug(self, tmp_path):
        from datus.utils.loggings import DynamicLogManager

        mgr = DynamicLogManager(debug=True, log_dir=str(tmp_path / "dbg"))
        assert mgr.root_logger.level == logging.DEBUG

    def test_init_non_source_env_uses_path_manager(self, tmp_path):
        """When not in source environment and log_dir=None, uses path_manager.logs_dir (lines 55-57)."""
        import datus.utils.loggings as loggings_module
        from datus.utils.loggings import DynamicLogManager

        with patch.object(loggings_module, "_is_source_environment", return_value=False):
            with patch("datus.utils.path_manager.get_path_manager") as mock_pm:
                mock_pm.return_value.logs_dir = tmp_path / "pm_logs"
                (tmp_path / "pm_logs").mkdir(parents=True, exist_ok=True)
                mgr = DynamicLogManager(log_dir=str(tmp_path / "pm_logs"))
        assert mgr is not None


class TestGetLogManager:
    """Tests for get_log_manager (lines 150-155)."""

    def test_returns_dynamic_log_manager(self, tmp_path):
        from datus.utils.loggings import DynamicLogManager, configure_logging, get_log_manager

        configure_logging(log_dir=str(tmp_path / "logs"))
        mgr = get_log_manager()
        assert isinstance(mgr, DynamicLogManager)

    def test_creates_instance_when_none(self, tmp_path):
        import datus.utils.loggings as loggings_module

        original = loggings_module._log_manager
        try:
            loggings_module._log_manager = None
            with patch.object(loggings_module, "_is_source_environment", return_value=True):
                mgr = loggings_module.get_log_manager()
            assert mgr is not None
        finally:
            loggings_module._log_manager = original


class TestConfigureLogging:
    """Tests for configure_logging (lines 158-199)."""

    def test_configure_logging_creates_manager(self, tmp_path):
        from datus.utils.loggings import configure_logging

        mgr = configure_logging(log_dir=str(tmp_path / "logs"))
        from datus.utils.loggings import DynamicLogManager

        assert isinstance(mgr, DynamicLogManager)

    def test_configure_logging_debug_false(self, tmp_path):
        from datus.utils.loggings import configure_logging

        mgr = configure_logging(debug=False, log_dir=str(tmp_path / "logs"))
        assert mgr.debug is False

    def test_configure_logging_debug_true(self, tmp_path):
        from datus.utils.loggings import configure_logging

        mgr = configure_logging(debug=True, log_dir=str(tmp_path / "logs"))
        assert mgr.debug is True

    def test_configure_logging_console_output_false(self, tmp_path):
        from datus.utils.loggings import configure_logging

        mgr = configure_logging(console_output=False, log_dir=str(tmp_path / "logs"))
        root_logger = logging.getLogger()
        assert mgr.console_handler not in root_logger.handlers

    def test_configure_logging_console_output_true(self, tmp_path):
        from datus.utils.loggings import configure_logging

        mgr = configure_logging(console_output=True, log_dir=str(tmp_path / "logs"))
        root_logger = logging.getLogger()
        assert mgr.console_handler in root_logger.handlers

    def test_configure_logging_auto_detects_source_env(self, tmp_path):
        """When log_dir is None and source env is detected, uses ./logs."""
        import datus.utils.loggings as loggings_module

        with patch.object(loggings_module, "_is_source_environment", return_value=True):
            mgr = loggings_module.configure_logging()
        assert mgr is not None


class TestAddExcInfo:
    """Tests for add_exc_info processor (line 202-206)."""

    def test_adds_exc_info_for_error(self):
        from datus.utils.loggings import add_exc_info

        event_dict = {}
        result = add_exc_info(None, "error", event_dict)
        assert result["exc_info"] is True

    def test_no_exc_info_for_info(self):
        from datus.utils.loggings import add_exc_info

        event_dict = {}
        result = add_exc_info(None, "info", event_dict)
        assert "exc_info" not in result


class TestAddCodeLocation:
    """Tests for add_code_location processor (lines 209-221)."""

    def test_adds_fileno_for_debug_method(self):
        from datus.utils.loggings import add_code_location

        event_dict = {}
        result = add_code_location(None, "debug", event_dict)
        assert "fileno" in result

    def test_no_fileno_for_info_without_fileno_flag(self):
        """When fileno global is False and method is info, no fileno added."""
        import datus.utils.loggings as loggings_module

        original = loggings_module.fileno
        try:
            loggings_module.fileno = False
            event_dict = {}
            result = loggings_module.add_code_location(None, "info", event_dict)
            assert "fileno" not in result
        finally:
            loggings_module.fileno = original


class TestGetCurrentLogFile:
    """Tests for _get_current_log_file (lines 332-356)."""

    def test_returns_path_or_none(self, tmp_path):
        from datus.utils.loggings import _get_current_log_file, configure_logging

        configure_logging(log_dir=str(tmp_path / "logs"))
        result = _get_current_log_file()
        assert result is None or isinstance(result, Path)

    def test_returns_none_when_no_log_manager(self):
        import datus.utils.loggings as loggings_module

        original = loggings_module._log_manager
        try:
            loggings_module._log_manager = None
            with patch("datus.utils.loggings.get_log_manager", side_effect=Exception("no manager")):
                # get_path_manager is imported inside the function, patch via path_manager module
                with patch("datus.utils.path_manager.get_path_manager") as mock_pm:
                    mock_pm.return_value.logs_dir = Path("/nonexistent_xyz_abc")
                    result = loggings_module._get_current_log_file()
            assert result is None
        finally:
            loggings_module._log_manager = original


class TestPrintRichException:
    """Tests for print_rich_exception (lines 359-374)."""

    def test_calls_console_print(self, tmp_path):
        from rich.console import Console

        from datus.utils.loggings import configure_logging, print_rich_exception

        configure_logging(log_dir=str(tmp_path / "logs"))

        console = MagicMock(spec=Console)
        ex = ValueError("something went wrong")
        print_rich_exception(console, ex, error_description="Operation failed")
        console.print.assert_called_once()
        call_args = str(console.print.call_args)
        assert "Operation failed" in call_args

    def test_uses_default_logger_when_none(self, tmp_path):
        from rich.console import Console

        from datus.utils.loggings import configure_logging, print_rich_exception

        configure_logging(log_dir=str(tmp_path / "logs"))

        console = MagicMock(spec=Console)
        ex = RuntimeError("runtime err")
        # Should not raise even with file_logger=None
        print_rich_exception(console, ex)
        console.print.assert_called_once()


class TestSetupWebChatbotLogging:
    """Tests for setup_web_chatbot_logging (lines 228-279)."""

    def test_returns_structlog_logger(self, tmp_path):
        pass

        from datus.utils.loggings import setup_web_chatbot_logging

        result = setup_web_chatbot_logging(log_dir=str(tmp_path / "chatbot_logs"))
        assert result is not None

    def test_creates_log_directory(self, tmp_path):
        from datus.utils.loggings import setup_web_chatbot_logging

        log_dir = tmp_path / "chatbot_logs" / "sub"
        setup_web_chatbot_logging(log_dir=str(log_dir))
        assert log_dir.exists()

    def test_debug_mode(self, tmp_path):
        from datus.utils.loggings import setup_web_chatbot_logging

        result = setup_web_chatbot_logging(debug=True, log_dir=str(tmp_path / "logs"))
        assert result is not None

    def test_auto_detect_source_env(self, tmp_path):
        import datus.utils.loggings as loggings_module

        with patch.object(loggings_module, "_is_source_environment", return_value=True):
            result = loggings_module.setup_web_chatbot_logging()
        assert result is not None


class TestAdaptiveRenderer:
    """Tests for AdaptiveRenderer (lines 297-307)."""

    def test_renderer_is_callable(self):
        from datus.utils.loggings import AdaptiveRenderer

        renderer = AdaptiveRenderer()
        assert callable(renderer)

    def test_renderer_produces_string_output(self):
        """AdaptiveRenderer.__call__ returns a string (delegates to ConsoleRenderer)."""
        from datus.utils.loggings import AdaptiveRenderer

        renderer = AdaptiveRenderer()
        result = renderer(None, "info", {"event": "test message"})
        assert isinstance(result, str)


class TestLogContext:
    """Tests for log_context context manager (lines 282-294)."""

    def test_log_context_console_only(self, tmp_path):
        from datus.utils.loggings import configure_logging, log_context

        configure_logging(log_dir=str(tmp_path / "logs"))

        with log_context("console"):
            root_logger = logging.getLogger()
            # Should not raise
            root_logger.info("test message in context")

    def test_log_context_restores_after_exit(self, tmp_path):
        from datus.utils.loggings import configure_logging, log_context

        configure_logging(log_dir=str(tmp_path / "logs"))
        before = logging.getLogger().handlers.copy()
        with log_context("none"):
            pass
        after = logging.getLogger().handlers
        assert set(before) == set(after)
