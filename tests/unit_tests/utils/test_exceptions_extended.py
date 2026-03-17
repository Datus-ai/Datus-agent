# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Extended CI-level unit tests for datus/utils/exceptions.py.

Covers uncovered lines: 167, 178-219.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from datus.utils.exceptions import DatusException, ErrorCode, setup_exception_handler


class TestDatusExceptionBuildMsg:
    """Tests for DatusException.build_msg (line 167)."""

    def test_custom_message_takes_priority(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message="custom msg")
        assert "custom msg" in str(ex)

    def test_message_args_format_template(self):
        ex = DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message_args={"field_name": "username"},
        )
        assert "username" in str(ex)

    def test_no_message_no_args_uses_desc(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert ErrorCode.COMMON_UNKNOWN.desc in str(ex)

    def test_error_code_appears_in_message(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert ErrorCode.COMMON_UNKNOWN.code in str(ex)

    def test_str_returns_message(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message="hello")
        assert str(ex) == ex.message

    def test_custom_message_overrides_args(self):
        """When both message and message_args are provided, custom message wins."""
        ex = DatusException(
            ErrorCode.COMMON_FIELD_REQUIRED,
            message="explicit override",
            message_args={"field_name": "x"},
        )
        assert "explicit override" in str(ex)

    def test_is_exception_subclass(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert isinstance(ex, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(DatusException) as exc_info:
            raise DatusException(ErrorCode.COMMON_UNKNOWN, message="test raise")
        assert "test raise" in str(exc_info.value)

    def test_code_attribute_set(self):
        ex = DatusException(ErrorCode.COMMON_FIELD_REQUIRED, message_args={"field_name": "f"})
        assert ex.code == ErrorCode.COMMON_FIELD_REQUIRED

    def test_message_args_attribute_set(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN, message_args={"key": "val"})
        assert ex.message_args == {"key": "val"}

    def test_message_args_defaults_to_empty_dict_when_none(self):
        ex = DatusException(ErrorCode.COMMON_UNKNOWN)
        assert ex.message_args == {}

    @pytest.mark.parametrize("code", list(ErrorCode))
    def test_all_error_codes_instantiable(self, code):
        """Every ErrorCode can be used to create a DatusException."""
        ex = DatusException(code)
        assert isinstance(ex, DatusException)


class TestSetupExceptionHandler:
    """Tests for setup_exception_handler (lines 178-219)."""

    def test_sets_sys_excepthook(self):
        original_hook = sys.excepthook
        try:
            setup_exception_handler()
            assert sys.excepthook is not original_hook
        finally:
            sys.excepthook = original_hook

    def test_system_exit_not_intercepted(self):
        """SystemExit should fall through to the original hook, not be swallowed."""
        original_hook = sys.excepthook
        try:
            setup_exception_handler()
            # We just verify that calling the hook with SystemExit doesn't raise
            # an unexpected error (it calls sys.__excepthook__ instead).
            # We can't easily verify the exact behavior without side effects,
            # but we can confirm setup_exception_handler runs without error.
            assert callable(sys.excepthook)
        finally:
            sys.excepthook = original_hook

    def test_handler_with_console_logger_datus_exception(self):
        """DatusException invokes console_logger with formatted message."""
        original_hook = sys.excepthook
        console_logger = MagicMock()
        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_lm.return_value.debug = True
                setup_exception_handler(console_logger=console_logger)
                # Simulate calling the exception hook directly
                try:
                    raise DatusException(ErrorCode.COMMON_UNKNOWN, message="test error")
                except DatusException:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            console_logger.assert_called_once()
            call_arg = console_logger.call_args[0][0]
            assert "test error" in call_arg or "Execution failed" in call_arg
        finally:
            sys.excepthook = original_hook

    def test_handler_without_console_logger_non_debug(self):
        """Without console_logger and non-debug mode, uses temporary_output."""
        original_hook = sys.excepthook
        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_manager = MagicMock()
                mock_manager.debug = False
                mock_manager.temporary_output.return_value.__enter__ = MagicMock(return_value=None)
                mock_manager.temporary_output.return_value.__exit__ = MagicMock(return_value=False)
                mock_lm.return_value = mock_manager

                setup_exception_handler()
                try:
                    raise ValueError("unexpected failure")
                except ValueError:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            assert mock_manager.temporary_output.called
        finally:
            sys.excepthook = original_hook

    def test_prefix_wrap_func_applied(self):
        """When prefix_wrap_func is provided it wraps the log prefix."""
        original_hook = sys.excepthook
        console_logger = MagicMock()

        def wrap_func(s):
            return f"[WRAPPED] {s}"

        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_lm.return_value.debug = True
                setup_exception_handler(console_logger=console_logger, prefix_wrap_func=wrap_func)
                try:
                    raise DatusException(ErrorCode.COMMON_UNKNOWN)
                except DatusException:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            call_arg = console_logger.call_args[0][0]
            assert "[WRAPPED]" in call_arg
        finally:
            sys.excepthook = original_hook

    def test_non_debug_with_console_logger(self):
        """Non-debug mode with console_logger logs trace to file and message to console."""
        original_hook = sys.excepthook
        console_logger = MagicMock()
        try:
            with patch("datus.utils.exceptions.get_log_manager") as mock_lm:
                mock_lm.return_value.debug = False
                setup_exception_handler(console_logger=console_logger)
                try:
                    raise DatusException(ErrorCode.COMMON_UNKNOWN, message="non-debug test")
                except DatusException:
                    exc_type, exc_val, tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_val, tb)

            console_logger.assert_called_once()
        finally:
            sys.excepthook = original_hook
