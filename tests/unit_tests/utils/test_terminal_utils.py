# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/utils/terminal_utils.py.

Tests cover:
- suppress_keyboard_input context manager (including non-terminal fallback)
- interrupt_on_escape context manager (including non-terminal fallback)

Since CI often runs without a real terminal (stdin is piped), these tests
exercise the graceful fallback paths that yield without modifying terminal
settings.

NO MOCK EXCEPT LLM. All objects under test are real implementations.
"""

import io
import os
import sys
import time

import pytest

from datus.cli.execution_state import InterruptController
from datus.utils.terminal_utils import interrupt_on_escape, suppress_keyboard_input


class TestSuppressKeyboardInput:
    """Tests for suppress_keyboard_input context manager."""

    def test_suppress_keyboard_input_is_context_manager(self):
        """suppress_keyboard_input yields control and does not raise."""
        entered = False
        with suppress_keyboard_input():
            entered = True
        assert entered is True

    def test_suppress_keyboard_input_noop_when_no_terminal(self):
        """In non-terminal environments, suppress_keyboard_input is a no-op."""
        # Save original stdin and replace with a non-terminal stream
        original_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO("test input")
            executed = False
            with suppress_keyboard_input():
                executed = True
            assert executed is True
        finally:
            sys.stdin = original_stdin

    def test_suppress_keyboard_input_body_runs_to_completion(self):
        """Code inside the context manager runs to completion."""
        result = []
        with suppress_keyboard_input():
            result.append(1)
            result.append(2)
        assert result == [1, 2]


class TestInterruptOnEscape:
    """Tests for interrupt_on_escape context manager."""

    def test_interrupt_on_escape_is_context_manager(self):
        """interrupt_on_escape yields control and does not raise."""
        ctrl = InterruptController()
        entered = False
        with interrupt_on_escape(ctrl):
            entered = True
        assert entered is True
        # Controller should not be interrupted (no ESC pressed)
        assert ctrl.is_interrupted is False

    def test_interrupt_on_escape_noop_when_no_terminal(self):
        """In non-terminal environments, interrupt_on_escape is a no-op."""
        original_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO("test input")
            ctrl = InterruptController()
            executed = False
            with interrupt_on_escape(ctrl):
                executed = True
            assert executed is True
            assert ctrl.is_interrupted is False
        finally:
            sys.stdin = original_stdin

    def test_interrupt_on_escape_body_runs_to_completion(self):
        """Code inside the context manager runs to completion."""
        ctrl = InterruptController()
        result = []
        with interrupt_on_escape(ctrl):
            result.append("a")
            result.append("b")
        assert result == ["a", "b"]

    def test_interrupt_on_escape_with_real_terminal(self):
        """On real terminal, interrupt_on_escape sets up listener thread and cleans up."""
        ctrl = InterruptController()
        # This test verifies no exceptions occur during setup/teardown
        # regardless of whether we have a real terminal or not
        with interrupt_on_escape(ctrl):
            pass
        # After exiting, controller should still be unset (no ESC)
        assert ctrl.is_interrupted is False


class TestTerminalUtilsEdgeCases:
    """Edge case tests for terminal utilities."""

    def test_suppress_keyboard_input_exception_propagates(self):
        """Exceptions inside suppress_keyboard_input propagate correctly."""
        with pytest.raises(ValueError, match="test error"):
            with suppress_keyboard_input():
                raise ValueError("test error")

    def test_interrupt_on_escape_exception_propagates(self):
        """Exceptions inside interrupt_on_escape propagate correctly."""
        ctrl = InterruptController()
        with pytest.raises(ValueError, match="test error"):
            with interrupt_on_escape(ctrl):
                raise ValueError("test error")


class TestSuppressKeyboardInputWithPty:
    """Tests for suppress_keyboard_input using a pty for real terminal code paths."""

    @pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty not available on this platform")
    def test_suppress_keyboard_input_with_real_terminal(self):
        """suppress_keyboard_input modifies and restores terminal settings on a real pty."""
        import termios

        master_fd, slave_fd = os.openpty()
        try:
            # Save original stdin
            original_stdin = sys.stdin
            # Replace stdin with the slave side of the pty
            sys.stdin = os.fdopen(slave_fd, "r", closefd=False)
            try:
                old_settings = termios.tcgetattr(slave_fd)
                with suppress_keyboard_input():
                    # Inside: terminal settings should be modified
                    new_settings = termios.tcgetattr(slave_fd)
                    # IXON should be disabled (bit should be unset)
                    import termios as t

                    assert (new_settings[0] & t.IXON) == 0, "IXON should be disabled during suppress"

                # After: terminal settings should be restored
                restored = termios.tcgetattr(slave_fd)
                assert restored[0] == old_settings[0], "Input flags should be restored"
                assert restored[3] == old_settings[3], "Local flags should be restored"
            finally:
                sys.stdin = original_stdin
        finally:
            os.close(master_fd)
            os.close(slave_fd)

    @pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty not available on this platform")
    def test_suppress_keyboard_input_exception_restores_settings(self):
        """suppress_keyboard_input restores terminal settings even when exception occurs."""
        import termios

        master_fd, slave_fd = os.openpty()
        try:
            original_stdin = sys.stdin
            sys.stdin = os.fdopen(slave_fd, "r", closefd=False)
            try:
                old_settings = termios.tcgetattr(slave_fd)
                with pytest.raises(RuntimeError):
                    with suppress_keyboard_input():
                        raise RuntimeError("test error")
                restored = termios.tcgetattr(slave_fd)
                assert restored[0] == old_settings[0], "Settings restored after exception"
            finally:
                sys.stdin = original_stdin
        finally:
            os.close(master_fd)
            os.close(slave_fd)


class TestInterruptOnEscapeWithPty:
    """Tests for interrupt_on_escape using a pty for real terminal code paths."""

    @pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty not available on this platform")
    def test_interrupt_on_escape_starts_listener(self):
        """interrupt_on_escape starts a listener thread on a real pty."""
        master_fd, slave_fd = os.openpty()
        try:
            original_stdin = sys.stdin
            sys.stdin = os.fdopen(slave_fd, "r", closefd=False)
            try:
                ctrl = InterruptController()
                with interrupt_on_escape(ctrl):
                    # Listener should be running; controller should not be interrupted yet
                    assert ctrl.is_interrupted is False
                    time.sleep(0.15)  # Brief pause to let listener start
                # After exit, listener should be stopped and settings restored
                assert ctrl.is_interrupted is False
            finally:
                sys.stdin = original_stdin
        finally:
            os.close(master_fd)
            os.close(slave_fd)

    @pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty not available on this platform")
    def test_interrupt_on_escape_detects_esc_key(self):
        """interrupt_on_escape detects ESC key and triggers interrupt."""
        master_fd, slave_fd = os.openpty()
        try:
            original_stdin = sys.stdin
            sys.stdin = os.fdopen(slave_fd, "r", closefd=False)
            try:
                ctrl = InterruptController()
                with interrupt_on_escape(ctrl):
                    # Write ESC key to master side (simulates user pressing ESC)
                    os.write(master_fd, b"\x1b")
                    # Give the listener thread time to detect the ESC
                    time.sleep(0.3)
                assert ctrl.is_interrupted is True
            finally:
                sys.stdin = original_stdin
        finally:
            os.close(master_fd)
            os.close(slave_fd)

    @pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty not available on this platform")
    def test_interrupt_on_escape_restores_after_exception(self):
        """interrupt_on_escape restores terminal settings after exception."""
        import termios

        master_fd, slave_fd = os.openpty()
        try:
            original_stdin = sys.stdin
            sys.stdin = os.fdopen(slave_fd, "r", closefd=False)
            try:
                old_settings = termios.tcgetattr(slave_fd)
                ctrl = InterruptController()
                with pytest.raises(RuntimeError):
                    with interrupt_on_escape(ctrl):
                        raise RuntimeError("test")
                restored = termios.tcgetattr(slave_fd)
                assert restored[3] == old_settings[3], "Local flags should be restored"
            finally:
                sys.stdin = original_stdin
        finally:
            os.close(master_fd)
            os.close(slave_fd)
