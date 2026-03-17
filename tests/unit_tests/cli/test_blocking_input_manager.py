# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/blocking_input_manager.py

Tests cover:
- BlockingInputManager initialization
- get_blocking_input: normal call, KeyboardInterrupt handling
- Global instance existence
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from datus.cli.blocking_input_manager import BlockingInputManager, blocking_input_manager


class TestBlockingInputManagerInit:
    def test_init_attributes(self):
        mgr = BlockingInputManager()
        assert isinstance(mgr._input_lock, type(threading.Lock()))
        assert mgr._output_redirected is False
        assert mgr._original_stdout is None
        assert mgr._original_stderr is None

    def test_global_instance_exists(self):
        assert blocking_input_manager is not None
        assert isinstance(blocking_input_manager, BlockingInputManager)


class TestBlockingInputManagerGetBlockingInput:
    def test_calls_prompt_func_and_returns_result(self):
        mgr = BlockingInputManager()
        prompt_func = MagicMock(return_value="user_input")

        with patch("time.sleep"):
            result = mgr.get_blocking_input(prompt_func)

        assert result == "user_input"
        prompt_func.assert_called_once()

    def test_flushes_stdout_stderr(self):
        mgr = BlockingInputManager()
        prompt_func = MagicMock(return_value="ok")

        with patch("sys.stdout") as mock_stdout, patch("sys.stderr") as mock_stderr, patch("time.sleep"):
            mgr.get_blocking_input(prompt_func)

        mock_stdout.flush.assert_called_once()
        mock_stderr.flush.assert_called_once()

    def test_keyboard_interrupt_propagates(self):
        mgr = BlockingInputManager()
        prompt_func = MagicMock(side_effect=KeyboardInterrupt)

        with patch("time.sleep"), pytest.raises(KeyboardInterrupt):
            mgr.get_blocking_input(prompt_func)

    def test_lock_used_for_thread_safety(self):
        """Verify the lock is acquired during input."""
        mgr = BlockingInputManager()
        call_log = []

        def prompt_func():
            call_log.append(mgr._input_lock.locked())
            return "result"

        with patch("time.sleep"):
            result = mgr.get_blocking_input(prompt_func)

        assert result == "result"
        # During the call the lock should have been held (True recorded)
        assert True in call_log
