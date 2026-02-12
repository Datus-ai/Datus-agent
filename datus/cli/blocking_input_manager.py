# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Blocking input manager for truly isolated user input."""

import sys
import threading
import time
from typing import Any, Callable

# Re-export from utils so existing imports continue to work
from datus.utils.terminal_utils import suppress_keyboard_input  # noqa: F401


class BlockingInputManager:
    """Manages blocking user input with complete output isolation."""

    def __init__(self):
        self._input_lock = threading.Lock()
        self._output_redirected = False
        self._original_stdout = None
        self._original_stderr = None

    def get_blocking_input(self, prompt_func: Callable) -> Any:
        """Get user input with minimal output isolation."""
        with self._input_lock:
            try:
                # Just flush buffers and get input
                sys.stdout.flush()
                sys.stderr.flush()

                # Small delay to ensure everything is settled
                time.sleep(0.05)

                # Get user input
                result = prompt_func()
                return result
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\nInput cancelled by user")
                raise


# Global instance
blocking_input_manager = BlockingInputManager()
