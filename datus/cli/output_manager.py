# -*- coding: utf-8 -*-
"""Output manager for controlling console output during user interactions."""

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from typing import List


class OutputManager:
    """Manages console output to ensure clean user interactions."""

    def __init__(self):
        self._output_lock = threading.Lock()
        self._suppressed_outputs: List[str] = []
        self._is_suppressing = False

    @asynccontextmanager
    async def suppress_all_output(self):
        """Context manager to suppress all console output."""
        with self._output_lock:
            if self._is_suppressing:
                yield
                return

            self._is_suppressing = True
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            try:
                # Create a suppressor stream
                class OutputSuppressor:
                    def __init__(self, manager):
                        self.manager = manager

                    def write(self, text):
                        if self.manager._is_suppressing:
                            self.manager._suppressed_outputs.append(text)
                        else:
                            sys.__stdout__.write(text)

                    def flush(self):
                        pass

                # Replace stdout and stderr
                suppressor = OutputSuppressor(self)
                sys.stdout = suppressor
                sys.stderr = suppressor

                # Force async tasks to settle
                await asyncio.sleep(0.1)

                # Flush any remaining output
                original_stdout.flush()
                original_stderr.flush()

                yield

            finally:
                # Restore original streams
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                self._is_suppressing = False

                # Small delay to ensure everything is settled
                await asyncio.sleep(0.05)

    def get_suppressed_outputs(self) -> List[str]:
        """Get all suppressed outputs."""
        with self._output_lock:
            return self._suppressed_outputs.copy()

    def clear_suppressed_outputs(self):
        """Clear suppressed outputs."""
        with self._output_lock:
            self._suppressed_outputs.clear()


# Global instance
output_manager = OutputManager()
