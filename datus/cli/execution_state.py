# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Execution state management for user interaction flow control."""

import asyncio
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Callable, Optional


class ExecutionState(Enum):
    """Execution states for agent flow control."""

    RUNNING = "running"
    WAITING_USER_INPUT = "waiting_user_input"
    PAUSED = "paused"
    COMPLETED = "completed"


class ExecutionFlowController:
    """Controls execution flow during user interactions."""

    def __init__(self):
        self._state = ExecutionState.RUNNING
        self._state_lock = asyncio.Lock()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
        self._user_input_result: Optional[Any] = None
        self._user_input_event = asyncio.Event()
        self._live_display = None  # Reference to active Rich Live display

    async def get_state(self) -> ExecutionState:
        """Get current execution state."""
        async with self._state_lock:
            return self._state

    async def set_state(self, state: ExecutionState):
        """Set execution state."""
        async with self._state_lock:
            old_state = self._state
            self._state = state

            if state == ExecutionState.PAUSED:
                self._pause_event.clear()
            elif state == ExecutionState.RUNNING:
                self._pause_event.set()

            logger.debug(f"Execution state changed: {old_state} -> {state}")

    @asynccontextmanager
    async def pause_execution(self):
        """Context manager to pause execution during user interaction."""
        await self.set_state(ExecutionState.PAUSED)
        try:
            yield
        finally:
            await self.set_state(ExecutionState.RUNNING)

    async def wait_for_resume(self):
        """Wait for execution to resume."""
        await self._pause_event.wait()

    async def request_user_input(self, prompt_func: Callable) -> Any:
        """
        Request user input and pause execution until response.

        Args:
            prompt_func: Function that prompts user and returns their input

        Returns:
            User's response
        """
        await self.set_state(ExecutionState.WAITING_USER_INPUT)

        try:
            # More aggressive pause: create a blocking input environment
            import sys
            from concurrent.futures import ThreadPoolExecutor

            # Create a thread-safe input environment
            def blocking_input():
                # Disable stdout buffering for this thread
                sys.stdout.reconfigure(line_buffering=True)
                sys.stderr.reconfigure(line_buffering=True)

                try:
                    return prompt_func()
                finally:
                    # Restore buffering
                    sys.stdout.reconfigure(line_buffering=False)
                    sys.stderr.reconfigure(line_buffering=False)

            # Use dedicated executor for user input to ensure isolation
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="user_input") as executor:
                future = executor.submit(blocking_input)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, future.result)

            return result
        finally:
            await self.set_state(ExecutionState.RUNNING)
            self._user_input_event.clear()

    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return not self._pause_event.is_set()

    def is_waiting_user_input(self) -> bool:
        """Check if waiting for user input."""
        return self._state == ExecutionState.WAITING_USER_INPUT

    def register_live_display(self, live_display):
        """Register the active Rich Live display."""
        self._live_display = live_display

    def unregister_live_display(self):
        """Unregister the Rich Live display."""
        self._live_display = None

    def stop_live_display(self):
        """Stop the Rich Live display if active."""
        if self._live_display:
            try:
                self._live_display.stop()
                return True
            except Exception as e:
                logger.warning(f"Failed to stop live display: {e}")
        return False

    def resume_live_display(self):
        """Resume the Rich Live display if it was active."""
        if self._live_display:
            try:
                self._live_display.start()
                return True
            except Exception as e:
                logger.warning(f"Failed to resume live display: {e}")
        return False


# Global instance
execution_controller = ExecutionFlowController()


# Import logger lazily to avoid circular imports
def get_logger():
    from datus.utils.loggings import get_logger

    return get_logger(__name__)


logger = get_logger()
