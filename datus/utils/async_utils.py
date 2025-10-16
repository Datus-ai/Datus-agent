# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Robust async utilities for running async code in various contexts.
Handles both synchronous and asynchronous environments gracefully.
"""

import asyncio
import logging
import sys
import threading
import weakref
from typing import Any, Coroutine, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Thread-local storage for tracking nested calls and loop ownership
_local = threading.local()

# Track all loops created by this module for cleanup
_created_loops = weakref.WeakSet()


def setup_windows_policy():
    """
    Setup Windows-specific event loop policy for better compatibility.
    ProactorEventLoop has limitations in terms of subprocess pipelines.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def is_event_loop_running() -> bool:
    """
    Check if an event loop is currently running.

    This is more robust than just checking get_running_loop().

    Returns:
        True if an event loop is running, False otherwise.
    """
    try:
        loop = asyncio.get_running_loop()
        # Double check that the loop is actually running
        return loop is not None and loop.is_running() and not loop.is_closed()
    except RuntimeError:
        # No running loop in current context
        return False


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop or create a new one if necessary.

    Returns:
        An event loop instance.

    Note:
        This function does NOT handle the case where a loop is already running.
        Use `run_async` for that scenario.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _created_loops.add(loop)
        return loop
    except RuntimeError:
        # No event loop in current thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _created_loops.add(loop)
        return loop


def run_async(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """
    Smart async coroutine runner that works in any context.

    This function can be called from:
    - Synchronous code (will create and manage event loop)
    - Inside an async function (will use thread pool)
    - From a thread with or without an event loop

    Args:
        coro: Coroutine to run
        timeout: Optional timeout in seconds

    Returns:
        The result of the coroutine

    Raises:
        asyncio.TimeoutError: If timeout is specified and exceeded
        Exception: Any exception raised by the coroutine
    """
    # Check for nested calls to prevent deadlock
    if hasattr(_local, "in_run_async") and _local.in_run_async:
        logger.warning("Nested run_async detected, using thread pool to avoid deadlock")
        return _run_in_thread(coro, timeout)

    # Check if we're in an async context
    if is_event_loop_running():
        # We're already in an async context, use thread pool
        logger.debug("Detected running event loop, using thread pool executor")
        return _run_in_thread(coro, timeout)
    else:
        # No running loop, we can safely create and use one
        logger.debug("No running event loop, creating new one")
        _local.in_run_async = True
        try:
            return _run_in_new_loop(coro, timeout)
        finally:
            _local.in_run_async = False


def _run_in_new_loop(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """
    Run coroutine in a new event loop with improved cleanup.

    Args:
        coro: Coroutine to run
        timeout: Optional timeout in seconds

    Returns:
        The result of the coroutine
    """
    loop = None
    original_loop = None

    try:
        # Store the current event loop for this thread, if any
        try:
            original_loop = asyncio.get_event_loop()
            if original_loop and original_loop.is_closed():
                original_loop = None
        except RuntimeError:
            original_loop = None

        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _created_loops.add(loop)

        # Wrap with timeout if specified
        if timeout is not None:

            async def with_timeout():
                return await asyncio.wait_for(coro, timeout)

            task_to_run = with_timeout()
        else:
            task_to_run = coro

        # Run the coroutine
        return loop.run_until_complete(task_to_run)

    finally:
        # Thorough cleanup
        if loop is not None:
            try:
                # Cancel any remaining tasks
                pending = asyncio.all_tasks(loop) if hasattr(asyncio, "all_tasks") else asyncio.Task.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Run the loop briefly to handle cancellations
                if pending:
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=1.0)
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

                # Make absolutely sure the loop is stopped
                loop.call_soon(loop.stop)
                if loop.is_running():
                    loop.run_until_complete(asyncio.sleep(0))
                    loop.stop()

                # Close the loop
                loop.close()

            except Exception as e:
                logger.warning(f"Error during loop cleanup: {e}")

        # Restore or clear the event loop for this thread
        if original_loop is not None and not original_loop.is_closed():
            asyncio.set_event_loop(original_loop)
        else:
            # IMPORTANT: Explicitly set to None to clear any loop reference
            asyncio.set_event_loop(None)

        logger.debug(f"Loop cleanup complete, restored: {original_loop}")


def _run_in_thread(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """
    Run coroutine in a separate thread with its own event loop.

    Args:
        coro: Coroutine to run
        timeout: Optional timeout in seconds

    Returns:
        The result of the coroutine

    Raises:
        Exception: Any exception raised by the coroutine
    """
    result_container: Dict[str, Any] = {"result": None, "exception": None, "loop": None}
    stop_event = threading.Event()

    def thread_target():
        """Target function for the thread."""
        loop = None
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_container["loop"] = loop

            async def run_with_stop_check():
                task = asyncio.create_task(coro)

                async def check_stop():
                    while not stop_event.is_set():
                        await asyncio.sleep(0.1)
                    task.cancel()

                stop_task = asyncio.create_task(check_stop())

                try:
                    if timeout:
                        return await asyncio.wait_for(task, timeout)
                    else:
                        return await task
                finally:
                    stop_task.cancel()
                    try:
                        await stop_task
                    except asyncio.CancelledError:
                        pass

            result = loop.run_until_complete(run_with_stop_check())
            result_container["result"] = result

        except Exception as e:
            result_container["exception"] = e
        finally:
            # Clean up the thread's loop
            if loop is not None:
                try:
                    # Cancel any remaining tasks
                    pending = asyncio.all_tasks(loop) if hasattr(asyncio, "all_tasks") else asyncio.Task.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        try:
                            loop.run_until_complete(
                                asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=0.5)
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass

                    if loop.is_running():
                        loop.stop()
                    loop.close()
                except Exception as e:
                    logger.warning(f"Error closing thread loop: {e}")
                finally:
                    # Always clear the loop for this thread
                    asyncio.set_event_loop(None)

    # Create and run the thread
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    # Check if thread is still alive (timeout case)
    if thread.is_alive():
        stop_event.set()
        if result_container["loop"]:
            try:
                result_container["loop"].call_soon_threadsafe(result_container["loop"].stop)
            except Exception:
                pass

        thread.join(timeout=0.5)

        if thread.is_alive():
            logger.error("Thread failed to stop gracefully")

        raise asyncio.TimeoutError(f"Coroutine execution exceeded timeout of {timeout} seconds")

    # Check for exceptions
    if result_container["exception"]:
        raise result_container["exception"]

    return result_container["result"]
