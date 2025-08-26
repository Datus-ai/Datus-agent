"""
Robust async utilities for running async code in various contexts.
Handles both synchronous and asynchronous environments gracefully.
"""

import asyncio
import concurrent.futures
import logging
import sys
import threading
import weakref
from functools import wraps
from typing import Any, Awaitable, Callable, Coroutine, Dict, Optional, TypeVar, Union

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


def force_cleanup_event_loop():
    """
    Force cleanup of any existing event loop in the current thread.

    This is useful before starting Textual or other frameworks that need
    complete control over the event loop.
    """
    try:
        # Try to get current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Cancel all tasks
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop) if hasattr(asyncio, "all_tasks") else asyncio.Task.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Give tasks a chance to cleanup
                if pending and loop.is_running():
                    try:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception:
                        pass

                # Stop and close the loop
                if loop.is_running():
                    loop.stop()

                loop.close()

        # Clear the event loop for this thread
        asyncio.set_event_loop(None)

        # Clear any reference in thread local storage
        if hasattr(_local, "event_loop"):
            delattr(_local, "event_loop")

        # Reset the event loop policy to default (important for prompt_toolkit)
        asyncio.set_event_loop_policy(None)

        logger.debug("Force cleaned up event loop")

    except Exception as e:
        logger.warning(f"Error during force cleanup: {e}")


def ensure_prompt_toolkit_compatible():
    """
    Ensure the environment is compatible with prompt_toolkit.

    This should be called after async operations and before returning
    control to prompt_toolkit.
    """
    try:
        # First, force cleanup any existing loop
        force_cleanup_event_loop()

        # Create a fresh event loop policy
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        # Don't create a new loop - let prompt_toolkit do that
        logger.debug("Environment prepared for prompt_toolkit")

    except Exception as e:
        logger.warning(f"Error ensuring prompt_toolkit compatibility: {e}")


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


def run_async_with_executor(
    coro: Coroutine[Any, Any, T],
    executor: Optional[concurrent.futures.Executor] = None,
    timeout: Optional[float] = None,
    cleanup_after: bool = True,
) -> T:
    """
    Run async coroutine using a specific executor.

    This is useful when you want to control the thread pool or process pool.

    Args:
        coro: Coroutine to run
        executor: Executor to use (defaults to ThreadPoolExecutor)
        timeout: Optional timeout in seconds
        cleanup_after: Whether to cleanup event loop after execution (default True)

    Returns:
        The result of the coroutine
    """
    created_executor = False
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        created_executor = True

    try:

        def run_coro():
            result = run_async(coro, timeout)
            # Cleanup in the thread after execution if requested
            if cleanup_after:
                force_cleanup_event_loop()
            return result

        future = executor.submit(run_coro)
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise asyncio.TimeoutError(f"Execution exceeded timeout of {timeout} seconds")
    finally:
        if created_executor:
            executor.shutdown(wait=True)  # Changed to wait=True for proper cleanup

        # Extra cleanup after executor work is done
        if cleanup_after:
            ensure_prompt_toolkit_compatible()


def ensure_async_context(func: Callable[..., Awaitable[T]]) -> Callable[..., Union[T, Awaitable[T]]]:
    """
    Decorator that ensures a function runs in an async context.

    If called from sync code, it will automatically run the function
    using run_async. If called from async code, it runs normally.
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        coro = func(*args, **kwargs)
        return run_async(coro)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_event_loop_running():
            return async_wrapper(*args, **kwargs)
        else:
            return sync_wrapper(*args, **kwargs)

    wrapper.__wrapped__ = func
    return wrapper


class AsyncContextManager:
    """
    Context manager for ensuring async operations run correctly.

    Usage:
        with AsyncContextManager() as async_runner:
            result = async_runner.run(some_async_function())
    """

    def __init__(self, timeout: Optional[float] = None):
        self.timeout = timeout
        self.loop = None
        self.thread = None
        self.executor = None

    def __enter__(self):
        """Enter the context."""
        if is_event_loop_running():
            # Use thread pool for async context
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and clean up."""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.loop and not self.loop.is_closed():
            self.loop.close()
        return False

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine in the context."""
        if self.executor:
            # We're in an async context, use executor
            return run_async_with_executor(coro, self.executor, self.timeout)
        else:
            # Direct execution
            return run_async(coro, self.timeout)


def async_to_sync(async_func: Callable) -> Callable:
    """
    Convert an async function to a synchronous function.

    Usage:
        @async_to_sync
        async def my_async_function(x):
            await asyncio.sleep(1)
            return x * 2

        # Now can be called synchronously
        result = my_async_function(5)
    """

    @wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        coro = async_func(*args, **kwargs)
        return run_async(coro)

    return sync_wrapper
