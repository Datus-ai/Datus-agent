import asyncio
import sys
from typing import Any, Coroutine, TypeVar


def setup_windows_policy():
    # ProactorEventLoop has limitations in terms of subprocess pipelines
    # Also for compatibility with other third-party packages.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


T = TypeVar("T")


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create event loop
    """
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Smart async coroutine runner

    Args:
        coro: Coroutine to run

    Returns:
        Coroutine result
    """
    # Check if already in async context
    loop = get_or_create_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
