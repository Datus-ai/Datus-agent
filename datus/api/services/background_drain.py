"""Shared registry for fire-and-forget background tasks spawned during request handling.

Tasks are tracked here so the FastAPI lifespan shutdown handler can await them
before the event loop closes, without depending on any optional route module.
"""

import asyncio

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_background_tasks: set[asyncio.Task] = set()


def track_background_task(task: asyncio.Task) -> None:
    """Register a background task for graceful drain on shutdown."""
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def drain_background_tasks() -> None:
    """Await all tracked background tasks (call from lifespan shutdown)."""
    if _background_tasks:
        results = await asyncio.gather(*list(_background_tasks), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Background task failed during drain", exc_info=result)
