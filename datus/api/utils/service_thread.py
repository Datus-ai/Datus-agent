"""Retain a service while native work outlives a request deadline."""

import asyncio

from datus.api.services.background_drain import track_background_task
from datus.utils.async_utils import drain_to_thread
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _report_worker_failure(worker):
    if not worker.cancelled():
        error = worker.exception()
        if error is not None:
            logger.error("Background service worker failed", exc_info=(type(error), error, error.__traceback__))


async def service_to_thread(service, function, /, *args, **kwargs):
    """Cancel the response promptly while the tracked worker owns its service pin."""
    acquire = getattr(service, "acquire_request", None)
    if not callable(acquire):
        return await asyncio.to_thread(function, *args, **kwargs)
    if not acquire(allow_closing=True):
        raise RuntimeError("DatusService is shutting down")

    async def run():
        try:
            return await drain_to_thread(function, *args, **kwargs)
        finally:
            service.release_request()

    worker = asyncio.create_task(run())
    track_background_task(worker)
    worker.add_done_callback(_report_worker_failure)
    return await asyncio.shield(worker)
