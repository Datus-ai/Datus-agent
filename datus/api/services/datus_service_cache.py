"""Global LRU cache: project_id -> DatusService.

Uses Future-based thundering herd prevention — concurrent requests for
the same project_id share a single factory call.
"""

import asyncio
import collections
from typing import Awaitable, Callable, Optional

from datus.api.services.datus_service import DatusService
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DatusServiceCache:
    """Async LRU cache for DatusService instances."""

    def __init__(self, max_size: int = 128):
        self._max_size = max_size
        self._cache: collections.OrderedDict[str, DatusService] = collections.OrderedDict()
        self._futures: dict[str, asyncio.Future[DatusService]] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        project_id: str,
        factory: Callable[[], Awaitable[DatusService]],
        expected_fingerprint: Optional[str] = None,
    ) -> DatusService:
        """Return cached DatusService or create via factory (thundering-herd safe).

        If ``expected_fingerprint`` is provided and does not match the cached
        instance's ``config_fingerprint``, the stale entry is evicted before
        creating a new one.
        """
        is_creator = False
        stale_svc: Optional[DatusService] = None

        async with self._lock:
            # Fast path: cache hit
            if project_id in self._cache:
                cached = self._cache[project_id]
                if expected_fingerprint is None or cached.config_fingerprint == expected_fingerprint:
                    self._cache.move_to_end(project_id)
                    return cached
                # Fingerprint mismatch — evict stale entry and rebuild
                stale_svc = self._cache.pop(project_id)
                logger.info(f"Evicting DatusService for project {project_id} due to AgentConfig fingerprint mismatch")

            # Another coroutine is already creating this entry — share its future
            if project_id in self._futures:
                fut = self._futures[project_id]
            else:
                # We are the creator — register a future for waiters
                fut = asyncio.get_running_loop().create_future()
                self._futures[project_id] = fut
                is_creator = True

        if stale_svc is not None:
            await self._dispose(project_id, stale_svc)

        if not is_creator:
            # Wait for the creator coroutine to finish
            return await fut

        # We are the creator — run the factory outside the lock
        try:
            svc = await factory()
        except Exception as e:
            async with self._lock:
                self._futures.pop(project_id, None)
            if not fut.done():
                fut.set_exception(e)
            raise

        async with self._lock:
            self._cache[project_id] = svc
            self._cache.move_to_end(project_id)
            self._futures.pop(project_id, None)

            # Evict oldest if over capacity, but skip services with active tasks
            evicted = []
            while len(self._cache) > self._max_size:
                # Find the oldest entry without active tasks
                candidate_pid = None
                for pid in self._cache:
                    if pid == project_id:
                        continue
                    if not self._cache[pid].has_active_tasks():
                        candidate_pid = pid
                        break
                if candidate_pid is None:
                    break  # all entries have active tasks — allow cache to exceed max_size
                old_svc = self._cache.pop(candidate_pid)
                evicted.append((candidate_pid, old_svc))

        # Resolve the future so waiters get the result
        if not fut.done():
            fut.set_result(svc)

        # Shutdown evicted services outside the lock
        for old_pid, old_svc in evicted:
            logger.info(f"LRU evicting DatusService for project {old_pid}")
            asyncio.create_task(old_svc.shutdown())

        return svc

    async def evict(self, project_id: str) -> None:
        """Evict a DatusService from cache (config change).

        Always removes from cache so new requests get fresh config.
        If the service has active tasks, defers shutdown until tasks drain.
        """
        async with self._lock:
            svc = self._cache.pop(project_id, None)
        if not svc:
            return
        await self._dispose(project_id, svc)

    async def _dispose(self, project_id: str, svc: DatusService) -> None:
        """Shutdown a service, deferring if it still has active tasks."""
        if svc.has_active_tasks():
            logger.info(f"Evicting DatusService for project {project_id} (deferring shutdown — active tasks)")
            asyncio.create_task(self._deferred_shutdown(project_id, svc))
        else:
            logger.info(f"Evicting DatusService for project {project_id}")
            await svc.shutdown()

    @staticmethod
    async def _deferred_shutdown(project_id: str, svc: DatusService) -> None:
        """Wait for active tasks to drain, then shutdown."""
        try:
            await svc.task_manager.wait_all_tasks()
            await svc.shutdown()
            logger.info(f"Deferred shutdown completed for project {project_id}")
        except Exception:
            logger.exception(f"Error in deferred shutdown for project {project_id}")

    async def shutdown(self) -> None:
        """Shutdown all cached DatusService instances (application exit)."""
        async with self._lock:
            items = list(self._cache.items())
            self._cache.clear()
        for pid, svc in items:
            try:
                await svc.shutdown()
            except Exception:
                logger.exception(f"Error shutting down DatusService for project {pid}")
        logger.info("DatusServiceCache shut down")
