"""Global LRU cache: project_id -> DatusService.

Uses Future-based thundering herd prevention — concurrent requests for
the same project_id share a single factory call.
"""

import asyncio
import collections
import time
from typing import Awaitable, Callable, Optional

from datus.api.services.datus_service import DatusService
from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DatusServiceCache:
    """Async LRU cache for DatusService instances."""

    def __init__(self, max_size: int = 128, *, idle_ttl: Optional[float] = None, sweep_interval: float = 30):
        if sweep_interval <= 0 or (idle_ttl is not None and idle_ttl < 0):
            raise ValueError("Sweep interval must be positive and idle TTL non-negative")
        self._max_size = max_size
        self._cache: collections.OrderedDict[str, DatusService] = collections.OrderedDict()
        self._futures: dict[str, asyncio.Future[DatusService]] = {}
        self._creations: dict[str, asyncio.Future[None]] = {}
        self._retired_services: set[DatusService] = set()
        self._lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task] = set()
        self._last_access: dict[str, float] = {}
        self._idle_ttl = idle_ttl
        self._sweep_interval = sweep_interval
        self._sweep_handle = None
        self._closed = False

    def _schedule_sweep(self):
        if not self._closed and self._sweep_handle is None:
            self._sweep_handle = asyncio.get_running_loop().call_later(self._sweep_interval, self._start_sweep)

    def _start_sweep(self):
        self._sweep_handle = None
        if not self._closed:
            self._track(asyncio.create_task(self._sweep()))

    async def _sweep(self):
        try:
            async with self._lock:
                evicted = []
                now = time.monotonic()
                for pid, svc in list(self._cache.items()):
                    if svc.has_active_tasks():
                        continue
                    idle = self._idle_ttl is not None and now - self._last_access[pid] >= self._idle_ttl
                    if len(self._cache) <= self._max_size and not idle:
                        continue
                    self._cache.pop(pid)
                    self._last_access.pop(pid, None)
                    evicted.append(svc)
            for svc in evicted:
                try:
                    await self._retire(svc, svc.shutdown())
                except Exception:
                    logger.exception("Error shutting down an expired DatusService")
        finally:
            self._schedule_sweep()

    def _track(self, task: asyncio.Task) -> asyncio.Task:
        """Register a background task so drain()/shutdown() can await it."""
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    def _retire(self, service, coroutine):
        """Keep disposal owners reachable so application shutdown can stop them."""
        self._retired_services.add(service)
        task = self._track(asyncio.create_task(coroutine))
        task.add_done_callback(lambda _done: self._retired_services.discard(service))
        return task

    async def get_or_create(
        self,
        project_id: str,
        factory: Callable[[], Awaitable[DatusService]],
        expected_fingerprint: Optional[str] = None,
        *,
        agent_config: Optional[AgentConfig] = None,
    ) -> DatusService:
        """Return cached DatusService or create via factory (thundering-herd safe).

        If ``expected_fingerprint`` is provided and does not match the cached
        instance's ``config_fingerprint``, a supplied ``agent_config`` creates
        a new facade sharing the project's task manager. Old turns stay
        reachable and keep their snapshots; new requests use the new config.
        Fingerprint-only callers retain the deferred-rebuild behavior.
        """
        if self._closed:
            raise RuntimeError("DatusServiceCache is shut down")
        self._schedule_sweep()
        if agent_config is not None and expected_fingerprint is None:
            expected_fingerprint = DatusService.compute_fingerprint(agent_config)

        is_creator = False
        stale_svc: Optional[DatusService] = None

        async with self._lock:
            if self._closed:
                raise RuntimeError("DatusServiceCache is shut down")
            self._last_access[project_id] = time.monotonic()
            # Fast path: cache hit
            if project_id in self._cache:
                cached = self._cache[project_id]
                if expected_fingerprint is None or cached.config_fingerprint == expected_fingerprint:
                    self._cache.move_to_end(project_id)
                    return cached
                if agent_config is not None:
                    refreshed = cached.with_config(agent_config)
                    self._cache[project_id] = refreshed
                    self._cache.move_to_end(project_id)
                    # Dispose only the old facade's exclusive resources. The
                    # task registry belongs to the replacement generation.
                    self._retire(cached, cached.shutdown(shutdown_tasks=False))
                    logger.info(f"Refreshed DatusService config for project {project_id} while preserving chat tasks")
                    return refreshed
                # Fingerprint mismatch. Rebuilding now would orphan any in-flight
                # chat task: its interaction broker lives in this instance's
                # task_manager, so a later /chat/user_interaction answer would
                # hit a fresh, empty manager and fail with "No active task found
                # for this session". Defer the config swap while the instance is
                # busy — keep serving it until its tasks drain, after which the
                # next request rebuilds with the new config.
                if cached.has_active_tasks():
                    self._cache.move_to_end(project_id)
                    logger.info(
                        f"Deferring DatusService rebuild for project {project_id}: "
                        f"AgentConfig fingerprint changed but tasks are still active"
                    )
                    return cached
                # Idle instance — safe to evict and rebuild with the new config.
                stale_svc = self._cache.pop(project_id)
                logger.info(f"Evicting DatusService for project {project_id} due to AgentConfig fingerprint mismatch")

            # Another coroutine is already creating this entry — share its future
            if project_id in self._futures:
                fut = self._futures[project_id]
            else:
                # We are the creator — register a future for waiters
                fut = asyncio.get_running_loop().create_future()
                self._futures[project_id] = fut
                creation_done = self._creations[project_id] = asyncio.get_running_loop().create_future()
                is_creator = True

        if not is_creator:
            # Wait for the creator coroutine to finish
            svc = await asyncio.shield(fut)
            if agent_config is not None:
                # The creator may have started before this request's config
                # change. Recheck against our snapshot after it finishes.
                return await self.get_or_create(project_id, factory, expected_fingerprint, agent_config=agent_config)
            return svc

        return await self._create_service(project_id, factory, fut, creation_done, stale_svc)

    async def _create_service(self, project_id, factory, fut, creation_done, stale_service):
        service = None
        published = False
        try:
            if stale_service is not None:
                await self._dispose(project_id, stale_service)
            service = await factory()
            async with self._lock:
                if not self._closed:
                    self._cache[project_id] = service
                    self._last_access[project_id] = time.monotonic()
                    self._cache.move_to_end(project_id)
                    published = True
                self._futures.pop(project_id, None)

                evicted = []
                while len(self._cache) > self._max_size:
                    candidate_pid = next(
                        (pid for pid, svc in self._cache.items() if pid != project_id and not svc.has_active_tasks()),
                        None,
                    )
                    if candidate_pid is None:
                        break
                    evicted.append((candidate_pid, self._cache.pop(candidate_pid)))
                    self._last_access.pop(candidate_pid, None)
            if not fut.done():
                fut.set_result(service)
            for old_pid, old_svc in evicted:
                logger.info(f"LRU evicting DatusService for project {old_pid}")
                self._retire(old_svc, old_svc.shutdown())
            if not published:
                self._last_access.pop(project_id, None)
                await asyncio.shield(self._retire(service, service.shutdown()))
            return service
        except BaseException as exc:
            async with self._lock:
                self._futures.pop(project_id, None)
                self._last_access.pop(project_id, None)
            if not fut.done():
                if isinstance(exc, asyncio.CancelledError):
                    fut.cancel()
                else:
                    fut.set_exception(exc)
                    fut.exception()
            if service is not None and not published and service not in self._retired_services:
                self._retire(service, service.shutdown())
            raise
        finally:
            if self._creations.get(project_id) is creation_done:
                self._creations.pop(project_id)
            creation_done.set_result(None)

    async def evict(self, project_id: str) -> None:
        """Evict a DatusService from cache (config change).

        Always removes from cache so new requests get fresh config.
        If the service has active tasks, defers shutdown until tasks drain.
        """
        async with self._lock:
            svc = self._cache.pop(project_id, None)
            self._last_access.pop(project_id, None)
        if not svc:
            return
        await self._dispose(project_id, svc)

    async def _dispose(self, project_id: str, svc: DatusService) -> None:
        """Shutdown a service, deferring if it still has active tasks."""
        if svc.has_active_tasks():
            logger.info(f"Evicting DatusService for project {project_id} (deferring shutdown — active tasks)")
            self._retire(svc, self._deferred_shutdown(project_id, svc))
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

    async def drain(self) -> None:
        """Await all pending background shutdown tasks before the loop closes."""
        while self._pending_tasks:
            pending = list(self._pending_tasks)
            self._pending_tasks.difference_update(pending)
            results = await asyncio.gather(*pending, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Background shutdown task failed", exc_info=result)

    async def shutdown(self) -> None:
        """Shutdown all cached DatusService instances (application exit)."""
        self._closed = True
        if self._sweep_handle is not None:
            self._sweep_handle.cancel()
            self._sweep_handle = None
        async with self._lock:
            creating = list(self._creations.values())
        if creating:
            await asyncio.gather(*(asyncio.shield(future) for future in creating), return_exceptions=True)
        async with self._lock:
            services = list(dict.fromkeys([*self._cache.values(), *self._retired_services]))
            self._cache.clear()
            self._last_access.clear()
        # Stop producers before awaiting deferred disposals or retired SSE users.
        results = await asyncio.gather(*(svc.shutdown() for svc in services), return_exceptions=True)
        for svc, result in zip(services, results):
            if isinstance(result, BaseException):
                logger.warning("Error shutting down DatusService %s", svc.project_id, exc_info=result)
        await self.drain()
        logger.info("DatusServiceCache shut down")
