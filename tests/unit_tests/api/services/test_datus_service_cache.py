"""Tests for datus.api.services.datus_service_cache — async LRU cache."""

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.api.services.datus_service_cache import DatusServiceCache


def _mock_service(project_id="p1", has_active=False, fingerprint="fp-default"):
    """Create a mock DatusService."""
    svc = MagicMock()
    svc.project_id = project_id
    svc.config_fingerprint = fingerprint
    svc.has_active_tasks.return_value = has_active
    svc.shutdown = AsyncMock()
    svc.task_manager = MagicMock()
    svc.task_manager.wait_all_tasks = AsyncMock()
    return svc


class TestDatusServiceCacheInit:
    """Tests for cache initialization."""

    def test_default_max_size(self):
        """Default max_size is 128."""
        cache = DatusServiceCache()
        assert cache._max_size == 128

    def test_custom_max_size(self):
        """Custom max_size is respected."""
        cache = DatusServiceCache(max_size=5)
        assert cache._max_size == 5

    def test_cache_starts_empty(self):
        """Cache and futures dicts are empty on init."""
        cache = DatusServiceCache()
        assert len(cache._cache) == 0
        assert len(cache._futures) == 0


@pytest.mark.asyncio
class TestGetOrCreate:
    """Tests for get_or_create — cache hit, miss, and thundering herd."""

    async def test_cache_miss_calls_factory(self):
        """Factory is called on first access for a project_id."""
        cache = DatusServiceCache()
        svc = _mock_service("proj-1")

        async def factory():
            return svc

        result = await cache.get_or_create("proj-1", factory)
        assert result is svc
        assert "proj-1" in cache._cache

    async def test_cache_hit_returns_existing(self):
        """Second call returns cached instance, not factory."""
        cache = DatusServiceCache()
        svc = _mock_service("proj-2")
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return svc

        await cache.get_or_create("proj-2", factory)
        result = await cache.get_or_create("proj-2", factory)
        assert result is svc
        assert call_count == 1

    async def test_different_projects_get_different_services(self):
        """Different project_ids get separate cache entries."""
        cache = DatusServiceCache()
        svc_a = _mock_service("a")
        svc_b = _mock_service("b")

        result_a = await cache.get_or_create("a", AsyncMock(return_value=svc_a))
        result_b = await cache.get_or_create("b", AsyncMock(return_value=svc_b))
        assert result_a is svc_a
        assert result_b is svc_b
        assert len(cache._cache) == 2

    async def test_factory_exception_propagates(self):
        """Factory exception propagates and doesn't leave stale future."""
        cache = DatusServiceCache()

        async def bad_factory():
            raise ValueError("config error")

        with pytest.raises(ValueError, match="config error"):
            await cache.get_or_create("fail", bad_factory)

        assert "fail" not in cache._cache
        assert "fail" not in cache._futures

    async def test_lru_eviction_when_over_capacity(self):
        """Oldest inactive entry is evicted when cache exceeds max_size."""
        cache = DatusServiceCache(max_size=2)

        svc_a = _mock_service("a", has_active=False)
        svc_b = _mock_service("b", has_active=False)
        svc_c = _mock_service("c", has_active=False)

        await cache.get_or_create("a", AsyncMock(return_value=svc_a))
        await cache.get_or_create("b", AsyncMock(return_value=svc_b))
        await cache.get_or_create("c", AsyncMock(return_value=svc_c))

        # 'a' should be evicted (oldest)
        assert "a" not in cache._cache
        assert "b" in cache._cache
        assert "c" in cache._cache

    async def test_active_tasks_skip_eviction(self):
        """Entries with active tasks are not evicted, cache may exceed max_size."""
        cache = DatusServiceCache(max_size=2)

        svc_a = _mock_service("a", has_active=True)
        svc_b = _mock_service("b", has_active=True)
        svc_c = _mock_service("c", has_active=False)

        await cache.get_or_create("a", AsyncMock(return_value=svc_a))
        await cache.get_or_create("b", AsyncMock(return_value=svc_b))
        await cache.get_or_create("c", AsyncMock(return_value=svc_c))

        # All should remain since a and b have active tasks
        assert len(cache._cache) == 3

    async def test_thundering_herd_shares_factory_result(self):
        """Concurrent requests for same project_id share a single factory call."""
        cache = DatusServiceCache()
        svc = _mock_service("shared")
        call_count = 0

        async def slow_factory():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return svc

        results = await asyncio.gather(
            cache.get_or_create("shared", slow_factory),
            cache.get_or_create("shared", slow_factory),
            cache.get_or_create("shared", slow_factory),
        )
        assert all(r is svc for r in results)
        assert call_count == 1


@pytest.mark.asyncio
class TestFingerprintEviction:
    """Tests for fingerprint-based stale-entry eviction in get_or_create."""

    async def test_matching_fingerprint_returns_cached(self):
        cache = DatusServiceCache()
        svc = _mock_service("p", fingerprint="fp-1")
        await cache.get_or_create("p", AsyncMock(return_value=svc))

        factory = AsyncMock()
        result = await cache.get_or_create("p", factory, expected_fingerprint="fp-1")
        assert result is svc
        factory.assert_not_called()

    async def test_mismatched_fingerprint_evicts_and_rebuilds(self):
        cache = DatusServiceCache()
        old = _mock_service("p", fingerprint="fp-old")
        await cache.get_or_create("p", AsyncMock(return_value=old))

        new = _mock_service("p", fingerprint="fp-new")
        factory = AsyncMock(return_value=new)
        result = await cache.get_or_create("p", factory, expected_fingerprint="fp-new")

        assert result is new
        factory.assert_awaited_once()
        old.shutdown.assert_awaited_once()
        assert cache._cache["p"] is new

    async def test_mismatched_fingerprint_with_active_tasks_defers_rebuild(self):
        """Fingerprint changes are deferred while tasks are active.

        Rebuilding would orphan the in-flight task's interaction broker, so the
        existing instance must keep serving requests (e.g. a pending
        /chat/user_interaction answer) until its tasks drain.
        """
        cache = DatusServiceCache()
        old = _mock_service("p", has_active=True, fingerprint="fp-old")
        await cache.get_or_create("p", AsyncMock(return_value=old))

        factory = AsyncMock()
        result = await cache.get_or_create("p", factory, expected_fingerprint="fp-new")

        # No rebuild: the busy instance is preserved and returned as-is.
        factory.assert_not_called()
        old.shutdown.assert_not_awaited()
        assert result is old
        assert cache._cache["p"] is old

    async def test_mismatched_fingerprint_rebuilds_after_tasks_drain(self):
        """Once the busy instance goes idle, the next request rebuilds."""
        cache = DatusServiceCache()
        old = _mock_service("p", has_active=True, fingerprint="fp-old")
        await cache.get_or_create("p", AsyncMock(return_value=old))

        # First mismatched request is deferred (tasks still active).
        await cache.get_or_create("p", AsyncMock(return_value=_mock_service("p")), expected_fingerprint="fp-new")
        assert cache._cache["p"] is old

        # Task drained — the next mismatched request evicts and rebuilds.
        old.has_active_tasks.return_value = False
        new = _mock_service("p", fingerprint="fp-new")
        result = await cache.get_or_create("p", AsyncMock(return_value=new), expected_fingerprint="fp-new")

        assert result is new
        old.shutdown.assert_awaited_once()
        assert cache._cache["p"] is new

    async def test_none_fingerprint_preserves_legacy_behavior(self):
        cache = DatusServiceCache()
        svc = _mock_service("p", fingerprint="fp-1")
        await cache.get_or_create("p", AsyncMock(return_value=svc))

        factory = AsyncMock()
        result = await cache.get_or_create("p", factory)  # no expected_fingerprint
        assert result is svc
        factory.assert_not_called()


@pytest.mark.asyncio
class TestEvict:
    """Tests for evict — cache removal and deferred shutdown."""

    async def test_evict_removes_from_cache(self):
        """Evict removes the service and calls shutdown."""
        cache = DatusServiceCache()
        svc = _mock_service("proj-e")
        await cache.get_or_create("proj-e", AsyncMock(return_value=svc))

        await cache.evict("proj-e")
        assert "proj-e" not in cache._cache
        svc.shutdown.assert_awaited_once()

    async def test_evict_nonexistent_is_noop(self):
        """Evicting a non-existent key doesn't raise."""
        cache = DatusServiceCache()
        await cache.evict("ghost")
        assert cache._cache == {}

    async def test_evict_with_active_tasks_defers_shutdown(self):
        """Evict service with active tasks schedules deferred shutdown."""
        cache = DatusServiceCache()
        svc = _mock_service("defer-proj", has_active=True)
        await cache.get_or_create("defer-proj", AsyncMock(return_value=svc))

        await cache.evict("defer-proj")
        assert "defer-proj" not in cache._cache
        # Deferred shutdown task was created — give it a moment
        await asyncio.sleep(0.1)
        # task_manager.wait_all_tasks should have been called
        svc.task_manager.wait_all_tasks.assert_awaited()


@pytest.mark.asyncio
class TestPendingTaskTracking:
    """Tests for _track() / drain() — background task lifecycle management."""

    async def test_lru_eviction_task_is_tracked(self):
        """LRU eviction create_task is registered in _pending_tasks."""
        cache = DatusServiceCache(max_size=1)
        svc_a = _mock_service("a", has_active=False)
        svc_b = _mock_service("b", has_active=False)

        # Pause the shutdown task so it stays in _pending_tasks when we check.
        pause = asyncio.Event()

        async def _paused_shutdown():
            await pause.wait()

        svc_a.shutdown = AsyncMock(side_effect=_paused_shutdown)

        await cache.get_or_create("a", AsyncMock(return_value=svc_a))
        await cache.get_or_create("b", AsyncMock(return_value=svc_b))

        assert len(cache._pending_tasks) == 1

        pause.set()
        await cache.drain()

    async def test_deferred_shutdown_task_is_tracked(self):
        """_dispose with active tasks registers the deferred-shutdown Task."""
        cache = DatusServiceCache()
        pause = asyncio.Event()

        async def _paused_wait():
            await pause.wait()

        svc = _mock_service("p", has_active=True)
        svc.task_manager.wait_all_tasks = AsyncMock(side_effect=_paused_wait)
        await cache.get_or_create("p", AsyncMock(return_value=svc))

        await cache._dispose("p", svc)

        assert len(cache._pending_tasks) == 1

        pause.set()
        await cache.drain()

    async def test_pending_task_removed_on_completion(self):
        """Completed task is discarded from _pending_tasks via done callback."""
        cache = DatusServiceCache()
        svc = _mock_service("p")
        await cache.get_or_create("p", AsyncMock(return_value=svc))
        await cache._dispose("p", svc)  # has_active=False → direct await, no task

        assert len(cache._pending_tasks) == 0

    async def test_drain_awaits_pending_tasks(self):
        """drain() awaits all registered background tasks."""
        cache = DatusServiceCache()
        completed = []

        async def _slow_work():
            await asyncio.sleep(0.01)
            completed.append(1)

        task = asyncio.create_task(_slow_work())
        cache._track(task)
        assert len(cache._pending_tasks) == 1

        await cache.drain()
        assert completed == [1]
        assert len(cache._pending_tasks) == 0

    async def test_drain_tolerates_task_exception(self):
        """drain() swallows task exceptions and clears the pending set."""
        cache = DatusServiceCache()

        async def _boom():
            raise RuntimeError("background failure")

        cache._track(asyncio.create_task(_boom()))
        await cache.drain()
        assert len(cache._pending_tasks) == 0

    async def test_shutdown_stops_services_before_waiting_for_background_users(self):
        """Producer shutdown releases consumers before draining their disposal tasks."""
        cache = DatusServiceCache()
        order = []
        stopped = asyncio.Event()

        async def _background():
            await stopped.wait()
            order.append("drained")

        async def stop_producer():
            order.append("service-shutdown")
            stopped.set()

        svc = _mock_service("p")
        svc.shutdown = AsyncMock(side_effect=stop_producer)
        await cache.get_or_create("p", AsyncMock(return_value=svc))
        cache._track(asyncio.create_task(_background()))

        await cache.shutdown()
        assert order == ["service-shutdown", "drained"]


@pytest.mark.asyncio
class TestShutdown:
    """Tests for shutdown — clean teardown of all services."""

    async def test_shutdown_clears_all_services(self):
        """Shutdown calls shutdown() on each cached service and clears cache."""
        cache = DatusServiceCache()
        svc_a = _mock_service("a")
        svc_b = _mock_service("b")
        await cache.get_or_create("a", AsyncMock(return_value=svc_a))
        await cache.get_or_create("b", AsyncMock(return_value=svc_b))

        await cache.shutdown()
        assert len(cache._cache) == 0
        svc_a.shutdown.assert_awaited_once()
        svc_b.shutdown.assert_awaited_once()

    async def test_shutdown_handles_exception_in_service(self):
        """Shutdown continues even if one service.shutdown() raises."""
        cache = DatusServiceCache()
        svc_ok = _mock_service("ok")
        svc_bad = _mock_service("bad")
        svc_bad.shutdown = AsyncMock(side_effect=RuntimeError("boom"))

        await cache.get_or_create("bad", AsyncMock(return_value=svc_bad))
        await cache.get_or_create("ok", AsyncMock(return_value=svc_ok))

        await cache.shutdown()  # should not raise
        assert len(cache._cache) == 0
        svc_ok.shutdown.assert_awaited_once()


@pytest.mark.asyncio
class TestConfigRefreshDuringCreation:
    async def test_waiter_uses_its_config_when_initial_factory_has_an_older_snapshot(self, real_agent_config):
        """An edit during first initialization must reach the waiting request."""
        from datus.api.services.datus_service import DatusService

        cache = DatusServiceCache()
        before = real_agent_config
        after = copy.deepcopy(before)
        after.services.datasources["warehouse"] = copy.deepcopy(before.current_db_config())
        factory_started = asyncio.Event()
        waiter_started = asyncio.Event()
        finish_factory = asyncio.Event()
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            factory_started.set()
            await finish_factory.wait()
            return DatusService(before, "project")

        async def request_new_config():
            waiter_started.set()
            return await cache.get_or_create("project", factory, agent_config=after)

        creator = asyncio.create_task(cache.get_or_create("project", factory, agent_config=before))
        waiter = None
        try:
            await asyncio.wait_for(factory_started.wait(), timeout=2)
            waiter = asyncio.create_task(request_new_config())
            await asyncio.wait_for(waiter_started.wait(), timeout=2)
            finish_factory.set()
            first, refreshed = await asyncio.wait_for(asyncio.gather(creator, waiter), timeout=2)

            assert first.agent_config is before
            assert refreshed.agent_config is after
            assert "warehouse" in refreshed.agent_config.datasource_configs
            assert refreshed.task_manager is first.task_manager
            assert calls == 1
            assert await cache.get_or_create("project", factory, agent_config=after) is refreshed
        finally:
            finish_factory.set()
            await asyncio.gather(creator, *([waiter] if waiter is not None else []), return_exceptions=True)
            await cache.shutdown()


@pytest.mark.asyncio
class TestPluginChangeRebuild:
    """Cross-component: real config + real service + real cache.

    Replays what ``get_datus_service`` does per request — recompute the
    fingerprint from the live AgentConfig, hand it to the cache — to pin that a
    plugin change actually swaps the cached instance instead of only moving a
    hash in isolation.
    """

    @staticmethod
    async def _request(cache, agent_config):
        from datus.api.services.datus_service import DatusService

        async def factory():
            return DatusService(agent_config=agent_config, project_id="p")

        return await cache.get_or_create(
            "p", factory, expected_fingerprint=DatusService.compute_fingerprint(agent_config)
        )

    async def test_plugin_profile_change_rebuilds_on_next_request(self, real_agent_config):
        cache = DatusServiceCache()
        try:
            real_agent_config.init_plugin_services({"hello": {"prod": {"api_base_url": "https://one"}}})
            first = await self._request(cache, real_agent_config)
            # Unchanged plugin state must reuse the cached instance.
            assert await self._request(cache, real_agent_config) is first

            real_agent_config.init_plugin_services({"hello": {"prod": {"api_base_url": "https://two"}}})

            assert await self._request(cache, real_agent_config) is not first
        finally:
            await cache.shutdown()

    async def test_plugin_disable_rebuilds_on_next_request(self, real_agent_config):
        cache = DatusServiceCache()
        try:
            first = await self._request(cache, real_agent_config)

            real_agent_config.plugins_enabled = False

            assert await self._request(cache, real_agent_config) is not first
        finally:
            await cache.shutdown()


@pytest.mark.asyncio
async def test_capacity_is_rechecked_after_active_services_drain():
    """No subsequent insertion is needed to undo an active-work capacity overflow."""
    cache = DatusServiceCache(max_size=1, sweep_interval=0.01)
    first = _mock_service("first", has_active=True)
    second = _mock_service("second", has_active=True)
    await cache.get_or_create("first", AsyncMock(return_value=first))
    await cache.get_or_create("second", AsyncMock(return_value=second))
    assert len(cache._cache) == 2
    first.has_active_tasks.return_value = False
    for _ in range(100):
        if first.shutdown.await_count:
            break
        await asyncio.sleep(0.01)
    assert len(cache._cache) == 1
    first.shutdown.assert_awaited_once()
    second.shutdown.assert_not_awaited()
    await cache.shutdown()


@pytest.mark.asyncio
async def test_real_service_eviction_closes_owned_managers_once(real_agent_config):
    """Exercise real cache/service/CLI teardown, replacing only connector allocation."""
    from datus.api.services.cli_service import CLIService
    from datus.api.services.datus_service import DatusService
    from datus.tools.db_tools.db_manager import DBManager

    first = DatusService(real_agent_config, "first")
    cli = CLIService.__new__(CLIService)
    cli.db_manager = DBManager({})
    connector = MagicMock()
    cli.db_manager._conn_dict["source"] = {"db": connector}
    cli._db_tool_cache = None
    cli._datasource_connectors = {}
    cli.current_db_connector = connector
    first._cli = cli
    first._explorers[None] = object()
    second = DatusService(real_agent_config, "second")
    cache = DatusServiceCache(max_size=1)
    await cache.get_or_create("first", AsyncMock(return_value=first))
    await cache.get_or_create("second", AsyncMock(return_value=second))
    await cache.drain()
    await first.shutdown()
    connector.close.assert_called_once()
    assert not first._explorers
    assert first._cli is None
    await cache.shutdown()


@pytest.mark.asyncio
async def test_config_refresh_disposes_facade_but_preserves_task_manager(real_agent_config):
    """Old facade resources drain without stopping tasks shared by the new facade."""
    from datus.api.services.datus_service import DatusService

    first = DatusService(real_agent_config, "project")
    first._cli = MagicMock()
    cli = first._cli
    task_shutdown = first.task_manager.shutdown = AsyncMock()
    first.acquire_request()
    cache = DatusServiceCache()
    await cache.get_or_create("project", AsyncMock(return_value=first))
    refreshed = await cache.get_or_create(
        "project", AsyncMock(), expected_fingerprint="changed", agent_config=real_agent_config
    )
    assert refreshed.task_manager is first.task_manager
    await asyncio.sleep(0)
    cli.close.assert_not_called()
    first.release_request()
    await cache.drain()
    cli.close.assert_called_once()
    task_shutdown.assert_not_awaited()
    await cache.shutdown()


@pytest.mark.asyncio
async def test_optional_idle_ttl_expires_service_without_requests():
    """The explicit idle policy works independently of capacity and later traffic."""
    cache = DatusServiceCache(max_size=128, idle_ttl=0.01, sweep_interval=0.01)
    service = _mock_service("idle")
    await cache.get_or_create("idle", AsyncMock(return_value=service))
    for _ in range(100):
        if service.shutdown.await_count:
            break
        await asyncio.sleep(0.01)
    service.shutdown.assert_awaited_once()
    assert not cache._cache
    await cache.shutdown()


@pytest.mark.asyncio
async def test_shutdown_drains_in_progress_creation(real_agent_config):
    """A factory finishing during shutdown cannot leave a newly owned service behind."""
    from datus.api.services.datus_service import DatusService

    cache = DatusServiceCache()
    service = DatusService(real_agent_config, "creating")
    service._cli = MagicMock()
    cli = service._cli
    started, release = asyncio.Event(), asyncio.Event()

    async def factory():
        started.set()
        await release.wait()
        return service

    creating = asyncio.create_task(cache.get_or_create("creating", factory))
    await started.wait()
    stopping = asyncio.create_task(cache.shutdown())
    try:
        await asyncio.sleep(0)
        assert not stopping.done()
    finally:
        release.set()
    await creating
    await asyncio.wait_for(stopping, timeout=2)
    cli.close.assert_called_once()
    assert not cache._cache


@pytest.mark.asyncio
async def test_cancelled_factory_does_not_strand_shutdown_or_retry():
    """Cancelled construction releases its pending slot so retries and shutdown can finish."""
    cache = DatusServiceCache()
    started = asyncio.Event()

    async def factory():
        started.set()
        await asyncio.Event().wait()

    creating = asyncio.create_task(cache.get_or_create("cancelled", factory))
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating
    service = _mock_service("cancelled")
    result = await asyncio.wait_for(cache.get_or_create("cancelled", AsyncMock(return_value=service)), timeout=2)
    assert result is service
    await asyncio.wait_for(cache.shutdown(), timeout=2)


@pytest.mark.asyncio
async def test_cancelled_joiner_does_not_cancel_shared_service_creation():
    cache = DatusServiceCache()
    service = _mock_service("shared")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def factory():
        calls.append("create")
        entered.set()
        await release.wait()
        return service

    creator = asyncio.create_task(cache.get_or_create("shared", factory))
    await entered.wait()
    joiner = asyncio.create_task(cache.get_or_create("shared", factory))
    await asyncio.sleep(0)
    joiner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await joiner
    survivor = asyncio.create_task(cache.get_or_create("shared", factory))
    try:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(creator, survivor), timeout=2)
        assert results == [service, service]
        assert calls == ["create"]
    finally:
        release.set()
        await asyncio.gather(creator, survivor, return_exceptions=True)
        await cache.shutdown()


@pytest.mark.asyncio
async def test_shutdown_waits_for_creator_even_when_shared_result_is_cancelled():
    cache = DatusServiceCache()
    service = _mock_service("creating")
    entered, release = asyncio.Event(), asyncio.Event()

    async def factory():
        entered.set()
        await release.wait()
        return service

    creator = asyncio.create_task(cache.get_or_create("creating", factory))
    await entered.wait()
    # Fault injection reproduces cancellation of the result Future by a peer;
    # it must not be used as evidence that the creator has actually finished.
    cache._futures["creating"].cancel()
    stopping = asyncio.create_task(cache.shutdown())
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(stopping), timeout=0.05)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(creator, stopping), timeout=2)
    assert cache._cache == {}
    service.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_retired_generation_request_protects_refreshed_project_from_idle_eviction(real_agent_config):
    from datus.api.services.datus_service import DatusService

    cache = DatusServiceCache(idle_ttl=0)
    first = DatusService(real_agent_config, "project")
    assert first.acquire_request() is True
    await cache.get_or_create("project", AsyncMock(return_value=first))
    try:
        refreshed = await cache.get_or_create(
            "project", AsyncMock(), expected_fingerprint="changed", agent_config=real_agent_config
        )
        await cache._sweep()
        assert cache._cache["project"] is refreshed
        assert refreshed.task_manager is first.task_manager
        assert refreshed.has_active_tasks() is True
    finally:
        first.release_request()
        await cache.drain()
    await cache._sweep()
    assert cache._cache == {}
    await cache.shutdown()
