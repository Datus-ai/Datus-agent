"""Tests for datus.api.services.datus_service_cache — async LRU cache."""

import asyncio
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

    async def test_mismatched_fingerprint_with_active_tasks_defers(self):
        cache = DatusServiceCache()
        old = _mock_service("p", has_active=True, fingerprint="fp-old")
        await cache.get_or_create("p", AsyncMock(return_value=old))

        new = _mock_service("p", fingerprint="fp-new")
        await cache.get_or_create("p", AsyncMock(return_value=new), expected_fingerprint="fp-new")

        # Old was not immediately shut down; deferred via active-tasks path
        old.shutdown.assert_not_awaited()
        await asyncio.sleep(0.05)
        old.task_manager.wait_all_tasks.assert_awaited()
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
        await cache.evict("ghost")  # should not raise

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
