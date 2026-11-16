"""Response cancellation must not release a native worker's service ownership."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import structlog
from structlog.testing import capture_logs

from datus.api.services.background_drain import drain_background_tasks
from datus.api.services.datus_service import DatusService
from datus.api.utils import service_thread
from datus.api.utils.service_thread import service_to_thread


@pytest.mark.asyncio
async def test_response_deadline_keeps_native_worker_and_service_alive(real_agent_config):
    service = DatusService(real_agent_config, "worker-deadline")
    entered, release = threading.Event(), threading.Event()
    order = []
    service._cli = SimpleNamespace(close=lambda: order.append("closed"))

    def native():
        entered.set()
        assert release.wait(timeout=5)
        order.append("native-finished")
        return 42

    request = asyncio.create_task(service_to_thread(service, native))
    stopping = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(request, timeout=0)
        assert service.has_active_tasks() is True
        stopping = asyncio.create_task(service.shutdown())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(stopping), timeout=0.05)
        assert order == []
    finally:
        release.set()
        await asyncio.gather(request, *([stopping] if stopping is not None else []), return_exceptions=True)
        await service.shutdown()
    assert order == ["native-finished", "closed"]


@pytest.mark.asyncio
async def test_detached_native_failure_is_logged_and_returns_service_pin(real_agent_config, monkeypatch):
    service = DatusService(real_agent_config, "failed-worker")
    entered, release = threading.Event(), threading.Event()

    def native():
        entered.set()
        assert release.wait(timeout=5)
        raise ValueError("late native failure")

    # Capture structured events before handler routing. Other tests can change
    # logging destinations, and a cached proxy can retain their processors.
    with capture_logs() as records:
        monkeypatch.setattr(service_thread, "logger", structlog.get_logger(service_thread.__name__))
        request = asyncio.create_task(service_to_thread(service, native))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert service.has_active_tasks() is True
        finally:
            release.set()
            await service.shutdown()
            await drain_background_tasks()
    assert service.has_active_tasks() is False
    failures = [record for record in records if record["event"] == "Background service worker failed"]
    assert len(failures) == 1
    assert failures[0]["log_level"] == "error"
    assert failures[0]["exc_info"][0] is ValueError
    assert str(failures[0]["exc_info"][1]) == "late native failure"


@pytest.mark.asyncio
async def test_legacy_facade_without_request_protocol_remains_supported():
    assert await service_to_thread(SimpleNamespace(), lambda: 42) == 42


@pytest.mark.asyncio
async def test_closed_service_rejects_native_work(real_agent_config):
    service = DatusService(real_agent_config, "closed-worker")
    await service.shutdown()
    with pytest.raises(RuntimeError, match="DatusService is shutting down"):
        await service_to_thread(service, lambda: 42)
