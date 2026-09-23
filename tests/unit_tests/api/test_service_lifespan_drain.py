"""Shutdown must drain the background tasks that cancelling in-flight turns creates."""

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from datus.api.services.background_drain import track_background_task

# ``datus.api`` exposes a ``service`` attribute that shadows the submodule.
service_module = importlib.import_module("datus.api.service")


@pytest.mark.asyncio
async def test_settlements_scheduled_by_cache_shutdown_are_drained(monkeypatch):
    settled = []

    class FakeService:
        agent_config = None

        def __init__(self, _args):
            pass

        async def initialize(self):
            return None

    class FakeCache:
        def __init__(self, max_size=128):
            pass

        async def shutdown(self):
            # What cancelling an in-flight turn does: settle it in the background.
            async def _settle():
                await asyncio.sleep(0.01)
                settled.append("billed")

            track_background_task(asyncio.create_task(_settle()))

    monkeypatch.setattr(service_module, "DatusAPIService", FakeService)
    monkeypatch.setattr(service_module, "DatusServiceCache", FakeCache)
    monkeypatch.setattr(service_module, "load_auth_provider", lambda *a, **k: object())
    monkeypatch.setattr(service_module, "init_deps", lambda *a, **k: None)
    monkeypatch.setattr("datus.utils.async_debug.install_task_dump_signal_handler", lambda: None)

    app = SimpleNamespace(state=SimpleNamespace(agent_args=None))
    async with service_module.lifespan(app):
        pass

    assert settled == ["billed"]
