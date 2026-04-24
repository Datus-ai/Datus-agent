# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.cli.background_sync.BackgroundSchemaSyncManager."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.cli.background_sync import BackgroundSchemaSyncManager

# --------------------------------------------------------------------------- #
# Helpers: a persistent background event loop mirroring DatusCLI._bg_loop
# --------------------------------------------------------------------------- #


class _BgLoop:
    """Spin up a daemon event-loop thread compatible with run_coroutine_threadsafe."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2)


@pytest.fixture()
def bg_loop():
    bl = _BgLoop()
    try:
        yield bl
    finally:
        bl.stop()


def _make_cli(bg_loop, *, enabled=True, on_startup=True, include_values=False, current="local_db"):
    cli = MagicMock()
    cli._bg_loop = bg_loop.loop
    cli.agent_config = SimpleNamespace(
        autocomplete=SimpleNamespace(
            background_sync_enabled=enabled,
            background_sync_on_startup=on_startup,
            background_sync_include_values=include_values,
        ),
        current_datasource=current,
    )
    cli.db_manager = MagicMock()
    cli.at_completer = MagicMock()
    cli.tui_app = None
    return cli


def _wait_for_future(fut, timeout=5.0):
    """Block until the manager's current future either completes or clears."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fut is None or fut.done():
            return
        time.sleep(0.01)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestScheduleRunsInitAndReloads:
    def test_schedule_invokes_init_and_reloads(self, bg_loop, monkeypatch):
        """A successful sync calls init_local_schema_async with the
        expected kwargs and then swaps the completer cache exactly once.
        """
        call_args = {}
        reload_events = []

        async def fake_init_async(store, agent_config, db_manager, **kwargs):
            call_args.update(kwargs)

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: MagicMock(),
        )

        cli = _make_cli(bg_loop)
        cli.at_completer.table_completer.reload_data = MagicMock(side_effect=lambda: reload_events.append("reloaded"))

        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="local_db", reason="switch")

        _wait_for_future(mgr._current_future)

        assert call_args["build_mode"] == "incremental"
        # include_values=False ⇒ table_type="table"
        assert call_args["table_type"] == "table"
        assert reload_events == ["reloaded"]

    def test_include_values_true_uses_full_table_type(self, bg_loop, monkeypatch):
        captured = {}

        async def fake_init_async(store, agent_config, db_manager, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: MagicMock(),
        )

        cli = _make_cli(bg_loop, include_values=True)
        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="local_db")
        _wait_for_future(mgr._current_future)

        assert captured["table_type"] == "full"


class TestScheduleCoalesce:
    def test_same_datasource_during_running_is_coalesced(self, bg_loop, monkeypatch):
        """Re-scheduling the same datasource while one is in-flight must
        not spawn a second init call.
        """
        started = threading.Event()
        proceed = threading.Event()
        call_count = {"n": 0}

        async def fake_init_async(store, agent_config, db_manager, **kwargs):
            call_count["n"] += 1
            started.set()
            await asyncio.get_running_loop().run_in_executor(None, proceed.wait)

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: MagicMock(),
        )

        cli = _make_cli(bg_loop)
        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="local_db")
        assert started.wait(timeout=2), "first sync did not start"
        mgr.schedule(datasource="local_db")  # coalesced
        proceed.set()
        _wait_for_future(mgr._current_future)

        assert call_count["n"] == 1


class TestScheduleCancelPrevious:
    def test_different_datasource_cancels_prior(self, bg_loop, monkeypatch):
        """A new schedule targeting a *different* datasource cancels the
        previous future so stale refreshes do not overwrite fresh data.
        """
        started = threading.Event()
        cancelled_observed = {"flag": False}
        invocations = []

        async def fake_init_async(store, agent_config, db_manager, **kwargs):
            invocations.append(kwargs)
            if len(invocations) == 1:
                started.set()
                try:
                    # Park long enough for a second schedule to cancel us.
                    await asyncio.sleep(2.0)
                except asyncio.CancelledError:
                    cancelled_observed["flag"] = True
                    raise

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: MagicMock(),
        )

        cli = _make_cli(bg_loop, current="local_db")
        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="local_db")
        assert started.wait(timeout=2)
        # Update current datasource before scheduling the next sync so the
        # drift guard does not skip it.
        cli.agent_config.current_datasource = "pg_db"
        mgr.schedule(datasource="pg_db")
        _wait_for_future(mgr._current_future)

        assert cancelled_observed["flag"] is True
        # Two init invocations overall: the first cancelled, the second ran.
        assert len(invocations) == 2


class TestDisabledIsNoOp:
    def test_disabled_short_circuits(self, bg_loop, monkeypatch):
        called = {"n": 0}

        async def fake_init_async(*args, **kwargs):
            called["n"] += 1

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        cli = _make_cli(bg_loop, enabled=False)
        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="local_db")
        time.sleep(0.1)
        assert called["n"] == 0
        assert mgr.is_running() is False


class TestFailureKeepsOldData:
    def test_init_exception_does_not_reload_and_does_not_raise(self, bg_loop, monkeypatch):
        async def fake_init_async(*args, **kwargs):
            raise RuntimeError("simulated init failure")

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: MagicMock(),
        )

        cli = _make_cli(bg_loop)
        cli.at_completer.table_completer.reload_data = MagicMock()

        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="local_db")
        _wait_for_future(mgr._current_future)

        cli.at_completer.table_completer.reload_data.assert_not_called()
        assert mgr.is_running() is False


class TestDriftGuard:
    def test_current_datasource_drift_skips_sync(self, bg_loop, monkeypatch):
        """When the CLI's active datasource changes between scheduling and
        execution, the orphaned task must skip the reload so the user sees
        the newly selected datasource's data, not the prior one.
        """
        init_called = {"n": 0}

        async def fake_init_async(*args, **kwargs):
            init_called["n"] += 1

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: MagicMock(),
        )

        cli = _make_cli(bg_loop, current="pg_db")  # already drifted
        cli.at_completer.table_completer.reload_data = MagicMock()

        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="local_db")  # stale target
        _wait_for_future(mgr._current_future)

        assert init_called["n"] == 0
        cli.at_completer.table_completer.reload_data.assert_not_called()


class TestIsRunningLifecycle:
    def test_is_running_toggles_during_task(self, bg_loop, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        async def fake_init_async(*args, **kwargs):
            started.set()
            await asyncio.get_running_loop().run_in_executor(None, release.wait)

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: MagicMock(),
        )

        cli = _make_cli(bg_loop)
        mgr = BackgroundSchemaSyncManager(cli)
        assert mgr.is_running() is False
        mgr.schedule(datasource="local_db")
        assert started.wait(timeout=2)
        assert mgr.is_running() is True
        release.set()
        _wait_for_future(mgr._current_future)
        assert mgr.is_running() is False


class TestShutdown:
    def test_shutdown_blocks_further_schedules(self, bg_loop, monkeypatch):
        called = {"n": 0}

        async def fake_init_async(*args, **kwargs):
            called["n"] += 1

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )

        cli = _make_cli(bg_loop)
        mgr = BackgroundSchemaSyncManager(cli)
        mgr.shutdown()
        mgr.schedule(datasource="local_db")
        time.sleep(0.1)
        assert called["n"] == 0


class TestEmptyDatasourceIgnored:
    def test_empty_datasource_name_is_noop(self, bg_loop, monkeypatch):
        called = {"n": 0}

        async def fake_init_async(*args, **kwargs):
            called["n"] += 1

        monkeypatch.setattr(
            "datus.storage.schema_metadata.local_init.init_local_schema_async",
            fake_init_async,
        )

        cli = _make_cli(bg_loop)
        mgr = BackgroundSchemaSyncManager(cli)
        mgr.schedule(datasource="")
        time.sleep(0.05)
        assert called["n"] == 0
