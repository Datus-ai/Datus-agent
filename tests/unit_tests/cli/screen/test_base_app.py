# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for :mod:`datus.cli.screen.base_app`.

These tests exercise the three branches of ``BaseApp.run`` without actually
mounting a Textual app: the ``no pt Application`` fast path, and the
``run_coroutine_threadsafe`` bridge that routes the Textual ``run`` through
a real background asyncio loop.
"""

from __future__ import annotations

import asyncio
import threading
from unittest import mock

import pytest

from datus.cli.screen.base_app import BaseApp


class _LoopOnThread:
    """Spin up a real asyncio loop on a helper thread for test fixtures."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=5)
        assert self.loop.is_running()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._ready.set)
        self.loop.run_forever()


@pytest.fixture
def pt_loop() -> _LoopOnThread:
    helper = _LoopOnThread()
    helper.start()
    try:
        yield helper
    finally:
        helper.stop()


@pytest.fixture
def base_app() -> BaseApp:
    # ``BaseApp.__init__`` calls into textual.App, which is fine — we never
    # actually .run() it below; we patch out ``textual.app.App.run``.
    return BaseApp()


class TestBaseAppRunNoPromptToolkit:
    def test_falls_through_when_no_pt_app(self, base_app: BaseApp) -> None:
        """With no pt Application live, ``run`` defers to the Textual parent."""
        with (
            mock.patch("prompt_toolkit.application.get_app_or_none", return_value=None),
            mock.patch("textual.app.App.run", return_value="textual-ok") as parent_run,
        ):
            result = base_app.run()

        assert result == "textual-ok"
        parent_run.assert_called_once()

    def test_falls_through_when_pt_loop_missing(self, base_app: BaseApp) -> None:
        """pt app present but no bound loop → still take the fast path."""
        stub_app = mock.MagicMock(loop=None)
        with (
            mock.patch("prompt_toolkit.application.get_app_or_none", return_value=stub_app),
            mock.patch("textual.app.App.run", return_value="textual-ok") as parent_run,
        ):
            result = base_app.run()

        assert result == "textual-ok"
        parent_run.assert_called_once()

    def test_falls_through_when_pt_loop_not_running(self, base_app: BaseApp) -> None:
        """A loop that has never started does not trigger the bridge."""
        dead_loop = asyncio.new_event_loop()
        try:
            stub_app = mock.MagicMock(loop=dead_loop)
            with (
                mock.patch("prompt_toolkit.application.get_app_or_none", return_value=stub_app),
                mock.patch("textual.app.App.run", return_value="textual-ok") as parent_run,
            ):
                result = base_app.run()
            assert result == "textual-ok"
            parent_run.assert_called_once()
        finally:
            dead_loop.close()


class TestBaseAppRunBridge:
    """Exercise the run_coroutine_threadsafe bridge with a real background loop."""

    def test_bridges_to_pt_loop_and_returns_textual_value(self, base_app: BaseApp, pt_loop: _LoopOnThread) -> None:
        """Textual ``run_async`` should execute on the pt loop thread and its return value propagate back."""
        textual_thread: dict = {}

        async def _fake_run_async(*_a, **_kw):  # awaited on the pt loop
            textual_thread["name"] = threading.current_thread().name
            return "textual-result"

        stub_app = mock.MagicMock(loop=pt_loop.loop)
        with (
            mock.patch("prompt_toolkit.application.get_app_or_none", return_value=stub_app),
            mock.patch("textual.app.App.run_async", new=_fake_run_async),
        ):
            result = base_app.run()

        assert result == "textual-result"
        # Textual must run on the pt loop thread (the main-thread loop) so
        # driver ``signal.signal`` calls succeed, not on the caller worker.
        assert textual_thread["name"] != threading.current_thread().name

    def test_bridge_propagates_exception(self, base_app: BaseApp, pt_loop: _LoopOnThread) -> None:
        """A crash in Textual should surface as the original exception — not InvalidStateError."""
        stub_app = mock.MagicMock(loop=pt_loop.loop)

        class BoomError(RuntimeError):
            pass

        async def _boom(*_a, **_kw):
            raise BoomError("kaboom")

        with (
            mock.patch("prompt_toolkit.application.get_app_or_none", return_value=stub_app),
            mock.patch("textual.app.App.run_async", new=_boom),
        ):
            with pytest.raises(BoomError, match="kaboom"):
                base_app.run()
