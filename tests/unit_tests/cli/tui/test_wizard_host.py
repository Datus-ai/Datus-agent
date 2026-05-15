# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.tui.wizard_host` and :class:`DatusApp`'s
embedded-wizard host API.

The wizard host introduces the contract that lets sub-wizards (``/model``,
``/agent``, ``ask_user`` …) render inside the main TUI's bottom slot
instead of taking over the full screen. These tests cover:

- ``resolve_with`` / ``resolve_cancel`` idempotence on the ``done_future``
- ``DatusApp.mount_wizard`` swaps the bottom slot to the wizard's
  container
- ``DatusApp.unmount_wizard`` restores the normal status+input HSplit
- ``DatusApp.run_wizard`` from a worker thread blocks on the
  ``done_future`` and unmounts on completion
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from datus.cli.tui.app import DatusApp
from datus.cli.tui.wizard_host import EmbeddedWizard, resolve_cancel, resolve_with


def _make_loop_with_future() -> tuple[asyncio.AbstractEventLoop, asyncio.Future]:
    loop = asyncio.new_event_loop()
    return loop, loop.create_future()


def test_resolve_with_sets_future_result():
    loop, fut = _make_loop_with_future()
    try:
        resolve_with(fut, "value")
        assert fut.done()
        assert fut.result() == "value"
    finally:
        loop.close()


def test_resolve_cancel_sets_none_result():
    loop, fut = _make_loop_with_future()
    try:
        resolve_cancel(fut)
        assert fut.done()
        assert fut.result() is None
    finally:
        loop.close()


def test_resolvers_are_idempotent_after_done():
    """Wizards may bind ESC and Ctrl+C to the same cancel path — second
    invocation must not raise InvalidStateError."""
    loop, fut = _make_loop_with_future()
    try:
        resolve_with(fut, "first")
        # Calling either resolver again is a no-op.
        resolve_with(fut, "second")
        resolve_cancel(fut)
        assert fut.result() == "first"
    finally:
        loop.close()


def _make_wizard(done_future: asyncio.Future) -> EmbeddedWizard:
    """Minimal EmbeddedWizard for host tests — a single focusable Window."""
    kb = KeyBindings()
    window = Window(content=FormattedTextControl(text=lambda: [("class:wiz", "wizard body")], focusable=True))
    return EmbeddedWizard(
        container=HSplit([window]),
        key_bindings=kb,
        first_focus=window,
        done_future=done_future,
    )


def _build_datus_app() -> DatusApp:
    return DatusApp(status_tokens_fn=lambda: [], dispatch_fn=lambda _: None)


def test_bottom_section_returns_normal_when_no_wizard():
    app = _build_datus_app()
    assert app._bottom_section() is app._normal_bottom_section


def test_mount_wizard_replaces_bottom_section():
    app = _build_datus_app()
    loop = asyncio.new_event_loop()
    try:
        fut = loop.create_future()
        wizard = _make_wizard(fut)

        app.mount_wizard(wizard)
        assert app._active_wizard is wizard
        assert app._bottom_section() is wizard.container

        app.unmount_wizard()
        assert app._active_wizard is None
        assert app._bottom_section() is app._normal_bottom_section
    finally:
        loop.close()


def test_run_wizard_blocks_until_done_future_resolves():
    """Worker-thread API: ``run_wizard`` must block until the wizard's
    bindings resolve ``done_future``."""
    app = _build_datus_app()

    loop = asyncio.new_event_loop()
    app._loop = loop

    # Run the loop on a background thread so the worker can submit
    # cross-thread coroutines (this mirrors the production setup
    # where DatusApp.run() owns the main loop).
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        captured_future: dict = {}

        def factory(done_future: asyncio.Future) -> EmbeddedWizard:
            captured_future["fut"] = done_future
            return _make_wizard(done_future)

        # The worker thread calls run_wizard which blocks here:
        worker_result: dict = {}

        def worker() -> None:
            worker_result["value"] = app.run_wizard(factory)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        # Give the factory a moment to install ``captured_future``.
        for _ in range(50):
            if "fut" in captured_future:
                break
            threading.Event().wait(0.01)
        assert "fut" in captured_future, "factory was not invoked"

        # Resolve via the loop thread; ``call_later`` is not thread-safe and
        # would not wake a loop already parked in ``select(timeout=None)``.
        loop.call_soon_threadsafe(lambda: captured_future["fut"].set_result({"ok": True}))
        worker_thread.join(timeout=2.0)
        assert not worker_thread.is_alive(), "run_wizard did not return"
        assert worker_result["value"] == {"ok": True}
        # After run_wizard returns the bottom slot must be back to normal.
        # mount/unmount happens via call_soon_threadsafe; give the loop a tick.
        for _ in range(50):
            if app._active_wizard is None:
                break
            threading.Event().wait(0.01)
        assert app._active_wizard is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()
        app._loop = None


def test_run_wizard_returns_none_on_cancel():
    app = _build_datus_app()
    loop = asyncio.new_event_loop()
    app._loop = loop
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        captured: dict = {}

        def factory(done_future: asyncio.Future) -> EmbeddedWizard:
            captured["fut"] = done_future
            return _make_wizard(done_future)

        worker_result: dict = {}

        def worker() -> None:
            worker_result["value"] = app.run_wizard(factory)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        for _ in range(50):
            if "fut" in captured:
                break
            threading.Event().wait(0.01)

        loop.call_soon_threadsafe(resolve_cancel, captured["fut"])
        worker_thread.join(timeout=2.0)
        assert worker_result["value"] is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()
        app._loop = None


def test_run_wizard_raises_without_event_loop():
    """Outside TUI mode the loop is None — callers must take the standalone path."""
    app = _build_datus_app()
    # _loop defaults to None until ``run()`` starts the asyncio loop.
    with pytest.raises(RuntimeError, match="event loop"):
        app.run_wizard(_make_wizard)
