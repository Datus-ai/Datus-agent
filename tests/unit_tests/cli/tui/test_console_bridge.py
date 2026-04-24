# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for :mod:`datus.cli.tui.console_bridge`.

The bridge has a single helper, ``run_in_terminal_sync``, whose contract is:

* If no prompt_toolkit Application is active, invoke ``func`` directly.
* If one is active, schedule ``func`` via ``run_in_terminal`` and block on
  the returned future.
"""

from __future__ import annotations

from unittest import mock

from datus.cli.tui.console_bridge import run_in_terminal_sync


def test_runs_inline_when_no_application() -> None:
    # The real ``get_app_or_none`` returns ``None`` in unit tests (there is
    # no Application), so patching is unnecessary here — exercise the real
    # code path to catch accidental regressions in the fallback logic.
    called: dict = {"count": 0}

    def _fn() -> None:
        called["count"] += 1

    run_in_terminal_sync(_fn)

    assert called["count"] == 1


def test_schedules_via_run_in_terminal_when_on_event_loop() -> None:
    """On the prompt_toolkit event loop, dispatch fires-and-forgets.

    Fire-and-forget because blocking on the returned future would deadlock
    the loop that has to run the scheduled callback. The no-await path is
    taken only when ``asyncio.get_running_loop()`` succeeds on the calling
    thread, so we simulate that with a patch.
    """
    fake_future = mock.MagicMock()
    fake_app = mock.MagicMock()
    fake_loop = mock.MagicMock()

    with (
        mock.patch("datus.cli.tui.console_bridge.get_app_or_none", return_value=fake_app),
        mock.patch("datus.cli.tui.console_bridge.asyncio.get_running_loop", return_value=fake_loop),
        mock.patch("datus.cli.tui.console_bridge.run_in_terminal", return_value=fake_future) as mocked_rit,
    ):
        fn = mock.MagicMock()
        run_in_terminal_sync(fn)

        mocked_rit.assert_called_once_with(fn)
        # On-loop path must NOT block on the future — it would deadlock.
        fake_future.result.assert_not_called()


def test_submits_coroutine_from_off_loop_thread() -> None:
    """Off the event loop with a live Application, submit via app.loop.

    ``run_in_terminal`` needs a running loop on the calling thread (Python
    3.12+); on the streaming refresh daemon there is none, so the bridge
    hops to ``app.loop`` via :func:`asyncio.run_coroutine_threadsafe` and
    blocks on the concurrent.futures.Future it returns.
    """
    fake_cf = mock.MagicMock()
    fake_cf.result.return_value = None
    fake_app = mock.MagicMock()

    with (
        mock.patch("datus.cli.tui.console_bridge.get_app_or_none", return_value=fake_app),
        mock.patch(
            "datus.cli.tui.console_bridge.asyncio.get_running_loop",
            side_effect=RuntimeError("no loop"),
        ),
        mock.patch(
            "datus.cli.tui.console_bridge.asyncio.run_coroutine_threadsafe",
            return_value=fake_cf,
        ) as mocked_submit,
    ):
        run_in_terminal_sync(lambda: None)

    mocked_submit.assert_called_once()
    _coro, loop_arg = mocked_submit.call_args.args
    assert loop_arg is fake_app.loop
    fake_cf.result.assert_called_once()


def test_swallows_future_exceptions() -> None:
    # Exceptions raised on the prompt_toolkit loop thread must not leak
    # back to callers: a worker-thread callback failing silently is
    # preferable to the REPL crashing mid-dispatch.
    fake_cf = mock.MagicMock()
    fake_cf.result.side_effect = RuntimeError("boom")
    fake_app = mock.MagicMock()

    with (
        mock.patch("datus.cli.tui.console_bridge.get_app_or_none", return_value=fake_app),
        mock.patch(
            "datus.cli.tui.console_bridge.asyncio.get_running_loop",
            side_effect=RuntimeError("no loop"),
        ),
        mock.patch(
            "datus.cli.tui.console_bridge.asyncio.run_coroutine_threadsafe",
            return_value=fake_cf,
        ) as mocked_submit,
    ):
        run_in_terminal_sync(lambda: None)

    # The bridge both scheduled the callback and awaited its result even
    # though the result raised — the exception was swallowed, not re-raised.
    mocked_submit.assert_called_once()
    fake_cf.result.assert_called_once()


def test_runs_inline_when_loop_missing_on_app() -> None:
    """Application without a running loop (e.g. pre-run) → invoke directly.

    There is no pinned area to preserve yet, so forwarding through
    ``run_in_terminal`` would be needlessly indirect — the helper falls
    through to an inline call.
    """
    fake_app = mock.MagicMock()
    fake_app.loop = None
    called = {"count": 0}

    def _fn() -> None:
        called["count"] += 1

    with (
        mock.patch("datus.cli.tui.console_bridge.get_app_or_none", return_value=fake_app),
        mock.patch(
            "datus.cli.tui.console_bridge.asyncio.get_running_loop",
            side_effect=RuntimeError("no loop"),
        ),
    ):
        run_in_terminal_sync(_fn)

    assert called["count"] == 1
