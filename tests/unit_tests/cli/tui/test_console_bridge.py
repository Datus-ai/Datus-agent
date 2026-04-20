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


def test_schedules_via_run_in_terminal_when_app_active() -> None:
    fake_future = mock.MagicMock()
    fake_future.result.return_value = None
    fake_app = mock.MagicMock()

    with (
        mock.patch("datus.cli.tui.console_bridge.get_app_or_none", return_value=fake_app),
        mock.patch("datus.cli.tui.console_bridge.run_in_terminal", return_value=fake_future) as mocked_rit,
    ):
        fn = mock.MagicMock()
        run_in_terminal_sync(fn)

        mocked_rit.assert_called_once_with(fn)
        # ``result()`` must be awaited so the caller can rely on the print
        # having completed before returning control.
        fake_future.result.assert_called_once()


def test_swallows_future_exceptions() -> None:
    # Exceptions raised on the prompt_toolkit loop thread must not leak
    # back to callers: a worker-thread callback failing silently is
    # preferable to the REPL crashing mid-dispatch.
    fake_future = mock.MagicMock()
    fake_future.result.side_effect = RuntimeError("boom")
    fake_app = mock.MagicMock()

    with (
        mock.patch("datus.cli.tui.console_bridge.get_app_or_none", return_value=fake_app),
        mock.patch("datus.cli.tui.console_bridge.run_in_terminal", return_value=fake_future) as mocked_rit,
    ):
        run_in_terminal_sync(lambda: None)

    # The bridge both scheduled the callback and awaited its result even
    # though the result raised — the exception was swallowed, not re-raised.
    mocked_rit.assert_called_once()
    fake_future.result.assert_called_once()


def test_passes_exact_callable_without_wrapping() -> None:
    # ``run_in_terminal`` accepts a no-arg callable; the bridge must not
    # wrap it in a lambda (which would defeat introspection in hooks).
    target = mock.MagicMock()
    fake_future = mock.MagicMock()
    fake_future.result.return_value = None
    fake_app = mock.MagicMock()

    with (
        mock.patch("datus.cli.tui.console_bridge.get_app_or_none", return_value=fake_app),
        mock.patch("datus.cli.tui.console_bridge.run_in_terminal", return_value=fake_future) as mocked_rit,
    ):
        run_in_terminal_sync(target)
        # Positional argument identity check — our helper should forward the
        # object unchanged.
        (passed,), _kwargs = mocked_rit.call_args
        assert passed is target
