# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.utils.async_debug`."""

import asyncio
import os
import signal

import pytest

from datus.utils import async_debug
from datus.utils.async_debug import (
    dump_async_tasks_to_log,
    format_async_tasks,
    install_task_dump_signal_handler,
)


@pytest.fixture(autouse=True)
def _reset_captured_loop():
    """Each test should start with no captured loop and restore afterwards."""
    async_debug._dump_loop = None
    yield
    async_debug._dump_loop = None
    # Best-effort: ensure the suite-wide SIGUSR1 handler is reset so a handler
    # bound to a closed test loop never fires later.
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)


def test_format_no_loop_returns_message():
    """Outside any running loop and with nothing captured -> explicit notice."""
    text = format_async_tasks()
    assert "no event loop available" in text


@pytest.mark.asyncio
async def test_format_includes_parked_task_and_name():
    """A task parked on an unresolved future must appear in the dump."""
    loop = asyncio.get_running_loop()
    parked_fut = loop.create_future()

    async def parked():
        await parked_fut

    task = asyncio.create_task(parked(), name="parked-worker")
    await asyncio.sleep(0)  # let the task start and park on the future

    try:
        text = format_async_tasks()
        # Task name surfaces so the operator can identify the stuck coroutine.
        assert "parked-worker" in text
        # The coroutine qualname is rendered, not just repr().
        assert "parked" in text
        # Header reports the live task count (>= our task + current task).
        assert "live task(s)" in text
        assert "end async task dump" in text
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_format_limit_caps_frames():
    """``limit`` is forwarded to ``Task.print_stack`` and bounds frame output."""
    loop = asyncio.get_running_loop()
    parked_fut = loop.create_future()

    async def parked():
        await parked_fut

    task = asyncio.create_task(parked(), name="limited")
    await asyncio.sleep(0)
    try:
        full = format_async_tasks(limit=None)
        capped = format_async_tasks(limit=1)
        # The capped dump should never be longer than the full one.
        assert len(capped) <= len(full)
        assert "limited" in capped
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_dump_to_log_emits_warning_and_returns_text(caplog):
    """``dump_async_tasks_to_log`` logs at WARNING and returns the same text."""
    with caplog.at_level("WARNING", logger="datus.utils.async_debug"):
        text = dump_async_tasks_to_log()
    assert "async task dump" in text
    assert any("Async task stack dump requested" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_install_captures_running_loop():
    """Installation records the running loop so later dumps can resolve it."""
    installed = install_task_dump_signal_handler()
    # On platforms without SIGUSR1 (e.g. Windows) install must report False and
    # capture no loop; everywhere else it installs and captures the running loop.
    has_sigusr1 = hasattr(signal, "SIGUSR1")
    assert installed is has_sigusr1
    assert (async_debug._dump_loop is asyncio.get_running_loop()) is has_sigusr1


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 unavailable on this platform")
@pytest.mark.asyncio
async def test_signal_handler_triggers_dump(caplog):
    """Sending SIGUSR1 routes through the loop handler and logs a dump."""
    parked_fut = asyncio.get_running_loop().create_future()

    async def parked():
        await parked_fut

    task = asyncio.create_task(parked(), name="signal-target")
    await asyncio.sleep(0)
    assert install_task_dump_signal_handler() is True

    try:
        with caplog.at_level("WARNING", logger="datus.utils.async_debug"):
            os.kill(os.getpid(), signal.SIGUSR1)
            # Yield control so the loop services the signal handler callback.
            await asyncio.sleep(0.1)
        joined = "\n".join(r.message for r in caplog.records)
        assert "Async task stack dump requested" in joined
        assert "signal-target" in joined
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_await_chain_descends_through_async_generator():
    """The await chain must cross the ``async for`` boundary to the real block.

    ``Task.print_stack`` stops at the consumer's ``async for`` line; the await
    chain has to descend into the generator and the coroutine it awaits to
    surface the innermost parked frame. This is the whole point of the feature
    for diagnosing the ``execute_write`` hang.
    """
    parked_fut = asyncio.get_running_loop().create_future()

    async def deep_block():
        await parked_fut  # the real block, inside a nested coroutine

    async def producer():
        await deep_block()
        yield 1

    async def consumer():
        async for _ in producer():
            pass

    task = asyncio.create_task(consumer(), name="chain-consumer")
    await asyncio.sleep(0)  # let it park
    try:
        text = format_async_tasks()
        # All three frames of the chain must appear, innermost included.
        assert "in consumer" in text
        assert "in producer" in text
        assert "in deep_block" in text
        assert "await chain (innermost last):" in text
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_explicit_loop_arg_overrides_capture():
    """An explicit loop argument is honored over the captured/running loop."""
    loop = asyncio.get_running_loop()
    text = format_async_tasks(loop=loop)
    assert "async task dump" in text
