# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import threading

import pytest

from datus.utils.exceptions import DatusException
from datus.utils.mcp_execution import MCPToolExecutor


@pytest.mark.parametrize("workers,queued", [(0, 1), (-1, 1), (1, -1)])
def test_invalid_capacity_is_rejected(workers, queued):
    with pytest.raises(DatusException, match="MCP max_workers"):
        MCPToolExecutor(max_workers=workers, max_queue_size=queued)


@pytest.mark.asyncio
async def test_capacity_recovers_after_failure():
    executor = MCPToolExecutor(max_workers=1, max_queue_size=0)

    def fail():
        raise ValueError("driver failed")

    try:
        with pytest.raises(ValueError, match="driver failed"):
            await executor.run(fail)
        assert await executor.run(lambda: "recovered") == "recovered"
    finally:
        executor.shutdown()


@pytest.mark.asyncio
async def test_cancelled_worker_releases_capacity_only_after_driver_returns():
    executor = MCPToolExecutor(max_workers=1, max_queue_size=0)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    late_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: late_errors.append(context))

    def driver():
        loop.call_soon_threadsafe(started.set)
        release.wait(2)
        raise ValueError("driver failed after cancellation")

    async def probe_until_free():
        while True:
            try:
                return await executor.run(lambda: "free")
            except DatusException:
                await asyncio.sleep(0)

    task = asyncio.create_task(executor.run(driver))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(DatusException, match="capacity exhausted"):
            await executor.run(lambda: "too early")
        release.set()
        assert await asyncio.wait_for(probe_until_free(), 2) == "free"
        # Late exceptions are retrieved even though the client has gone away.
        await asyncio.sleep(0)
        assert late_errors == []
    finally:
        release.set()
        executor.shutdown()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_shutdown_rejects_further_work():
    executor = MCPToolExecutor(max_workers=1, max_queue_size=0)
    executor.shutdown()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="shutdown"):
            await executor.run(lambda: None)
