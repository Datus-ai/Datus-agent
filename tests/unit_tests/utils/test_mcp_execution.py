# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import threading

import pytest

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.mcp_execution import MCPToolExecutor


@pytest.mark.parametrize(
    "raw",
    [
        {"workers": 2},
        {"max_workers": True},
        {"max_queue_size": False},
        {"max_workers": "invalid"},
        {"max_workers": 1.5},
        {"max_workers": None},
        {"max_workers": "0"},
        {"max_queue_size": "-1"},
        {"max_workers": "1.5"},
    ],
)
def test_malformed_executor_configuration_has_a_config_error(raw):
    with pytest.raises(DatusException) as caught:
        MCPToolExecutor.from_config(raw)
    assert caught.value.code == ErrorCode.COMMON_CONFIG_ERROR


@pytest.mark.asyncio
async def test_quoted_capacity_values_apply_exact_admission_limits():
    executor = MCPToolExecutor.from_config({"max_workers": "1", "max_queue_size": "0"})
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def block():
        loop.call_soon_threadsafe(started.set)
        release.wait(2)

    task = asyncio.create_task(executor.run(block))
    try:
        await asyncio.wait_for(started.wait(), 2)
        with pytest.raises(DatusException) as caught:
            await executor.run(lambda: "excess")
        assert caught.value.code == ErrorCode.TOOL_EXECUTION_FAILED
    finally:
        release.set()
        await task
        executor.shutdown()


@pytest.mark.parametrize("workers,queued", [(0, 1), (-1, 1), (1, -1)])
def test_invalid_capacity_is_rejected(workers, queued):
    with pytest.raises(DatusException, match="MCP max_workers"):
        MCPToolExecutor(max_workers=workers, max_queue_size=queued)


@pytest.mark.parametrize("key", ["max_workers", "max_queue_size"])
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_direct_constructor_rejects_non_integer_capacity(key, value):
    with pytest.raises(DatusException) as caught:
        MCPToolExecutor(**{key: value})
    assert caught.value.code == ErrorCode.COMMON_CONFIG_ERROR


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


@pytest.mark.asyncio
async def test_rejected_job_releases_its_resource():
    from datus.mcp_server import ToolContext

    executor = MCPToolExecutor(max_workers=1, max_queue_size=0)
    executor.shutdown()
    context = ToolContext("test", None, None, {})
    closed = threading.Event()
    try:
        with pytest.raises(RuntimeError, match="shutdown"):
            await executor.run_with_resource(context, lambda: None)
        context.retire(lambda ctx: closed.set())
        assert closed.is_set(), "submission failure leaked the resource pin"
    finally:
        executor.shutdown()
