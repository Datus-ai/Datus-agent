# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ToolResultChannel."""

import asyncio

import pytest

from datus.tools.proxy.tool_result_channel import ToolResultChannel


@pytest.mark.ci
class TestToolResultChannel:
    @pytest.mark.asyncio
    async def test_wait_and_publish(self):
        channel = ToolResultChannel()

        async def publisher():
            await asyncio.sleep(0.01)
            await channel.publish("call_1", {"success": 1, "result": "hello"})

        task = asyncio.create_task(publisher())
        result = await channel.wait_for("call_1")
        await task

        assert result == {"success": 1, "result": "hello"}

    @pytest.mark.asyncio
    async def test_publish_before_wait(self):
        """publish arrives before wait_for — result must not be lost."""
        channel = ToolResultChannel()
        await channel.publish("call_early", "early_result")
        result = await channel.wait_for("call_early")
        assert result == "early_result"

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        channel = ToolResultChannel()

        async def waiter():
            return await channel.wait_for("call_cancel")

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        channel.cancel_all("test shutdown")

        with pytest.raises(RuntimeError, match="test shutdown"):
            await task

    @pytest.mark.asyncio
    async def test_cancel_all_clears_futures(self):
        channel = ToolResultChannel()

        async def waiter():
            try:
                return await channel.wait_for("call_x")
            except RuntimeError:
                pass

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        channel.cancel_all()
        await task

        assert len(channel._futures) == 0

    @pytest.mark.asyncio
    async def test_future_done_after_wait(self):
        channel = ToolResultChannel()

        async def publisher():
            await asyncio.sleep(0.01)
            await channel.publish("call_cleanup", "result")

        task = asyncio.create_task(publisher())
        result = await channel.wait_for("call_cleanup")
        await task

        assert result == "result"
        assert channel._futures["call_cleanup"].done()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_waits(self):
        channel = ToolResultChannel()

        async def publisher():
            await asyncio.sleep(0.01)
            await channel.publish("call_a", "result_a")
            await channel.publish("call_b", "result_b")

        task = asyncio.create_task(publisher())
        result_a, result_b = await asyncio.gather(
            channel.wait_for("call_a"),
            channel.wait_for("call_b"),
        )
        await task

        assert result_a == "result_a"
        assert result_b == "result_b"

    @pytest.mark.asyncio
    async def test_publish_to_done_future_is_noop(self):
        channel = ToolResultChannel()

        await channel.publish("call_done", "first")
        await channel.publish("call_done", "second")  # should be ignored (future already done)

        result = await channel.wait_for("call_done")
        assert result == "first"
