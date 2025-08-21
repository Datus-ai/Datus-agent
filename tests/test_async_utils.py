import asyncio

import pytest

from datus.utils.async_utils import run_async


async def sample_async_func(value: int) -> str:
    """Sample async function for testing"""
    await asyncio.sleep(0.01)
    return f"async_{value}"


class TestAsyncRunner:
    """Test cases for AsyncRunner"""

    def test_run_async_in_sync_context(self):
        """Test running async function in sync context"""
        result = run_async(sample_async_func(42))
        assert result == "async_42"

    @pytest.mark.asyncio
    async def test_run_in_async_context(self):
        """Test running in existing async context"""
        result = await sample_async_func(400)
        assert result == "async_400"

    def test_thread_safety(self):
        """Test running in different thread"""
        import concurrent.futures

        def run_in_thread():
            return run_async(sample_async_func(600))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            result = future.result()
            assert result == "async_600"

    @pytest.mark.asyncio
    async def test_nested_async_calls(self):
        """Test nested async function calls"""

        async def outer():
            result1 = await sample_async_func(800)
            result2 = await sample_async_func(900)
            return f"{result1}_{result2}"

        result = await outer()
        assert result == "async_800_async_900"
