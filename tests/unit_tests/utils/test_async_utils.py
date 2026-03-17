import asyncio
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from datus.utils.async_utils import (
    _run_in_new_loop,
    _run_in_thread,
    get_or_create_event_loop,
    is_event_loop_running,
    run_async,
    setup_windows_policy,
)


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


# ---------------------------------------------------------------------------
# setup_windows_policy
# ---------------------------------------------------------------------------


def test_setup_windows_policy_non_windows():
    """On non-Windows platforms this should be a no-op (no exception)."""
    with patch.object(sys, "platform", "linux"):
        setup_windows_policy()  # should not raise


def test_setup_windows_policy_windows():
    """On Windows, sets the event loop policy (mocked to avoid real change)."""
    mock_policy = MagicMock()
    with patch.object(sys, "platform", "win32"):
        with patch("asyncio.set_event_loop_policy") as mock_set:
            with patch.object(asyncio, "WindowsSelectorEventLoopPolicy", mock_policy, create=True):
                setup_windows_policy()
                mock_set.assert_called_once()


# ---------------------------------------------------------------------------
# is_event_loop_running
# ---------------------------------------------------------------------------


def test_is_event_loop_running_false_in_sync():
    """In a plain sync context there is no running loop."""
    assert is_event_loop_running() is False


@pytest.mark.asyncio
async def test_is_event_loop_running_true_in_async():
    """Inside an async function there IS a running loop."""
    assert is_event_loop_running() is True


# ---------------------------------------------------------------------------
# get_or_create_event_loop
# ---------------------------------------------------------------------------


def test_get_or_create_event_loop_returns_loop():
    loop = get_or_create_event_loop()
    assert loop is not None
    assert not loop.is_closed()


def test_get_or_create_event_loop_creates_new_when_closed():
    """If current loop is closed, a new one should be created."""
    old_loop = asyncio.new_event_loop()
    old_loop.close()
    asyncio.set_event_loop(old_loop)
    try:
        new_loop = get_or_create_event_loop()
        assert new_loop is not old_loop
        assert not new_loop.is_closed()
    finally:
        asyncio.set_event_loop(None)


def test_get_or_create_event_loop_in_thread_without_loop():
    """In a new thread without a loop, a new loop should be created."""
    result_holder = {}

    def thread_fn():
        asyncio.set_event_loop(None)
        try:
            loop = get_or_create_event_loop()
            result_holder["loop"] = loop
            result_holder["closed"] = loop.is_closed()
        finally:
            asyncio.set_event_loop(None)

    t = threading.Thread(target=thread_fn)
    t.start()
    t.join(timeout=5)
    assert "loop" in result_holder
    assert result_holder["closed"] is False


# ---------------------------------------------------------------------------
# _run_in_new_loop
# ---------------------------------------------------------------------------


def test_run_in_new_loop_basic():
    async def add(a, b):
        return a + b

    result = _run_in_new_loop(add(2, 3))
    assert result == 5


def test_run_in_new_loop_with_timeout():
    async def fast():
        await asyncio.sleep(0.001)
        return "ok"

    result = _run_in_new_loop(fast(), timeout=5.0)
    assert result == "ok"


def test_run_in_new_loop_timeout_raises():
    async def slow():
        await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        _run_in_new_loop(slow(), timeout=0.05)


def test_run_in_new_loop_exception_propagates():
    async def boom():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        _run_in_new_loop(boom())


# ---------------------------------------------------------------------------
# _run_in_thread
# ---------------------------------------------------------------------------


def test_run_in_thread_basic():
    async def greet(name):
        return f"hello {name}"

    result = _run_in_thread(greet("world"))
    assert result == "hello world"


def test_run_in_thread_exception_propagates():
    async def raise_error():
        raise RuntimeError("thread error")

    with pytest.raises(RuntimeError, match="thread error"):
        _run_in_thread(raise_error())


def test_run_in_thread_timeout():
    async def slow():
        await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        _run_in_thread(slow(), timeout=0.1)


# ---------------------------------------------------------------------------
# run_async - core scenarios
# ---------------------------------------------------------------------------


def test_run_async_basic_sync_context():
    async def double(x):
        return x * 2

    assert run_async(double(7)) == 14


def test_run_async_exception_propagates():
    async def fail():
        raise KeyError("missing")

    with pytest.raises(KeyError, match="missing"):
        run_async(fail())


def test_run_async_with_timeout_passes():
    async def quick():
        await asyncio.sleep(0.001)
        return 99

    assert run_async(quick(), timeout=5.0) == 99


def test_run_async_timeout_raises():
    async def slow():
        await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        run_async(slow(), timeout=0.05)


@pytest.mark.asyncio
async def test_run_async_in_async_context_uses_thread():
    """When called from an async context, run_async should use a thread."""

    async def compute():
        return "thread_result"

    result = run_async(compute())
    assert result == "thread_result"


def test_run_async_nested_detection():
    """Nested run_async calls should not deadlock (use thread pool path)."""

    async def inner():
        return "inner"

    async def outer():
        return "outer"

    # Simulate nested by setting the flag directly
    from datus.utils import async_utils

    old_flag = getattr(async_utils._local, "in_run_async", False)
    async_utils._local.in_run_async = True
    try:
        result = run_async(inner())
        assert result == "inner"
    finally:
        async_utils._local.in_run_async = old_flag


def test_run_async_multithreaded():
    """Multiple threads should each be able to call run_async independently."""
    results = []
    errors = []

    async def compute(val):
        await asyncio.sleep(0.01)
        return val * 10

    def worker(val):
        try:
            r = run_async(compute(val))
            results.append(r)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(results) == 5
    assert sorted(results) == [0, 10, 20, 30, 40]
