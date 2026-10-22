# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Bounded execution of blocking MCP work, independent of asyncio's I/O pool."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from threading import BoundedSemaphore
from typing import Any, Callable, TypeVar

from datus.utils.exceptions import DatusException, ErrorCode

T = TypeVar("T")


class MCPToolExecutor:
    """Reject excess work instead of growing the executor queue without bound.

    Capacity belongs to the concurrent future, not its asyncio waiter. Cancelling
    a request cannot stop a running driver read and must not admit replacement
    work until that read really finishes. Even queued work retains its slot on
    cancellation, avoiding a queue full of cancelled executor work items.
    """

    def __init__(self, max_workers: int = 8, max_queue_size: int = 16):
        if max_workers < 1 or max_queue_size < 0:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message="MCP max_workers must be positive and max_queue_size must be nonnegative",
            )
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="datus-mcp")
        self._capacity = BoundedSemaphore(max_workers + max_queue_size)

    def submit(self, function: Callable[..., T], *args, **kwargs) -> Future[T]:
        """Submit admitted work so owners can attach resource cleanup callbacks."""
        if not self._capacity.acquire(blocking=False):
            raise DatusException(ErrorCode.TOOL_EXECUTION_FAILED, message="MCP worker capacity exhausted; retry later")
        try:
            future = self._pool.submit(copy_context().run, partial(function, *args, **kwargs))
        except BaseException:
            self._capacity.release()
            raise
        future.add_done_callback(lambda _: self._capacity.release())
        return future

    @staticmethod
    async def wait(future: Future[T]) -> T:
        """Await a submitted job without cancelling its worker on disconnect."""
        result = asyncio.wrap_future(future)
        # A disconnected client may never retrieve a late driver exception.
        result.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(result)

    async def run(self, function: Callable[..., T], *args, **kwargs) -> T:
        return await self.wait(self.submit(function, *args, **kwargs))

    async def run_with_resource(self, resource: Any, function: Callable[..., T], *args, **kwargs) -> T:
        """Retain a resource until the job finishes, even after waiter cancellation."""
        resource.acquire()
        try:
            future = self.submit(function, *args, **kwargs)
        except BaseException:
            resource.release()
            raise
        future.add_done_callback(lambda _: resource.release())
        return await self.wait(future)

    def shutdown(self) -> None:
        """Stop accepting work without blocking the event loop on stuck reads."""
        self._pool.shutdown(wait=False, cancel_futures=True)


# Registration helpers used without a server share one bounded pool as well.
default_mcp_executor = MCPToolExecutor()
