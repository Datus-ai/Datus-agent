# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Async channel for proxy tool results.

Allows proxy tools to await results that are published from stdin dispatch.
Wait and publish are order-independent: either side can arrive first.
"""

import asyncio
from typing import Any, Dict, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Safety net: a proxied tool (e.g. write_file/edit_file executed on the client)
# blocks the agent loop until the client POSTs its result. If the client never
# reports — tab closed, crash, or a frontend bug that swallows the report — the
# loop would otherwise hang forever. ``wait_for`` defaults to this bound so the
# turn fails cleanly instead.
DEFAULT_RESULT_TIMEOUT: float = 600.0


class ToolResultChannel:
    """Async pub/sub channel for proxy tool call results.

    Both ``wait_for`` and ``publish`` lazily create a Future on first access,
    so the result is never lost regardless of which side arrives first.
    """

    def __init__(self):
        self._futures: Dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()

    def _get_or_create_future(self, call_id: str) -> asyncio.Future[Any]:
        fut = self._futures.get(call_id)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            self._futures[call_id] = fut
        return fut

    async def wait_for(self, call_id: str, timeout: Optional[float] = DEFAULT_RESULT_TIMEOUT) -> Any:
        """Wait for a result to be published for the given call_id.

        ``timeout`` (seconds) bounds the wait so a never-reported client tool
        cannot block the agent loop forever; it defaults to
        ``DEFAULT_RESULT_TIMEOUT`` and only an explicit ``timeout=None`` opts
        into an unbounded wait. On expiry ``asyncio.wait_for`` cancels the
        future (leaving it settled) and ``asyncio.TimeoutError`` propagates. The
        settled future is retained so a late ``publish`` for the same
        ``call_id`` is recognised as late rather than silently re-created.
        """
        async with self._lock:
            future = self._get_or_create_future(call_id)
        if timeout is None:
            return await future
        return await asyncio.wait_for(future, timeout)

    async def publish(self, call_id: str, result: Any) -> bool:
        """Publish a result for the given call_id.

        Returns ``True`` if a live waiter received it, ``False`` if the result
        was dropped — a duplicate, or one that arrived after the waiter timed
        out — so callers can tell ``matched`` from ``late`` reports.
        """
        async with self._lock:
            future = self._get_or_create_future(call_id)
            if future.done():
                # Already settled — typically a duplicate report, or one that
                # arrived after wait_for timed out and dropped its waiter.
                logger.warning(f"Tool result for call_id={call_id} ignored; waiter already settled")
                return False
            future.set_result(result)
        logger.info(f"Tool result published for call_id={call_id}")
        return True

    def cancel_all(self, reason: str = "Channel closed") -> None:
        """Cancel all pending futures.

        Note: This is a synchronous method and must be called from the
        same event-loop thread that owns the futures.
        """
        pending = [call_id for call_id, fut in self._futures.items() if not fut.done()]
        if pending:
            logger.info(f"Cancelling {len(pending)} pending tool result(s): {reason}; call_ids={pending}")
        for future in self._futures.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self._futures.clear()
