# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Async channel for proxy tool results.

Allows proxy tools to await results that are published from stdin dispatch.
Wait and publish are order-independent: either side can arrive first.
"""

import asyncio
import os
from typing import Any, Dict, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Safety net: a proxied tool (e.g. write_file/edit_file executed on the client)
# blocks the agent loop until the client POSTs its result. If the client never
# reports — tab closed, crash, or a frontend bug that swallows the report — the
# loop would otherwise hang forever. ``wait_for`` defaults to this bound so the
# turn fails cleanly instead. Override via DATUS_PROXY_TOOL_RESULT_TIMEOUT (s).
DEFAULT_RESULT_TIMEOUT: float = float(os.getenv("DATUS_PROXY_TOOL_RESULT_TIMEOUT", "600"))


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

    async def wait_for(self, call_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a result to be published for the given call_id.

        ``timeout`` (seconds) bounds the wait so a never-reported client tool
        cannot block the agent loop forever; on expiry the dead future is
        dropped and ``asyncio.TimeoutError`` propagates to the caller.
        """
        async with self._lock:
            future = self._get_or_create_future(call_id)
        if timeout is None:
            return await future
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            # Drop the abandoned future so a late publish doesn't target a
            # waiter that has already given up.
            async with self._lock:
                if self._futures.get(call_id) is future:
                    self._futures.pop(call_id, None)
            raise

    async def publish(self, call_id: str, result: Any) -> None:
        """Publish a result for the given call_id."""
        async with self._lock:
            future = self._get_or_create_future(call_id)
            if future.done():
                # Already settled — typically a duplicate report, or one that
                # arrived after wait_for timed out and dropped its waiter.
                logger.warning(f"Tool result for call_id={call_id} ignored; waiter already settled")
                return
            future.set_result(result)
        logger.info(f"Tool result published for call_id={call_id}")

    def cancel_all(self, reason: str = "Channel closed"):
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
