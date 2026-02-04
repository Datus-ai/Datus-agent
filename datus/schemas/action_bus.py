# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ActionBus – single-channel action stream merger.

Tools call ``bus.put(action)`` to inject sub-actions (e.g. explorer
sub-agent tool calls).  The node calls ``bus.merge(primary, *secondaries)``
to yield everything in one stream for the CLI / web UI.

Lifecycle follows the owning ``AgenticNode``.
"""

from __future__ import annotations

import asyncio
import queue
from typing import AsyncGenerator, Dict, Optional, Set

from datus.schemas.action_history import ActionHistory
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ActionBus:
    """Single-channel action bus with N-stream merge.

    * **put(action)** – thread-safe push for tool sub-actions.
    * **merge(primary, \\*secondaries)** – async generator that yields
      actions from the primary stream, all secondary streams, *and*
      the internal queue, interleaved via ``asyncio.wait``.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[ActionHistory] = queue.Queue()

    # -- push side ----------------------------------------------------------

    def put(self, action: ActionHistory) -> None:
        """Inject an action (non-blocking, thread-safe)."""
        self._queue.put_nowait(action)

    @property
    def has_pending(self) -> bool:
        return not self._queue.empty()

    # -- merge --------------------------------------------------------------

    async def merge(
        self,
        primary: AsyncGenerator[ActionHistory, None],
        *secondaries: AsyncGenerator[ActionHistory, None],
    ) -> AsyncGenerator[ActionHistory, None]:
        """Merge *primary* + *secondaries* + internal queue.

        Terminates when primary exhausts **and** the queue is empty.
        """

        _EXHAUSTED = object()

        # Build named stream map
        streams: Dict[str, AsyncGenerator[ActionHistory, None]] = {"primary": primary}
        for idx, sec in enumerate(secondaries):
            streams[f"secondary_{idx}"] = sec
        # Internal queue exposed as an async generator
        streams["_bus_queue"] = self._fetch()

        iters = {name: s.__aiter__() for name, s in streams.items()}
        tasks: Dict[str, asyncio.Task] = {}
        exhausted: Set[str] = set()

        async def _safe_anext(it):  # type: ignore[no-untyped-def]
            try:
                return await it.__anext__()
            except StopAsyncIteration:
                return _EXHAUSTED

        try:
            while True:
                # Terminate when primary is done and queue is drained
                if "primary" in exhausted and not self.has_pending:
                    break

                # Create tasks for non-exhausted streams without active tasks
                for name, it in iters.items():
                    if name not in exhausted and name not in tasks:
                        tasks[name] = asyncio.create_task(
                            _safe_anext(it), name=name,
                        )

                if not tasks:
                    break

                # Process already-done tasks before waiting
                already_done = {n: t for n, t in tasks.items() if t.done()}
                for name, task in already_done.items():
                    result = task.result()
                    del tasks[name]
                    if result is _EXHAUSTED:
                        exhausted.add(name)
                    else:
                        yield result

                if "primary" in exhausted and not self.has_pending:
                    break

                active_tasks = {n: t for n, t in tasks.items() if not t.done()}
                if not active_tasks:
                    continue

                done, _ = await asyncio.wait(
                    active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    name = task.get_name()
                    result = task.result()
                    tasks.pop(name, None)

                    if result is _EXHAUSTED:
                        exhausted.add(name)
                        logger.debug("ActionBus: stream '%s' exhausted", name)
                    else:
                        yield result

        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    # -- internal -----------------------------------------------------------

    async def _fetch(self) -> AsyncGenerator[ActionHistory, None]:
        """Drain the internal queue as an async generator."""
        while True:
            item = await self._queue_get(timeout=0.1)
            if item is not None:
                yield item

    async def _queue_get(self, timeout: float = 0.1) -> Optional[ActionHistory]:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._queue.get, True, timeout),
                timeout=timeout + 0.1,
            )
        except (queue.Empty, asyncio.TimeoutError):
            return None
