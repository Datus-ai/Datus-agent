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
from typing import AsyncGenerator, Callable, Dict, List, Optional, Set

from datus.schemas.action_history import ActionHistory
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ActionBus:
    """Single-channel action bus with N-stream merge.

    * **put(action)** – push for tool sub-actions (must be called from
      the event-loop thread, e.g. inside an ``async`` function).
    * **merge(primary, \\*secondaries)** – async generator that yields
      actions from the primary stream, all secondary streams, *and*
      the internal queue, interleaved via ``asyncio.wait``.
    * **close()** – place a sentinel in the queue so ``_fetch()`` terminates
      naturally.  Called automatically when the primary stream exhausts.
    """

    _STOP = object()  # sentinel value

    def __init__(self) -> None:
        # Lazy init so that __init__ can run before an event loop exists.
        # The queue is (re-)created for the current event loop on each
        # merge() call so that the ActionBus survives across different loops
        # (e.g. successive asyncio.run / run_until_complete calls).
        self._queue: Optional[asyncio.Queue] = None
        self._closed: bool = False
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None

    def reset(self) -> None:
        """Drop all pending items and reset state for a new execution.

        Must be called at the start of each top-level execution to prevent
        leftover queued items from a previous (possibly interrupted) run
        from being replayed via ``_rebind_queue()``.
        """
        self._queue = None
        self._closed = False
        self._bound_loop = None

    def _ensure_queue(self) -> asyncio.Queue:
        """Return the internal queue, rebinding if the event loop has changed."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._queue is not None and current_loop is not None and current_loop is not self._bound_loop:
            # Event loop changed — rebind the queue to avoid RuntimeError
            self._queue = self._rebind_queue()

        if self._queue is None:
            self._queue = asyncio.Queue()
            self._bound_loop = current_loop
        return self._queue

    def _rebind_queue(self) -> asyncio.Queue:
        """Re-create the internal queue for the *current* event loop.

        Transfers any items that were put() before merge() started.
        This is necessary because asyncio.Queue is bound to the loop
        where it was first used; if the ActionBus is reused across
        different loops (e.g. successive CLI commands) the old queue
        becomes invalid.
        """
        old = self._queue
        self._queue = asyncio.Queue()
        try:
            self._bound_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._bound_loop = None
        if old is not None:
            while True:
                try:
                    item = old.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is self._STOP:
                    continue
                self._queue.put_nowait(item)
        return self._queue

    # -- push side ----------------------------------------------------------

    def put(self, action: ActionHistory) -> None:
        """Inject an action (non-blocking, must be called from the event-loop thread)."""
        if self._closed:
            logger.warning("ActionBus.put() called after close()")
            return
        self._ensure_queue().put_nowait(action)

    @property
    def has_pending(self) -> bool:
        return self._queue is not None and not self._queue.empty()

    def close(self) -> None:
        """Place a sentinel in the queue so ``_fetch()`` terminates naturally.

        Idempotent – calling close() more than once is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        self._ensure_queue().put_nowait(self._STOP)

    # -- merge --------------------------------------------------------------

    async def merge(
        self,
        primary: AsyncGenerator[ActionHistory, None],
        *secondaries: AsyncGenerator[ActionHistory, None],
        on_primary_done: Optional[Callable[[], None]] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Merge *primary* + *secondaries* + internal queue.

        Terminates when all streams are exhausted (including the internal
        queue, which is closed via sentinel when primary exhausts).

        Args:
            primary: The main action stream.
            *secondaries: Additional action streams (e.g. interaction broker).
            on_primary_done: Optional callback invoked when the primary stream
                exhausts.  Typically used to close secondary streams so they
                also terminate naturally.
        """

        _EXHAUSTED = object()

        # Reset closed state for a new merge cycle.
        self._closed = False

        # (Re-)create the queue for the current event loop, transferring
        # any items that were put() before merge() was called.
        q = self._rebind_queue()

        # Build named stream map – _bus_queue first so injected actions
        # are always yielded before primary/secondary when ready simultaneously.
        stream_order: List[str] = ["_bus_queue", "primary"]
        streams: Dict[str, AsyncGenerator[ActionHistory, None]] = {
            "_bus_queue": self._fetch(q),
            "primary": primary,
        }
        for idx, sec in enumerate(secondaries):
            name = f"secondary_{idx}"
            stream_order.append(name)
            streams[name] = sec

        iters = {name: s.__aiter__() for name, s in streams.items()}
        tasks: Dict[str, asyncio.Task] = {}
        exhausted: Set[str] = set()

        async def _safe_anext(it):  # type: ignore[no-untyped-def]
            try:
                return await it.__anext__()
            except StopAsyncIteration:
                return _EXHAUSTED

        def _on_stream_exhausted(name: str) -> None:
            """Handle side-effects when a stream exhausts."""
            exhausted.add(name)
            logger.debug("ActionBus: stream exhausted", stream=name)
            if name == "primary":
                self.close()
                if on_primary_done is not None:
                    on_primary_done()

        def _drain_done() -> list:
            """Pop and return results from all already-done tasks in stream_order."""
            results = []
            for name in stream_order:
                if name not in tasks or not tasks[name].done():
                    continue
                task = tasks.pop(name)
                result = task.result()
                if result is _EXHAUSTED:
                    _on_stream_exhausted(name)
                else:
                    results.append((name, result))
            return results

        try:
            while True:
                # Create tasks for non-exhausted streams without active tasks
                for name, it in iters.items():
                    if name not in exhausted and name not in tasks:
                        tasks[name] = asyncio.create_task(
                            _safe_anext(it),
                            name=name,
                        )

                if not tasks:
                    break

                # Process already-done tasks in stream_order
                for name, result in _drain_done():
                    _at = getattr(result, "action_type", "?")
                    _role = getattr(result, "role", "?")
                    _depth = getattr(result, "depth", "?")
                    logger.debug(
                        "ActionBus.merge yield (already-done)",
                        stream=name,
                        action_type=_at,
                        role=str(_role),
                        depth=_depth,
                    )
                    yield result

                active_tasks = {n: t for n, t in tasks.items() if not t.done()}
                if not active_tasks:
                    continue

                done, _ = await asyncio.wait(
                    active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                done_names = {t.get_name() for t in done}

                for name in stream_order:
                    if name not in done_names:
                        continue
                    task = tasks.pop(name)
                    result = task.result()

                    if result is _EXHAUSTED:
                        _on_stream_exhausted(name)
                    else:
                        _at = getattr(result, "action_type", "?")
                        _role = getattr(result, "role", "?")
                        _depth = getattr(result, "depth", "?")
                        logger.debug(
                            "ActionBus.merge yield",
                            stream=name,
                            action_type=_at,
                            role=str(_role),
                            depth=_depth,
                        )
                        yield result

            # Drain remaining done tasks after the loop exits normally
            for name, result in _drain_done():
                _at = getattr(result, "action_type", "?")
                _role = getattr(result, "role", "?")
                _depth = getattr(result, "depth", "?")
                logger.debug(
                    "ActionBus.merge yield (post-loop drain)",
                    stream=name,
                    action_type=_at,
                    role=str(_role),
                    depth=_depth,
                )
                yield result

        finally:
            # Only cleanup here — no yielding to avoid RuntimeError
            # when async generator is closed via aclose() or GC.
            for task in tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    # -- internal -----------------------------------------------------------

    async def _fetch(self, q: asyncio.Queue) -> AsyncGenerator[ActionHistory, None]:
        """Drain the internal queue as an async generator.

        Blocks on ``q.get()`` and terminates when the sentinel ``_STOP``
        is dequeued.  FIFO ordering guarantees all items enqueued before
        the sentinel are yielded first.
        """
        while True:
            item = await q.get()
            if item is self._STOP:
                return
            yield item
