# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ActionBus – single-channel action stream merger.

Tools call ``bus.put(action)`` to inject sub-actions (e.g. explorer
sub-agent tool calls).  The node calls ``bus.merge(primary, *secondaries)``
to yield everything in one stream for the CLI / web UI.

Implementation is a classic fan-in: one shared ``asyncio.Queue`` plus one
pump task per stream, all feeding the same queue.  ``put()`` writes directly
into that queue, so injected actions interleave with primary/secondary
actions in FIFO (real arrival) order.  Downstream consumers re-group actions
by ``parent_action_id`` / ``depth``, so no cross-stream priority ordering is
needed.

Lifecycle follows the owning ``AgenticNode``.  ``put()`` must be called from
the event-loop thread that runs ``merge()`` (e.g. inside an ``async``
function); a mismatch is logged as a warning.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Callable, List, Optional, Tuple

from datus.schemas.action_history import ActionHistory
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ActionBus:
    """Single-channel action bus with N-stream fan-in merge.

    * **put(action)** – push for tool sub-actions (must be called from the
      event-loop thread that runs ``merge()``).
    * **merge(primary, \\*secondaries)** – async generator that yields actions
      from the primary stream, all secondary streams, *and* anything injected
      via ``put()``, interleaved in FIFO order through one shared queue.
    * **close()** – place a sentinel in the queue so the injection channel
      terminates.  Called automatically when the primary stream exhausts.
    """

    _STOP = object()  # injection-channel sentinel (enqueued by close())
    _DONE = object()  # per-stream sentinel (enqueued when a pump exhausts)

    def __init__(self) -> None:
        # Lazy init so that __init__ can run before an event loop exists.
        # The queue is (re-)created on the loop that runs merge(); ``_loop``
        # is tracked only to warn on cross-loop misuse of put().
        self._queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed: bool = False

    def reset(self) -> None:
        """Drop all pending items and reset state for a new execution.

        Must be called at the start of each top-level execution to prevent
        leftover queued items from a previous (possibly interrupted) run from
        being replayed.
        """
        self._queue = None
        self._loop = None
        self._closed = False

    @staticmethod
    def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def _ensure_queue(self) -> asyncio.Queue:
        """Return the shared queue, lazily creating it on the current loop."""
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._loop = self._running_loop()
        return self._queue

    # -- push side ----------------------------------------------------------

    def put(self, action: ActionHistory) -> None:
        """Inject an action (non-blocking, must run on the event-loop thread)."""
        if self._closed:
            logger.warning("ActionBus.put() called after close()")
            return
        current = self._running_loop()
        if self._loop is not None and current is not self._loop:
            # The shared queue's getter futures are bound to merge()'s loop;
            # waking them from another loop — or from a thread with no running
            # loop at all (current is None) — is unsafe.  Surface the misuse
            # instead of silently enqueueing onto a queue whose waiters merge()
            # never reads.  We intentionally do not reschedule cross-loop here:
            # by design every put() runs on merge()'s loop (see module docstring).
            logger.warning(
                "ActionBus.put() called from a different event loop than merge() "
                "(or from a thread with no running loop); action may not be "
                "delivered — put() must run on the event-loop thread"
            )
        self._ensure_queue().put_nowait(action)

    @property
    def has_pending(self) -> bool:
        return self._queue is not None and not self._queue.empty()

    def close(self) -> None:
        """Enqueue a sentinel so the injection channel terminates.

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
        """Merge *primary* + *secondaries* + injected actions into one stream.

        Each stream is drained by its own pump task into a single shared queue;
        the main loop yields whatever arrives, in FIFO order.  Terminates once
        every pump has exhausted and the injection channel is closed.

        Args:
            primary: The main action stream.
            *secondaries: Additional action streams (e.g. interaction broker).
            on_primary_done: Optional callback invoked exactly once when the
                primary stream exhausts (or errors).  Typically closes the
                secondary streams so they terminate naturally
                (e.g. ``on_primary_done=broker.close``).
        """
        # Reuse a queue holding actions put() before merge() started; otherwise
        # create one bound to the current loop.
        q = self._ensure_queue()
        self._loop = self._running_loop()
        self._closed = False

        streams: List[Tuple[str, AsyncGenerator[ActionHistory, None], bool]] = [("primary", primary, True)]
        for idx, sec in enumerate(secondaries):
            streams.append((f"secondary_{idx}", sec, False))

        # One _DONE per pump stream + one _STOP from close() (primary exhaust).
        expected = len(streams) + 1
        primary_error: Optional[BaseException] = None

        async def pump(name: str, gen: AsyncGenerator[ActionHistory, None], is_primary: bool) -> None:
            nonlocal primary_error
            cancelled = False
            try:
                async for action in gen:
                    q.put_nowait(action)
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception as exc:  # noqa: BLE001
                if is_primary:
                    primary_error = exc
                else:
                    logger.error("ActionBus secondary stream error: %s", name, exc_info=True)
            finally:
                # Skip primary-done side effects on cancellation (abnormal
                # teardown), but always emit the per-stream sentinel.
                if is_primary and not cancelled:
                    if on_primary_done is not None:
                        try:
                            on_primary_done()
                        except Exception:  # noqa: BLE001
                            logger.debug("ActionBus on_primary_done raised", exc_info=True)
                    self.close()  # enqueues _STOP (idempotent)
                q.put_nowait(self._DONE)

        tasks = [
            asyncio.create_task(pump(name, gen, is_primary), name=f"action_bus:{name}")
            for name, gen, is_primary in streams
        ]

        seen = 0
        try:
            while seen < expected:
                item = await q.get()
                if item is self._STOP:
                    # _STOP is enqueued only by the primary pump's completion
                    # (via close()).  If the primary errored, fail fast and
                    # raise immediately instead of waiting for secondaries that
                    # may never terminate — the finally block cancels them.
                    seen += 1
                    if primary_error is not None:
                        break
                    continue
                if item is self._DONE:
                    seen += 1
                    continue
                logger.debug(
                    "ActionBus.merge yield",
                    action_type=getattr(item, "action_type", "?"),
                    role=str(getattr(item, "role", "?")),
                    depth=getattr(item, "depth", "?"),
                )
                yield item
            if primary_error is not None:
                raise primary_error
        finally:
            # Cleanup only — no yielding here to avoid RuntimeError when the
            # async generator is closed via aclose() or GC.
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    logger.debug("ActionBus pump task error during cleanup", exc_info=True)
