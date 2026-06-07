# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level tests for ActionBus – put / has_pending / close / merge (fan-in)."""

import asyncio
from typing import AsyncGenerator

import pytest

from datus.schemas.action_bus import ActionBus
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

# ── helpers ────────────────────────────────────────────────────────


def _action(action_type: str = "test", role: ActionRole = ActionRole.TOOL) -> ActionHistory:
    return ActionHistory.create_action(
        role=role,
        action_type=action_type,
        messages="",
        input_data={},
        status=ActionStatus.PROCESSING,
    )


async def _agen(*items: ActionHistory) -> AsyncGenerator[ActionHistory, None]:
    """Simple async generator that yields the given items."""
    for item in items:
        yield item


async def _collect(agen: AsyncGenerator[ActionHistory, None]) -> list:
    return [a async for a in agen]


# ── put / has_pending ──────────────────────────────────────────────


@pytest.mark.ci
class TestPutAndPending:
    def test_empty_initially(self):
        bus = ActionBus()
        assert not bus.has_pending

    def test_put_makes_pending(self):
        bus = ActionBus()
        bus.put(_action())
        assert bus.has_pending

    def test_put_multiple(self):
        bus = ActionBus()
        bus.put(_action("a"))
        bus.put(_action("b"))
        assert bus.has_pending


# ── merge: primary only ───────────────────────────────────────────


@pytest.mark.ci
class TestMergePrimaryOnly:
    @pytest.mark.asyncio
    async def test_yields_all_primary_actions(self):
        bus = ActionBus()
        a1, a2 = _action("a1"), _action("a2")
        results = await _collect(bus.merge(_agen(a1, a2)))
        assert results == [a1, a2]

    @pytest.mark.asyncio
    async def test_empty_primary(self):
        bus = ActionBus()
        results = await _collect(bus.merge(_agen()))
        assert results == []


# ── merge: bus queue items ────────────────────────────────────────


@pytest.mark.ci
class TestMergeBusQueue:
    @pytest.mark.asyncio
    async def test_bus_items_yielded(self):
        """Items put() before merge starts are yielded."""
        bus = ActionBus()
        bus_action = _action("from_bus")
        bus.put(bus_action)

        primary_action = _action("from_primary")
        results = await _collect(bus.merge(_agen(primary_action)))

        action_types = [r.action_type for r in results]
        assert "from_bus" in action_types
        assert "from_primary" in action_types

    @pytest.mark.asyncio
    async def test_bus_items_during_primary(self):
        """Items put() while primary is yielding are not lost."""
        bus = ActionBus()
        primary_a = _action("primary")
        bus_a = _action("bus_during")

        async def _primary_that_puts() -> AsyncGenerator[ActionHistory, None]:
            yield primary_a
            bus.put(bus_a)
            # Small delay so merge can pick up the bus item
            await asyncio.sleep(0.05)

        results = await _collect(bus.merge(_primary_that_puts()))
        action_types = [r.action_type for r in results]
        assert "primary" in action_types
        assert "bus_during" in action_types

    @pytest.mark.asyncio
    async def test_multiple_bus_items_not_lost(self):
        """Multiple rapid bus.put() calls don't lose items."""
        bus = ActionBus()
        expected_types = [f"bus_{i}" for i in range(5)]

        async def _primary() -> AsyncGenerator[ActionHistory, None]:
            yield _action("start")
            for t in expected_types:
                bus.put(_action(t))
            await asyncio.sleep(0.15)
            yield _action("end")

        results = await _collect(bus.merge(_primary()))
        result_types = [r.action_type for r in results]
        for t in expected_types:
            assert t in result_types, f"{t} was lost"

    @pytest.mark.asyncio
    async def test_all_roles_preserved(self):
        """Actions with role=USER are not filtered or lost."""
        bus = ActionBus()
        user_action = _action("user_task", role=ActionRole.USER)
        tool_action = _action("tool_call", role=ActionRole.TOOL)
        assistant_action = _action("response", role=ActionRole.ASSISTANT)

        async def _primary() -> AsyncGenerator[ActionHistory, None]:
            yield _action("trigger")
            bus.put(user_action)
            bus.put(tool_action)
            bus.put(assistant_action)
            await asyncio.sleep(0.15)

        results = await _collect(bus.merge(_primary()))
        result_types = [r.action_type for r in results]
        assert "user_task" in result_types
        assert "tool_call" in result_types
        assert "response" in result_types


# ── merge: secondaries ────────────────────────────────────────────


@pytest.mark.ci
class TestMergeSecondaries:
    @pytest.mark.asyncio
    async def test_secondary_stream_yielded(self):
        bus = ActionBus()
        p = _action("primary")
        s = _action("secondary")
        results = await _collect(bus.merge(_agen(p), _agen(s)))
        action_types = [r.action_type for r in results]
        assert "primary" in action_types
        assert "secondary" in action_types


# ── merge: termination ────────────────────────────────────────────


@pytest.mark.ci
class TestMergeTermination:
    @pytest.mark.asyncio
    async def test_terminates_after_primary_exhausted_and_queue_drained(self):
        """merge() terminates cleanly when primary exhausts and queue is empty."""
        bus = ActionBus()
        results = await _collect(bus.merge(_agen(_action("only"))))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_does_not_terminate_while_queue_has_items(self):
        """merge() waits for bus queue items even after primary exhausts."""
        bus = ActionBus()

        async def _primary() -> AsyncGenerator[ActionHistory, None]:
            yield _action("p1")
            # Put item after primary yields but before it exhausts
            bus.put(_action("late_bus"))
            await asyncio.sleep(0.05)

        results = await _collect(bus.merge(_primary()))
        action_types = [r.action_type for r in results]
        assert "late_bus" in action_types


# ── merge: ordering ───────────────────────────────────────────────


@pytest.mark.ci
class TestMergeOrdering:
    @pytest.mark.asyncio
    async def test_injected_actions_preserve_fifo_order(self):
        """Injected actions are yielded in the order they were put()."""
        bus = ActionBus()
        bus.put(_action("first"))
        bus.put(_action("second"))
        bus.put(_action("third"))

        results = await _collect(bus.merge(_agen(_action("primary"))))
        types = [r.action_type for r in results]
        # FIFO among injected actions (no cross-stream priority reordering)
        assert types.index("first") < types.index("second") < types.index("third")
        assert "primary" in types

    @pytest.mark.asyncio
    async def test_injected_before_primary_yielded_first(self):
        """Actions put() before merge starts arrive before later primary output."""
        bus = ActionBus()
        bus.put(_action("injected"))

        async def _primary() -> AsyncGenerator[ActionHistory, None]:
            # Yield only after a tick so the pre-queued item is drained first.
            await asyncio.sleep(0.01)
            yield _action("primary_late")

        results = await _collect(bus.merge(_primary()))
        types = [r.action_type for r in results]
        assert types.index("injected") < types.index("primary_late")


# ── close / sentinel ─────────────────────────────────────────────


@pytest.mark.ci
class TestClose:
    def test_close_sets_closed(self):
        bus = ActionBus()
        assert bus._closed is False
        bus.close()
        assert bus._closed is True

    def test_close_is_idempotent(self):
        """Calling close() twice does not enqueue a second sentinel."""
        bus = ActionBus()
        bus._ensure_queue()
        bus.close()
        bus.close()  # second call is no-op
        assert bus._closed is True
        # Only one sentinel should be in the queue
        assert bus._queue.qsize() == 1

    def test_put_after_close_is_ignored(self):
        """put() after close() logs a warning and drops the action."""
        bus = ActionBus()
        bus.close()
        bus.put(_action("ignored"))
        # The queue should only contain the sentinel, not the ignored action
        assert bus._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_put_after_primary_exhausts_is_ignored(self):
        """Once primary exhausts, the bus auto-closes and late put() is dropped."""
        bus = ActionBus()

        async def _primary() -> AsyncGenerator[ActionHistory, None]:
            yield _action("p")
            # primary exhausts here → merge() calls close()

        results = await _collect(bus.merge(_primary()))
        # After merge, the bus is closed; a late put() must be ignored.
        bus.put(_action("too_late"))
        types = [r.action_type for r in results]
        assert "p" in types
        assert "too_late" not in types
        # Validate the drop actually happened: the late put() left nothing queued.
        assert not bus.has_pending

    @pytest.mark.asyncio
    async def test_primary_error_propagates(self):
        """An exception raised by the primary stream surfaces to the consumer."""
        bus = ActionBus()

        async def _boom() -> AsyncGenerator[ActionHistory, None]:
            yield _action("before_error")
            raise ValueError("primary blew up")

        with pytest.raises(ValueError, match="primary blew up"):
            await _collect(bus.merge(_boom()))

    @pytest.mark.asyncio
    async def test_primary_error_fails_fast_with_hanging_secondary(self):
        """A primary error propagates even if a secondary never terminates."""
        bus = ActionBus()

        async def _boom() -> AsyncGenerator[ActionHistory, None]:
            yield _action("before_error")
            raise ValueError("primary blew up")

        async def _never_ending() -> AsyncGenerator[ActionHistory, None]:
            # Never produces and never returns until cancelled by cleanup.
            await asyncio.Event().wait()
            yield _action("never")  # pragma: no cover

        # Without fail-fast this would hang forever waiting on the secondary;
        # the primary error must surface promptly instead.
        with pytest.raises(ValueError, match="primary blew up"):
            await _collect(bus.merge(_boom(), _never_ending()))


# ── merge: on_primary_done callback ──────────────────────────────


@pytest.mark.ci
class TestOnPrimaryDone:
    @pytest.mark.asyncio
    async def test_on_primary_done_called(self):
        """on_primary_done callback is invoked when primary exhausts."""
        bus = ActionBus()
        called = []

        def cb():
            called.append(True)

        results = await _collect(bus.merge(_agen(_action("p")), on_primary_done=cb))
        assert len(results) == 1
        assert called == [True]

    @pytest.mark.asyncio
    async def test_on_primary_done_not_called_if_none(self):
        """merge() works fine when on_primary_done is None (default)."""
        bus = ActionBus()
        results = await _collect(bus.merge(_agen(_action("p"))))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_merge_secondary_closed_via_on_primary_done(self):
        """Secondary stream terminates when on_primary_done closes it."""
        bus = ActionBus()
        secondary_q: asyncio.Queue = asyncio.Queue()

        async def secondary_gen() -> AsyncGenerator[ActionHistory, None]:
            while True:
                item = await secondary_q.get()
                if item is None:
                    return
                yield item

        def close_secondary():
            secondary_q.put_nowait(None)

        # Pre-queue a secondary action
        secondary_q.put_nowait(_action("sec_1"))

        results = await _collect(
            bus.merge(
                _agen(_action("primary")),
                secondary_gen(),
                on_primary_done=close_secondary,
            )
        )

        action_types = [r.action_type for r in results]
        assert "primary" in action_types
        assert "sec_1" in action_types


# ── put: cross-loop misuse ───────────────────────────────────────


@pytest.mark.ci
class TestPutCrossLoop:
    @pytest.mark.asyncio
    async def test_put_from_different_loop_warns(self):
        """put() from a loop other than merge()'s logs a warning but still enqueues."""
        bus = ActionBus()
        bus._ensure_queue()  # binds the queue/_loop to the current (test) loop
        other_loop = asyncio.new_event_loop()
        try:
            # Simulate merge() having bound to a *different* loop than the
            # caller of put() is currently running on.
            bus._loop = other_loop
            bus.put(_action("cross_loop"))
            # The action is still enqueued (warning only, no drop).
            assert bus.has_pending
        finally:
            other_loop.close()


# ── merge: secondary errors & callback failures ──────────────────


@pytest.mark.ci
class TestMergeErrorHandling:
    @pytest.mark.asyncio
    async def test_secondary_stream_error_is_swallowed(self):
        """A secondary stream raising does not abort merge; primary still finishes."""
        bus = ActionBus()

        async def bad_secondary() -> AsyncGenerator[ActionHistory, None]:
            yield _action("sec_ok")
            raise RuntimeError("secondary boom")

        results = await _collect(bus.merge(_agen(_action("p")), bad_secondary()))
        types = [r.action_type for r in results]
        assert "p" in types
        assert "sec_ok" in types

    @pytest.mark.asyncio
    async def test_on_primary_done_exception_is_swallowed(self):
        """An exception from on_primary_done does not break merge()."""
        bus = ActionBus()

        def boom():
            raise RuntimeError("callback boom")

        results = await _collect(bus.merge(_agen(_action("p")), on_primary_done=boom))
        assert len(results) == 1
        assert results[0].action_type == "p"


# ── merge: cleanup cancels running pumps ─────────────────────────


@pytest.mark.ci
class TestMergeCleanup:
    @pytest.mark.asyncio
    async def test_aclose_cancels_running_pump_tasks(self):
        """Closing the merge generator early cancels still-running pump tasks."""
        bus = ActionBus()

        async def primary() -> AsyncGenerator[ActionHistory, None]:
            yield _action("p")
            # Stay alive so the primary pump is still running at aclose() time.
            await asyncio.sleep(5)

        async def blocking_secondary() -> AsyncGenerator[ActionHistory, None]:
            # Never produces and never returns until cancelled.
            await asyncio.Event().wait()
            yield _action("never")  # pragma: no cover

        gen = bus.merge(primary(), blocking_secondary())
        first = await gen.__anext__()
        assert first.action_type == "p"

        # aclose() runs the finally block, which cancels the pending pump tasks
        # (both primary's sleep and the blocking secondary) and awaits them.
        await gen.aclose()
