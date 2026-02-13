# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/execution_state.py.

Tests cover:
- PendingInteraction dataclass
- InteractionCancelled exception
- InteractionBroker: init, request, submit, fetch, has_pending, is_queue_empty
- merge_interaction_stream: merging execute_stream and broker output

NO MOCK EXCEPT LLM. All classes under test are real implementations.
"""

import asyncio

import pytest

from datus.cli.execution_state import (
    InteractionBroker,
    InteractionCancelled,
    PendingInteraction,
    merge_interaction_stream,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

# ===========================================================================
# PendingInteraction Tests
# ===========================================================================


class TestPendingInteractionInit:
    """Tests for PendingInteraction dataclass creation."""

    def test_pending_interaction_creation(self):
        """PendingInteraction stores action_id, future, and choices correctly."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        choices = {"y": "Yes", "n": "No"}

        pending = PendingInteraction(action_id="test-id", future=future, choices=choices)

        assert pending.action_id == "test-id"
        assert pending.future is future
        assert pending.choices == {"y": "Yes", "n": "No"}
        assert pending.created_at is not None
        loop.close()

    def test_pending_interaction_created_at_auto_set(self):
        """PendingInteraction auto-sets created_at to current datetime."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        pending = PendingInteraction(action_id="test-id-2", future=future, choices={})

        assert pending.created_at is not None
        # created_at should be a datetime object
        from datetime import datetime

        assert isinstance(pending.created_at, datetime)
        loop.close()


# ===========================================================================
# InteractionCancelled Tests
# ===========================================================================


class TestInteractionCancelled:
    """Tests for InteractionCancelled exception."""

    def test_exception_is_exception_subclass(self):
        """InteractionCancelled is a subclass of Exception."""
        assert issubclass(InteractionCancelled, Exception)

    def test_exception_message(self):
        """InteractionCancelled stores and returns the error message."""
        exc = InteractionCancelled("test cancellation")
        assert str(exc) == "test cancellation"

    def test_exception_can_be_raised_and_caught(self):
        """InteractionCancelled can be raised and caught specifically."""
        with pytest.raises(InteractionCancelled, match="cancelled"):
            raise InteractionCancelled("cancelled")


# ===========================================================================
# InteractionBroker Tests
# ===========================================================================


class TestInteractionBrokerInit:
    """Tests for InteractionBroker initialization."""

    def test_broker_init_empty_pending(self):
        """Newly created broker has no pending interactions."""
        broker = InteractionBroker()
        assert broker.has_pending is False

    def test_broker_init_empty_queue(self):
        """Newly created broker has an empty output queue."""
        broker = InteractionBroker()
        assert broker.is_queue_empty() is True


class TestInteractionBrokerRequest:
    """Tests for InteractionBroker.request() method."""

    @pytest.mark.asyncio
    async def test_request_creates_pending_and_queues_action(self):
        """request() adds a pending interaction and queues an ActionHistory for the UI."""
        broker = InteractionBroker()

        # Start the request in background; it will block until submit
        async def do_request():
            return await broker.request(
                content="Pick one",
                choices={"a": "Option A", "b": "Option B"},
                default_choice="a",
            )

        task = asyncio.create_task(do_request())

        # Give the request coroutine time to queue the action
        await asyncio.sleep(0.05)

        assert broker.has_pending is True
        assert broker.is_queue_empty() is False

        # Fetch the queued action
        action = broker._output_queue.get_nowait()
        assert action.role == ActionRole.INTERACTION
        assert action.status == ActionStatus.PROCESSING
        assert action.action_type == "request_choice"
        assert action.input["content"] == "Pick one"
        assert action.input["choices"] == {"a": "Option A", "b": "Option B"}
        assert action.input["default_choice"] == "a"

        # Submit response so the task completes
        action_id = action.action_id
        await broker.submit(action_id, "a")
        result, callback = await task

        assert result == "a"
        assert callable(callback)

    @pytest.mark.asyncio
    async def test_request_returns_callback_that_queues_success_action(self):
        """The callback returned by request() queues a SUCCESS ActionHistory."""
        broker = InteractionBroker()

        async def do_request():
            return await broker.request(
                content="Confirm?",
                choices={"y": "Yes", "n": "No"},
                default_choice="y",
                content_type="text",
            )

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.05)

        # Get the queued action to find the action_id
        action = broker._output_queue.get_nowait()
        action_id = action.action_id

        # Submit and get callback
        await broker.submit(action_id, "y")
        choice, callback = await task

        assert choice == "y"

        # Call callback and verify it queues a SUCCESS action
        await callback("Done!", "text")

        success_action = broker._output_queue.get_nowait()
        assert success_action.role == ActionRole.INTERACTION
        assert success_action.status == ActionStatus.SUCCESS
        assert success_action.action_id == action_id
        assert success_action.output["content"] == "Done!"
        assert success_action.output["content_type"] == "text"
        assert success_action.output["user_choice"] == "y"

    @pytest.mark.asyncio
    async def test_request_cancelled_raises_interaction_cancelled(self):
        """When the future is cancelled, request() raises InteractionCancelled."""
        broker = InteractionBroker()

        async def do_request():
            return await broker.request(
                content="Pick one",
                choices={"a": "A"},
                default_choice="a",
            )

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.05)

        # Cancel the task (which cancels the future inside request())
        task.cancel()

        with pytest.raises((InteractionCancelled, asyncio.CancelledError)):
            await task


class TestInteractionBrokerSubmit:
    """Tests for InteractionBroker.submit() method."""

    @pytest.mark.asyncio
    async def test_submit_unknown_action_id_returns_false(self):
        """submit() returns False when action_id is not found."""
        broker = InteractionBroker()
        result = await broker.submit("nonexistent-id", "choice")
        assert result is False

    @pytest.mark.asyncio
    async def test_submit_invalid_choice_returns_false(self):
        """submit() returns False when user_choice is not in the valid choices."""
        broker = InteractionBroker()

        async def do_request():
            return await broker.request(
                content="Pick",
                choices={"a": "A", "b": "B"},
                default_choice="a",
            )

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.05)

        action = broker._output_queue.get_nowait()
        action_id = action.action_id

        # Submit an invalid choice
        result = await broker.submit(action_id, "z")
        assert result is False

        # The pending should still be there
        assert broker.has_pending is True

        # Clean up: submit valid choice to unblock the task
        await broker.submit(action_id, "a")
        await task

    @pytest.mark.asyncio
    async def test_submit_valid_choice_returns_true(self):
        """submit() returns True when the choice is valid."""
        broker = InteractionBroker()

        async def do_request():
            return await broker.request(
                content="Pick",
                choices={"x": "X"},
                default_choice="x",
            )

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.05)

        action = broker._output_queue.get_nowait()
        action_id = action.action_id

        result = await broker.submit(action_id, "x")
        assert result is True
        assert broker.has_pending is False

        choice, _ = await task
        assert choice == "x"

    @pytest.mark.asyncio
    async def test_submit_empty_choices_accepts_any_text(self):
        """When choices is empty dict, any free-text input is accepted."""
        broker = InteractionBroker()

        async def do_request():
            return await broker.request(
                content="Enter text",
                choices={},
                default_choice="",
            )

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.05)

        action = broker._output_queue.get_nowait()
        action_id = action.action_id

        result = await broker.submit(action_id, "free text input")
        assert result is True

        choice, _ = await task
        assert choice == "free text input"


class TestInteractionBrokerFetch:
    """Tests for InteractionBroker.fetch() async generator."""

    @pytest.mark.asyncio
    async def test_fetch_yields_queued_actions(self):
        """fetch() yields ActionHistory objects that were queued."""
        broker = InteractionBroker()

        # Manually put an action in the queue
        action = ActionHistory(
            action_id="test-action",
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            action_type="request_choice",
            messages="test",
            input={},
            output=None,
        )
        broker._output_queue.put_nowait(action)

        # Fetch should yield the action
        fetched_actions = []
        fetch_gen = broker.fetch()

        async def collect():
            async for a in fetch_gen:
                fetched_actions.append(a)
                break  # Only collect one

        await asyncio.wait_for(collect(), timeout=1.0)
        assert len(fetched_actions) == 1
        assert fetched_actions[0].action_id == "test-action"


class TestInteractionBrokerEdgeCases:
    """Edge case tests for InteractionBroker."""

    def test_has_pending_reflects_state(self):
        """has_pending property accurately reflects pending dict state."""
        broker = InteractionBroker()
        assert broker.has_pending is False

    def test_is_queue_empty_reflects_state(self):
        """is_queue_empty() reflects the queue state correctly."""
        broker = InteractionBroker()
        assert broker.is_queue_empty() is True

        action = ActionHistory(
            action_id="q-test",
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            action_type="test",
            messages="test",
            input={},
            output=None,
        )
        broker._output_queue.put_nowait(action)
        assert broker.is_queue_empty() is False

    @pytest.mark.asyncio
    async def test_queue_get_returns_none_on_empty(self):
        """_queue_get returns None when queue is empty after timeout."""
        broker = InteractionBroker()
        result = await broker._queue_get(timeout=0.05)
        assert result is None


# ===========================================================================
# merge_interaction_stream Tests
# ===========================================================================


class TestMergeInteractionStream:
    """Tests for merge_interaction_stream function."""

    @pytest.mark.asyncio
    async def test_merge_yields_execute_stream_actions(self):
        """merge_interaction_stream yields actions from execute_stream."""
        broker = InteractionBroker()

        async def execute_stream():
            yield ActionHistory(
                action_id="exec-1",
                role=ActionRole.ASSISTANT,
                status=ActionStatus.SUCCESS,
                action_type="response",
                messages="Hello",
                input={},
                output={"raw_output": "Hello"},
            )

        actions = []
        async for action in merge_interaction_stream(execute_stream(), broker):
            actions.append(action)

        assert len(actions) == 1
        assert actions[0].action_id == "exec-1"
        assert actions[0].role == ActionRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_merge_yields_interaction_actions(self):
        """merge_interaction_stream yields actions from broker queue."""
        broker = InteractionBroker()

        # Queue an interaction action before starting the merge
        interaction_action = ActionHistory(
            action_id="interact-1",
            role=ActionRole.INTERACTION,
            status=ActionStatus.SUCCESS,
            action_type="request_choice",
            messages="Result",
            input={},
            output={"content": "Done"},
        )
        broker._output_queue.put_nowait(interaction_action)

        async def execute_stream():
            yield ActionHistory(
                action_id="exec-1",
                role=ActionRole.ASSISTANT,
                status=ActionStatus.SUCCESS,
                action_type="response",
                messages="Hi",
                input={},
                output={"raw_output": "Hi"},
            )

        actions = []
        async for action in merge_interaction_stream(execute_stream(), broker):
            actions.append(action)

        action_ids = {a.action_id for a in actions}
        assert "exec-1" in action_ids
        assert "interact-1" in action_ids

    @pytest.mark.asyncio
    async def test_merge_empty_execute_stream(self):
        """merge_interaction_stream handles empty execute_stream correctly."""
        broker = InteractionBroker()

        async def execute_stream():
            if False:
                yield  # Make it an async generator

        actions = []
        async for action in merge_interaction_stream(execute_stream(), broker):
            actions.append(action)

        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_merge_multiple_execute_actions(self):
        """merge_interaction_stream yields multiple actions from execute_stream in order."""
        broker = InteractionBroker()

        async def execute_stream():
            for i in range(3):
                yield ActionHistory(
                    action_id=f"exec-{i}",
                    role=ActionRole.TOOL,
                    status=ActionStatus.SUCCESS,
                    action_type="tool_call",
                    messages=f"Tool {i}",
                    input={},
                    output={"raw_output": f"result-{i}"},
                )

        actions = []
        async for action in merge_interaction_stream(execute_stream(), broker):
            actions.append(action)

        assert len(actions) == 3
        exec_ids = [a.action_id for a in actions if a.action_id.startswith("exec-")]
        assert len(exec_ids) == 3

    @pytest.mark.asyncio
    async def test_merge_interleaves_execute_and_interaction(self):
        """merge_interaction_stream interleaves execute_stream and broker actions."""
        broker = InteractionBroker()

        async def execute_stream():
            # First yield an action
            yield ActionHistory(
                action_id="exec-0",
                role=ActionRole.ASSISTANT,
                status=ActionStatus.SUCCESS,
                action_type="response",
                messages="step 0",
                input={},
                output={"raw_output": "step 0"},
            )
            # Simulate a delay where broker action gets queued
            await asyncio.sleep(0.05)
            yield ActionHistory(
                action_id="exec-1",
                role=ActionRole.ASSISTANT,
                status=ActionStatus.SUCCESS,
                action_type="response",
                messages="step 1",
                input={},
                output={"raw_output": "step 1"},
            )

        # Queue a broker action that should be picked up during the stream
        broker._output_queue.put_nowait(
            ActionHistory(
                action_id="broker-0",
                role=ActionRole.INTERACTION,
                status=ActionStatus.SUCCESS,
                action_type="request_choice",
                messages="interaction",
                input={},
                output={"content": "done"},
            )
        )

        actions = []
        async for action in merge_interaction_stream(execute_stream(), broker):
            actions.append(action)

        action_ids = [a.action_id for a in actions]
        assert "exec-0" in action_ids
        assert "exec-1" in action_ids
        assert "broker-0" in action_ids
        assert len(actions) == 3
