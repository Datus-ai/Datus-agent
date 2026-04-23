# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Interaction broker for async user interaction flow control."""

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.schemas.interaction_event import InteractionEvent

logger = get_logger(__name__)


class ExecutionInterrupted(Exception):
    """Raised when the user interrupts the current execution."""

    pass


class InterruptController:
    """Thread-safe interrupt controller for graceful execution cancellation."""

    def __init__(self):
        self._interrupted = threading.Event()

    def interrupt(self):
        """Signal that execution should be interrupted."""
        self._interrupted.set()

    @property
    def is_interrupted(self) -> bool:
        """Check if interrupt has been signaled."""
        return self._interrupted.is_set()

    def check(self):
        """Raise ExecutionInterrupted if interrupted."""
        if self._interrupted.is_set():
            raise ExecutionInterrupted("Execution interrupted by user")

    def reset(self):
        """Clear the interrupt signal for a new execution cycle."""
        self._interrupted.clear()


@dataclass
class PendingInteraction:
    """Pending interaction waiting for user response"""

    action_id: str
    future: asyncio.Future
    choices: List[Dict[str, str]]
    allow_free_text: bool = False
    action_type: str = "request_choice"
    input_data: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.now)


class InteractionCancelled(Exception):
    """Raised when interaction is cancelled."""


class InteractionBroker:
    """Per-node broker for async user interactions.

    All answers use a unified ``List[List[str]]`` format:

    - Single question, single-select:  ``[["y"]]``
    - Single question, multi-select:   ``[["1", "3"]]``
    - Multiple questions:              ``[["ans1"], ["ans2"]]``
    - ESC / cancel:                    ``[[""]]``

    Single question::

        answers = await broker.request([
            InteractionEvent(title="Permission", content="Allow?",
                             choices={"y": "Yes", "n": "No"}, default_choice="n"),
        ])
        # answers == [["y"]]

    Batch questions::

        answers = await broker.request([
            InteractionEvent(title="DB", content="Which DB?",
                             choices={"1": "MySQL", "2": "PG"}),
            InteractionEvent(title="Desc", content="Description?",
                             allow_free_text=True),
        ])
        # answers == [["1"], ["some text"]]
    """

    _STOP_SENTINEL = object()

    def __init__(self):
        self._pending: Dict[str, PendingInteraction] = {}
        self._output_queue: asyncio.Queue[ActionHistory] = asyncio.Queue()
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False

    def reset_queue(self) -> None:
        """Recreate the asyncio.Queue bound to the current event loop."""
        self._output_queue = asyncio.Queue()
        self._closed = False

    def close(self) -> None:
        """Place a sentinel so ``fetch()`` terminates naturally.

        Also cancels any pending interactions so callers blocked in
        ``request()`` are released with ``InteractionCancelled``.
        """
        if self._closed:
            return
        self._closed = True
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for interaction in pending:
            if not interaction.future.done():
                try:
                    loop = interaction.future.get_loop()
                    loop.call_soon_threadsafe(
                        interaction.future.set_exception,
                        InteractionCancelled("Broker closed"),
                    )
                except RuntimeError:
                    pass
        self._output_queue.put_nowait(self._STOP_SENTINEL)

    async def request(self, events: List["InteractionEvent"]) -> List[List[str]]:
        """Request user input. Blocks until user responds.

        Args:
            events: One or more ``InteractionEvent`` objects.

        Returns:
            ``List[List[str]]`` — one inner list per event.
            Single-select answers have one element; multi-select may have more.

        Raises:
            InteractionCancelled: If the broker is closed while waiting.
        """
        if self._closed:
            raise InteractionCancelled("Broker is already closed")
        if not events:
            raise InteractionCancelled("No questions to ask (empty events)")

        action_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        action_type = "request_batch" if len(events) > 1 else "request_choice"
        input_data = {"events": [ev.model_dump() for ev in events]}

        pending = PendingInteraction(
            action_id=action_id,
            future=future,
            choices=[ev.choices for ev in events],
            allow_free_text=any(ev.allow_free_text for ev in events),
            action_type=action_type,
            input_data=input_data,
        )

        with self._lock:
            self._pending[action_id] = pending

        if len(events) == 1:
            display_content = events[0].content
        else:
            lines = []
            for i, ev in enumerate(events, 1):
                lines.append(f"**{i}. {ev.content}**")
                if ev.choices:
                    opts = " / ".join(ev.choices.values())
                    lines.append(f"   Options: {opts}")
                else:
                    lines.append("   _(free text)_")
                lines.append("")
            display_content = "\n".join(lines)

        action = ActionHistory(
            action_id=action_id,
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            action_type=action_type,
            messages=display_content,
            input=input_data,
            output=None,
        )

        if not self._closed:
            self._output_queue.put_nowait(action)
        logger.debug(f"InteractionBroker: request queued with action_id={action_id}")

        try:
            result = await future
            logger.debug(f"InteractionBroker: received response for action_id={action_id}: {result}")
            return result
        except asyncio.CancelledError:
            with self._lock:
                self._pending.pop(action_id, None)
            raise InteractionCancelled("Request cancelled")

    async def fetch(self) -> AsyncGenerator[ActionHistory, None]:
        """Async generator that yields ActionHistory objects for interactions."""
        while True:
            try:
                item = await self._output_queue.get()
                if item is self._STOP_SENTINEL:
                    return
                yield item
            except asyncio.CancelledError:
                break

    async def submit(self, action_id: str, answers: List[List[str]]) -> bool:
        """Submit user response for a pending interaction.

        After resolving the pending future, a SUCCESS ``ActionHistory`` is
        automatically enqueued.

        Args:
            action_id: The action_id from the INTERACTION ActionHistory.
            answers: ``List[List[str]]`` — one inner list per question.
                Single-select: ``[["y"]]``.  Multi-select: ``[["1","3"]]``.
                Batch: ``[["a"], ["b"]]``.

        Returns:
            True if submission was successful, False otherwise.
        """
        if not isinstance(answers, list) or not all(
            isinstance(inner, list) and all(isinstance(v, str) for v in inner) for inner in answers
        ):
            logger.warning(
                f"InteractionBroker: submit requires List[List[str]], got {type(answers).__name__}={answers!r}"
            )
            return False

        with self._lock:
            if action_id not in self._pending:
                logger.warning(f"InteractionBroker: submit called with unknown action_id={action_id}")
                return False

            pending = self._pending.get(action_id)

            # Validate: for each question with concrete choices (non free-text),
            # every answer value must be a valid choice key.
            for i, answer_list in enumerate(answers):
                if i >= len(pending.choices):
                    break
                ch = pending.choices[i]
                if not ch or pending.allow_free_text:
                    continue
                for val in answer_list:
                    if val and val not in ch:
                        logger.warning(
                            f"InteractionBroker: invalid choice '{val}' for question {i}, not in {list(ch.keys())}"
                        )
                        return False

            self._pending.pop(action_id, None)

        if not pending.future.done():
            pending.future.get_loop().call_soon_threadsafe(pending.future.set_result, answers)
            logger.debug(f"InteractionBroker: submitted response for action_id={action_id}")

        if not self._closed:
            success_action = ActionHistory(
                action_id=action_id,
                role=ActionRole.INTERACTION,
                status=ActionStatus.SUCCESS,
                action_type=pending.action_type,
                messages="",
                input=pending.input_data,
                output={"user_choice": answers},
            )
            self._output_queue.put_nowait(success_action)

        return True

    @property
    def has_pending(self) -> bool:
        """Check if there are pending interactions waiting for response."""
        return len(self._pending) > 0

    def is_queue_empty(self) -> bool:
        """Check if the output queue is empty."""
        return self._output_queue.empty()


async def auto_submit_interaction(broker: InteractionBroker, action: ActionHistory) -> None:
    """Auto-submit default choice for a PROCESSING interaction action.

    Used by non-interactive CLI mode and Web executor to automatically
    resolve pending interactions without user input.
    """
    from datus.schemas.interaction_event import InteractionEvent

    events = InteractionEvent.from_broker_input(action.input or {})
    if not events:
        await broker.submit(action.action_id, [[""]])
        logger.warning("Auto-submit: empty events list, submitted empty string")
        return

    answers: List[List[str]] = []
    for ev in events:
        if ev.choices and ev.default_choice:
            answers.append([ev.default_choice])
        elif ev.choices:
            answers.append([next(iter(ev.choices.keys()))])
        else:
            answers.append([""])
    await broker.submit(action.action_id, answers)
    logger.info(f"Auto-submitted {len(answers)} answer(s)")


async def merge_interaction_stream(
    execute_stream: AsyncGenerator[ActionHistory, None],
    broker: InteractionBroker,
) -> AsyncGenerator[ActionHistory, None]:
    """Merge execute_stream output with interaction broker output."""
    from datus.schemas.action_bus import ActionBus

    bus = ActionBus()
    async for action in bus.merge(execute_stream, broker.fetch(), on_primary_done=broker.close):
        yield action
