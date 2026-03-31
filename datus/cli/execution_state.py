# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Interaction broker for async user interaction flow control."""

import asyncio
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

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
    choices: List[Dict[str, str]]  # per-question choices; each dict is {shortcut: display}
    allow_free_text: bool = False  # When True, accept values outside choices
    created_at: datetime = field(default_factory=datetime.now)


class InteractionCancelled(Exception):
    """Raised when interaction is cancelled."""


class InteractionBroker:
    """
    Per-node broker for async user interactions.

    Provides:
    - request(): Async method for hooks to request user input (blocks until response),
                 returns (choice, callback) where callback generates SUCCESS action
    - fetch(): AsyncGenerator for node to consume interaction ActionHistory objects
    - submit(): For UI to submit responses
    - close(): Place a sentinel so fetch() terminates naturally

    All parameters use list format. ``action_type`` is auto-inferred:
    ``"request_choice"`` (single) or ``"request_batch"`` (multiple questions).

    Single question::

        choice, callback = await broker.request(
            contents=["Sync to Knowledge Base?"],
            choices=[{"y": "Accept - Sync to KB", "n": "Reject - Delete file"}],
            default_choices=["y"],
        )

    Batch questions::

        choice, callback = await broker.request(
            contents=["Which DB?", "Description?"],
            choices=[{"1": "MySQL", "2": "PostgreSQL"}, {}],
            default_choices=["1", ""],
            allow_free_text=True,
        )
    """

    _STOP_SENTINEL = object()

    def __init__(self):
        self._pending: Dict[str, PendingInteraction] = {}
        self._output_queue: asyncio.Queue[ActionHistory] = asyncio.Queue()
        # Use threading.Lock for thread-safe access to _pending
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False

    def reset_queue(self) -> None:
        """Recreate the asyncio.Queue bound to the current event loop.

        Must be called inside an async context (i.e. within asyncio.run())
        before each execution cycle. This ensures the queue is always bound
        to the active event loop, preventing 'bound to a different event loop'
        errors when a node is reused across separate asyncio.run() calls.
        """
        self._output_queue = asyncio.Queue()
        self._closed = False

    def close(self) -> None:
        """Place a sentinel so ``fetch()`` terminates naturally.

        Also cancels any pending interactions so callers blocked in
        ``request()`` are released with ``InteractionCancelled``.

        Idempotent – calling close() more than once is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        # Release callers blocked in request()
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
                    pass  # Loop already closed
        self._output_queue.put_nowait(self._STOP_SENTINEL)

    async def _queue_put(self, item: ActionHistory) -> None:
        """Put item into queue (non-blocking)."""
        if self._closed:
            logger.warning("InteractionBroker._queue_put() called after close()")
            return
        self._output_queue.put_nowait(item)

    async def _queue_get(self, timeout: float = 0.1) -> Optional[ActionHistory]:
        """Get item from queue with timeout, returns None if empty."""
        try:
            return await asyncio.wait_for(self._output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def request(
        self,
        contents: List[str],
        choices: List[Dict[str, str]],
        default_choices: Optional[List[str]] = None,
        content_type: str = "markdown",
        allow_free_text: bool = False,
        multi_selects: Optional[List[bool]] = None,
    ) -> Tuple[str, Callable[[str, str], Awaitable[None]]]:
        """
        Request user input with choices. Blocks until user responds.

        Args:
            contents: List of question strings. Single question: ``["Q?"]``.
            choices: List of choice dicts, one per question. ``{}`` means free text.
            default_choices: Default choice key per question. Defaults to ``[""]``.
            content_type: How to render the content (markdown, sql, yaml, text).
            allow_free_text: When True, accept values outside choices.
            multi_selects: Per-question flag for multi-select mode. Defaults to all False.

        Returns:
            Tuple of (choice, callback):
            - choice: The selected choice key (single) or JSON array of answers (batch)
            - callback: Async function to generate SUCCESS action with result content.

        Raises:
            InteractionCancelled: If broker is closed while waiting
        """
        # Fail fast if broker is already closed or contents is empty
        if self._closed:
            raise InteractionCancelled("Broker is already closed")
        if not contents:
            raise InteractionCancelled("No questions to ask (empty contents)")

        choices_list = choices
        if default_choices is None:
            default_choices = [""] * len(contents)
        while len(default_choices) < len(contents):
            default_choices.append("")

        action_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Create pending interaction
        pending = PendingInteraction(
            action_id=action_id,
            future=future,
            choices=choices_list,
            allow_free_text=allow_free_text,
        )

        with self._lock:
            self._pending[action_id] = pending

        # Build display content for messages field
        if len(contents) == 1:
            display_content = contents[0]
        else:
            lines = []
            for i, (q, ch) in enumerate(zip(contents, choices_list), 1):
                lines.append(f"**{i}. {q}**")
                if ch:
                    opts = " / ".join(ch.values())
                    lines.append(f"   Options: {opts}")
                else:
                    lines.append("   _(free text)_")
                lines.append("")
            display_content = "\n".join(lines)

        # Auto-infer action_type
        action_type = "request_batch" if len(contents) > 1 else "request_choice"

        if multi_selects is None:
            multi_selects = [False] * len(contents)

        input_data = {
            "contents": contents,
            "content_type": content_type,
            "choices": choices_list,
            "default_choices": default_choices,
            "allow_free_text": allow_free_text,
            "multi_selects": multi_selects,
        }

        action = ActionHistory(
            action_id=action_id,
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            action_type=action_type,
            messages=display_content,
            input=input_data,
            output=None,
        )

        await self._queue_put(action)
        logger.debug(f"InteractionBroker: request queued with action_id={action_id}")

        # Wait for user response
        try:
            result = await future
            logger.debug(f"InteractionBroker: received response for action_id={action_id}: {result}")

            # Create callback for generating SUCCESS action
            async def success_callback(
                callback_content: str,
                callback_content_type: str = "markdown",
            ) -> None:
                """Generate a SUCCESS interaction action with the given content."""
                success_action = ActionHistory(
                    action_id=action_id,
                    role=ActionRole.INTERACTION,
                    status=ActionStatus.SUCCESS,
                    action_type=action_type,
                    messages=callback_content,
                    input=input_data,
                    output={
                        "content": callback_content,
                        "content_type": callback_content_type,
                        "user_choice": result,
                    },
                )

                await self._queue_put(success_action)
                logger.debug(f"InteractionBroker: success callback queued for action_id={action_id}")

            return result, success_callback
        except asyncio.CancelledError:
            with self._lock:
                self._pending.pop(action_id, None)
            raise InteractionCancelled("Request cancelled")

    async def fetch(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Async generator that yields ActionHistory objects for interactions.

        Blocks on ``queue.get()`` and terminates when the sentinel
        ``_STOP_SENTINEL`` is dequeued.  FIFO ordering guarantees all
        items enqueued before the sentinel are yielded first.

        Yields:
            ActionHistory objects with INTERACTION role
        """
        while True:
            try:
                item = await self._output_queue.get()
                if item is self._STOP_SENTINEL:
                    return
                yield item
            except asyncio.CancelledError:
                break

    async def submit(self, action_id: str, user_choice: Optional[str]) -> bool:
        """
        Submit user response for a pending interaction.

        Args:
            action_id: The action_id from the INTERACTION ActionHistory
            user_choice: The user's response. For single-question with choices, must be
                a valid choice key. For batch questions, must be a JSON-encoded list
                of answer strings. ``None`` indicates a collector failure.

        Returns:
            True if submission was successful, False if action_id not found or invalid choice
        """

        with self._lock:
            if action_id not in self._pending:
                logger.warning(f"InteractionBroker: submit called with unknown action_id={action_id}")
                return False

            pending = self._pending.get(action_id)

            # Validate choice: only for single-question with concrete choices
            if (
                user_choice is not None
                and len(pending.choices) == 1
                and pending.choices[0]
                and not pending.allow_free_text
                and user_choice not in pending.choices[0]
            ):
                logger.warning(
                    f"InteractionBroker: invalid choice '{user_choice}', not in {list(pending.choices[0].keys())}"
                )
                return False

            self._pending.pop(action_id, None)

        # Resolve the future with the user's choice
        if not pending.future.done():
            pending.future.get_loop().call_soon_threadsafe(pending.future.set_result, user_choice)
            logger.debug(f"InteractionBroker: submitted response for action_id={action_id}")

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
    input_data = action.input or {}
    contents = input_data.get("contents", [])
    choices_list = input_data.get("choices", [])
    default_choices = input_data.get("default_choices", [])

    if len(contents) > 1:
        # Batch: auto-submit first option value or empty for each question
        answers = []
        for ch in choices_list:
            answers.append(next(iter(ch.keys())) if ch else "")
        await broker.submit(action.action_id, json.dumps(answers))
        logger.info(f"Auto-submitted batch answers: {len(answers)}")
    elif len(contents) == 1:
        ch = choices_list[0] if choices_list else {}
        default = default_choices[0] if default_choices else ""
        if ch and default:
            await broker.submit(action.action_id, default)
            logger.info(f"Auto-submitted default choice: {default}")
        elif not ch:
            await broker.submit(action.action_id, "")
            logger.info("Auto-submitted empty string for free-text input")
        elif ch:
            first_key = next(iter(ch))
            await broker.submit(action.action_id, first_key)
            logger.info(f"Auto-submitted first choice (no default): {first_key}")
    else:
        await broker.submit(action.action_id, "")
        logger.warning("Auto-submit: empty contents list, submitted empty string")


async def merge_interaction_stream(
    execute_stream: AsyncGenerator[ActionHistory, None],
    broker: InteractionBroker,
) -> AsyncGenerator[ActionHistory, None]:
    """
    Merge execute_stream output with interaction broker output.

    Delegates to ``ActionBus.merge()`` with ``on_primary_done=broker.close``
    so that all streams terminate naturally via sentinel.

    Args:
        execute_stream: The node's execute_stream() generator
        broker: The InteractionBroker instance for this node

    Yields:
        ActionHistory objects from both streams, interleaved
    """
    from datus.schemas.action_bus import ActionBus

    bus = ActionBus()
    async for action in bus.merge(execute_stream, broker.fetch(), on_primary_done=broker.close):
        yield action
