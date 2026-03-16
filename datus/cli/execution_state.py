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
from typing import AsyncGenerator, Awaitable, Callable, Dict, Optional, Tuple

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
    choices: Dict[str, str]  # key=submit value, value=display text
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

    Usage in hooks:
        choice, callback = await broker.request(
            content="## Generated YAML\\n```yaml\\n...\\n```\\n\\nSync to Knowledge Base?",
            choices={"y": "Yes - Save to KB", "n": "No - Keep file only"},
            default_choice="y",
            content_type="markdown",
        )
        if choice == "y":
            await sync_to_storage(...)
            await callback("**Successfully synced to Knowledge Base**")
        else:
            await callback("File saved locally only")

    Usage in node (merging with execute_stream):
        async for action in merge_interaction_stream(node.execute_stream(), broker):
            yield action

    Usage in UI:
        # CLI - distinguish by status (PROCESSING = waiting for input, SUCCESS = show result)
        for action in merged_stream:
            if action.role == ActionRole.INTERACTION and action.action_type == "request_choice":
                if action.status == ActionStatus.PROCESSING:
                    choice = display_and_get_user_choice(action)
                    broker.submit(action.action_id, choice)
                elif action.status == ActionStatus.SUCCESS:
                    display_success_content(action)
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
        content: str,
        choices: Dict[str, str],
        default_choice: str = "",
        content_type: str = "markdown",
        allow_free_text: bool = False,
    ) -> Tuple[str, Callable[[str, str], Awaitable[None]]]:
        """
        Request user input with choices. Blocks until user responds.

        Args:
            content: Display content/prompt for user (supports markdown)
            choices: Dict of {key: display_text}. Empty dict means free-text input.
            default_choice: Key of default choice (required when choices is non-empty)
            content_type: Type of content ("text", "yaml", "sql", "markdown")
            allow_free_text: When True, accept values outside choices (e.g. "Other" input).

        Returns:
            Tuple of (choice, callback):
            - choice: The selected choice key (or free text if choices is empty)
            - callback: Async function to generate SUCCESS action with result content.
                        Signature: async def callback(content: str, content_type: str = "markdown") -> None

        Raises:
            InteractionCancelled: If broker is closed while waiting
        """
        action_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Create pending interaction
        pending = PendingInteraction(
            action_id=action_id,
            future=future,
            choices=choices,
            allow_free_text=allow_free_text,
        )

        with self._lock:
            self._pending[action_id] = pending

        # Create ActionHistory with INTERACTION role
        action = ActionHistory(
            action_id=action_id,
            role=ActionRole.INTERACTION,
            status=ActionStatus.PROCESSING,
            action_type="request_choice",
            messages=content,
            input={
                "content": content,
                "content_type": content_type,
                "choices": choices,
                "default_choice": default_choice,
                "allow_free_text": allow_free_text,
            },
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

                # Use same action_id and action_type, but status=SUCCESS to indicate completion
                success_action = ActionHistory(
                    action_id=action_id,  # Same action_id to link with the original request
                    role=ActionRole.INTERACTION,
                    status=ActionStatus.SUCCESS,  # SUCCESS indicates completion
                    action_type="request_choice",  # Same action_type, UI distinguishes by status
                    messages=callback_content,
                    input={
                        "content": content,  # Original request content
                        "content_type": content_type,
                        "choices": choices,
                        "default_choice": default_choice,
                    },
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
            ActionHistory objects with INTERACTION role (request_choice and success types)
        """
        while True:
            try:
                item = await self._output_queue.get()
                if item is self._STOP_SENTINEL:
                    return
                yield item
            except asyncio.CancelledError:
                break

    async def submit(self, action_id: str, user_choice: str) -> bool:
        """
        Submit user response for a pending interaction.

        Args:
            action_id: The action_id from the INTERACTION ActionHistory
            user_choice: The user's selected choice key (must be in choices keys if choices is non-empty)

        Returns:
            True if submission was successful, False if action_id not found or invalid choice
        """

        with self._lock:
            if action_id not in self._pending:
                logger.warning(f"InteractionBroker: submit called with unknown action_id={action_id}")
                return False

            pending = self._pending.get(action_id)

            # Validate choice: if choices is non-empty and free text is not allowed,
            # user_choice must be a valid key
            if pending.choices and not pending.allow_free_text and user_choice not in pending.choices:
                logger.warning(
                    f"InteractionBroker: invalid choice '{user_choice}', not in {list(pending.choices.keys())}"
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
