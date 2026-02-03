# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Interaction broker for async user interaction flow control."""

import asyncio
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Awaitable, Callable, Dict, Optional, Tuple

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class PendingInteraction:
    """Pending interaction waiting for user response"""

    action_id: str
    future: asyncio.Future
    choices: Dict[str, str]  # key=submit value, value=display text
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

    def __init__(self):
        self._pending: Dict[str, PendingInteraction] = {}
        # Use thread-safe queue.Queue to share across different event loops
        self._output_queue: queue.Queue[ActionHistory] = queue.Queue()
        # Use threading.Lock for thread-safe access to _pending
        self._lock: threading.Lock = threading.Lock()

    async def _queue_put(self, item: ActionHistory) -> None:
        """Put item into queue (non-blocking, thread-safe)."""
        self._output_queue.put_nowait(item)

    async def _queue_get(self, timeout: float = 0.1) -> Optional[ActionHistory]:
        """Get item from queue with timeout, returns None if empty."""
        loop = asyncio.get_running_loop()
        try:
            # Run blocking get in executor to avoid blocking the event loop
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._output_queue.get, True, timeout),
                timeout=timeout + 0.1,
            )
        except (queue.Empty, asyncio.TimeoutError):
            return None

    async def request(
        self,
        content: str,
        choices: Dict[str, str],
        default_choice: str = "",
        content_type: str = "markdown",
    ) -> Tuple[str, Callable[[str, str], Awaitable[None]]]:
        """
        Request user input with choices. Blocks until user responds.

        Args:
            content: Display content/prompt for user (supports markdown)
            choices: Dict of {key: display_text}. Empty dict means free-text input.
            default_choice: Key of default choice (required when choices is non-empty)
            content_type: Type of content ("text", "yaml", "sql", "markdown")

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

        Used by the node to consume interaction actions and merge with
        execute_stream output.

        Yields:
            ActionHistory objects with INTERACTION role (request_choice and success types)
        """
        while True:
            try:
                action = await self._queue_get(timeout=0.1)
                if action is not None:
                    yield action
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

            # Validate choice: if choices is non-empty, user_choice must be a valid key
            if pending.choices and user_choice not in pending.choices:
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

    This allows the UI to receive both:
    1. Normal execution actions (TOOL, ASSISTANT, etc.)
    2. Interaction actions (INTERACTION role): request_choice and success

    Args:
        execute_stream: The node's execute_stream() generator
        broker: The InteractionBroker instance for this node

    Yields:
        ActionHistory objects from both streams, interleaved
    """
    execute_iter = execute_stream.__aiter__()
    fetch_iter = broker.fetch().__aiter__()

    execute_exhausted = False
    execute_task: Optional[asyncio.Task] = None
    fetch_task: Optional[asyncio.Task] = None

    _EXHAUSTED = object()  # Sentinel for exhausted iterator

    async def safe_anext(iterable, sentinel):
        """Safely get next item, return sentinel on exhaustion."""
        try:
            return await iterable.__anext__()
        except StopAsyncIteration:
            return sentinel

    try:
        while not execute_exhausted or broker.has_pending or not broker.is_queue_empty():
            tasks_to_wait = []

            # Handle execute task - process if done during yield, otherwise add to wait list
            if execute_task is not None:
                if execute_task.done():
                    # Task completed during yield, process it now
                    result = execute_task.result()
                    execute_task = None
                    if result is _EXHAUSTED:
                        execute_exhausted = True
                        logger.debug("merge_interaction_stream: execute_stream exhausted (during yield)")
                    else:
                        yield result
                else:
                    tasks_to_wait.append(execute_task)

            # Create new execute task if needed
            if not execute_exhausted and execute_task is None:
                execute_task = asyncio.create_task(safe_anext(execute_iter, _EXHAUSTED), name="execute")
                tasks_to_wait.append(execute_task)

            # Handle fetch task - process if done during yield, otherwise add to wait list
            if fetch_task is not None:
                if fetch_task.done():
                    # Task completed during yield, process it now
                    result = fetch_task.result()
                    fetch_task = None
                    if result is not _EXHAUSTED:
                        yield result
                else:
                    tasks_to_wait.append(fetch_task)

            # Create new fetch task if needed
            if fetch_task is None:
                fetch_task = asyncio.create_task(safe_anext(fetch_iter, _EXHAUSTED), name="fetch")
                tasks_to_wait.append(fetch_task)

            if not tasks_to_wait:
                # Both exhausted and no pending
                break

            # Wait for first completed task
            done, _ = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                result = task.result()

                if task.get_name() == "execute":
                    execute_task = None
                    if result is _EXHAUSTED:
                        execute_exhausted = True
                        logger.debug("merge_interaction_stream: execute_stream exhausted")
                    else:
                        yield result

                elif task.get_name() == "fetch":
                    fetch_task = None
                    if result is not _EXHAUSTED:
                        yield result

    finally:
        # Clean up any remaining tasks
        for task in [execute_task, fetch_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
