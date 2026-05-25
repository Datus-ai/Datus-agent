# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``RunHooks`` adapter that drives compact passes from inside a Runner loop.

The OpenAI Agents SDK invokes ``on_tool_end`` after every successful tool
call. We hook in there to:

1. Increment the node's per-session tool-call counter (used as the rolling
   window precondition).
2. Ask the node whether to compact next, via ``_decide_compact_mode``.
3. Dispatch:

   * ``major`` — run synchronously **before** returning control to the SDK.
     Without it the next turn would already be over the model's context limit
     and the run would crash with ``MaxTurnsExceeded`` / token-limit errors.
   * ``minor`` — schedule asynchronously via ``asyncio.create_task`` so the
     Runner loop doesn't stall on disk I/O. The compact pass acquires the
     node-level ``asyncio.Lock`` so concurrent triggers serialize naturally.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agents import RunHooks

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)


class CompactHook(RunHooks):
    """Forwards ``on_tool_end`` to the owning node's compact dispatcher."""

    def __init__(self, node: "AgenticNode") -> None:
        self._node = node

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: str) -> None:  # noqa: D401
        node = self._node
        # The minor-compact eligibility window is now derived from the
        # session items themselves (number of ``role == "user"`` messages),
        # so the hook does not need to mutate any per-tool counter. It just
        # dispatches the decide-and-act pipeline once per tool completion.
        try:
            mode = await node._decide_compact_mode()
        except Exception as exc:  # noqa: BLE001 — never crash the run loop
            logger.debug("compact mode decision failed in on_tool_end: %s", exc)
            return
        if mode == "noop":
            return
        if mode == "major":
            # Block the run loop here: the next SDK turn will re-read the
            # session, so we must persist the summary before yielding control.
            try:
                await node.compact(mode="major", reason="hook_major")
            except Exception as exc:
                logger.warning("Hook-triggered major compact failed: %s", exc)
            return
        # Minor: fire-and-forget. The lock inside ``compact`` ensures any
        # concurrent CLI / overflow trigger serializes correctly. The task
        # is registered on the node so asyncio's weak-ref scheduler does not
        # GC it before it runs; the done-callback discards the entry so the
        # set never grows unbounded.
        try:
            task = asyncio.create_task(node.compact(mode="minor", reason="hook_minor"))
        except RuntimeError as exc:
            # No running loop (unusual — on_tool_end runs inside the SDK
            # event loop). Fall back to inline await rather than dropping
            # the trigger silently.
            logger.debug("create_task unavailable, awaiting minor compact inline: %s", exc)
            try:
                await node.compact(mode="minor", reason="hook_minor")
            except Exception as exc2:
                logger.warning("Hook-triggered minor compact failed: %s", exc2)
            return
        pending = getattr(node, "_pending_compact_tasks", None)
        if pending is None:
            pending = set()
            node._pending_compact_tasks = pending
        pending.add(task)
        task.add_done_callback(pending.discard)
