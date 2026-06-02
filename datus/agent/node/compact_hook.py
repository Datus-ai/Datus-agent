# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``RunHooks`` adapter that drives compact passes from inside a Runner loop.

The OpenAI Agents SDK invokes ``on_tool_end`` after every successful tool
call. We hook in there to:

1. Increment the node's per-session tool-call counter (used as the rolling
   window precondition).
2. Ask the node whether to compact next, via ``_decide_compact_mode``.
3. Dispatch the chosen pass **synchronously**, before returning control to the
   SDK loop:

   * ``major`` — must block: the next turn re-reads the session, so the summary
     has to be persisted first or the run would crash with ``MaxTurnsExceeded``
     / token-limit errors.
   * ``minor`` — a fast, local, rule-based archive of long tool I/O (no LLM
     call); we block until it finishes too, so the archive is committed before
     the loop issues the next model call. It barely delays the agent and keeps
     the trigger simple (no fire-and-forget task bookkeeping).

   The node-level ``asyncio.Lock`` inside ``compact`` serializes either pass
   against any concurrent CLI / pre-user-turn trigger.
"""

from __future__ import annotations

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
            # ``mid_turn=True``: only major is decided per tool call. Minor's
            # gate is the user-turn count, which is constant within a turn, so
            # it is decided once at turn start (``pre_user_turn``) instead.
            mode = await node._decide_compact_mode(mid_turn=True)
        except Exception as exc:  # noqa: BLE001 — never crash the run loop
            logger.debug("compact mode decision failed in on_tool_end: %s", exc)
            return
        if mode == "noop":
            return
        # Run the chosen pass synchronously before returning control to the SDK
        # loop. ``major`` must block (the next turn would otherwise re-read an
        # over-limit session). ``minor`` is a fast, local, rule-based archive of
        # long tool I/O — no LLM call — so we block on it too: it barely delays
        # the agent and guarantees the archive is committed before the next
        # model call reads the session. The node-level lock inside ``compact``
        # serializes against any concurrent CLI / pre-user-turn trigger; for
        # ``major``, the user-facing display (pinned hint + summary panel) is
        # injected inside ``node.compact``.
        try:
            await node.compact(mode=mode, reason=f"hook_{mode}")
        except Exception as exc:  # noqa: BLE001 — a compact failure must not crash the loop
            logger.warning("Hook-triggered %s compact failed: %s", mode, exc)
