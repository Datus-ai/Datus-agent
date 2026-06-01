# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``RunHooks`` adapter that emits per-LLM-call token-usage updates.

The OpenAI Agents SDK fires ``on_llm_end`` after every individual LLM
response. We hook in there to:

1. Read the SDK's running ``RunContextWrapper.usage`` accumulator so we
   always have an up-to-date turn-cumulative snapshot regardless of which
   model path produced the response.
2. Compute a per-call delta against the prior frame so consumers can
   render both turn totals and the single LLM call's contribution.
3. Persist the snapshot via :meth:`SessionManager.upsert_running_turn_usage`
   so a mid-turn crash / resume still surfaces partial progress.
4. Inject a ``token_usage`` :class:`ActionHistory` into the active
   :class:`ActionHistoryManager` so the existing stream → SSE pipeline
   forwards it without bespoke plumbing.
5. Trigger the node's status-dirty callback (wired by the CLI to
   ``DatusApp.invalidate``) so the bottom toolbar refreshes immediately
   instead of waiting on the periodic redraw.

The hook is a no-op when no ``ActionHistoryManager`` is bound on the node
(e.g. test doubles, non-streaming paths that haven't opted in yet).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from agents import RunHooks

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)


_USAGE_FIELDS = (
    "requests",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
)


def _delta(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, int]:
    prev = previous or {}
    out: Dict[str, int] = {}
    for key in _USAGE_FIELDS:
        try:
            out[key] = max(0, int(current.get(key, 0) or 0) - int(prev.get(key, 0) or 0))
        except (TypeError, ValueError):
            out[key] = 0
    return out


class TokenUsageHook(RunHooks):
    """Bridge between SDK ``on_llm_end`` and Datus's streaming UI."""

    def __init__(self, node: "AgenticNode") -> None:
        self._node = node
        self._last_cumulative: Optional[Dict[str, Any]] = None

    async def on_start(self, context: Any, agent: Any) -> None:  # noqa: D401
        # New SDK run: reset the per-call delta baseline so the first
        # ``on_llm_end`` reports the full first-call usage as a delta.
        self._last_cumulative = None

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:  # noqa: D401
        # Handoff opens a fresh sub-agent run; treat as a baseline reset so
        # we don't attribute the previous agent's totals as a delta on the
        # next call.
        self._last_cumulative = None

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:  # noqa: D401
        try:
            await self._emit(context)
        except Exception:  # noqa: BLE001 — hook must never break the run loop
            logger.debug("TokenUsageHook.on_llm_end failed", exc_info=True)

    # ------------------------------------------------------------------
    # Manual fan-in for models that don't drive RunHooks (e.g. Codex)
    # ------------------------------------------------------------------

    async def emit_manual(self, usage_info: Dict[str, Any]) -> None:
        """Push a usage snapshot through the same pipeline as ``on_llm_end``.

        Used by model paths that bypass the OpenAI Agents SDK Runner (and
        therefore don't naturally trigger ``on_llm_end``). ``usage_info`` is
        the standardized dict produced by ``_extract_usage_info``.
        """
        try:
            self._publish(dict(usage_info or {}))
        except Exception:  # noqa: BLE001
            logger.debug("TokenUsageHook.emit_manual failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _emit(self, context: Any) -> None:
        usage_obj = getattr(context, "usage", None)
        if usage_obj is None:
            return
        model = getattr(self._node, "model", None)
        extractor = getattr(model, "_extract_usage_info", None) if model is not None else None
        if extractor is None:
            return
        try:
            cumulative = extractor(usage_obj) or {}
        except Exception:  # noqa: BLE001
            logger.debug("TokenUsageHook: _extract_usage_info failed", exc_info=True)
            return
        self._publish(dict(cumulative))

    def _publish(self, cumulative: Dict[str, Any]) -> None:
        delta = _delta(cumulative, self._last_cumulative)
        self._last_cumulative = cumulative

        context_length = self._resolve_context_length()
        last_call_input_tokens = int(cumulative.get("last_call_input_tokens", 0) or 0)

        self._update_node_snapshot(cumulative, context_length, last_call_input_tokens)
        self._persist_snapshot(cumulative, context_length)
        self._enqueue_action(cumulative, delta, context_length, last_call_input_tokens)
        self._notify_status_dirty()

    def _resolve_context_length(self) -> int:
        try:
            value = getattr(self._node, "context_length", 0)
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _update_node_snapshot(
        self,
        cumulative: Dict[str, Any],
        context_length: int,
        last_call_input_tokens: int,
    ) -> None:
        try:
            from datus.schemas.token_usage import TokenUsage

            self._node.running_turn_usage = TokenUsage.from_usage_dict(
                cumulative,
                session_total_tokens=last_call_input_tokens or int(cumulative.get("input_tokens", 0) or 0),
                context_length=context_length,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TokenUsageHook: failed to update node.running_turn_usage", exc_info=True)

    def _persist_snapshot(self, cumulative: Dict[str, Any], context_length: int) -> None:
        session_id = getattr(self._node, "session_id", None)
        if not session_id:
            return
        try:
            session_manager = self._node.session_manager
        except Exception:  # noqa: BLE001
            session_manager = None
        if session_manager is None or not hasattr(session_manager, "upsert_running_turn_usage"):
            return
        turn_number = self._current_turn_number()
        try:
            session_manager.upsert_running_turn_usage(
                session_id=session_id,
                user_turn_number=turn_number,
                cumulative=cumulative,
                context_length=context_length,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TokenUsageHook: upsert_running_turn_usage failed", exc_info=True)

        # Persist the context-window occupancy to the on-disk session_state so
        # a resume of an already-completed turn (when the running snapshot
        # above has been cleared at turn end) can still render the context
        # bar. ``last_call_input_tokens`` is the real context-window usage of
        # the most recent LLM call. Decoupled from SQLite on purpose — see
        # ``AgenticNode.persist_context_state`` / ``datus.storage.session_state``.
        persist = getattr(self._node, "persist_context_state", None)
        if not callable(persist):
            return
        last_call_input_tokens = int(cumulative.get("last_call_input_tokens", 0) or 0)
        try:
            persist(last_call_input_tokens, context_length)
        except Exception:  # noqa: BLE001
            logger.debug("TokenUsageHook: persist_context_state failed", exc_info=True)

    def _current_turn_number(self) -> int:
        # Best-effort: count root-level USER actions on the node. This matches
        # the SDK's ``user_turn_number`` semantics (1-based, monotonic per
        # user message) without coupling to internal SDK accounting.
        actions = getattr(self._node, "actions", None) or []
        count = 0
        for action in actions:
            if getattr(action, "role", None) == ActionRole.USER and getattr(action, "depth", 0) == 0:
                count += 1
        # The in-flight USER action still lives in the active manager (not yet
        # flushed into ``self._node.actions``); count it so a mid-turn snapshot
        # is persisted against the current turn rather than the previous one.
        manager = getattr(self._node, "_current_action_history", None)
        if manager is not None:
            try:
                if any(
                    getattr(action, "role", None) == ActionRole.USER and getattr(action, "depth", 0) == 0
                    for action in manager.get_actions()
                ):
                    count += 1
            except Exception:  # noqa: BLE001
                logger.debug("TokenUsageHook: failed to inspect current action history", exc_info=True)
        return max(count, 1)

    def _enqueue_action(
        self,
        cumulative: Dict[str, Any],
        delta: Dict[str, int],
        context_length: int,
        last_call_input_tokens: int,
    ) -> None:
        manager = getattr(self._node, "_current_action_history", None)
        if manager is None:
            return
        action = ActionHistory(
            action_id=f"token_usage_{uuid.uuid4().hex[:8]}",
            role=ActionRole.ASSISTANT,
            messages="Token usage update",
            action_type="token_usage",
            input={},
            output={
                "cumulative": cumulative,
                "delta": delta,
                "context_length": context_length,
                "last_call_input_tokens": last_call_input_tokens,
                # Session of the node that produced this usage (main agent or a
                # sub-agent). Lets the API attribute a forwarded ``token_usage``
                # to the correct LLM session instead of the parent's.
                "agent_session_id": getattr(self._node, "session_id", None),
            },
            status=ActionStatus.SUCCESS,
        )
        try:
            manager.add_action(action)
        except Exception:  # noqa: BLE001
            logger.debug("TokenUsageHook: failed to enqueue token_usage action", exc_info=True)
            return
        # ``action_history_manager`` only persists the record; the live
        # stream is driven by :class:`ActionBus`, so we also ``put`` so the
        # CLI / API see the action mid-turn instead of after the user-turn
        # completes. ``put`` is non-blocking and safe to call from the
        # event-loop thread that hosts ``on_llm_end``.
        action_bus = getattr(self._node, "action_bus", None)
        put = getattr(action_bus, "put", None) if action_bus is not None else None
        if callable(put):
            try:
                put(action)
            except Exception:  # noqa: BLE001
                logger.debug("TokenUsageHook: action_bus.put failed", exc_info=True)

    def _notify_status_dirty(self) -> None:
        notify = getattr(self._node, "_notify_status_dirty", None)
        if callable(notify):
            try:
                notify()
            except Exception:  # noqa: BLE001
                logger.debug("TokenUsageHook: _notify_status_dirty failed", exc_info=True)
