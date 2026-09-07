# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Mid-turn context rewriter: compaction that the *next* model call actually sees.

Why this exists
---------------
The agents SDK rebuilds every model call's input from memory
(``streamed_result.input + _model_input_items``) and never re-reads the
session while a run is in flight. A mid-turn compact that only rewrites
SQLite therefore changes nothing the model receives. The one hook that can
change the input of the *next* call is ``RunConfig.call_model_input_filter``:
it fires right before every model call with the full input list and returns a
replacement. Its result is used for that single call, so a compaction has to
be a **stateful view** that is re-applied on every call.

:class:`MidTurnCompactor` owns that view for one run:

* ``rewrite_sdk_input(raw)`` (SDK paths, Responses-API items) overlays the
  recorded replacement on the append-only raw list and decides whether to
  compact before this call.
* ``rewrite_native_messages(messages)`` (Claude native loop, Anthropic-format
  messages) does the same in place — the native loop owns its own list.

The safe boundary is the moment the filter fires: every ``function_call`` of
the previous round already has its ``function_call_output``, no tool is
running, and the session mirrors the in-memory list.

The compactor never raises into the SDK loop: any failure keeps the current
view and, after ``max_failures`` consecutive failures, disables itself for the
rest of the run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from agents import AgentHooks

from datus.agent.node.compact_prompts import build_mid_turn_resume_message, is_mid_turn_resume_text
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode

logger = get_logger(__name__)

ItemFormat = Literal["responses", "anthropic"]

#: Upper bound on the output-token headroom reserved when judging occupancy.
MAX_OUTPUT_RESERVE_TOKENS = 20480
#: Fallback headroom when the model does not expose ``max_tokens``.
DEFAULT_OUTPUT_RESERVE_TOKENS = 8192

_RESPONSES_ITEM_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "reasoning",
        "web_search_call",
        "file_search_call",
        "computer_call",
        "computer_call_output",
        "item_reference",
    }
)
_RESPONSES_BLOCK_TYPES = frozenset({"input_text", "output_text", "input_image", "input_file", "refusal"})
_ANTHROPIC_BLOCK_TYPES = frozenset(
    {
        "text",
        "tool_use",
        "tool_result",
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "image",
        "document",
    }
)


# ---------------------------------------------------------------------------
# Pure helpers (format detection, estimation, item builders)
# ---------------------------------------------------------------------------


def estimate_items_tokens(items: List[Any]) -> int:
    """Rough token estimate: serialized JSON length / 4.

    Same heuristic Codex and Claude Code use for the items appended since the
    last usage report. Never raises — unserializable items fall back to
    ``str()``.
    """
    total = 0
    for item in items:
        try:
            total += len(json.dumps(item, ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001 — estimation must never break the run loop
            total += len(str(item))
    return total // 4


def detect_item_format(items: List[Any]) -> ItemFormat:
    """Guess whether ``items`` are Responses-API items or Anthropic messages.

    Responses items carry ``type`` (``message`` / ``function_call`` / …) and
    ``input_text`` / ``output_text`` blocks; Anthropic messages have no
    ``type`` wrapper and use ``text`` / ``tool_use`` / ``tool_result`` blocks.
    A string-content user message is ambiguous and harmless either way, so the
    default is ``"responses"``.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message" or item_type in _RESPONSES_ITEM_TYPES:
            return "responses"
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in _RESPONSES_BLOCK_TYPES:
                    return "responses"
                if block_type in _ANTHROPIC_BLOCK_TYPES:
                    return "anthropic"
    return "responses"


def _text_blocks(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        return []
    texts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("input_text", "text", "output_text"):
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return texts


def extract_user_text(item: Any) -> Optional[str]:
    """Return the text of a *real* user message, or ``None``.

    Tool results travel as user messages on the Anthropic path
    (``tool_result`` blocks); those carry no text and yield ``None``.
    """
    if not isinstance(item, dict) or item.get("role") != "user":
        return None
    texts = _text_blocks(item.get("content"))
    if not texts:
        return None
    return "\n".join(texts)


def is_resume_message(item: Any) -> bool:
    """Whether ``item`` is a resume instruction left by an earlier compaction."""
    text = extract_user_text(item)
    return text is not None and is_mid_turn_resume_text(text)


def build_user_item(text: str, item_format: ItemFormat) -> Dict[str, Any]:
    if item_format == "anthropic":
        return {"role": "user", "content": [{"type": "text", "text": text}]}
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def build_assistant_item(text: str, item_format: ItemFormat) -> Dict[str, Any]:
    if item_format == "anthropic":
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _sanitize_user_item(item: Dict[str, Any], item_format: ItemFormat) -> Optional[Dict[str, Any]]:
    """Keep a user message verbatim, dropping tool-result blocks that would dangle.

    Returns ``None`` when nothing textual remains.
    """
    content = item.get("content")
    if not isinstance(content, list):
        return item
    if item_format == "anthropic":
        kept = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]
    else:
        kept = list(content)
    if not any(isinstance(b, dict) and b.get("type") in ("text", "input_text") and b.get("text") for b in kept):
        return None
    if len(kept) == len(content):
        return item
    return {**item, "content": kept}


def find_turn_request(items: List[Any]) -> Optional[Dict[str, Any]]:
    """Return the newest real user message that is not a resume instruction.

    Called on the first model call of a run, when the raw list is
    ``[prior history…, this turn's prompt]``, so the result is the request the
    current turn is working on.
    """
    for item in reversed(items):
        if extract_user_text(item) is not None and not is_resume_message(item):
            return item
    return None


def select_current_turn_user_items(
    items: List[Any],
    turn_request: Optional[Dict[str, Any]],
    item_format: ItemFormat,
) -> List[Dict[str, Any]]:
    """Verbatim user messages of the current turn: the request plus later inserts.

    ``turn_request`` anchors the turn (matched by identity, then equality,
    newest match wins). Without an anchor the newest real user message is
    used. Resume instructions from earlier compactions are dropped.
    """
    start = -1
    if turn_request is not None:
        for idx in range(len(items) - 1, -1, -1):
            candidate = items[idx]
            if candidate is turn_request or candidate == turn_request:
                start = idx
                break
    if start < 0:
        for idx in range(len(items) - 1, -1, -1):
            if extract_user_text(items[idx]) is not None and not is_resume_message(items[idx]):
                start = idx
                break
    if start < 0:
        return []
    out: List[Dict[str, Any]] = []
    for item in items[start:]:
        if extract_user_text(item) is None or is_resume_message(item):
            continue
        sanitized = _sanitize_user_item(item, item_format)
        if sanitized is not None:
            out.append(sanitized)
    return out


def build_mid_turn_view(
    items: List[Any],
    continuation: str,
    *,
    item_format: ItemFormat,
    turn_request: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Assemble the post-compaction input for a turn still in progress.

    Shape (starts with ``user`` for Anthropic, ends with ``user`` so the model
    keeps working instead of replying to the summary)::

        [user: this turn's request (verbatim)]
        [user: mid-run inserts of this turn (verbatim, if any)]
        [assistant: summary continuation]
        [user: resume instruction]
    """
    view = select_current_turn_user_items(items, turn_request, item_format)
    view.append(build_assistant_item(continuation, item_format))
    view.append(build_user_item(build_mid_turn_resume_message(), item_format))
    return view


# ---------------------------------------------------------------------------
# The per-run compactor
# ---------------------------------------------------------------------------


class MidTurnCompactor(AgentHooks):
    """Per-run stateful view that compacts before a model call when needed.

    Also an :class:`agents.AgentHooks` (``on_start``) so composing it into
    the node's hooks resets the view whenever the model layer starts a new
    ``Runner`` run — including the retry of a whole run after a transport
    error, which re-reads the session and would otherwise be spliced with a
    stale overlay.
    """

    def __init__(
        self,
        node: "AgenticNode",
        *,
        interrupt_controller: Any = None,
        system_instruction: str = "",
        max_failures: int = 3,
    ) -> None:
        self._node = node
        self._interrupt_controller = interrupt_controller
        self._system_instruction = system_instruction or ""
        self._max_failures = max(1, int(max_failures))
        self._warned_no_context_length = False
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the overlay and counters; called at the start of every run."""
        self._prefix_len = 0
        self._replacement: List[Dict[str, Any]] = []
        self._boundary_item: Any = None
        self._pins: List[Tuple[int, Dict[str, Any]]] = []
        self._prev_view_len = 0
        self._calls = 0
        self._compactions = 0
        self._failures = 0
        self._disabled = False
        self._requests_fence: Optional[int] = None
        self._turn_request: Optional[Dict[str, Any]] = None

    async def on_start(self, context: Any, agent: Any) -> None:  # noqa: D401
        self.reset()

    @property
    def compactions(self) -> int:
        """Number of rewrites installed during the current run."""
        return self._compactions

    # ------------------------------------------------------------------
    # SDK path (Responses-format items, overlay re-applied every call)
    # ------------------------------------------------------------------

    def view_of(self, raw: List[Any]) -> List[Any]:
        """Apply the recorded overlay to the SDK's raw input list."""
        if self._prefix_len == 0 and not self._pins:
            return list(raw)
        prefix_len = min(self._prefix_len, len(raw))
        out: List[Any] = list(self._replacement)
        cursor = prefix_len
        for offset, item in self._pins:
            offset = min(max(offset, cursor), len(raw))
            out.extend(raw[cursor:offset])
            out.append(item)
            cursor = offset
        out.extend(raw[cursor:])
        return out

    def pin_insert(self, raw_len: int, item: Dict[str, Any]) -> None:
        """Keep a drained mid-run insert visible on every later call of this run."""
        self._pins.append((max(raw_len, self._prefix_len), item))
        self._prev_view_len += 1

    def _overlay_is_stale(self, raw: List[Any]) -> bool:
        if self._prefix_len == 0:
            return False
        if len(raw) < self._prefix_len:
            return True
        if self._boundary_item is not None and raw[self._prefix_len - 1] != self._boundary_item:
            return True
        return False

    async def rewrite_sdk_input(self, raw: List[Any]) -> List[Any]:
        """Return the input the model should see for this call. Never raises."""
        try:
            if self._overlay_is_stale(raw):
                logger.warning("Mid-turn compaction overlay no longer matches the run input; resetting it.")
                self.reset()
            view = self.view_of(raw)
            try:
                new_view = await self._maybe_compact(view, item_format="responses")
            except Exception:  # noqa: BLE001 — keep the current view on any failure
                logger.exception("Mid-turn compaction failed; keeping the current context view.")
                self._record_failure()
                new_view = None
            if new_view is not None:
                self._prefix_len = len(raw)
                self._replacement = list(new_view)
                self._boundary_item = raw[-1] if raw else None
                self._pins = []
                view = list(new_view)
            self._prev_view_len = len(view)
            self._calls += 1
            return view
        except Exception:  # noqa: BLE001 — the filter must never abort the SDK run
            logger.exception("Context rewriter failed; passing the input through unchanged.")
            self._calls += 1
            return list(raw)

    # ------------------------------------------------------------------
    # Claude native path (Anthropic messages, rewritten in place)
    # ------------------------------------------------------------------

    async def rewrite_native_messages(self, messages: List[Any]) -> Optional[List[Any]]:
        """Return a replacement message list, or ``None`` to keep ``messages``."""
        new_view: Optional[List[Any]] = None
        try:
            new_view = await self._maybe_compact(messages, item_format="anthropic")
        except Exception:  # noqa: BLE001 — never break the native loop
            logger.exception("Mid-turn compaction failed on the native path; keeping messages.")
            self._record_failure()
        self._prev_view_len = len(new_view) if new_view is not None else len(messages)
        self._calls += 1
        # Hand the loop its own list: it appends the next rounds in place.
        return list(new_view) if new_view is not None else None

    # ------------------------------------------------------------------
    # Decision + delegation
    # ------------------------------------------------------------------

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._max_failures and not self._disabled:
            self._disabled = True
            logger.warning(
                "Mid-turn compaction disabled for the rest of this run after %d consecutive failures.",
                self._failures,
            )

    def _context_length(self) -> int:
        try:
            return int(getattr(self._node, "context_length", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _base_tokens(self, view: List[Any]) -> Tuple[int, int]:
        """``(base_tokens, tail_start)`` for the occupancy estimate.

        Uses the last model call's real input tokens when the usage snapshot
        is newer than the last rewrite, plus everything appended since that
        call (``view[tail_start:]``). Otherwise estimates the whole view.
        """
        running = getattr(self._node, "running_turn_usage", None)
        requests = int(getattr(running, "requests", 0) or 0) if running is not None else 0
        fresh = running is not None and (self._requests_fence is None or requests > self._requests_fence)
        tokens = 0
        if running is not None:
            tokens = int(getattr(running, "session_total_tokens", 0) or getattr(running, "input_tokens", 0) or 0)
        if fresh and tokens > 0:
            return tokens, min(self._prev_view_len, len(view))
        return estimate_items_tokens(view), len(view)

    async def _maybe_compact(self, view: List[Any], *, item_format: ItemFormat) -> Optional[List[Any]]:
        if self._calls == 0:
            # First call of the run: the turn-start compact just ran and there
            # is no previous call to measure against. Anchor the turn instead.
            self._turn_request = find_turn_request(view)
            return None
        if self._disabled:
            return None
        controller = self._interrupt_controller
        if controller is not None and getattr(controller, "is_interrupted", False):
            return None
        context_length = self._context_length()
        if context_length <= 0:
            if not self._warned_no_context_length:
                self._warned_no_context_length = True
                logger.info("Mid-turn compaction inactive: the model's context window is unknown.")
            return None
        if self._turn_request is None:
            self._turn_request = find_turn_request(view)

        base_tokens, tail_start = self._base_tokens(view)
        result = await self._node.compact_mid_turn(
            view,
            item_format=item_format,
            base_tokens=base_tokens,
            tail_start=tail_start,
            instruction=self._system_instruction,
            turn_request=self._turn_request,
            reason=f"mid_turn_{item_format}",
        )
        if not isinstance(result, dict):
            return None
        mode = result.get("mode", "noop")
        success = bool(result.get("success", False))
        failed = (not success) or bool(result.get("major_error"))
        if failed:
            self._record_failure()
        elif mode != "noop":
            self._failures = 0
        if mode == "noop" or not success:
            return None

        new_items = result.get("items")
        if not isinstance(new_items, list) or new_items is view:
            return None

        running = getattr(self._node, "running_turn_usage", None)
        self._requests_fence = int(getattr(running, "requests", 0) or 0) if running is not None else None
        self._compactions += 1
        threshold = float(getattr(getattr(self._node._compact_cfg, "major", None), "token_threshold", 0.9) or 0.9)
        if (estimate_items_tokens(new_items) + self._node._mid_turn_output_reserve()) / context_length >= threshold:
            self._disabled = True
            logger.warning("Mid-turn compaction cannot bring the context under the threshold; disabled for this run.")
        return new_items
