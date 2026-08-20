# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Hand-off of AI review verdicts from the permission gate to the tool action.

``PermissionHooks`` reviews a planned bash/SQL action in ``on_tool_start``,
long before the tool's completed ``ActionHistory`` exists. This module is the
seam between the two: the gate records a verdict keyed by ``tool_call_id``, and
whoever builds the completed action for that same call stamps it into
``output["permission_review"]`` (see ``PERMISSION_REVIEW_OUTPUT_KEY``).

Keying works because every producer of a finished tool action uses the same
``complete_{call_id}`` id convention (``openai_compatible``, ``claude_model``,
``codex_model``, ``bash_mode``), and ``call_id`` is exactly the SDK's
``ToolContext.tool_call_id`` handed to the gate.

Entries are consumed once (:func:`take`). The registry is bounded and evicts
oldest-first so an abandoned entry — a review whose tool never ran because the
user denied it, or a call id the producer never completed — cannot accumulate.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Key under which the verdict is stamped into a tool action's ``output``.
PERMISSION_REVIEW_OUTPUT_KEY = "permission_review"

# Reviews are consumed by the very next completed action for the same call, so
# the live set is normally 0-1 entries. The cap only bounds pathological cases
# (denied calls, producers that never complete) and is far above any real
# concurrent tool fan-out.
_MAX_PENDING = 256

_pending: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_lock = Lock()


def record(call_id: Optional[str], payload: Optional[Dict[str, Any]]) -> None:
    """Store the review outcome for ``call_id``; no-op on missing inputs.

    Re-recording the same call id overwrites: the gate refines ``outcome``
    (``auto_allowed`` → ``user_approved``) after the user answers a prompt.
    """
    if not call_id or not payload:
        return
    with _lock:
        _pending[call_id] = payload
        _pending.move_to_end(call_id)
        while len(_pending) > _MAX_PENDING:
            evicted, _ = _pending.popitem(last=False)
            logger.debug("Evicted unconsumed permission review for call %s", evicted)


def take(call_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Pop the review recorded for ``call_id``, or ``None`` when there is none."""
    if not call_id:
        return None
    with _lock:
        return _pending.pop(call_id, None)


def clear() -> None:
    """Drop every pending entry (session teardown, tests)."""
    with _lock:
        _pending.clear()


def stamp(output: Any, call_id: Optional[str]) -> Any:
    """Attach the pending review for ``call_id`` to a tool action's ``output``.

    Returns ``output`` unchanged when there is no review, so producers can call
    this unconditionally. A non-dict ``output`` is wrapped so the verdict is
    never silently dropped.
    """
    review = take(call_id)
    if review is None:
        return output
    if isinstance(output, dict):
        output[PERMISSION_REVIEW_OUTPUT_KEY] = review
        return output
    return {"result": output, PERMISSION_REVIEW_OUTPUT_KEY: review}


def call_id_of(action_id: Optional[str]) -> Optional[str]:
    """Recover the tool ``call_id`` from a completed action's id.

    Every producer names a finished tool action ``complete_{call_id}``; the
    in-flight frame uses the bare ``call_id``. Both map back to the same key.
    """
    if not action_id:
        return None
    return action_id[len("complete_") :] if action_id.startswith("complete_") else action_id


def enrich_action(action: Any) -> None:
    """Stamp a pending review onto a completed tool action, in place.

    Registered with ``ActionHistoryManager`` so every model implementation
    picks this up without knowing the permission layer exists. Only SUCCESS or
    FAILED tool actions are considered: the PROCESSING frame is emitted before
    the tool runs and would consume the entry too early, leaving the completed
    action — the one that persists and gets replayed — without it.
    """
    from datus.schemas.action_history import ActionRole, ActionStatus

    if action.role != ActionRole.TOOL or action.status == ActionStatus.PROCESSING:
        return
    if isinstance(action.output, dict) and PERMISSION_REVIEW_OUTPUT_KEY in action.output:
        return
    call_id = call_id_of(action.action_id)
    if call_id and _peek(call_id) is None:
        return
    action.output = stamp(action.output if action.output is not None else {}, call_id)


def _peek(call_id: str) -> Optional[Dict[str, Any]]:
    """Non-consuming lookup, so a miss never rewrites ``output``."""
    with _lock:
        return _pending.get(call_id)
