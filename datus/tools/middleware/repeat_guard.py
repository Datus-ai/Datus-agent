# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""No-progress guard: stop a model from replaying a tool call that changes nothing.

A reasoning model that already holds the answer can still emit the same function
call on every turn. The tool succeeds, the context grows by one more identical
``(call, result)`` pair, and the next prompt therefore differs from the previous
one *only* by that pair -- a fixed point the run leaves only when ``max_turns``
trips. Seen in production as 28 identical ``execute_sql`` calls returning the
same 23 rows, ~190k tokens, ending in ``Maximum turns (30) exceeded`` with the
correct answer in hand since call 3.

The guard never suppresses execution: tools stay authoritative and side effects
happen exactly as before. It only watches whether a call *changed anything*. A
repeat whose result differs is progress -- polling a table until a job lands,
re-reading a file after writing it -- and resets the streak. Only a repeat whose
result is byte-identical counts:

* from :data:`WARN_AFTER` identical results in a row the payload carries a
  :data:`NOTE_KEY` note telling the model the call is not advancing;
* from :data:`DENY_AFTER` the guard returns a failure payload instead of the
  result the model already has verbatim in its context.

State is per run and per call signature, held in a ContextVar so concurrent runs
(sub-agents included) never share counters. Call :func:`reset_repeat_guard` once
per agent run; a run that forgets to simply carries the previous window's
counters, which is why the reset lives at the single stream chokepoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from contextvars import ContextVar
from typing import Any, Dict, Optional

from agents import FunctionTool

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Marker on a rebuilt tool's ``on_invoke_tool`` so :func:`apply_repeat_guard` can
# recognise already-guarded tools. Mirrors the transformer layer's marker so the
# two wrappers compose in either order without double-wrapping.
_GUARDED_MARKER = "_datus_repeat_guarded"

# Streak thresholds, counted in *identical results for the same call*. A streak
# of 1 is the first repeat, so WARN_AFTER=2 fires on the second identical result
# and leaves the model two more chances to self-correct before the deny.
WARN_AFTER = 2
DENY_AFTER = 4

# Top-level key the note is attached under. Namespaced so it cannot collide with
# a tool's own payload keys, and ignorable by any consumer that does not know it.
NOTE_KEY = "datus_guard"

# Tools whose whole purpose is to be asked the same thing again. ``ask_user``
# repeats are driven by the human, not by the model failing to converge.
_REPEATS_ALWAYS_ALLOWED = frozenset({"ask_user"})


@dataclasses.dataclass
class _CallState:
    """How many times in a row this exact call returned this exact result."""

    fingerprint: Optional[str] = None
    streak: int = 0


_repeat_state_var: ContextVar[Optional[Dict[str, _CallState]]] = ContextVar("datus_tool_repeat_state", default=None)


def reset_repeat_guard() -> None:
    """Open a fresh no-progress window. Call once per agent run."""
    _repeat_state_var.set({})


def _state() -> Dict[str, _CallState]:
    store = _repeat_state_var.get()
    if store is None:
        store = {}
        _repeat_state_var.set(store)
    return store


def _signature(tool_name: str, args_str: str) -> str:
    """Identify a call by name plus canonicalised arguments.

    Key ordering is not stable across turns, so the raw argument string cannot
    be compared directly. Arguments that do not parse fall back to the raw
    string: a malformed call repeated verbatim is still a repeat.
    """
    try:
        parsed = json.loads(args_str) if args_str else {}
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        canonical = json.dumps(parsed, sort_keys=True, ensure_ascii=False, default=str)
    else:
        canonical = args_str or ""
    return f"{tool_name}:{hashlib.sha256(canonical.encode('utf-8', 'replace')).hexdigest()}"


def _fingerprint(result: Any) -> str:
    """Hash a tool result so large payloads cost a digest, not a retained copy."""
    try:
        rendered = json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(result)
    return hashlib.sha256(rendered.encode("utf-8", "replace")).hexdigest()


def _note(tool_name: str, streak: int) -> str:
    return (
        f"'{tool_name}' has now returned an identical result {streak} times in a row for identical "
        "arguments. Repeating it cannot produce new information — use the result you already have "
        "and write your final answer instead of calling this tool again."
    )


def _no_progress_payload(tool_name: str, streak: int) -> dict:
    """Standard failure payload returned once a call is provably not advancing.

    Shaped like ``FuncToolResult`` (and the transformer layer's denial) so the
    model sees an ordinary tool error rather than a crashed run.
    """
    return {
        "success": 0,
        "error": (
            f"Tool call '{tool_name}' was blocked: the last {streak} identical calls all returned the "
            "same result, so calling it again cannot produce new information. That result is already "
            "in this conversation — use it and write your final answer now."
        ),
        "result": None,
    }


def _annotate(result: Any, note: str) -> Any:
    """Attach ``note`` to a dict payload without mutating the tool's own object.

    Non-dict results pass through untouched: there is nowhere to put the note
    that would not corrupt the tool's contract with its renderers.
    """
    if not isinstance(result, dict):
        return result
    annotated = dict(result)
    annotated[NOTE_KEY] = note
    return annotated


def tool_is_repeat_guarded(tool: Any) -> bool:
    """Return True if ``tool`` was already wrapped by :func:`wrap_tool_with_repeat_guard`."""
    return bool(getattr(getattr(tool, "on_invoke_tool", None), _GUARDED_MARKER, False))


def wrap_tool_with_repeat_guard(
    original: FunctionTool,
    warn_after: int = WARN_AFTER,
    deny_after: int = DENY_AFTER,
) -> FunctionTool:
    """Rebuild ``original`` so identical calls returning identical results are flagged.

    The wrapper is transparent until a streak forms: the first call of any
    signature, and every call whose result differs from the previous one, is
    returned exactly as the tool produced it.
    """

    async def guarded_invoke(tool_ctx: Any, args_str: str) -> Any:
        result = await original.on_invoke_tool(tool_ctx, args_str)
        if original.name in _REPEATS_ALWAYS_ALLOWED:
            return result

        signature = _signature(original.name, args_str)
        fingerprint = _fingerprint(result)
        store = _state()
        state = store.get(signature)

        if state is None or state.fingerprint != fingerprint:
            store[signature] = _CallState(fingerprint=fingerprint, streak=1)
            return result

        state.streak += 1
        if state.streak >= deny_after:
            logger.warning(
                "[Repeat guard] '%s' returned an identical result %d times in a row; blocking the call.",
                original.name,
                state.streak,
            )
            return _no_progress_payload(original.name, state.streak)
        if state.streak >= warn_after:
            logger.info(
                "[Repeat guard] '%s' returned an identical result %d times in a row; nudging the model.",
                original.name,
                state.streak,
            )
            return _annotate(result, _note(original.name, state.streak))
        return result

    guarded_invoke._datus_repeat_guarded = True  # type: ignore[attr-defined]

    # Forward every declared field by name so the rebuild stays faithful across
    # SDK versions (same rationale as ``wrap_tool_with_transformers``): rebuilding
    # must not silently re-enable a gated tool or drop its guardrails.
    carried = {
        field.name: getattr(original, field.name, None)
        for field in dataclasses.fields(FunctionTool)
        if field.init and field.name != "on_invoke_tool"
    }
    carried["on_invoke_tool"] = guarded_invoke
    return FunctionTool(**carried)


def apply_repeat_guard(node: Any) -> int:
    """Wrap every tool on ``node`` with the no-progress guard, in place.

    Returns the number of tools wrapped. Already-guarded tools are skipped, so
    callers may re-run this after a tool-list rebuild (a runtime ``/model``
    switch remounts web tools) without double-wrapping.
    """
    tools = getattr(node, "tools", None)
    if not tools:
        return 0

    wrapped = 0
    for idx, tool in enumerate(tools):
        if not isinstance(tool, FunctionTool) or tool_is_repeat_guarded(tool):
            continue
        tools[idx] = wrap_tool_with_repeat_guard(tool)
        wrapped += 1
    return wrapped
