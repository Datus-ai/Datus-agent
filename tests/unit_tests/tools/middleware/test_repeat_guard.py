# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the tool no-progress guard.

Covers the streak rules end to end: transparency on the first call and on any
call that changes something, the note at ``WARN_AFTER``, the denial at
``DENY_AFTER``, argument canonicalisation, per-run isolation, and the
``apply_repeat_guard`` wrapping/skipping behavior.
"""

import json
from types import SimpleNamespace

import pytest
from agents import FunctionTool

from datus.tools.middleware.repeat_guard import (
    DENY_AFTER,
    NOTE_KEY,
    WARN_AFTER,
    apply_repeat_guard,
    reset_repeat_guard,
    tool_is_repeat_guarded,
    wrap_tool_with_repeat_guard,
)


def _tool(name, handler):
    """Build a minimal FunctionTool whose invocation runs ``handler(args_str)``."""

    async def on_invoke_tool(tool_ctx, args_str):
        return handler(args_str)

    return FunctionTool(
        name=name,
        description=f"{name} test double",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke_tool,
    )


def _constant_tool(name="execute_sql", result=None):
    result = {"success": 1, "error": None, "result": {"rows": 23}} if result is None else result
    return _tool(name, lambda _args: result)


@pytest.fixture(autouse=True)
def _fresh_window():
    reset_repeat_guard()
    yield
    reset_repeat_guard()


async def _call(tool, args='{"sql": "SELECT 1"}'):
    return await tool.on_invoke_tool(SimpleNamespace(), args)


class TestStreakBehaviour:
    @pytest.mark.asyncio
    async def test_first_call_is_untouched(self):
        """The guard is invisible until a streak forms."""
        guarded = wrap_tool_with_repeat_guard(_constant_tool())
        result = await _call(guarded)
        assert result == {"success": 1, "error": None, "result": {"rows": 23}}
        assert NOTE_KEY not in result

    @pytest.mark.asyncio
    async def test_note_appears_once_the_result_stops_changing(self):
        """From WARN_AFTER identical results the payload carries the nudge."""
        guarded = wrap_tool_with_repeat_guard(_constant_tool())

        for _ in range(WARN_AFTER - 1):
            assert NOTE_KEY not in await _call(guarded)

        noted = await _call(guarded)
        assert noted[NOTE_KEY]
        # The tool's own payload is preserved alongside the note.
        assert noted["success"] == 1
        assert noted["result"] == {"rows": 23}

    @pytest.mark.asyncio
    async def test_call_is_blocked_once_it_is_provably_stuck(self):
        """At DENY_AFTER the guard answers with a failure instead of the known result."""
        guarded = wrap_tool_with_repeat_guard(_constant_tool())

        for _ in range(DENY_AFTER - 1):
            await _call(guarded)

        blocked = await _call(guarded)
        assert blocked["success"] == 0
        assert blocked["result"] is None
        assert "cannot produce new information" in blocked["error"]

    @pytest.mark.asyncio
    async def test_the_tool_still_runs_every_time(self):
        """The guard observes; it never suppresses execution or its side effects."""
        calls = []

        def handler(args_str):
            calls.append(args_str)
            return {"success": 1, "result": "same"}

        guarded = wrap_tool_with_repeat_guard(_tool("write_thing", handler))
        for _ in range(DENY_AFTER + 2):
            await _call(guarded)

        assert len(calls) == DENY_AFTER + 2

    @pytest.mark.asyncio
    async def test_a_changing_result_is_progress_and_resets_the_streak(self):
        """Polling until something lands must never trip the guard."""
        counter = {"n": 0}

        def handler(_args):
            counter["n"] += 1
            return {"success": 1, "result": counter["n"]}

        guarded = wrap_tool_with_repeat_guard(_tool("list_tables", handler))
        for _ in range(DENY_AFTER + 3):
            result = await _call(guarded)
            assert result["success"] == 1
            assert NOTE_KEY not in result

    @pytest.mark.asyncio
    async def test_streak_resets_when_the_result_changes_mid_run(self):
        """A single changed result clears an accumulated streak."""
        state = {"value": "same"}
        guarded = wrap_tool_with_repeat_guard(_tool("probe", lambda _a: {"result": state["value"]}))

        for _ in range(DENY_AFTER - 1):
            await _call(guarded)

        state["value"] = "changed"
        assert NOTE_KEY not in await _call(guarded)

        # The streak restarted: the next identical result is the second one, so
        # it earns a note rather than the denial the old streak had reached.
        second = await _call(guarded)
        assert second[NOTE_KEY]
        assert second.get("success") != 0


class TestSignature:
    @pytest.mark.asyncio
    async def test_key_order_does_not_hide_a_repeat(self):
        """Arguments are canonicalised, so a reordered dict is still the same call."""
        guarded = wrap_tool_with_repeat_guard(_constant_tool())

        await _call(guarded, json.dumps({"a": 1, "b": 2}))
        noted = await _call(guarded, json.dumps({"b": 2, "a": 1}))
        assert noted[NOTE_KEY]

    @pytest.mark.asyncio
    async def test_different_arguments_are_different_calls(self):
        """Two distinct queries never accumulate a shared streak."""
        guarded = wrap_tool_with_repeat_guard(_constant_tool())

        for i in range(DENY_AFTER + 2):
            result = await _call(guarded, json.dumps({"sql": f"SELECT {i}"}))
            assert NOTE_KEY not in result

    @pytest.mark.asyncio
    async def test_malformed_arguments_still_count_as_a_repeat(self):
        """A call that does not parse as JSON is compared by its raw string."""
        guarded = wrap_tool_with_repeat_guard(_constant_tool())

        await _call(guarded, "not json")
        noted = await _call(guarded, "not json")
        assert noted[NOTE_KEY]


class TestPayloadShapes:
    @pytest.mark.asyncio
    async def test_non_dict_results_pass_through_unannotated(self):
        """There is nowhere to put a note on a string without corrupting it."""
        guarded = wrap_tool_with_repeat_guard(_tool("read_file", lambda _a: "file contents"))

        for _ in range(WARN_AFTER + 1):
            assert await _call(guarded) == "file contents"

    @pytest.mark.asyncio
    async def test_the_tools_own_payload_object_is_not_mutated(self):
        """Annotating must not leak the note back into the tool's own state."""
        payload = {"success": 1, "result": "x"}
        guarded = wrap_tool_with_repeat_guard(_tool("t", lambda _a: payload))

        for _ in range(WARN_AFTER):
            await _call(guarded)

        assert NOTE_KEY not in payload

    @pytest.mark.asyncio
    async def test_ask_user_repeats_are_the_humans_business(self):
        """Asking the same question twice is the user's prerogative, not a loop."""
        guarded = wrap_tool_with_repeat_guard(_tool("ask_user", lambda _a: {"success": 1, "result": "yes"}))

        for _ in range(DENY_AFTER + 2):
            result = await _call(guarded)
            assert result["success"] == 1
            assert NOTE_KEY not in result


class TestRunIsolation:
    @pytest.mark.asyncio
    async def test_reset_opens_a_fresh_window(self):
        """A new run must not inherit the previous run's streak."""
        guarded = wrap_tool_with_repeat_guard(_constant_tool())

        for _ in range(DENY_AFTER + 1):
            await _call(guarded)

        reset_repeat_guard()
        assert NOTE_KEY not in await _call(guarded)


class TestApplyRepeatGuard:
    def test_wraps_every_tool_and_marks_them(self):
        node = SimpleNamespace(tools=[_constant_tool("a"), _constant_tool("b")])
        assert apply_repeat_guard(node) == 2
        assert all(tool_is_repeat_guarded(t) for t in node.tools)

    def test_is_idempotent(self):
        node = SimpleNamespace(tools=[_constant_tool("a")])
        assert apply_repeat_guard(node) == 1
        assert apply_repeat_guard(node) == 0

    def test_preserves_the_declared_tool_fields(self):
        original = _constant_tool("keep_me")
        node = SimpleNamespace(tools=[original])
        apply_repeat_guard(node)

        wrapped = node.tools[0]
        assert wrapped.name == original.name
        assert wrapped.description == original.description
        assert wrapped.params_json_schema == original.params_json_schema

    def test_tolerates_a_node_without_tools(self):
        assert apply_repeat_guard(SimpleNamespace(tools=[])) == 0
        assert apply_repeat_guard(SimpleNamespace()) == 0
