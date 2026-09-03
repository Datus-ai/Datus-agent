# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus/agent/node/context_rewriter.py``.

Covers the pure helpers (format detection, token estimate, view shaping) and
the per-run :class:`MidTurnCompactor` state machine against a scripted node
double. The end-to-end behaviour inside a real agents-SDK loop lives in
``tests/unit_tests/models/test_base.py``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.agent.node.compact_prompts import MID_TURN_RESUME_PREFIX, build_mid_turn_resume_message
from datus.agent.node.context_rewriter import (
    MidTurnCompactor,
    build_assistant_item,
    build_mid_turn_view,
    build_user_item,
    detect_item_format,
    estimate_items_tokens,
    extract_user_text,
    find_turn_request,
    select_current_turn_user_items,
)
from datus.configuration.agent_config import CompactConfig
from datus.schemas.token_usage import TokenUsage

# ---------------------------------------------------------------------------
# Item builders
# ---------------------------------------------------------------------------


def _user(text: str) -> Dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _plain_user(text: str) -> Dict[str, Any]:
    """The shape the SDK gives a string prompt: no ``type``, string content."""
    return {"role": "user", "content": text}


def _assistant(text: str) -> Dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _call(call_id: str, name: str = "execute_sql") -> Dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _out(call_id: str, text: str = "ok") -> Dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def _anth_user(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _anth_tool_result(tool_use_id: str, text: str = "ok") -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text}]}


def _anth_assistant_tool_use(tool_use_id: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "running"},
            {"type": "tool_use", "id": tool_use_id, "name": "execute_sql", "input": {}},
        ],
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestEstimateItemsTokens:
    def test_is_serialized_length_over_four(self):
        items = [_user("x" * 400)]
        assert estimate_items_tokens(items) == len(json.dumps(items[0], ensure_ascii=False)) // 4

    def test_empty_is_zero(self):
        assert estimate_items_tokens([]) == 0

    def test_unserializable_item_falls_back_to_str(self):
        weird = object()
        assert estimate_items_tokens([weird]) == len(json.dumps(weird, default=str)) // 4


class TestDetectItemFormat:
    def test_responses_message_item(self):
        assert detect_item_format([_user("hi")]) == "responses"

    def test_responses_function_call(self):
        assert detect_item_format([_plain_user("hi"), _call("c1")]) == "responses"

    def test_anthropic_text_blocks(self):
        assert detect_item_format([_anth_user("hi")]) == "anthropic"

    def test_anthropic_tool_result(self):
        assert detect_item_format([_anth_tool_result("t1")]) == "anthropic"

    def test_string_content_defaults_to_responses(self):
        assert detect_item_format([_plain_user("hi")]) == "responses"

    def test_empty_defaults_to_responses(self):
        assert detect_item_format([]) == "responses"


class TestExtractUserText:
    def test_string_content(self):
        assert extract_user_text(_plain_user("hello")) == "hello"

    def test_input_text_blocks(self):
        assert extract_user_text(_user("hello")) == "hello"

    def test_anthropic_text_blocks(self):
        assert extract_user_text(_anth_user("hello")) == "hello"

    def test_tool_result_only_is_not_user_text(self):
        assert extract_user_text(_anth_tool_result("t1")) is None

    def test_assistant_is_none(self):
        assert extract_user_text(_assistant("x")) is None

    def test_non_dict_is_none(self):
        assert extract_user_text("nope") is None


class TestSelectCurrentTurnUserItems:
    def test_anchors_on_the_turn_request_by_identity(self):
        request = _plain_user("current ask")
        items = [_user("old ask"), _assistant("old answer"), request, _call("c1"), _out("c1")]
        assert select_current_turn_user_items(items, request, "responses") == [request]

    def test_anchors_by_equality_when_identity_is_lost(self):
        items = [_user("old ask"), _assistant("old answer"), _plain_user("current ask"), _call("c1"), _out("c1")]
        anchor = {"role": "user", "content": "current ask"}  # equal, not identical
        assert select_current_turn_user_items(items, anchor, "responses") == [items[2]]

    def test_keeps_later_inserts_and_drops_resume_messages(self):
        request = _plain_user("current ask")
        resume = build_user_item(build_mid_turn_resume_message(), "responses")
        insert = _user("also check refunds")
        items = [request, _assistant("summary"), resume, _call("c2"), _out("c2"), insert]
        assert select_current_turn_user_items(items, request, "responses") == [request, insert]

    def test_falls_back_to_newest_real_user_message_without_anchor(self):
        items = [_user("old"), _assistant("a"), _user("newest"), _call("c1"), _out("c1")]
        assert select_current_turn_user_items(items, None, "responses") == [items[2]]

    def test_anthropic_tool_results_are_not_user_messages(self):
        request = _anth_user("ask")
        items = [request, _anth_assistant_tool_use("t1"), _anth_tool_result("t1", "big"), _anth_tool_result("t2")]
        assert select_current_turn_user_items(items, request, "anthropic") == [request]

    def test_anthropic_mixed_user_message_drops_tool_result_blocks(self):
        request = _anth_user("ask")
        mixed = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r"},
                {"type": "text", "text": "and also this"},
            ],
        }
        items = [request, _anth_assistant_tool_use("t1"), mixed]
        kept = select_current_turn_user_items(items, request, "anthropic")
        assert kept[0] is request
        assert kept[1] == {"role": "user", "content": [{"type": "text", "text": "and also this"}]}

    def test_no_user_messages_gives_empty(self):
        assert select_current_turn_user_items([_assistant("a"), _call("c"), _out("c")], None, "responses") == []


class TestFindTurnRequest:
    def test_newest_real_user_message(self):
        items = [_user("old"), _assistant("a"), _plain_user("new"), _call("c1")]
        assert find_turn_request(items) is items[2]

    def test_skips_resume_messages(self):
        items = [_user("ask"), build_user_item(build_mid_turn_resume_message(), "responses")]
        assert find_turn_request(items) is items[0]

    def test_none_without_users(self):
        assert find_turn_request([_assistant("a")]) is None


class TestBuildMidTurnView:
    def test_responses_shape(self):
        request = _plain_user("analyse refunds")
        items = [
            _user("older"),
            _assistant("earlier"),
            request,
            _call("c1"),
            _out("c1", "x" * 50),
            _call("c2"),
            _out("c2"),
        ]
        view = build_mid_turn_view(items, "SUMMARY --- jsonl", item_format="responses", turn_request=request)
        assert view == [
            request,
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "SUMMARY --- jsonl"}]},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": build_mid_turn_resume_message()}],
            },
        ]
        # Nothing from before the boundary survives: no tool calls, no old turns.
        assert not any(it.get("type") in ("function_call", "function_call_output") for it in view)
        assert view[-1]["content"][0]["text"].startswith(MID_TURN_RESUME_PREFIX)

    def test_anthropic_shape_starts_and_ends_with_user(self):
        request = _anth_user("analyse refunds")
        items = [request, _anth_assistant_tool_use("t1"), _anth_tool_result("t1")]
        view = build_mid_turn_view(items, "SUMMARY", item_format="anthropic", turn_request=request)
        assert view == [
            request,
            {"role": "assistant", "content": [{"type": "text", "text": "SUMMARY"}]},
            {"role": "user", "content": [{"type": "text", "text": build_mid_turn_resume_message()}]},
        ]
        assert view[0]["role"] == "user" and view[-1]["role"] == "user"
        assert all(isinstance(m["content"], list) and m["content"] for m in view)

    def test_inserts_of_the_turn_are_kept_verbatim_before_the_summary(self):
        request = _plain_user("ask")
        insert = _user("also refunds by region")
        items = [request, _call("c1"), _out("c1"), insert, _call("c2"), _out("c2")]
        view = build_mid_turn_view(items, "S", item_format="responses", turn_request=request)
        assert view[:2] == [request, insert]
        assert view[2] == build_assistant_item("S", "responses")


# ---------------------------------------------------------------------------
# MidTurnCompactor
# ---------------------------------------------------------------------------


class _FakeNode:
    """Node double exposing exactly what ``MidTurnCompactor`` touches."""

    def __init__(self, *, context_length: int = 1000, output_reserve: int = 0) -> None:
        self.context_length = context_length
        self.running_turn_usage: Optional[TokenUsage] = None
        self._compact_cfg = CompactConfig()
        self._output_reserve = output_reserve
        self.compact_mid_turn = AsyncMock(return_value={"mode": "noop", "success": True})

    def _mid_turn_output_reserve(self) -> int:
        return self._output_reserve

    def script_major(self, summary: str = "SUMMARY") -> None:
        """Make ``compact_mid_turn`` rebuild the view like the real node does."""

        async def _compact(items, *, item_format, base_tokens, tail_start, instruction, turn_request, reason):
            view = build_mid_turn_view(items, summary, item_format=item_format, turn_request=turn_request)
            return {"mode": "major", "success": True, "items": view, "summary": summary}

        self.compact_mid_turn = AsyncMock(side_effect=_compact)


def _usage(requests: int, occupancy: int) -> TokenUsage:
    return TokenUsage(requests=requests, input_tokens=occupancy, session_total_tokens=occupancy, context_length=1000)


class TestViewOf:
    def test_no_overlay_returns_a_copy(self):
        compactor = MidTurnCompactor(_FakeNode())
        raw = [_plain_user("a"), _call("c1")]
        view = compactor.view_of(raw)
        assert view == raw and view is not raw

    def test_overlay_replaces_the_recorded_prefix_only(self):
        compactor = MidTurnCompactor(_FakeNode())
        compactor._prefix_len = 3
        compactor._replacement = [_user("R")]
        raw = [_plain_user("a"), _call("c1"), _out("c1"), _call("c2"), _out("c2")]
        assert compactor.view_of(raw) == [_user("R"), _call("c2"), _out("c2")]

    def test_pins_are_spliced_at_their_raw_offset(self):
        compactor = MidTurnCompactor(_FakeNode())
        raw = [_plain_user("a"), _call("c1"), _out("c1")]
        compactor.pin_insert(3, _user("INSERT"))
        raw_later = raw + [_call("c2"), _out("c2")]
        assert compactor.view_of(raw_later) == raw + [_user("INSERT"), _call("c2"), _out("c2")]

    def test_pin_offset_never_precedes_the_overlay(self):
        compactor = MidTurnCompactor(_FakeNode())
        compactor._prefix_len = 2
        compactor._replacement = [_user("R")]
        compactor.pin_insert(1, _user("INSERT"))  # stale offset inside the compacted prefix
        raw = [_plain_user("a"), _call("c1"), _out("c1")]
        assert compactor.view_of(raw) == [_user("R"), _user("INSERT"), _out("c1")]


class TestRewriteSdkInput:
    @pytest.mark.asyncio
    async def test_first_call_only_anchors_the_turn_request(self):
        node = _FakeNode()
        compactor = MidTurnCompactor(node)
        raw = [_user("old"), _assistant("a"), _plain_user("current")]
        assert await compactor.rewrite_sdk_input(raw) == raw
        node.compact_mid_turn.assert_not_awaited()
        assert compactor._turn_request is raw[2]

    @pytest.mark.asyncio
    async def test_second_call_installs_the_view_and_replays_it(self):
        node = _FakeNode()
        node.script_major()
        node.running_turn_usage = _usage(requests=1, occupancy=950)
        compactor = MidTurnCompactor(node)
        request = _plain_user("current")
        raw1 = [request]
        await compactor.rewrite_sdk_input(raw1)
        raw2 = raw1 + [_call("c1"), _out("c1", "x" * 200)]
        view2 = await compactor.rewrite_sdk_input(raw2)
        assert view2 == build_mid_turn_view(raw2, "SUMMARY", item_format="responses", turn_request=request)
        assert compactor.compactions == 1
        # Next call: the context is small again, so the node reports noop; the
        # overlay is re-applied and the new items are appended.
        node.compact_mid_turn = AsyncMock(return_value={"mode": "noop", "success": True})
        raw3 = raw2 + [_call("c2"), _out("c2")]
        node.running_turn_usage = _usage(requests=2, occupancy=100)
        view3 = await compactor.rewrite_sdk_input(raw3)
        assert view3 == view2 + [_call("c2"), _out("c2")]
        assert compactor.compactions == 1

    @pytest.mark.asyncio
    async def test_base_tokens_come_from_fresh_usage_plus_tail(self):
        node = _FakeNode()
        node.running_turn_usage = _usage(requests=1, occupancy=500)
        compactor = MidTurnCompactor(node)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        raw2 = raw1 + [_call("c1"), _out("c1")]
        await compactor.rewrite_sdk_input(raw2)
        kwargs = node.compact_mid_turn.await_args.kwargs
        assert kwargs["base_tokens"] == 500
        assert kwargs["tail_start"] == 1  # everything after the previous call's view
        assert kwargs["item_format"] == "responses"
        assert kwargs["turn_request"] is raw1[0]

    @pytest.mark.asyncio
    async def test_stale_usage_falls_back_to_estimating_the_whole_view(self):
        node = _FakeNode()
        node.script_major()
        node.running_turn_usage = _usage(requests=1, occupancy=950)
        compactor = MidTurnCompactor(node)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        raw2 = raw1 + [_call("c1"), _out("c1")]
        await compactor.rewrite_sdk_input(raw2)  # compacts; fence = requests 1
        node.compact_mid_turn = AsyncMock(return_value={"mode": "noop", "success": True})
        raw3 = raw2 + [_call("c2"), _out("c2")]
        view3 = await compactor.rewrite_sdk_input(raw3)  # usage NOT refreshed yet
        kwargs = node.compact_mid_turn.await_args.kwargs
        assert kwargs["base_tokens"] == estimate_items_tokens(view3)
        assert kwargs["tail_start"] == len(view3)

    @pytest.mark.asyncio
    async def test_noop_passes_the_raw_input_through(self):
        node = _FakeNode()
        compactor = MidTurnCompactor(node)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        raw2 = raw1 + [_call("c1"), _out("c1")]
        assert await compactor.rewrite_sdk_input(raw2) == raw2
        assert compactor.compactions == 0

    @pytest.mark.asyncio
    async def test_exception_keeps_the_view_and_counts_a_failure(self):
        node = _FakeNode()
        node.compact_mid_turn = AsyncMock(side_effect=RuntimeError("boom"))
        compactor = MidTurnCompactor(node, max_failures=2)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        raw2 = raw1 + [_call("c1"), _out("c1")]
        assert await compactor.rewrite_sdk_input(raw2) == raw2
        assert await compactor.rewrite_sdk_input(raw2) == raw2
        assert compactor._disabled is True
        node.compact_mid_turn.reset_mock()
        await compactor.rewrite_sdk_input(raw2)
        node.compact_mid_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_three_failed_results_trip_the_breaker(self):
        node = _FakeNode()
        node.compact_mid_turn = AsyncMock(return_value={"mode": "major", "success": False})
        compactor = MidTurnCompactor(node)
        raw = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw)
        for _ in range(5):
            await compactor.rewrite_sdk_input(raw + [_call("c"), _out("c")])
        assert node.compact_mid_turn.await_count == 3
        assert compactor._disabled is True

    @pytest.mark.asyncio
    async def test_major_error_after_a_successful_archive_installs_the_view_but_counts(self):
        node = _FakeNode()
        archived = [_plain_user("current"), _call("c1"), _out("c1", "[DATUS_ARCHIVED] path=p preview=x")]
        node.compact_mid_turn = AsyncMock(
            return_value={"mode": "minor", "success": True, "items": archived, "major_error": "llm down"}
        )
        compactor = MidTurnCompactor(node)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        view = await compactor.rewrite_sdk_input(raw1 + [_call("c1"), _out("c1", "x" * 500)])
        assert view == archived
        assert compactor._failures == 1

    @pytest.mark.asyncio
    async def test_success_resets_the_failure_counter(self):
        node = _FakeNode()
        node.compact_mid_turn = AsyncMock(return_value={"mode": "major", "success": False})
        compactor = MidTurnCompactor(node)
        raw = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw)
        await compactor.rewrite_sdk_input(raw + [_call("c"), _out("c")])
        assert compactor._failures == 1
        node.script_major()
        await compactor.rewrite_sdk_input(raw + [_call("c"), _out("c")])
        assert compactor._failures == 0

    @pytest.mark.asyncio
    async def test_interrupted_run_skips_compaction(self):
        node = _FakeNode()
        controller = MagicMock(is_interrupted=True)
        compactor = MidTurnCompactor(node, interrupt_controller=controller)
        raw = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw)
        await compactor.rewrite_sdk_input(raw + [_call("c"), _out("c")])
        node.compact_mid_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_context_window_is_inert(self):
        node = _FakeNode(context_length=0)
        compactor = MidTurnCompactor(node)
        raw = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw)
        await compactor.rewrite_sdk_input(raw + [_call("c"), _out("c")])
        node.compact_mid_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_overlay_is_dropped_when_the_raw_list_no_longer_matches(self):
        node = _FakeNode()
        node.script_major()
        node.running_turn_usage = _usage(requests=1, occupancy=950)
        compactor = MidTurnCompactor(node)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        await compactor.rewrite_sdk_input(raw1 + [_call("c1"), _out("c1")])
        assert compactor._prefix_len == 3
        # A brand-new run (e.g. a retried Runner) hands over a shorter raw list.
        fresh = [_plain_user("current")]
        assert await compactor.rewrite_sdk_input(fresh) == fresh
        assert compactor._prefix_len == 0

    @pytest.mark.asyncio
    async def test_on_start_resets_everything(self):
        node = _FakeNode()
        node.script_major()
        node.running_turn_usage = _usage(requests=1, occupancy=950)
        compactor = MidTurnCompactor(node)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        await compactor.rewrite_sdk_input(raw1 + [_call("c1"), _out("c1")])
        assert compactor.compactions == 1
        await compactor.on_start(None, None)
        assert compactor.compactions == 0
        assert compactor._prefix_len == 0
        assert compactor._calls == 0
        assert compactor.view_of(raw1) == raw1

    @pytest.mark.asyncio
    async def test_disabled_when_the_compacted_view_is_still_over_threshold(self):
        node = _FakeNode(context_length=10)  # absurdly small window: nothing fits
        node.script_major(summary="x" * 200)
        node.running_turn_usage = _usage(requests=1, occupancy=9)
        compactor = MidTurnCompactor(node)
        raw1 = [_plain_user("current")]
        await compactor.rewrite_sdk_input(raw1)
        await compactor.rewrite_sdk_input(raw1 + [_call("c1"), _out("c1")])
        assert compactor.compactions == 1
        assert compactor._disabled is True


class TestRewriteNativeMessages:
    @pytest.mark.asyncio
    async def test_first_call_returns_none_and_anchors(self):
        node = _FakeNode()
        compactor = MidTurnCompactor(node)
        messages = [
            _anth_user("old"),
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            _anth_user("now"),
        ]
        assert await compactor.rewrite_native_messages(messages) is None
        assert compactor._turn_request is messages[2]
        node.compact_mid_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_second_call_returns_the_node_view(self):
        node = _FakeNode()
        node.script_major()
        node.running_turn_usage = _usage(requests=1, occupancy=950)
        compactor = MidTurnCompactor(node)
        request = _anth_user("now")
        first = [request]
        await compactor.rewrite_native_messages(first)
        second = first + [_anth_assistant_tool_use("t1"), _anth_tool_result("t1", "x" * 100)]
        view = await compactor.rewrite_native_messages(second)
        assert view == build_mid_turn_view(second, "SUMMARY", item_format="anthropic", turn_request=request)
        assert node.compact_mid_turn.await_args.kwargs["item_format"] == "anthropic"
        assert node.compact_mid_turn.await_args.kwargs["tail_start"] == 1
        assert compactor._prev_view_len == len(view)

    @pytest.mark.asyncio
    async def test_noop_returns_none(self):
        node = _FakeNode()
        compactor = MidTurnCompactor(node)
        first = [_anth_user("now")]
        await compactor.rewrite_native_messages(first)
        assert await compactor.rewrite_native_messages(first + [_anth_tool_result("t1")]) is None
