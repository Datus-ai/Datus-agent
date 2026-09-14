# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pytest
from agents.models.chatcmpl_converter import Converter

from datus.models.litellm_model import normalize_kimi_tool_replay_items
from datus.models.reasoning_replay import (
    REASONING_ENDPOINT_KEY,
    reasoning_endpoint_identity,
    should_replay_reasoning_content,
)


def _call(call_id: str) -> dict:
    return {
        "type": "function_call",
        "name": "read_file",
        "arguments": "{}",
        "call_id": call_id,
    }


def _output(call_id: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": "done"}


def _empty_message() -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "", "annotations": []}],
    }


def _converted_messages(items: list[dict], model: str) -> list[dict]:
    normalized = normalize_kimi_tool_replay_items(items)
    return Converter.items_to_messages(normalized, model=model)


def test_removes_empty_message_after_parallel_kimi_calls():
    items = [_call("call_1"), _call("call_2"), _empty_message(), _output("call_1"), _output("call_2")]

    messages = _converted_messages(items, "moonshot/kimi-k3")

    assert [message["role"] for message in messages] == ["assistant", "tool", "tool"]
    assert [call["id"] for call in messages[0]["tool_calls"]] == ["call_1", "call_2"]


def test_removes_empty_message_between_parallel_kimi_calls():
    items = [_call("call_1"), _empty_message(), _call("call_2"), _output("call_1"), _output("call_2")]

    messages = _converted_messages(items, "moonshot/kimi-k2.6")

    assert [message["role"] for message in messages] == ["assistant", "tool", "tool"]
    assert [call["id"] for call in messages[0]["tool_calls"]] == ["call_1", "call_2"]


def test_preserves_empty_message_before_tool_calls():
    empty = _empty_message()
    items = [empty, _call("call_1"), _output("call_1")]

    assert normalize_kimi_tool_replay_items(items) is items


def test_moves_nonempty_assistant_message_before_pending_call():
    message = _empty_message()
    message["content"][0]["text"] = "I will read both files."
    items = [_call("call_1"), message, _output("call_1")]

    assert normalize_kimi_tool_replay_items(items) == [message, _call("call_1"), _output("call_1")]


def test_keeps_parallel_calls_with_nonempty_content_in_one_converted_message():
    message = _empty_message()
    message["content"][0]["text"] = "I will read both files."
    items = [
        {"type": "reasoning", "summary": []},
        _call("call_1"),
        message,
        _call("call_2"),
        _output("call_1"),
        _output("call_2"),
    ]

    messages = _converted_messages(items, "moonshot/kimi-k3")

    assert [item["role"] for item in messages] == ["assistant", "tool", "tool"]
    assert messages[0]["content"] == "I will read both files."
    assert [call["id"] for call in messages[0]["tool_calls"]] == ["call_1", "call_2"]


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("moonshot/kimi-k2.6", "https://api.moonshot.cn/v1"),
        ("moonshot/kimi-k3", "https://api.moonshot.cn/v1"),
        ("anthropic/kimi-for-coding", "https://api.kimi.com/coding/"),
    ],
)
def test_kimi_models_replay_reasoning_content_with_tool_call(model: str, base_url: str):
    items = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Inspect both sources."}],
            "provider_data": {
                "model": model,
                REASONING_ENDPOINT_KEY: reasoning_endpoint_identity(base_url),
            },
        },
        _call("call_1"),
        _output("call_1"),
    ]

    messages = Converter.items_to_messages(
        items,
        model=model,
        base_url=base_url,
        should_replay_reasoning_content=should_replay_reasoning_content,
    )

    assert messages[0]["reasoning_content"] == "Inspect both sources."
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_call_id"] == "call_1"


def test_preserves_non_sequence_input():
    assert normalize_kimi_tool_replay_items("hello") == "hello"
