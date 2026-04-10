# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``CacheControlLitellmModel`` and ``apply_cache_control``."""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock, patch

import pytest

from datus.models.litellm_cache_control import (
    CacheControlLitellmModel,
    apply_cache_control,
)

EPHEMERAL = {"type": "ephemeral"}


def test_apply_cache_control_tags_system_user_and_tool():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    ]
    tools = [{"name": "t1"}, {"name": "t2"}]

    new_messages, new_tools = apply_cache_control(messages, tools)

    # System prompt: wrapped to list, last block tagged
    assert isinstance(new_messages[0]["content"], list)
    assert new_messages[0]["content"][-1]["cache_control"] == EPHEMERAL
    assert new_messages[0]["content"][-1]["text"] == "you are helpful"

    # Last user message: last block tagged, others not
    last_user_blocks = new_messages[3]["content"]
    assert last_user_blocks[-1]["cache_control"] == EPHEMERAL
    assert "cache_control" not in last_user_blocks[0]

    # Earlier user message not tagged
    assert new_messages[1]["content"] == "hi"

    # Last tool tagged
    assert new_tools[-1]["cache_control"] == EPHEMERAL
    assert "cache_control" not in new_tools[0]


def test_apply_cache_control_deepcopy_safety():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
    ]
    tools = [{"name": "x"}]
    original_messages = copy.deepcopy(messages)
    original_tools = copy.deepcopy(tools)

    apply_cache_control(messages, tools)

    assert messages == original_messages
    assert tools == original_tools


def test_apply_cache_control_handles_empty_inputs():
    new_messages, new_tools = apply_cache_control(None, None)
    assert new_messages is None and new_tools is None

    new_messages, new_tools = apply_cache_control([], [])
    assert new_messages == [] and new_tools == []


def test_apply_cache_control_tool_message_tagged():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "result"},
    ]
    new_messages, _ = apply_cache_control(messages, None)
    # Last tool message last block tagged
    assert new_messages[-1]["content"][-1]["cache_control"] == EPHEMERAL


@pytest.mark.asyncio
async def test_cache_control_model_anthropic_patches_acompletion():
    model = CacheControlLitellmModel(model="anthropic/claude-sonnet-4", api_key="sk-test")

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return "ret"

    # Patch parent _fetch_response to invoke litellm.acompletion with known payload
    async def fake_super_fetch(self_inner, *args, **kwargs):
        import litellm

        return await litellm.acompletion(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            tools=[{"name": "t"}],
        )

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        with patch(
            "agents.extensions.models.litellm_model.LitellmModel._fetch_response",
            new=fake_super_fetch,
        ):
            await model._fetch_response()

    assert isinstance(captured["messages"][0]["content"], list)
    assert captured["messages"][0]["content"][-1]["cache_control"] == EPHEMERAL
    assert captured["messages"][-1]["content"][-1]["cache_control"] == EPHEMERAL
    assert captured["tools"][-1]["cache_control"] == EPHEMERAL


@pytest.mark.asyncio
async def test_cache_control_model_non_anthropic_passthrough():
    model = CacheControlLitellmModel(model="openai/gpt-4", api_key="sk-test")

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return "ret"

    async def fake_super_fetch(self_inner, *args, **kwargs):
        import litellm

        return await litellm.acompletion(
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=[{"name": "t"}],
        )

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        with patch(
            "agents.extensions.models.litellm_model.LitellmModel._fetch_response",
            new=fake_super_fetch,
        ):
            await model._fetch_response()

    # No cache_control anywhere
    assert captured["messages"][0]["content"] == "sys"
    assert captured["messages"][-1]["content"] == "hi"
    assert "cache_control" not in captured["tools"][-1]
