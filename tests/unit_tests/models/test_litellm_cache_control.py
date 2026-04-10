# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``CacheControlLitellmModel`` and ``apply_cache_control``."""

from __future__ import annotations

import copy
from unittest.mock import patch

import litellm
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
async def test_cache_control_wrapper_applies_markers_when_flag_set():
    """The globally-installed wrapper injects cache markers when _apply_cache is True."""
    from datus.models.litellm_cache_control import _apply_cache

    captured: dict = {}

    async def fake_original(*args, **kwargs):
        captured.update(kwargs)
        return "ret"

    # Temporarily re-install wrapper around our fake
    import datus.models.litellm_cache_control as ccmod

    saved = litellm.acompletion
    litellm._datus_cache_control_installed = False  # type: ignore[attr-defined]
    litellm.acompletion = fake_original
    ccmod._install_cache_control_wrapper()
    wrapper_fn = litellm.acompletion

    try:
        # With flag set: should apply cache markers
        token = _apply_cache.set(True)
        try:
            await wrapper_fn(
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                tools=[{"name": "t"}],
            )
        finally:
            _apply_cache.reset(token)

        assert isinstance(captured["messages"][0]["content"], list)
        assert captured["messages"][0]["content"][-1]["cache_control"] == EPHEMERAL
        assert captured["messages"][-1]["content"][-1]["cache_control"] == EPHEMERAL
        assert captured["tools"][-1]["cache_control"] == EPHEMERAL

        # Without flag: should pass through unchanged
        captured.clear()
        await wrapper_fn(
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=[{"name": "t"}],
        )
        assert captured["messages"][0]["content"] == "sys"
        assert captured["messages"][-1]["content"] == "hi"
        assert "cache_control" not in captured["tools"][-1]
    finally:
        litellm.acompletion = saved
        litellm._datus_cache_control_installed = True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fetch_response_sets_flag_for_anthropic():
    """CacheControlLitellmModel sets _apply_cache=True for anthropic models."""
    from datus.models.litellm_cache_control import _apply_cache

    model = CacheControlLitellmModel(model="anthropic/claude-sonnet-4", api_key="sk-test")
    observed_flag: list = []

    async def fake_super_fetch(self_inner, *args, **kwargs):
        observed_flag.append(_apply_cache.get(False))
        return "ret"

    with patch(
        "agents.extensions.models.litellm_model.LitellmModel._fetch_response",
        new=fake_super_fetch,
    ):
        await model._fetch_response()

    assert observed_flag == [True]
    # After _fetch_response returns, flag should be reset
    assert _apply_cache.get(False) is False


@pytest.mark.asyncio
async def test_fetch_response_skips_flag_for_non_anthropic():
    """CacheControlLitellmModel does NOT set _apply_cache for non-anthropic models."""
    from datus.models.litellm_cache_control import _apply_cache

    model = CacheControlLitellmModel(model="openai/gpt-4", api_key="sk-test")
    observed_flag: list = []

    async def fake_super_fetch(self_inner, *args, **kwargs):
        observed_flag.append(_apply_cache.get(False))
        return "ret"

    with patch(
        "agents.extensions.models.litellm_model.LitellmModel._fetch_response",
        new=fake_super_fetch,
    ):
        await model._fetch_response()

    assert observed_flag == [False]


@pytest.mark.asyncio
async def test_context_var_isolation_between_tasks():
    """Concurrent tasks with different models should not interfere via ContextVar."""
    import asyncio

    from datus.models.litellm_cache_control import _apply_cache

    observed: dict = {}

    async def check_flag(label: str, expected: bool):
        await asyncio.sleep(0)
        observed[label] = _apply_cache.get(False)
        assert _apply_cache.get(False) == expected

    async def anthropic_task():
        token = _apply_cache.set(True)
        try:
            await check_flag("anthropic", True)
        finally:
            _apply_cache.reset(token)

    async def openai_task():
        await check_flag("openai", False)

    await asyncio.gather(anthropic_task(), openai_task())
    assert observed["anthropic"] is True
    assert observed["openai"] is False
