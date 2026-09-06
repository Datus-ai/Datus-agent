# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/reasoning_replay.py.

Covers:
- Provider detection helpers (is_kimi_model / is_deepseek_model / reasoning_provider_family)
- should_replay_reasoning_content: the SDK replay hook decision matrix
- ensure_reasoning_content_placeholders: empty-string placeholders, never cross-turn copies
"""

from types import SimpleNamespace

import pytest

from datus.models.reasoning_replay import (
    ensure_reasoning_content_placeholders,
    is_deepseek_model,
    is_kimi_model,
    is_reasoning_echo_provider,
    reasoning_provider_family,
    should_replay_reasoning_content,
)


class TestProviderDetection:
    @pytest.mark.parametrize(
        "name", ["kimi-k2.5", "moonshot/kimi-k2.6", "moonshot-v1-8k", "kimi-k2-thinking", "k2.5-x"]
    )
    def test_kimi_models(self, name):
        assert is_kimi_model(name) is True
        assert reasoning_provider_family(name) == "kimi"

    @pytest.mark.parametrize("name", ["deepseek/deepseek-v4-pro", "deepseek-chat", "DeepSeek-Reasoner"])
    def test_deepseek_models(self, name):
        assert is_deepseek_model(name) is True
        assert reasoning_provider_family(name) == "deepseek"

    @pytest.mark.parametrize("name", ["gpt-5.4", "anthropic/claude-sonnet-5", "qwen-max", "", None])
    def test_other_models(self, name):
        assert is_kimi_model(name) is False
        assert is_deepseek_model(name) is False
        assert reasoning_provider_family(name) is None
        assert is_reasoning_echo_provider(name) is False


def _context(model, origin_model=None, provider_data=None):
    return SimpleNamespace(
        model=model,
        base_url=None,
        reasoning=SimpleNamespace(item={}, origin_model=origin_model, provider_data=provider_data or {}),
    )


class TestShouldReplayReasoningContent:
    def test_deepseek_target_with_deepseek_origin(self):
        assert should_replay_reasoning_content(_context("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro")) is True

    def test_kimi_target_with_kimi_origin(self):
        assert should_replay_reasoning_content(_context("moonshot/kimi-k2.6", "moonshot/kimi-k2.6")) is True

    def test_cross_family_origin_is_not_replayed(self):
        assert should_replay_reasoning_content(_context("deepseek/deepseek-v4-pro", "moonshot/kimi-k2.6")) is False
        assert should_replay_reasoning_content(_context("moonshot/kimi-k2.6", "deepseek/deepseek-v4-pro")) is False

    def test_non_echo_target_never_replays(self):
        assert should_replay_reasoning_content(_context("gpt-5.4", "gpt-5.4")) is False
        assert should_replay_reasoning_content(_context("anthropic/claude-sonnet-5", None)) is False

    def test_untracked_origin_is_trusted_only_without_provider_data(self):
        assert should_replay_reasoning_content(_context("deepseek/deepseek-v4-pro", None, {})) is True
        assert should_replay_reasoning_content(_context("moonshot/kimi-k2.6", None, {})) is True
        assert (
            should_replay_reasoning_content(_context("deepseek/deepseek-v4-pro", None, {"model": None, "x": 1}))
            is False
        )


class TestEnsureReasoningContentPlaceholders:
    def _history(self):
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
                "reasoning_content": "turn-1 thought",
            },
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "{}"},
            {"role": "assistant", "content": "final answer"},
        ]

    def test_non_echo_provider_untouched(self):
        messages = self._history()
        result = ensure_reasoning_content_placeholders(messages, "gpt-5.4")
        assert result is messages
        assert "reasoning_content" not in messages[4]
        assert messages[4]["content"] is None

    def test_placeholders_are_empty_strings_never_other_turns_reasoning(self):
        messages = self._history()
        ensure_reasoning_content_placeholders(messages, "deepseek/deepseek-v4-pro")

        assert messages[2]["reasoning_content"] == "turn-1 thought"
        assert messages[4]["reasoning_content"] == ""
        assert messages[6]["reasoning_content"] == ""
        assert all("reasoning_content" not in m for m in messages if m["role"] != "assistant")

    def test_tool_call_content_none_becomes_empty_string(self):
        messages = self._history()
        ensure_reasoning_content_placeholders(messages, "moonshot/kimi-k2.6")
        assert messages[2]["content"] == ""
        assert messages[4]["content"] == ""
        assert messages[6]["content"] == "final answer"

    def test_no_reasoning_anywhere_means_not_thinking_mode(self):
        messages = self._history()
        del messages[2]["reasoning_content"]
        ensure_reasoning_content_placeholders(messages, "deepseek/deepseek-v4-pro")

        assert all("reasoning_content" not in m for m in messages)
        assert messages[2]["content"] == ""

    def test_non_list_payload_untouched(self):
        assert ensure_reasoning_content_placeholders("prompt", "deepseek/deepseek-v4-pro") == "prompt"
        assert ensure_reasoning_content_placeholders(None, "deepseek/deepseek-v4-pro") is None
