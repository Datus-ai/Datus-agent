# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/reasoning_replay.py.

Covers:
- Provider detection helpers (is_kimi_model / is_deepseek_model / reasoning_provider_family)
- should_replay_reasoning_content: the SDK replay hook decision matrix
"""

from types import SimpleNamespace

import pytest

from datus.models.reasoning_replay import (
    is_deepseek_model,
    is_kimi_model,
    is_reasoning_echo_provider,
    reasoning_endpoint_identity,
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

    @pytest.mark.parametrize("name", ["deepseek/deepseek-v4-pro", "deepseek-reasoner", "moonshot/kimi-k3", "kimi-k2.6"])
    def test_thinking_models_echo_reasoning(self, name):
        assert is_reasoning_echo_provider(name) is True

    @pytest.mark.parametrize(
        ("name", "family"),
        [
            ("deepseek/deepseek-chat", "deepseek"),
            ("deepseek-chat", "deepseek"),
            ("moonshot/moonshot-v1-8k", "kimi"),
            ("kimi-k2", "kimi"),
        ],
    )
    def test_known_non_thinking_models_do_not_echo_reasoning(self, name, family):
        """Vendor models on the non-thinking deny-list are still Kimi/DeepSeek, but never replay or pad."""
        assert reasoning_provider_family(name) == family
        assert is_reasoning_echo_provider(name) is False


def _context(model, origin_model=None, provider_data=None):
    base_url = "https://models.example.test/v1"
    if provider_data is None:
        provider_data = {"datus_reasoning_endpoint": reasoning_endpoint_identity(base_url)}
    return SimpleNamespace(
        model=model,
        base_url=base_url,
        reasoning=SimpleNamespace(item={}, origin_model=origin_model, provider_data=provider_data),
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

    def test_missing_provenance_never_replays(self):
        """Items without an origin model may come from another provider; they are never replayed."""
        assert should_replay_reasoning_content(_context("deepseek/deepseek-v4-pro", None, {})) is False
        assert should_replay_reasoning_content(_context("moonshot/kimi-k3", None, {})) is False
        assert should_replay_reasoning_content(_context("moonshot/kimi-k3", None, {"model": None})) is False

    @pytest.mark.parametrize("target", ["moonshot/moonshot-v1-8k", "moonshot/kimi-k2", "deepseek/deepseek-chat"])
    def test_non_thinking_targets_never_replay(self, target):
        origin = "moonshot/kimi-k2.5" if "moonshot" in target else "deepseek/deepseek-v4-pro"
        assert should_replay_reasoning_content(_context(target, origin)) is False

    @pytest.mark.parametrize(
        ("target", "origin"),
        [
            ("moonshot/kimi-k3", "moonshot/moonshot-v1-8k"),
            ("moonshot/kimi-k3", "kimi-k2"),
            ("deepseek/deepseek-v4-pro", "deepseek-chat"),
        ],
    )
    def test_non_thinking_origins_never_replay(self, target, origin):
        """A non-thinking model cannot have produced reasoning; same-family origin is not enough."""
        assert should_replay_reasoning_content(_context(target, origin)) is False
