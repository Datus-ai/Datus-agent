# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/sdk_patches.py.

Tests cover:
- _is_kimi_model: detection of Kimi/Moonshot model names
- _normalize_provider_data: dict and Pydantic-style object normalization
- _preprocess_items_for_reasoning: model name normalization for reasoning
- _ReasoningContentStreamWrapper: streaming reasoning_content capture
- _postprocess_messages_for_reasoning: reasoning_content injection into messages
- apply_sdk_patches / remove_sdk_patches: full lifecycle

NO MOCK EXCEPT LLM. All objects are real.
"""

import copy

import pytest

from datus.models.sdk_patches import (
    _is_kimi_model,
    _normalize_provider_data,
    _postprocess_messages_for_reasoning,
    _preprocess_items_for_reasoning,
    _reasoning_content_cache,
    _ReasoningContentStreamWrapper,
    apply_sdk_patches,
    remove_sdk_patches,
)


class TestIsKimiModel:
    """Tests for _is_kimi_model detection."""

    def test_kimi_model_detected(self):
        """Kimi model names are correctly detected."""
        assert _is_kimi_model("kimi-1.5") is True
        assert _is_kimi_model("moonshot-v1") is True
        assert _is_kimi_model("k2.5-large") is True
        assert _is_kimi_model("k2-base") is True

    def test_non_kimi_model_not_detected(self):
        """Non-Kimi model names return False."""
        assert _is_kimi_model("gpt-4") is False
        assert _is_kimi_model("deepseek-chat") is False
        assert _is_kimi_model("claude-3") is False
        assert _is_kimi_model("") is False


class TestNormalizeProviderData:
    """Tests for _normalize_provider_data with dict and object items."""

    def test_dict_item_with_kimi_model_is_normalized(self):
        """Dict item with Kimi provider_data.model gets 'deepseek-' prefix."""
        item = {"provider_data": {"model": "kimi-1.5", "other": "data"}, "content": "hello"}
        result = _normalize_provider_data(item)
        assert result["provider_data"]["model"] == "deepseek-kimi-1.5"
        assert result["content"] == "hello"
        # Original should not be modified
        assert item["provider_data"]["model"] == "kimi-1.5"

    def test_dict_item_without_provider_data_unchanged(self):
        """Dict item without provider_data is returned unchanged."""
        item = {"content": "hello", "role": "user"}
        result = _normalize_provider_data(item)
        assert result is item

    def test_dict_item_with_non_kimi_model_unchanged(self):
        """Dict item with non-Kimi provider_data.model is returned unchanged."""
        item = {"provider_data": {"model": "gpt-4"}, "content": "hello"}
        result = _normalize_provider_data(item)
        assert result is item

    def test_dict_item_with_none_provider_data_unchanged(self):
        """Dict item with None provider_data is returned unchanged."""
        item = {"provider_data": None, "content": "hello"}
        result = _normalize_provider_data(item)
        assert result is item

    def test_dict_item_with_non_dict_provider_data_unchanged(self):
        """Dict item with non-dict provider_data is returned unchanged."""
        item = {"provider_data": "string_value", "content": "hello"}
        result = _normalize_provider_data(item)
        assert result is item

    def test_object_item_with_kimi_model_is_normalized(self):
        """Pydantic-style object with Kimi provider_data is normalized."""

        class FakeItem:
            def __init__(self):
                self.provider_data = {"model": "moonshot-v1"}
                self.content = "hello"

            def model_copy(self, deep=False):
                new = copy.deepcopy(self)
                return new

        item = FakeItem()
        result = _normalize_provider_data(item)
        assert result.provider_data["model"] == "deepseek-moonshot-v1"
        # Original should not be modified
        assert item.provider_data["model"] == "moonshot-v1"

    def test_object_item_with_copy_method(self):
        """Object with copy() but not model_copy() is handled."""

        class FakeItem:
            def __init__(self):
                self.provider_data = {"model": "k2.5-large"}
                self.content = "hello"

            def copy(self, deep=False):
                return copy.deepcopy(self)

        item = FakeItem()
        result = _normalize_provider_data(item)
        assert result.provider_data["model"] == "deepseek-k2.5-large"

    def test_object_item_without_copy_methods(self):
        """Object without model_copy/copy falls back to deepcopy."""

        class FakeItem:
            def __init__(self):
                self.provider_data = {"model": "k2-base"}
                self.content = "hello"

        item = FakeItem()
        result = _normalize_provider_data(item)
        assert result.provider_data["model"] == "deepseek-k2-base"

    def test_object_item_without_provider_data_unchanged(self):
        """Object without provider_data attribute is returned unchanged."""

        class FakeItem:
            content = "hello"

        item = FakeItem()
        result = _normalize_provider_data(item)
        assert result is item

    def test_object_item_with_non_kimi_model_unchanged(self):
        """Object with non-Kimi provider_data is returned unchanged."""

        class FakeItem:
            provider_data = {"model": "gpt-4"}

        item = FakeItem()
        result = _normalize_provider_data(item)
        assert result is item


class TestPreprocessItemsForReasoning:
    """Tests for _preprocess_items_for_reasoning."""

    def test_kimi_model_name_is_normalized(self):
        """Kimi model name gets 'deepseek-' prefix."""
        items = [{"provider_data": {"model": "kimi-1.5"}, "content": "hi"}]
        result_items, result_model = _preprocess_items_for_reasoning(items, "kimi-1.5")
        assert result_model == "deepseek-kimi-1.5"
        assert result_items[0]["provider_data"]["model"] == "deepseek-kimi-1.5"

    def test_non_kimi_model_unchanged(self):
        """Non-Kimi model is returned unchanged."""
        items = [{"content": "hi"}]
        result_items, result_model = _preprocess_items_for_reasoning(items, "gpt-4")
        assert result_model == "gpt-4"

    def test_string_items_returned_as_is(self):
        """String items are returned without modification."""
        result_items, result_model = _preprocess_items_for_reasoning("hello", "kimi-1.5")
        assert result_items == "hello"
        assert result_model == "deepseek-kimi-1.5"

    def test_none_model(self):
        """None model is handled gracefully."""
        result_items, result_model = _preprocess_items_for_reasoning([], None)
        assert result_model is None


class TestReasoningContentStreamWrapper:
    """Tests for _ReasoningContentStreamWrapper async iterator."""

    @pytest.mark.asyncio
    async def test_wrapper_captures_reasoning_content(self):
        """Stream wrapper captures reasoning_content from delta chunks."""
        _reasoning_content_cache.clear()

        class FakeChoice:
            def __init__(self, rc):
                self.delta = type("Delta", (), {"reasoning_content": rc})()

        class FakeChunk:
            def __init__(self, rc):
                self.choices = [FakeChoice(rc)]

        class FakeStream:
            def __init__(self):
                self._chunks = iter([FakeChunk("Step 1."), FakeChunk(" Step 2.")])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._chunks)
                except StopIteration:
                    raise StopAsyncIteration

        wrapper = _ReasoningContentStreamWrapper(FakeStream(), "kimi-test-model")
        chunks = []
        async for chunk in wrapper:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "kimi-test-model" in _reasoning_content_cache
        assert _reasoning_content_cache["kimi-test-model"] == "Step 1. Step 2."

    @pytest.mark.asyncio
    async def test_wrapper_no_reasoning_content_no_cache(self):
        """Stream wrapper does not cache if no reasoning_content in chunks."""
        _reasoning_content_cache.clear()

        class FakeChunk:
            choices = []

        class FakeStream:
            def __init__(self):
                self._chunks = iter([FakeChunk()])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._chunks)
                except StopIteration:
                    raise StopAsyncIteration

        wrapper = _ReasoningContentStreamWrapper(FakeStream(), "kimi-nocache")
        async for _ in wrapper:
            pass

        assert "kimi-nocache" not in _reasoning_content_cache

    @pytest.mark.asyncio
    async def test_wrapper_exception_in_choice_processing_is_ignored(self):
        """Stream wrapper ignores exceptions when processing choices."""
        _reasoning_content_cache.clear()

        class BadChunk:
            """A chunk where accessing choices raises an error."""

            @property
            def choices(self):
                raise AttributeError("No choices")

        class FakeStream:
            def __init__(self):
                self._chunks = iter([BadChunk()])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._chunks)
                except StopIteration:
                    raise StopAsyncIteration

        wrapper = _ReasoningContentStreamWrapper(FakeStream(), "kimi-bad")
        chunks = []
        async for chunk in wrapper:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "kimi-bad" not in _reasoning_content_cache

    def test_wrapper_getattr_delegates(self):
        """__getattr__ delegates to the underlying stream."""

        class FakeStream:
            custom_attr = "hello"

        wrapper = _ReasoningContentStreamWrapper(FakeStream(), "model")
        assert wrapper.custom_attr == "hello"


class TestPostprocessMessagesForReasoning:
    """Tests for _postprocess_messages_for_reasoning."""

    def test_non_kimi_model_returns_unchanged(self):
        """Non-Kimi model messages are returned unchanged."""
        messages = [{"role": "assistant", "tool_calls": [{"id": "1"}]}]
        result = _postprocess_messages_for_reasoning(messages, "gpt-4")
        assert result is messages

    def test_none_model_returns_unchanged(self):
        """None model returns messages unchanged."""
        messages = [{"role": "user", "content": "hi"}]
        result = _postprocess_messages_for_reasoning(messages, None)
        assert result is messages

    def test_kimi_injects_reasoning_content(self):
        """Kimi model injects reasoning_content into assistant+tool_calls messages."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should think...",
                "tool_calls": [{"id": "1"}],
            },
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
        ]
        result = _postprocess_messages_for_reasoning(messages, "kimi-1.5")
        # First message already has reasoning_content
        assert result[0]["reasoning_content"] == "I should think..."
        # Second assistant message should get injected reasoning_content
        assert result[2]["reasoning_content"] == "I should think..."
        # content=None should be set to ""
        assert result[0]["content"] == ""
        assert result[2]["content"] == ""

    def test_kimi_uses_cached_reasoning_content(self):
        """When no reasoning_content in messages, uses cached value."""
        _reasoning_content_cache.clear()
        _reasoning_content_cache["kimi-cached"] = "Cached thinking..."

        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ]
        result = _postprocess_messages_for_reasoning(messages, "kimi-cached")
        assert result[0]["reasoning_content"] == "Cached thinking..."

        _reasoning_content_cache.clear()

    def test_kimi_empty_reasoning_when_no_source(self):
        """When no reasoning_content anywhere, sets empty string."""
        _reasoning_content_cache.clear()

        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ]
        result = _postprocess_messages_for_reasoning(messages, "kimi-norcache")
        assert result[0]["reasoning_content"] == ""

        _reasoning_content_cache.clear()


class TestApplyAndRemoveSdkPatches:
    """Tests for apply_sdk_patches and remove_sdk_patches lifecycle."""

    def test_apply_and_remove_patches(self):
        """apply_sdk_patches and remove_sdk_patches complete without error."""
        # Apply patches
        apply_sdk_patches()

        # Verify patches were applied by checking the cache was not affected
        # (patches modify class methods, hard to test without full integration)

        # Remove patches
        remove_sdk_patches()

        # After removal, _reasoning_content_cache should be cleared
        assert len(_reasoning_content_cache) == 0

    def test_remove_patches_clears_cache(self):
        """remove_sdk_patches clears the reasoning_content_cache."""
        _reasoning_content_cache["test-model"] = "test content"
        remove_sdk_patches()
        assert len(_reasoning_content_cache) == 0

    def test_apply_patches_idempotent(self):
        """Calling apply_sdk_patches twice does not raise."""
        apply_sdk_patches()
        apply_sdk_patches()  # Should not raise
        remove_sdk_patches()
