# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/sdk_patches.py.

Tests cover:
- _is_kimi_model / _is_deepseek_model / _needs_reasoning_injection: provider detection
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
    _is_deepseek_model,
    _is_kimi_model,
    _needs_reasoning_injection,
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


class TestIsDeepSeekModel:
    """Tests for _is_deepseek_model detection."""

    def test_deepseek_model_detected(self):
        """DeepSeek model names are correctly detected across aliases and litellm prefixes."""
        assert _is_deepseek_model("deepseek-chat") is True
        assert _is_deepseek_model("deepseek-reasoner") is True
        assert _is_deepseek_model("deepseek-v4") is True
        assert _is_deepseek_model("deepseek/deepseek-v4") is True
        assert _is_deepseek_model("DeepSeek-V4") is True

    def test_non_deepseek_model_not_detected(self):
        """Other providers and empty strings return False."""
        assert _is_deepseek_model("kimi-k2.5") is False
        assert _is_deepseek_model("gpt-4") is False
        assert _is_deepseek_model("claude-sonnet-4-5") is False
        assert _is_deepseek_model("") is False
        assert _is_deepseek_model(None) is False  # type: ignore[arg-type]


class TestNeedsReasoningInjection:
    """Tests for _needs_reasoning_injection combined gate."""

    def test_returns_true_for_kimi_and_deepseek(self):
        assert _needs_reasoning_injection("kimi-k2.5") is True
        assert _needs_reasoning_injection("moonshot-v1") is True
        assert _needs_reasoning_injection("deepseek-v4") is True
        assert _needs_reasoning_injection("deepseek/deepseek-reasoner") is True

    def test_returns_false_for_other_providers(self):
        assert _needs_reasoning_injection("gpt-4") is False
        assert _needs_reasoning_injection("claude-sonnet-4-5") is False
        assert _needs_reasoning_injection("") is False
        assert _needs_reasoning_injection(None) is False  # type: ignore[arg-type]


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

    def test_deepseek_injects_reasoning_content_from_cache(self):
        """DeepSeek model uses cached reasoning_content as fallback for tool_calls messages."""
        _reasoning_content_cache.clear()
        _reasoning_content_cache["deepseek/deepseek-v4"] = "cached deepseek thinking"

        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
        ]
        result = _postprocess_messages_for_reasoning(messages, "deepseek/deepseek-v4")
        assert result[0]["reasoning_content"] == "cached deepseek thinking"
        assert result[2]["reasoning_content"] == "cached deepseek thinking"
        # content=None must be normalized to "" so the provider accepts tool_calls-only messages
        assert result[0]["content"] == ""
        assert result[2]["content"] == ""

        _reasoning_content_cache.clear()

    def test_deepseek_does_not_inject_empty_placeholder(self):
        """DeepSeek must NOT get an empty reasoning_content placeholder when no source exists.

        DeepSeek's API rejects empty reasoning_content in thinking mode with the same error
        we're trying to fix; Kimi tolerates it. Only Kimi gets the "" fallback.
        """
        _reasoning_content_cache.clear()

        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ]
        result = _postprocess_messages_for_reasoning(messages, "deepseek-v4")
        # Key should NOT be present (leave the message alone)
        assert "reasoning_content" not in result[0]

        _reasoning_content_cache.clear()

    def test_deepseek_reuses_existing_reasoning_content_in_messages(self):
        """Existing non-empty reasoning_content in messages is propagated to later tool_calls msgs."""
        _reasoning_content_cache.clear()

        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "first turn thinking",
                "tool_calls": [{"id": "1"}],
            },
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
        ]
        result = _postprocess_messages_for_reasoning(messages, "deepseek-reasoner")
        assert result[0]["reasoning_content"] == "first turn thinking"
        assert result[2]["reasoning_content"] == "first turn thinking"

        _reasoning_content_cache.clear()

    def test_deepseek_injects_reasoning_content_into_final_assistant_message(self):
        """DeepSeek V4 Pro requires final assistant messages from tool turns to keep reasoning_content."""
        _reasoning_content_cache.clear()
        _reasoning_content_cache["deepseek/deepseek-v4-pro"] = "final answer thinking"

        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Final answer after tool."},
            {"role": "user", "content": "next question"},
        ]
        result = _postprocess_messages_for_reasoning(messages, "deepseek/deepseek-v4-pro")

        assert result[0]["reasoning_content"] == "final answer thinking"
        assert result[0]["content"] == ""
        assert result[2]["reasoning_content"] == "final answer thinking"
        assert result[2]["content"] == "Final answer after tool."

        _reasoning_content_cache.clear()

    def test_kimi_does_not_inject_reasoning_content_into_final_assistant_message(self):
        """Kimi keeps the historical narrower assistant+tool_calls patch scope."""
        _reasoning_content_cache.clear()
        _reasoning_content_cache["kimi-k2.5"] = "cached thinking"

        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Final answer after tool."},
        ]
        result = _postprocess_messages_for_reasoning(messages, "kimi-k2.5")

        assert result[0]["reasoning_content"] == "cached thinking"
        assert "reasoning_content" not in result[2]

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
        """Calling apply_sdk_patches twice must not re-capture the already-patched
        litellm functions as 'originals'. Otherwise remove_sdk_patches() would
        restore the patched version instead of the true original.
        """
        import litellm

        from datus.models import sdk_patches

        # Capture the true originals before any patching.
        true_original_completion = litellm.completion
        true_original_acompletion = litellm.acompletion

        apply_sdk_patches()
        captured_after_first = sdk_patches._original_completion
        captured_acompletion_after_first = sdk_patches._original_acompletion
        patched_completion_first = litellm.completion

        apply_sdk_patches()  # second call must be a no-op for capture
        try:
            # The stored "original" must still be the pre-patch function,
            # not the patched wrapper captured on the first call.
            assert sdk_patches._original_completion is true_original_completion
            assert sdk_patches._original_acompletion is true_original_acompletion
            assert sdk_patches._original_completion is captured_after_first
            assert sdk_patches._original_acompletion is captured_acompletion_after_first
            # The live litellm.completion should still be the patched wrapper
            # (not re-wrapped into a double-patched function).
            assert litellm.completion is patched_completion_first
        finally:
            remove_sdk_patches()

        # After removal, litellm.completion is restored to the true original.
        assert litellm.completion is true_original_completion
        assert litellm.acompletion is true_original_acompletion


class TestPatchedCompletionSync:
    """Tests for the sync litellm.completion patch (Kimi reasoning_content)."""

    def _make_fake_response(self, content="", reasoning_content=None):
        """Create a fake litellm response object."""

        class FakeMessage:
            pass

        msg = FakeMessage()
        msg.content = content
        if reasoning_content is not None:
            msg.reasoning_content = reasoning_content

        class FakeChoice:
            pass

        choice = FakeChoice()
        choice.message = msg

        class FakeResponse:
            pass

        resp = FakeResponse()
        resp.choices = [choice]
        return resp

    def test_patched_completion_caches_kimi_reasoning_content(self):
        """Patched litellm.completion caches reasoning_content for Kimi models."""
        import litellm

        # Ensure clean state
        remove_sdk_patches()
        _reasoning_content_cache.clear()

        fake_response = self._make_fake_response(content="", reasoning_content="Thinking step by step...")
        original_real = litellm.completion
        litellm.completion = lambda *args, **kwargs: fake_response

        try:
            apply_sdk_patches()
            litellm.completion(model="kimi-test-sync", messages=[{"role": "user", "content": "hi"}])

            assert "kimi-test-sync" in _reasoning_content_cache
            assert _reasoning_content_cache["kimi-test-sync"] == "Thinking step by step..."
            # Empty content should be replaced with reasoning_content
            assert fake_response.choices[0].message.content == "Thinking step by step..."
        finally:
            remove_sdk_patches()
            litellm.completion = original_real
            _reasoning_content_cache.clear()

    def test_patched_completion_non_kimi_skips_caching(self):
        """Patched litellm.completion skips reasoning_content caching for non-Kimi models."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        fake_response = self._make_fake_response(content="Hello", reasoning_content="Some reasoning")
        original_real = litellm.completion
        litellm.completion = lambda *args, **kwargs: fake_response

        try:
            apply_sdk_patches()
            litellm.completion(model="gpt-4", messages=[{"role": "user", "content": "hi"}])

            assert "gpt-4" not in _reasoning_content_cache
        finally:
            remove_sdk_patches()
            litellm.completion = original_real
            _reasoning_content_cache.clear()

    def test_patched_completion_preserves_non_empty_content(self):
        """Patched litellm.completion does not overwrite non-empty content."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        fake_response = self._make_fake_response(content="Real answer", reasoning_content="Thinking...")
        original_real = litellm.completion
        litellm.completion = lambda *args, **kwargs: fake_response

        try:
            apply_sdk_patches()
            litellm.completion(model="kimi-keep-content", messages=[{"role": "user", "content": "hi"}])

            assert _reasoning_content_cache["kimi-keep-content"] == "Thinking..."
            # Non-empty content should NOT be overwritten
            assert fake_response.choices[0].message.content == "Real answer"
        finally:
            remove_sdk_patches()
            litellm.completion = original_real
            _reasoning_content_cache.clear()

    def test_patched_completion_caches_deepseek_reasoning_content(self):
        """Patched litellm.completion caches reasoning_content for DeepSeek models."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        fake_response = self._make_fake_response(content="Real answer", reasoning_content="DeepSeek thinking...")
        original_real = litellm.completion
        litellm.completion = lambda *args, **kwargs: fake_response

        try:
            apply_sdk_patches()
            litellm.completion(model="deepseek/deepseek-v4", messages=[{"role": "user", "content": "hi"}])

            assert "deepseek/deepseek-v4" in _reasoning_content_cache
            assert _reasoning_content_cache["deepseek/deepseek-v4"] == "DeepSeek thinking..."
            # DeepSeek: content must NOT be overwritten even if it were empty — the API returns real content
            assert fake_response.choices[0].message.content == "Real answer"
        finally:
            remove_sdk_patches()
            litellm.completion = original_real
            _reasoning_content_cache.clear()

    def test_patched_completion_deepseek_does_not_overwrite_empty_content(self):
        """For DeepSeek, even when response.content is empty, we do NOT inject reasoning_content into it."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        fake_response = self._make_fake_response(content="", reasoning_content="DeepSeek thinking...")
        original_real = litellm.completion
        litellm.completion = lambda *args, **kwargs: fake_response

        try:
            apply_sdk_patches()
            litellm.completion(model="deepseek-reasoner", messages=[{"role": "user", "content": "hi"}])

            assert _reasoning_content_cache["deepseek-reasoner"] == "DeepSeek thinking..."
            # DeepSeek path must preserve the empty content — only Kimi rewrites it
            assert fake_response.choices[0].message.content == ""
        finally:
            remove_sdk_patches()
            litellm.completion = original_real
            _reasoning_content_cache.clear()

    def test_patched_completion_handles_exception_in_caching_logs_debug(self):
        """Patched litellm.completion logs debug message when caching fails."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        class BrokenMessage:
            content = "answer"

            @property
            def reasoning_content(self):
                raise RuntimeError("Broken attribute access")

        class BrokenChoice:
            message = BrokenMessage()

        class BrokenResponse:
            choices = [BrokenChoice()]

        original_real = litellm.completion
        litellm.completion = lambda *args, **kwargs: BrokenResponse()

        try:
            apply_sdk_patches()
            # Should not raise despite broken reasoning_content property
            result = litellm.completion(model="kimi-broken", messages=[{"role": "user", "content": "hi"}])
            assert result.choices[0].message.content == "answer"
            # Cache should not contain the broken model
            assert "kimi-broken" not in _reasoning_content_cache
        finally:
            remove_sdk_patches()
            litellm.completion = original_real
            _reasoning_content_cache.clear()


class TestPatchedAcompletionAsync:
    """Tests for the async litellm.acompletion patch (Kimi + DeepSeek reasoning_content)."""

    def _make_fake_response(self, content="", reasoning_content=None):
        class FakeMessage:
            pass

        msg = FakeMessage()
        msg.content = content
        if reasoning_content is not None:
            msg.reasoning_content = reasoning_content

        class FakeChoice:
            pass

        choice = FakeChoice()
        choice.message = msg

        class FakeResponse:
            pass

        resp = FakeResponse()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_patched_acompletion_caches_deepseek_reasoning_content(self):
        """Patched async litellm.acompletion caches reasoning_content for DeepSeek models."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        fake_response = self._make_fake_response(content="Answer", reasoning_content="DeepSeek thought")
        original_real = litellm.acompletion

        async def fake_acompletion(*args, **kwargs):
            return fake_response

        litellm.acompletion = fake_acompletion

        try:
            apply_sdk_patches()
            result = await litellm.acompletion(model="deepseek-v4", messages=[{"role": "user", "content": "hi"}])

            assert result is fake_response
            assert _reasoning_content_cache["deepseek-v4"] == "DeepSeek thought"
        finally:
            remove_sdk_patches()
            litellm.acompletion = original_real
            _reasoning_content_cache.clear()

    @pytest.mark.asyncio
    async def test_patched_acompletion_non_thinking_provider_skips_caching(self):
        """Async patch must skip caching for providers outside the thinking-injection set."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        fake_response = self._make_fake_response(content="Answer", reasoning_content="Something")
        original_real = litellm.acompletion

        async def fake_acompletion(*args, **kwargs):
            return fake_response

        litellm.acompletion = fake_acompletion

        try:
            apply_sdk_patches()
            await litellm.acompletion(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
            assert "gpt-4" not in _reasoning_content_cache
        finally:
            remove_sdk_patches()
            litellm.acompletion = original_real
            _reasoning_content_cache.clear()

    @pytest.mark.asyncio
    async def test_patched_acompletion_handles_exception_in_caching(self):
        """Patched litellm.acompletion logs debug when caching fails."""
        import litellm

        remove_sdk_patches()
        _reasoning_content_cache.clear()

        class BrokenMessage:
            content = "answer"

            @property
            def reasoning_content(self):
                raise RuntimeError("Broken attribute access")

        class BrokenChoice:
            message = BrokenMessage()

        class BrokenResponse:
            choices = [BrokenChoice()]

        original_real = litellm.acompletion

        async def fake_acompletion(*args, **kwargs):
            return BrokenResponse()

        litellm.acompletion = fake_acompletion

        try:
            apply_sdk_patches()
            # Should not raise despite broken reasoning_content property
            result = await litellm.acompletion(model="kimi-broken-async", messages=[{"role": "user", "content": "hi"}])
            assert result.choices[0].message.content == "answer"
            assert "kimi-broken-async" not in _reasoning_content_cache
        finally:
            remove_sdk_patches()
            litellm.acompletion = original_real
            _reasoning_content_cache.clear()
