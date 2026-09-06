# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/sdk_patches.py.

Covers:
- _normalize_text_content_blocks / _normalize_items: Chat-style text block normalization
- _extract_reasoning_content: robust reasoning extraction from provider payloads
- litellm.(a)completion wrappers: reasoning_content placeholders and Kimi empty-content recovery
- apply_sdk_patches / remove_sdk_patches: full lifecycle
"""

from types import SimpleNamespace

import pytest

from datus.models.sdk_patches import (
    _extract_reasoning_content,
    _normalize_items,
    _normalize_text_content_blocks,
    _recover_empty_kimi_content,
    apply_sdk_patches,
    remove_sdk_patches,
)


class TestNormalizeItems:
    def test_text_blocks_are_normalized_for_chat_completions_converter(self):
        """Session replay may contain Chat-style text blocks; SDK input expects input_text."""
        items = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        result_items = _normalize_items(items)

        assert result_items[0]["content"] == [{"type": "input_text", "text": "hello"}]
        assert items[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_assistant_response_message_text_blocks_are_normalized_to_output_text(self):
        """Response output messages expect output_text, not input_text."""
        items = [{"type": "message", "role": "assistant", "content": [{"type": "text", "text": "final answer"}]}]

        result_items = _normalize_items(items)

        assert result_items[0]["content"] == [{"type": "output_text", "text": "final answer"}]

    def test_string_items_returned_as_is(self):
        assert _normalize_items("plain prompt") == "plain prompt"


class TestNormalizeTextContentBlocks:
    def test_non_dict_item_is_returned_unchanged(self):
        item = object()
        assert _normalize_text_content_blocks(item) is item

    def test_item_without_text_blocks_is_returned_unchanged(self):
        item = {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        assert _normalize_text_content_blocks(item) is item

    def test_tool_output_text_blocks_are_normalized(self):
        item = {"type": "function_call_output", "output": [{"type": "text", "text": "tool result"}]}

        result = _normalize_text_content_blocks(item)

        assert result["output"] == [{"type": "input_text", "text": "tool result"}]
        assert item["output"] == [{"type": "text", "text": "tool result"}]


class TestReasoningContentExtraction:
    def test_extracts_from_dict_and_nested_provider_fields(self):
        value = {"provider_specific_fields": {"reasoning_content": "nested thought"}}
        assert _extract_reasoning_content(value) == "nested thought"

    def test_extracts_from_object_model_extra(self):
        class Value:
            model_extra = {"reasoning": {"text": "model-extra thought"}}

        assert _extract_reasoning_content(Value()) == "model-extra thought"

    def test_does_not_treat_normal_content_as_reasoning(self):
        value = {"content": "visible assistant text"}
        assert _extract_reasoning_content(value) is None


class TestRecoverEmptyKimiContent:
    def test_empty_content_is_replaced_by_reasoning(self):
        message = SimpleNamespace(content="", reasoning_content="hidden thought")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        _recover_empty_kimi_content(response)

        assert message.content == "hidden thought"

    def test_visible_content_is_kept(self):
        message = SimpleNamespace(content="answer", reasoning_content="hidden thought")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        _recover_empty_kimi_content(response)

        assert message.content == "answer"


class TestApplyAndRemoveSdkPatches:
    """Tests for apply_sdk_patches and remove_sdk_patches lifecycle."""

    @pytest.fixture(autouse=True)
    def unpatched_baseline(self):
        """Start every lifecycle test from the un-patched state.

        ``datus/models/__init__.py`` calls ``apply_sdk_patches()`` at import
        time, so by the time any test runs ``litellm.completion`` is already
        the patched wrapper and ``sdk_patches._original_*`` already hold the
        true originals. Tests here capture ``litellm.completion`` as "the true
        original" and re-apply, which only holds when nothing is patched yet.
        """
        remove_sdk_patches()
        yield
        remove_sdk_patches()

    def test_apply_and_remove_patches(self):
        """apply_sdk_patches and remove_sdk_patches complete without error."""
        import litellm

        original = litellm.completion
        apply_sdk_patches()
        assert litellm.completion is not original
        remove_sdk_patches()
        assert litellm.completion is original

    def test_patched_converter_accepts_session_text_blocks_for_deepseek(self):
        """Regression for DeepSeek session replay: Chat-style text blocks must not raise Unknown content."""
        from agents.models.chatcmpl_converter import Converter

        apply_sdk_patches()
        try:
            messages = Converter.items_to_messages(
                [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                model="deepseek/deepseek-v4-flash",
            )
            assert messages == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        finally:
            remove_sdk_patches()

    def test_patched_converter_replays_each_turns_own_reasoning_for_kimi(self):
        """With the replay hook, every assistant message carries the reasoning of its own turn only."""
        from agents.models.chatcmpl_converter import Converter

        from datus.models.reasoning_replay import should_replay_reasoning_content

        model = "moonshot/kimi-k2.6"
        items = [
            {"role": "user", "content": "hi"},
            {
                "id": "r1",
                "type": "reasoning",
                "summary": [{"text": "think-1", "type": "summary_text"}],
                "provider_data": {"model": model},
            },
            {
                "id": "c1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
                "provider_data": {"model": model},
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
            {
                "id": "c2",
                "type": "function_call",
                "call_id": "call_2",
                "name": "get_weather",
                "arguments": "{}",
                "provider_data": {"model": model},
            },
            {"type": "function_call_output", "call_id": "call_2", "output": "{}"},
            {
                "id": "r3",
                "type": "reasoning",
                "summary": [{"text": "think-3", "type": "summary_text"}],
                "provider_data": {"model": model},
            },
            {
                "id": "m3",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
                "provider_data": {"model": model},
            },
        ]

        apply_sdk_patches()
        try:
            messages = Converter.items_to_messages(
                items, model=model, should_replay_reasoning_content=should_replay_reasoning_content
            )
        finally:
            remove_sdk_patches()

        assistant = [m for m in messages if m.get("role") == "assistant"]
        assert [m.get("reasoning_content") for m in assistant] == ["think-1", None, "think-3"]

    def test_patched_completion_adds_placeholders_and_recovers_kimi_content(self, monkeypatch):
        """The sync wrapper fills reasoning_content gaps with '' and surfaces Kimi reasoning-only replies."""
        import litellm

        captured = {}
        reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", reasoning_content="why"))])

        def fake_completion(*args, **kwargs):
            captured["messages"] = kwargs["messages"]
            return reply

        monkeypatch.setattr(litellm, "completion", fake_completion)
        apply_sdk_patches()
        try:
            litellm.completion(
                model="moonshot/kimi-k2.6",
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}], "reasoning_content": "t1"},
                    {"role": "tool", "tool_call_id": "c1", "content": "{}"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "c2"}]},
                ],
            )
        finally:
            remove_sdk_patches()

        assert captured["messages"][3]["reasoning_content"] == ""
        assert captured["messages"][3]["content"] == ""
        assert captured["messages"][1]["reasoning_content"] == "t1"
        assert reply.choices[0].message.content == "why"

    def test_apply_patches_idempotent(self):
        """Calling apply_sdk_patches twice must not re-capture the already-patched
        litellm functions as 'originals'. Otherwise remove_sdk_patches() would
        restore the patched version instead of the true original.
        """
        import litellm

        from datus.models import sdk_patches

        true_original_completion = litellm.completion
        true_original_acompletion = litellm.acompletion

        apply_sdk_patches()
        captured_after_first = sdk_patches._original_completion
        captured_acompletion_after_first = sdk_patches._original_acompletion
        patched_completion_first = litellm.completion

        apply_sdk_patches()  # second call must be a no-op for capture
        try:
            assert sdk_patches._original_completion is true_original_completion
            assert sdk_patches._original_acompletion is true_original_acompletion
            assert sdk_patches._original_completion is captured_after_first
            assert sdk_patches._original_acompletion is captured_acompletion_after_first
            assert litellm.completion is patched_completion_first
        finally:
            remove_sdk_patches()

        assert litellm.completion is true_original_completion
        assert litellm.acompletion is true_original_acompletion
