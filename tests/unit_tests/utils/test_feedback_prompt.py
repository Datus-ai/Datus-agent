# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/feedback_prompt.py."""

from datus.utils.feedback_prompt import build_reaction_feedback_prompt


class TestBuildReactionFeedbackPrompt:
    def test_basic_format(self):
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="thumbsup",
            reference_msg="Sales for Q1 total 12345",
        )
        assert prompt == '[The user reacted to this message "Sales for Q1 total 12345" with [thumbsup]]'

    def test_non_ascii_reference_passes_through(self):
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="thumbsdown",
            reference_msg="这里是返回的表格数据",
        )
        assert "这里是返回的表格数据" in prompt
        assert "[thumbsdown]" in prompt

    def test_optional_reaction_msg_appended(self):
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="thumbsup",
            reference_msg="hello",
            reaction_msg="good job",
        )
        assert prompt.endswith("good job")
        assert prompt.startswith("[The user reacted to this message")

    def test_reaction_msg_whitespace_is_stripped(self):
        """Blank/whitespace-only reaction_msg should not be appended."""
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="thumbsup",
            reference_msg="hello",
            reaction_msg="   ",
        )
        assert prompt == '[The user reacted to this message "hello" with [thumbsup]]'

    def test_reference_msg_truncated_with_ellipsis(self):
        """References longer than max_ref_len are truncated with '...' marker."""
        long_ref = "a" * 1000
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="x",
            reference_msg=long_ref,
            max_ref_len=50,
        )
        # Extract quoted content between the first pair of '"'
        start = prompt.index('"') + 1
        end = prompt.index('"', start)
        quoted = prompt[start:end]
        assert len(quoted) == 50
        assert quoted.endswith("...")
        # Preserved prefix is 47 'a's (50 - len("..."))
        assert quoted[:-3] == "a" * 47

    def test_truncation_disabled_with_zero(self):
        long_ref = "a" * 1000
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="x",
            reference_msg=long_ref,
            max_ref_len=0,
        )
        assert long_ref in prompt

    def test_emoji_whitespace_trimmed(self):
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="  thumbsup  ",
            reference_msg="hi",
        )
        assert "[thumbsup]" in prompt
        assert "[  thumbsup  ]" not in prompt

    def test_empty_reference(self):
        prompt = build_reaction_feedback_prompt(
            reaction_emoji="thumbsup",
            reference_msg="",
        )
        assert prompt == '[The user reacted to this message "" with [thumbsup]]'

    def test_truncate_respects_tiny_max_len(self):
        """When max_len is smaller than the ellipsis, the quoted portion must still fit in max_len."""
        long_ref = "a" * 100
        for tiny in (1, 2, 3):
            prompt = build_reaction_feedback_prompt(
                reaction_emoji="x",
                reference_msg=long_ref,
                max_ref_len=tiny,
            )
            start = prompt.index('"') + 1
            end = prompt.index('"', start)
            quoted = prompt[start:end]
            assert len(quoted) == tiny, f"max_len={tiny} produced {quoted!r}"
            assert quoted == "..."[:tiny]
