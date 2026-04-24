# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`MarkdownStreamBuffer`.

Each test feeds the buffer a controlled sequence of deltas and asserts on the
exact ``(stable_segments, tail)`` split. The invariants these tests pin down:

* Stable segments + tail always reconstruct the concatenated input.
* An unclosed fenced code block defers every commit.
* Blank-line boundaries are the only block cut (no mid-paragraph cut).
* Oversize tails trigger a last-newline commit to keep Rich re-parse bounded.
"""

from __future__ import annotations

import pytest

from datus.cli.action_display.markdown_stream import MAX_TAIL_BYTES, MarkdownStreamBuffer


def _concat(stable: list[str], tail: str) -> str:
    return "".join(stable) + tail


class TestBasicBoundaries:
    def test_empty_delta_is_noop(self) -> None:
        buf = MarkdownStreamBuffer()
        assert buf.append("") == []
        assert buf.get_tail() == ""
        assert not buf.has_tail()

    def test_no_newline_keeps_everything_as_tail(self) -> None:
        buf = MarkdownStreamBuffer()
        assert buf.append("hello") == []
        assert buf.get_tail() == "hello"
        assert buf.has_tail()

    def test_single_newline_is_not_a_block_boundary(self) -> None:
        buf = MarkdownStreamBuffer()
        assert buf.append("hello\n") == []
        assert buf.get_tail() == "hello\n"

    def test_blank_line_commits_paragraph(self) -> None:
        buf = MarkdownStreamBuffer()
        stable = buf.append("hello\n\n")
        assert stable == ["hello\n\n"]
        assert buf.get_tail() == ""
        assert not buf.has_tail()

    def test_two_paragraphs_split_at_blank_line(self) -> None:
        buf = MarkdownStreamBuffer()
        # Partial first delta: both paragraphs present, tail started.
        stable = buf.append("hello\n\nworld")
        assert stable == ["hello\n\n"]
        assert buf.get_tail() == "world"

    def test_incremental_chunks_reconstruct_input(self) -> None:
        buf = MarkdownStreamBuffer()
        all_stable: list[str] = []
        for chunk in ["hel", "lo", "\n", "\n", "w", "orld", "\n\n", "end"]:
            all_stable.extend(buf.append(chunk))
        assert _concat(all_stable, buf.get_tail()) == "hello\n\nworld\n\nend"
        # only two committed paragraphs; "end" remains
        assert all_stable == ["hello\n\n", "world\n\n"]
        assert buf.get_tail() == "end"


class TestFenceGuard:
    def test_unclosed_fence_keeps_entire_text_as_tail(self) -> None:
        buf = MarkdownStreamBuffer()
        assert buf.append("```py\nprint(1)\n") == []
        assert buf.get_tail() == "```py\nprint(1)\n"

    def test_closed_fence_with_trailing_blank_commits(self) -> None:
        buf = MarkdownStreamBuffer()
        stable = buf.append("```py\nprint(1)\n```\n\nnext")
        # fence balanced (2 triple-backticks) + blank line after close → commit
        assert stable == ["```py\nprint(1)\n```\n\n"]
        assert buf.get_tail() == "next"

    def test_prose_before_fence_waits_for_fence_close(self) -> None:
        buf = MarkdownStreamBuffer()
        # Even though "hello\n\n" is a clean block boundary, the unbalanced
        # fence count forces the whole text back into the tail — we must
        # never split a half-open code block away from its prefix and risk
        # Rich rendering a mismatched fence on its own.
        assert buf.append("hello\n\n```py\nprint(1)") == []
        assert buf.get_tail() == "hello\n\n```py\nprint(1)"

    def test_closed_fence_then_reopens_waits(self) -> None:
        buf = MarkdownStreamBuffer()
        stable = buf.append("```a\nx\n```\n\n```b\ny")
        # 3 triple-backticks so far → odd → hold everything
        assert stable == []
        assert buf.get_tail() == "```a\nx\n```\n\n```b\ny"


class TestTablesAndLists:
    def test_complete_table_commits_after_blank_line(self) -> None:
        buf = MarkdownStreamBuffer()
        text = "| a | b |\n| - | - |\n| 1 | 2 |\n\ntrail"
        stable = buf.append(text)
        assert stable == ["| a | b |\n| - | - |\n| 1 | 2 |\n\n"]
        assert buf.get_tail() == "trail"

    def test_table_without_blank_line_stays_tail(self) -> None:
        buf = MarkdownStreamBuffer()
        text = "| a | b |\n| - | - |\n| 1 | 2 |\n"
        assert buf.append(text) == []
        assert buf.get_tail() == text

    def test_list_commits_only_after_blank_line(self) -> None:
        buf = MarkdownStreamBuffer()
        # Streaming list items without a terminating blank line: no commit.
        assert buf.append("- a\n- b\n- c\n") == []
        assert buf.get_tail() == "- a\n- b\n- c\n"
        # Blank line terminates the list → commit.
        stable = buf.append("\n")
        assert stable == ["- a\n- b\n- c\n\n"]
        assert buf.get_tail() == ""


class TestFlushAndClear:
    def test_flush_returns_and_clears_tail(self) -> None:
        buf = MarkdownStreamBuffer()
        buf.append("partial")
        assert buf.flush() == "partial"
        assert buf.get_tail() == ""
        assert not buf.has_tail()

    def test_flush_on_empty_buffer(self) -> None:
        buf = MarkdownStreamBuffer()
        assert buf.flush() == ""

    def test_clear_discards_tail_without_returning(self) -> None:
        buf = MarkdownStreamBuffer()
        buf.append("will be dropped")
        buf.clear()
        assert buf.get_tail() == ""
        assert not buf.has_tail()


class TestOversizeGuard:
    def test_oversize_tail_commits_at_last_newline(self) -> None:
        buf = MarkdownStreamBuffer()
        # Build a tail well above MAX_TAIL_BYTES with balanced fences (none)
        # and no blank-line boundary, but with internal newlines so the
        # oversize commit can latch on.
        line = "x" * 200 + "\n"
        n = (MAX_TAIL_BYTES // len(line)) + 3
        huge = line * n
        stable = buf.append(huge)
        # Oversize path must have committed at least one segment
        assert stable
        # The committed segment ends on a newline
        assert stable[-1].endswith("\n")
        # And reconstruction is exact
        assert _concat(stable, buf.get_tail()) == huge

    def test_oversize_does_not_fire_with_unclosed_fence(self) -> None:
        buf = MarkdownStreamBuffer()
        # Unclosed fence must keep the whole thing as tail regardless of size.
        body = "`" * 0  # placeholder
        text = "```py\n" + ("y" * (MAX_TAIL_BYTES + 100))
        _ = body
        assert buf.append(text) == []
        assert buf.get_tail() == text


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "hello",
            "hello\n\nworld\n\n",
            "```py\nprint(1)\n```\n\n",
            "a paragraph\nthat wraps\n\nanother\n\n",
            "| x | y |\n| - | - |\n| 1 | 2 |\n\ntrailing text",
        ],
    )
    def test_chunked_feed_matches_whole_input(self, text: str) -> None:
        # Feed the text one character at a time and make sure the
        # concatenated commits + final tail equal the input byte-for-byte.
        buf = MarkdownStreamBuffer()
        stable: list[str] = []
        for ch in text:
            stable.extend(buf.append(ch))
        assert _concat(stable, buf.get_tail()) == text
