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

from datus.cli.action_display.markdown_stream import (
    MAX_TAIL_BYTES,
    MAX_TAIL_LINES,
    MarkdownStreamBuffer,
)


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


class TestLineOverflow:
    """The overflow trigger fires when the residual tail grows past
    :data:`MAX_TAIL_LINES` newlines, regardless of whether a ``\\n\\n``
    boundary exists. The commit happens at the last ``\\n`` so the
    pinned region is left with just the unfinished current line."""

    def test_line_overflow_without_blank_commits_at_last_newline(self) -> None:
        buf = MarkdownStreamBuffer()
        # Stream of single-line "lines" separated by a single newline —
        # no ``\n\n`` ever appears. The overflow guard must still fire
        # once the tail crosses the line budget, and the commit must
        # split at the last ``\n`` so only the unfinished line remains.
        lines = "".join(f"line {i}\n" for i in range(MAX_TAIL_LINES + 5))
        text = lines + "unfinished"
        stable = buf.append(text)
        assert stable, "expected a line-overflow commit"
        # Every committed segment ends on a newline — the last-line
        # suffix must never be handed off mid-line.
        assert stable[-1].endswith("\n")
        # Only the unfinished trailing line stays live.
        assert buf.get_tail() == "unfinished"
        # Reconstruction: spilled + tail == original.
        assert "".join(stable) + buf.get_tail() == text
        # Spill latch reflects that a commit occurred.
        assert buf.has_spilled()

    def test_under_line_budget_does_not_force_commit(self) -> None:
        buf = MarkdownStreamBuffer()
        # Just under the line budget and no ``\n\n`` boundary → nothing
        # committed yet; the tail rides live in the pinned region.
        lines = "".join(f"line {i}\n" for i in range(MAX_TAIL_LINES - 1))
        stable = buf.append(lines)
        assert stable == []
        assert buf.get_tail() == lines
        assert not buf.has_spilled()

    def test_unclosed_fence_blocks_line_overflow(self) -> None:
        buf = MarkdownStreamBuffer()
        # Once we're inside an unterminated code fence, balance is odd
        # and no commit is allowed — not even the line-overflow guard
        # may hand off a half-open code block to the scrollback.
        body = "```py\n" + "".join(f"code line {i}\n" for i in range(MAX_TAIL_LINES + 5))
        stable = buf.append(body)
        assert stable == []
        assert buf.get_tail() == body
        assert not buf.has_spilled()

    def test_paragraph_boundary_still_takes_precedence(self) -> None:
        buf = MarkdownStreamBuffer()
        # When a ``\n\n`` boundary is already in the tail, the
        # paragraph-boundary path commits up to that boundary first.
        # The residual tail is below the line budget so the overflow
        # guard doesn't fire.
        text = "para\n\nnext"
        stable = buf.append(text)
        assert stable == ["para\n\n"]
        assert buf.get_tail() == "next"
        assert buf.has_spilled()

    def test_spill_latch_cleared_on_flush(self) -> None:
        buf = MarkdownStreamBuffer()
        buf.append("".join(f"line {i}\n" for i in range(MAX_TAIL_LINES + 5)) + "rest")
        assert buf.has_spilled()
        _ = buf.flush()
        assert not buf.has_spilled()
        assert buf.get_tail() == ""

    def test_spill_latch_cleared_on_clear(self) -> None:
        buf = MarkdownStreamBuffer()
        buf.append("".join(f"line {i}\n" for i in range(MAX_TAIL_LINES + 5)) + "rest")
        assert buf.has_spilled()
        buf.clear()
        assert not buf.has_spilled()
        assert buf.get_tail() == ""

    def test_incremental_feed_reconstructs_long_body(self) -> None:
        buf = MarkdownStreamBuffer()
        body = "".join(f"paragraph {i} line A\nline B\n\n" for i in range(MAX_TAIL_LINES + 10))
        stable_all: list[str] = []
        for ch in body:
            stable_all.extend(buf.append(ch))
        # Combined commits plus final flush reproduce the full input
        # byte-for-byte — no duplication, no loss.
        assert "".join(stable_all) + buf.flush() == body


class TestTableGuard:
    """Open markdown tables must receive the same hold-until-closure
    protection that fenced code blocks get. Otherwise the overflow guard
    would cut header-less row fragments into the scrollback, where Rich
    renders them as a single mashed-pipe line (no header → no table
    layout → every ``|`` collapsed onto one row)."""

    @staticmethod
    def _build_table(row_count: int) -> str:
        header = "| c1 | c2 |\n| -- | -- |\n"
        rows = "".join(f"| r{i} | v{i} |\n" for i in range(row_count))
        return header + rows

    def test_open_table_blocks_line_overflow(self) -> None:
        buf = MarkdownStreamBuffer()
        # 30 rows + header + separator = 32 lines, well above MAX_TAIL_LINES,
        # and no trailing blank line: the table is still open. The overflow
        # guard must hold the whole thing instead of slicing at the last
        # newline.
        body = self._build_table(30)
        stable = buf.append(body)
        assert stable == []
        assert buf.get_tail() == body
        assert not buf.has_spilled()

    def test_closed_table_flushes_whole_block(self) -> None:
        buf = MarkdownStreamBuffer()
        body = self._build_table(30)
        # Open table: held.
        assert buf.append(body) == []
        # Closing ``\n`` completes the blank-line boundary; the entire
        # table becomes one stable segment via the normal ``\n\n`` path.
        stable = buf.append("\n")
        assert stable == [body + "\n"]
        assert buf.get_tail() == ""
        assert buf.has_spilled()

    def test_pipe_in_prose_does_not_trigger_guard(self) -> None:
        buf = MarkdownStreamBuffer()
        # Inline pipe references in prose — each line has at most one
        # ``|`` — must not be mistaken for a table. The overflow guard
        # stays active and splits at the last newline as usual.
        lines = "".join(f"Use the pipe operator `a | b` on line {i}\n" for i in range(MAX_TAIL_LINES + 5))
        text = lines + "trailing"
        stable = buf.append(text)
        assert stable, "prose with inline single-pipe must still overflow-split"
        assert stable[-1].endswith("\n")
        assert buf.get_tail() == "trailing"
        assert "".join(stable) + buf.get_tail() == text

    def test_table_then_prose_splits_at_table_close(self) -> None:
        buf = MarkdownStreamBuffer()
        text = "| a | b |\n| - | - |\n| 1 | 2 |\n\nnext prose"
        stable = buf.append(text)
        assert stable == ["| a | b |\n| - | - |\n| 1 | 2 |\n\n"]
        assert buf.get_tail() == "next prose"
        assert buf.has_spilled()

    def test_long_prose_still_overflow_splits(self) -> None:
        """Regression: non-table prose over the line budget keeps splitting."""
        buf = MarkdownStreamBuffer()
        lines = "".join(f"line {i}\n" for i in range(MAX_TAIL_LINES + 5))
        text = lines + "unfinished"
        stable = buf.append(text)
        assert stable, "plain prose overflow must still hand off a prefix"
        assert buf.get_tail() == "unfinished"

    def test_long_code_block_still_holds(self) -> None:
        """Regression: open fenced code block continues to hold regardless
        of the new table guard."""
        buf = MarkdownStreamBuffer()
        body = "```py\n" + "".join(f"code_line_{i}()\n" for i in range(MAX_TAIL_LINES + 5))
        stable = buf.append(body)
        assert stable == []
        assert buf.get_tail() == body

    def test_reconstruction_round_trip_with_long_table(self) -> None:
        buf = MarkdownStreamBuffer()
        body = self._build_table(40) + "\nafter the table text\n\ntrailer"
        stable_all: list[str] = []
        for ch in body:
            stable_all.extend(buf.append(ch))
        # Final flush plus all committed segments must reproduce the
        # exact input — no duplication, no loss, regardless of where
        # the guard fired.
        assert "".join(stable_all) + buf.flush() == body


class TestAppendRawSimple:
    """``append_raw`` is the plain accumulator — it never commits mid-
    stream so the whole text rides in the tail until :meth:`flush`."""

    def test_append_raw_holds_everything_until_flush(self) -> None:
        buf = MarkdownStreamBuffer()
        # Even a body large enough to blow every threshold stays live
        # until the caller flushes it.
        body = "".join(f"line {i}\n" for i in range(MAX_TAIL_LINES + 5))
        buf.append_raw(body)
        assert buf.get_tail() == body
        assert not buf.has_spilled()
        assert buf.flush() == body
        assert buf.get_tail() == ""

    def test_append_raw_empty_delta_is_noop(self) -> None:
        buf = MarkdownStreamBuffer()
        buf.append_raw("")
        assert buf.get_tail() == ""
        assert not buf.has_spilled()


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
