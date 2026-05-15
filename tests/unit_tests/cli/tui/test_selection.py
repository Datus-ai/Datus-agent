# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.tui.selection`.

Covers:

* :class:`TranscriptSelection` state transitions (begin / update_head /
  finish / clear) and forward-ordered :meth:`range`.
* Forward and reverse drags collapse to the same ``range`` result.
* :func:`split_line_for_selection` correctly slices fragments at visual
  columns for ASCII, CJK fullwidth glyphs, and ANSI-styled fragments.
* :func:`extract_plain_text_between` returns visually-bounded plain text
  preserving fullwidth glyphs as wholes.
* :class:`SelectionAutoscroll` schedules ticks via :meth:`due`.
"""

from __future__ import annotations

import pytest

from datus.cli.tui.selection import (
    COLUMN_TO_LINE_END,
    SelectionAutoscroll,
    SelectionPoint,
    TranscriptSelection,
    extract_plain_text_between,
    line_char_count,
    split_line_for_selection,
)

# ── TranscriptSelection state machine ────────────────────────────────


def test_selection_starts_empty():
    sel = TranscriptSelection()
    assert sel.is_empty()
    assert sel.range() is None
    assert sel.version == 0


def test_begin_sets_anchor_head_and_bumps_version():
    sel = TranscriptSelection()
    sel.begin(SelectionPoint(line=2, column=3))
    assert sel.anchor == SelectionPoint(line=2, column=3)
    assert sel.head == SelectionPoint(line=2, column=3)
    assert sel.dragging is True
    assert sel.version == 1
    # Single-point selection is still considered empty for highlight
    # purposes — no characters lie between anchor and head.
    assert sel.is_empty()


def test_update_head_moves_head_and_bumps_version():
    sel = TranscriptSelection()
    sel.begin(SelectionPoint(line=1, column=0))
    v_after_begin = sel.version
    sel.update_head(SelectionPoint(line=1, column=5))
    assert sel.head == SelectionPoint(line=1, column=5)
    assert sel.version == v_after_begin + 1
    # Idempotent on identical head.
    sel.update_head(SelectionPoint(line=1, column=5))
    assert sel.version == v_after_begin + 1


def test_update_head_is_noop_without_anchor():
    sel = TranscriptSelection()
    sel.update_head(SelectionPoint(line=1, column=5))
    assert sel.anchor is None
    assert sel.head is None
    assert sel.version == 0


def test_range_is_forward_regardless_of_drag_direction():
    forward = TranscriptSelection()
    forward.begin(SelectionPoint(line=1, column=2))
    forward.update_head(SelectionPoint(line=3, column=4))

    reverse = TranscriptSelection()
    reverse.begin(SelectionPoint(line=3, column=4))
    reverse.update_head(SelectionPoint(line=1, column=2))

    assert forward.range() == reverse.range()
    start, end = forward.range()
    assert (start.line, start.column) == (1, 2)
    assert (end.line, end.column) == (3, 4)


def test_finish_clears_dragging_but_keeps_selection():
    sel = TranscriptSelection()
    sel.begin(SelectionPoint(line=0, column=0))
    sel.update_head(SelectionPoint(line=0, column=4))
    sel.finish()
    assert sel.dragging is False
    assert sel.range() is not None


def test_clear_removes_anchor_and_head_and_bumps_version():
    sel = TranscriptSelection()
    sel.begin(SelectionPoint(line=0, column=0))
    sel.update_head(SelectionPoint(line=0, column=4))
    v_before = sel.version
    sel.clear()
    assert sel.anchor is None
    assert sel.head is None
    assert sel.dragging is False
    assert sel.version == v_before + 1
    # Second clear is a no-op (no extra version bump).
    sel.clear()
    assert sel.version == v_before + 1


def test_columns_for_line_single_row():
    sel = TranscriptSelection()
    sel.begin(SelectionPoint(line=2, column=3))
    sel.update_head(SelectionPoint(line=2, column=7))
    assert sel.columns_for_line(2) == (3, 7)
    assert sel.columns_for_line(1) is None
    assert sel.columns_for_line(3) is None


def test_columns_for_line_multi_row_uses_sentinels():
    sel = TranscriptSelection()
    sel.begin(SelectionPoint(line=1, column=4))
    sel.update_head(SelectionPoint(line=3, column=2))
    assert sel.columns_for_line(1) == (4, COLUMN_TO_LINE_END)
    assert sel.columns_for_line(2) == (0, COLUMN_TO_LINE_END)
    assert sel.columns_for_line(3) == (0, 2)


# ── split_line_for_selection ─────────────────────────────────────────


def test_split_line_inside_single_ascii_fragment():
    line = [("", "hello world")]
    out = split_line_for_selection(line, start_col=2, end_col=7)
    # Prefix "he" untouched, "llo w" highlighted, "orld" untouched.
    assert out == [
        ("", "he"),
        ("class:selection", "llo w"),
        ("", "orld"),
    ]


def test_split_line_across_two_fragments_preserves_styles():
    line = [("class:a", "foo"), ("class:b", "bar")]
    out = split_line_for_selection(line, start_col=1, end_col=5)
    # Selected slice covers last 2 chars of "foo" and first 2 of "bar".
    assert out == [
        ("class:a", "f"),
        ("class:a class:selection", "oo"),
        ("class:b class:selection", "ba"),
        ("class:b", "r"),
    ]


def test_split_line_full_line_with_sentinel_end_col():
    line = [("", "abc")]
    out = split_line_for_selection(line, start_col=0, end_col=COLUMN_TO_LINE_END)
    assert out == [("class:selection", "abc")]


def test_split_line_with_fullwidth_glyph_uses_char_indices():
    # Char indices treat each glyph as 1 unit regardless of visual width;
    # this matches the coordinate system prompt_toolkit's MouseEvent uses.
    line = [("", "中A")]
    # char_idx 0..1 = first char = the whole "中" glyph.
    out = split_line_for_selection(line, start_col=0, end_col=1)
    assert out == [
        ("class:selection", "中"),
        ("", "A"),
    ]
    # char_idx 0..2 = both characters.
    out = split_line_for_selection(line, start_col=0, end_col=2)
    assert out == [("class:selection", "中A")]


def test_split_line_returns_input_when_range_empty():
    line = [("", "abc")]
    assert split_line_for_selection(line, 5, 5) == line
    assert split_line_for_selection([], 0, 5) == []


def test_split_line_clamps_end_col_to_line_width():
    line = [("", "abc")]
    # end_col=100 → clamp to 3 → highlight everything.
    out = split_line_for_selection(line, start_col=1, end_col=100)
    assert out == [
        ("", "a"),
        ("class:selection", "bc"),
    ]


def test_split_line_preserves_mouse_handler_tuple_element():
    handler = lambda evt: None  # noqa: E731
    line = [("", "hello", handler)]
    out = split_line_for_selection(line, start_col=1, end_col=4)
    # Prefix, inside, suffix all keep the handler attached.
    assert len(out) == 3
    for fragment in out:
        assert len(fragment) == 3
        assert fragment[2] is handler


# ── extract_plain_text_between ───────────────────────────────────────


def test_extract_plain_text_ascii():
    line = [("class:a", "hello "), ("class:b", "world")]
    assert extract_plain_text_between(line, 0, 11) == "hello world"
    assert extract_plain_text_between(line, 3, 8) == "lo wo"


def test_extract_plain_text_handles_fullwidth_chars():
    line = [("", "你好")]
    # Char indices: 你 → 0, 好 → 1. Selecting [0, 1) gives just "你"; [0, 2) gives both.
    assert extract_plain_text_between(line, 0, 1) == "你"
    assert extract_plain_text_between(line, 0, 2) == "你好"


def test_line_char_count_for_mixed_content():
    # ab=2 chars + 中=1 char = 3 characters total.
    line = [("", "ab"), ("", "中")]
    assert line_char_count(line) == 3


# ── SelectionAutoscroll ──────────────────────────────────────────────


def test_autoscroll_inactive_by_default():
    auto = SelectionAutoscroll()
    assert auto.is_active() is False
    assert auto.due() is False


def test_autoscroll_arm_then_due():
    auto = SelectionAutoscroll(interval_seconds=10.0)
    auto.arm(1)
    assert auto.is_active() is True
    # First tick fires immediately (next_tick_monotonic was reset to 0).
    assert auto.due(now=100.0) is True
    # Next tick gated by interval — 100 + 10 = 110, so 109 is too early.
    assert auto.due(now=109.0) is False
    assert auto.due(now=111.0) is True


def test_autoscroll_disarm():
    auto = SelectionAutoscroll()
    auto.arm(-1)
    auto.disarm()
    assert auto.is_active() is False
    assert auto.due(now=999.0) is False


def test_autoscroll_invalid_direction_raises():
    auto = SelectionAutoscroll()
    with pytest.raises(ValueError):
        auto.arm(2)


def test_autoscroll_direction_flip_resets_clock():
    auto = SelectionAutoscroll(interval_seconds=10.0)
    auto.arm(1)
    auto.due(now=100.0)  # consume the immediate first tick → next_tick=110
    auto.arm(-1)
    # Flip: next_tick reset to 0, so any now value fires.
    assert auto.due(now=101.0) is True
