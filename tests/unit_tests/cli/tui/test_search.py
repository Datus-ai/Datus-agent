# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.tui.search`.

Covers:

* :func:`find_matches` substring semantics: case-insensitivity,
  overlap, empty / no-match queries, multi-row scans.
* :class:`SearchState` lifecycle (``clear`` / ``update`` / ``set_current``
  / ``matches_on_line``) and ``version`` bumping rules consumed by the
  renderer's UIContent cache.
"""

from __future__ import annotations

from datus.cli.tui.search import SearchMatch, SearchState, find_matches


def _line_provider(rows):
    """Helper: convert a list of plain strings into a ``get_line`` callable."""
    fragments = [[("", row)] if row else [] for row in rows]

    def get_line(idx):
        return fragments[idx]

    return get_line, len(rows)


# ── find_matches ─────────────────────────────────────────────────────


def test_empty_query_returns_no_matches():
    get_line, n = _line_provider(["hello", "world"])
    assert find_matches(get_line, n, "") == []


def test_substring_case_insensitive_single_line():
    get_line, n = _line_provider(["Hello World"])
    matches = find_matches(get_line, n, "world")
    assert matches == [SearchMatch(line=0, start=6, end=11)]
    # Uppercase query, lowercase haystack — still matches.
    matches = find_matches(get_line, n, "HELLO")
    assert matches == [SearchMatch(line=0, start=0, end=5)]


def test_overlapping_matches_on_single_line():
    # ``aa`` in ``aaaa`` should produce three overlapping matches.
    get_line, n = _line_provider(["aaaa"])
    matches = find_matches(get_line, n, "aa")
    assert matches == [
        SearchMatch(line=0, start=0, end=2),
        SearchMatch(line=0, start=1, end=3),
        SearchMatch(line=0, start=2, end=4),
    ]


def test_multi_row_matches_are_row_major_ordered():
    get_line, n = _line_provider(["alpha BETA", "gamma alpha", "BeTa alpha"])
    matches = find_matches(get_line, n, "alpha")
    assert [(m.line, m.start, m.end) for m in matches] == [
        (0, 0, 5),
        (1, 6, 11),
        (2, 5, 10),
    ]


def test_blank_rows_are_skipped_without_crashing():
    get_line, n = _line_provider(["first", "", "third"])
    matches = find_matches(get_line, n, "first")
    assert matches == [SearchMatch(line=0, start=0, end=5)]


def test_fragments_with_mouse_handler_tuple_are_scanned():
    """3-tuple fragments (``(style, text, handler)``) must contribute text too."""

    def get_line(idx):
        return [("", "lead "), ("", "needle ", lambda _e: None), ("", "tail")]

    matches = find_matches(get_line, 1, "needle")
    assert matches == [SearchMatch(line=0, start=5, end=11)]


def test_no_match_returns_empty_list():
    get_line, n = _line_provider(["foo bar baz"])
    assert find_matches(get_line, n, "qux") == []


# ── SearchState ──────────────────────────────────────────────────────


def test_state_starts_idle():
    state = SearchState()
    assert state.query == ""
    assert state.matches == []
    assert state.current_idx == -1
    assert state.version == 0
    assert state.is_active() is False
    assert state.current() is None
    assert state.matches_on_line(0) == []


def test_update_replaces_state_and_bumps_version():
    state = SearchState()
    state.update(
        query="x",
        matches=[SearchMatch(line=0, start=0, end=1), SearchMatch(line=2, start=1, end=2)],
        current_idx=0,
    )
    assert state.query == "x"
    assert state.is_active() is True
    assert state.current_idx == 0
    assert state.current() == SearchMatch(line=0, start=0, end=1)
    assert state.version == 1


def test_update_with_no_matches_forces_current_idx_minus_one():
    state = SearchState()
    state.update(query="x", matches=[], current_idx=5)  # caller's hint is ignored
    assert state.current_idx == -1
    assert state.current() is None


def test_set_current_wraps_around_and_bumps_version_only_on_change():
    matches = [SearchMatch(0, 0, 1), SearchMatch(1, 0, 1), SearchMatch(2, 0, 1)]
    state = SearchState()
    state.update(query="x", matches=matches, current_idx=0)
    v0 = state.version
    state.set_current(2)
    assert state.current_idx == 2
    assert state.version == v0 + 1
    # Wrapping past end.
    state.set_current(3)
    assert state.current_idx == 0
    # Repeating same idx → no bump.
    v1 = state.version
    state.set_current(0)
    assert state.version == v1


def test_clear_resets_state_and_bumps_version_once():
    state = SearchState()
    state.update(query="x", matches=[SearchMatch(0, 0, 1)], current_idx=0)
    v_before = state.version
    state.clear()
    assert state.query == ""
    assert state.matches == []
    assert state.current_idx == -1
    assert state.version == v_before + 1
    # Idempotent: second clear does nothing.
    state.clear()
    assert state.version == v_before + 1


def test_matches_on_line_marks_current_correctly():
    matches = [
        SearchMatch(line=0, start=0, end=2),
        SearchMatch(line=0, start=5, end=7),
        SearchMatch(line=1, start=3, end=5),
    ]
    state = SearchState()
    state.update(query="xx", matches=matches, current_idx=1)
    line0 = state.matches_on_line(0)
    assert line0 == [(0, 2, False), (5, 7, True)]
    line1 = state.matches_on_line(1)
    assert line1 == [(3, 5, False)]
    # Unrelated line: empty list.
    assert state.matches_on_line(2) == []


def test_matches_on_line_is_sorted_by_start():
    # Construct matches whose source order is reversed on the same line —
    # the helper must sort them so the renderer paints left-to-right.
    matches = [
        SearchMatch(line=0, start=10, end=12),
        SearchMatch(line=0, start=2, end=4),
        SearchMatch(line=0, start=6, end=8),
    ]
    state = SearchState()
    state.update(query="xx", matches=matches, current_idx=-1)
    assert [t[0] for t in state.matches_on_line(0)] == [2, 6, 10]
