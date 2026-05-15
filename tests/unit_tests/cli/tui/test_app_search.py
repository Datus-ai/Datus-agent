# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""End-to-end-ish tests for the Ctrl+F search wiring on :class:`DatusApp`.

The app construction is real (it builds the full layout), but ``run()``
is never called — we exercise the search public surface and verify the
state machine and viewport-centering math from the outside.
"""

from __future__ import annotations

import pytest

from datus.cli.tui.app import DatusApp
from datus.cli.tui.output_buffer import TUIOutputBuffer


@pytest.fixture
def tui_app(monkeypatch: pytest.MonkeyPatch) -> DatusApp:
    buf = TUIOutputBuffer()
    # Populate enough lines that scroll math is meaningful.
    for i in range(30):
        buf.write(f"line {i:02d} content\n")
    # Add a known sentinel deep in the buffer.
    buf.write("the magic needle is here\n")
    for i in range(30, 60):
        buf.write(f"line {i:02d} content\n")

    app = DatusApp(
        status_tokens_fn=lambda: [],
        dispatch_fn=lambda _t: None,
        output_buffer=buf,
        output_line_count_fn=buf.line_count,
        output_tokens_fn=buf.tokens,
    )
    monkeypatch.setattr(app, "_output_viewport_rows", lambda: 10)
    return app


# ── open / close lifecycle ───────────────────────────────────────────


def test_open_search_flips_active_and_clears_state(tui_app: DatusApp):
    tui_app._open_search()
    assert tui_app._search_active is True
    assert tui_app._search_buffer.text == ""
    assert tui_app._search_state.query == ""
    assert tui_app._search_state.matches == []


def test_close_search_resets_state_and_drops_overlay(tui_app: DatusApp):
    tui_app._open_search()
    tui_app._search_buffer.text = "needle"  # triggers _on_search_text_changed
    assert tui_app._search_state.matches  # sanity: query matched
    tui_app._close_search()
    assert tui_app._search_active is False
    assert tui_app._search_state.query == ""
    assert tui_app._search_state.matches == []
    assert tui_app._search_buffer.text == ""


# ── incremental scan ─────────────────────────────────────────────────


def test_typing_query_scans_incrementally_and_centres_first_match(tui_app: DatusApp):
    tui_app._open_search()
    tui_app._search_buffer.text = "needle"
    state = tui_app._search_state
    assert state.query == "needle"
    assert len(state.matches) == 1
    match = state.matches[0]
    assert match.line == 30  # the sentinel line index
    assert state.current_idx == 0
    # Viewport (10 rows) centred → top = 30 - 5 = 25.
    assert tui_app._output_scroll_offset == 25


def test_no_match_query_clears_results_without_scrolling(tui_app: DatusApp):
    tui_app._open_search()
    # First a hit so viewport is parked mid-buffer.
    tui_app._search_buffer.text = "needle"
    scroll_after_first = tui_app._output_scroll_offset
    # Then a query that doesn't match anything.
    tui_app._search_buffer.text = "definitely-not-in-the-buffer"
    assert tui_app._search_state.matches == []
    assert tui_app._search_state.current_idx == -1
    # Viewport intentionally NOT yanked back when there's nothing to jump to.
    assert tui_app._output_scroll_offset == scroll_after_first


def test_empty_query_clears_matches_without_scrolling(tui_app: DatusApp):
    tui_app._open_search()
    tui_app._search_buffer.text = "needle"
    scroll_after = tui_app._output_scroll_offset
    tui_app._search_buffer.text = ""
    assert tui_app._search_state.matches == []
    assert tui_app._search_state.query == ""
    assert tui_app._output_scroll_offset == scroll_after


# ── navigation ───────────────────────────────────────────────────────


def test_jump_to_match_cycles_forward_and_recentres(tui_app: DatusApp):
    tui_app._open_search()
    # "line 1" matches "line 10", "line 11", … "line 19" — ten hits
    # spread across rows 10..19, plenty of distance to verify the
    # viewport shifts.
    tui_app._search_buffer.text = "line 1"
    matches = tui_app._search_state.matches
    assert len(matches) >= 2

    initial_scroll = tui_app._output_scroll_offset
    # Fixture lines "line 10".."line 19" guarantee distinct rows for the
    # first two matches, so the viewport must shift on +1 unconditionally.
    assert matches[0].line != matches[1].line
    tui_app._jump_to_match(+1)
    assert tui_app._search_state.current_idx == 1
    expected_top = max(0, matches[1].line - 5)
    assert tui_app._output_scroll_offset == expected_top
    assert tui_app._output_scroll_offset != initial_scroll


def test_jump_to_match_wraps_around(tui_app: DatusApp):
    tui_app._open_search()
    tui_app._search_buffer.text = "line 0"
    n = len(tui_app._search_state.matches)
    # Go to last, then one more → wraps back to 0.
    tui_app._search_state.set_current(n - 1)
    tui_app._jump_to_match(+1)
    assert tui_app._search_state.current_idx == 0


def test_jump_to_match_backwards_from_first_wraps_to_last(tui_app: DatusApp):
    tui_app._open_search()
    tui_app._search_buffer.text = "line 0"
    n = len(tui_app._search_state.matches)
    tui_app._jump_to_match(-1)
    assert tui_app._search_state.current_idx == n - 1


def test_jump_with_no_matches_is_noop(tui_app: DatusApp):
    tui_app._open_search()
    tui_app._search_buffer.text = "nope-no-match-anywhere"
    before = tui_app._output_scroll_offset
    tui_app._jump_to_match(+1)
    assert tui_app._search_state.current_idx == -1
    assert tui_app._output_scroll_offset == before


# ── search bar layout + status tokens ────────────────────────────────


def test_search_bar_in_normal_bottom_section(tui_app: DatusApp):
    """The bar must sit in the bottom HSplit so the layout reserves a row
    for it once ``_search_active`` flips on."""
    bottom = tui_app._normal_bottom_section
    children = list(bottom.get_children())
    assert tui_app._search_bar in children


def test_search_status_tokens_reflect_state(tui_app: DatusApp):
    # Idle / no query.
    tokens = tui_app._search_status_tokens()
    assert tokens[0][0] == "class:search-meta"
    assert "type to search" in tokens[0][1]

    tui_app._open_search()
    tui_app._search_buffer.text = "needle"
    tokens = tui_app._search_status_tokens()
    assert "1/1" in tokens[0][1]

    tui_app._search_buffer.text = "no-such-string"
    tokens = tui_app._search_status_tokens()
    assert tokens[0][0] == "class:search-meta.no-match"


def test_search_provider_returns_state_only_when_relevant(tui_app: DatusApp):
    # The BufferedOutputControl uses this provider to decide whether to
    # render the overlay. Idle app + no matches → provider returns None
    # (cheaper code path; overlay disabled).
    provider = tui_app._output_window.content._search_provider
    assert provider() is None

    tui_app._open_search()
    assert provider() is tui_app._search_state
    tui_app._close_search()
    assert provider() is None
