# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the search overlay path on :class:`BufferedOutputControl`."""

from __future__ import annotations

from datus.cli.tui.output_buffer import BufferedOutputControl, TUIOutputBuffer
from datus.cli.tui.search import SearchMatch, SearchState
from datus.cli.tui.selection import SelectionPoint, TranscriptSelection


def _styled_chars(line):
    """Expand a fragment list to ``[(style, char), ...]`` so per-char styling
    can be asserted without depending on whether prompt_toolkit collapses
    runs of identical fragments."""
    out = []
    for fragment in line:
        style = fragment[0]
        text = fragment[1] if len(fragment) > 1 else ""
        for ch in text:
            out.append((style, ch))
    return out


def test_match_lines_get_search_style_applied():
    buf = TUIOutputBuffer()
    buf.write("alpha BETA gamma\n")
    state = SearchState()
    state.update(query="beta", matches=[SearchMatch(line=0, start=6, end=10)], current_idx=-1)
    control = BufferedOutputControl(buf, search_provider=lambda: state)
    content = control.create_content(width=80, height=10)
    chars = _styled_chars(content.get_line(0))
    plain = "".join(c for _, c in chars)
    assert plain == "alpha BETA gamma"
    # Match span carries the style; surrounding chars do not.
    match_styles = {style for style, ch in chars[6:10]}
    assert match_styles == {"class:search-match"}
    assert all("class:search-match" not in style for style, _ in chars[:6])
    assert all("class:search-match" not in style for style, _ in chars[10:])


def test_current_match_uses_current_style():
    buf = TUIOutputBuffer()
    buf.write("foo bar foo bar\n")
    state = SearchState()
    state.update(
        query="bar",
        matches=[
            SearchMatch(line=0, start=4, end=7),
            SearchMatch(line=0, start=12, end=15),
        ],
        current_idx=1,
    )
    control = BufferedOutputControl(buf, search_provider=lambda: state)
    content = control.create_content(width=80, height=10)
    chars = _styled_chars(content.get_line(0))
    # First "bar" (idx 4..7) is non-current.
    assert {style for style, _ in chars[4:7]} == {"class:search-match"}
    # Second "bar" (idx 12..15) is the current match.
    assert {style for style, _ in chars[12:15]} == {"class:search-match.current"}


def test_uicontent_cache_invalidates_on_search_version_bump():
    buf = TUIOutputBuffer()
    buf.write("alpha beta gamma\n")
    state = SearchState()
    control = BufferedOutputControl(buf, search_provider=lambda: state)

    no_search = control.create_content(width=80, height=10)
    # Same call: cache hit.
    assert control.create_content(width=80, height=10) is no_search

    state.update(query="beta", matches=[SearchMatch(0, 6, 10)], current_idx=0)
    with_search = control.create_content(width=80, height=10)
    assert with_search is not no_search
    # Current_idx flip without query change must also invalidate.
    state.set_current(0)  # same idx → no bump → cache hit
    assert control.create_content(width=80, height=10) is with_search


def test_search_and_selection_compose_with_selection_on_top():
    buf = TUIOutputBuffer()
    buf.write("hello world\n")
    selection = TranscriptSelection()
    selection.begin(SelectionPoint(line=0, column=0))
    selection.update_head(SelectionPoint(line=0, column=5))
    state = SearchState()
    state.update(query="hello", matches=[SearchMatch(0, 0, 5)], current_idx=0)
    control = BufferedOutputControl(
        buf,
        selection_provider=lambda: selection,
        search_provider=lambda: state,
    )
    content = control.create_content(width=80, height=10)
    chars = _styled_chars(content.get_line(0))
    # The covered range must carry both styles so prompt_toolkit's style
    # merger can decide the final appearance; we don't lose the search
    # marker just because the selection overlays the same chars.
    covered_styles = {style for style, _ in chars[0:5]}
    assert any("class:search-match.current" in s for s in covered_styles)
    assert any("class:selection" in s for s in covered_styles)


def test_search_without_matches_returns_unchanged_lines():
    buf = TUIOutputBuffer()
    buf.write("just text here\n")
    state = SearchState()  # idle / no matches
    control = BufferedOutputControl(buf, search_provider=lambda: state)
    content = control.create_content(width=80, height=10)
    # When state has no matches, the search overlay must not introduce
    # any extra fragments — the line should look exactly like the raw
    # snapshot (plus blank-line padding for empty rows, which doesn't
    # apply here).
    assert "".join(f[1] for f in content.get_line(0)) == "just text here"
