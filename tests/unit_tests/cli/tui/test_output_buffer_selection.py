# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for selection-aware rendering and text extraction in :mod:`output_buffer`.

Complements :mod:`tests.unit_tests.cli.tui.test_output_buffer` (which covers
the snapshot / token-stream contract) with cases specific to the
:class:`TranscriptSelection` integration added on top.
"""

from __future__ import annotations

from datus.cli.tui.output_buffer import BufferedOutputControl, TUIOutputBuffer, extract_selection_text
from datus.cli.tui.selection import SelectionPoint, TranscriptSelection


def _populate(buf: TUIOutputBuffer, lines: list[str]) -> None:
    """Helper: append ``lines`` (each followed by ``\\n``) into the buffer."""
    for line in lines:
        buf.write(line + "\n")


def test_buffered_control_returns_unhighlighted_content_without_selection():
    """An empty selection must not perturb the renderer's fast path."""
    buf = TUIOutputBuffer()
    _populate(buf, ["alpha", "beta"])
    selection = TranscriptSelection()
    control = BufferedOutputControl(buf, selection_provider=lambda: selection)

    content = control.create_content(width=80, height=10)
    # No selection range → returned UIContent.get_line forwards the
    # buffer's own fragments untouched.
    assert content.get_line(0) == buf.snapshot().get_line(0)
    assert content.get_line(1) == buf.snapshot().get_line(1)


def test_uicontent_cache_invalidates_on_selection_version_bump():
    """A drag bumps ``TranscriptSelection.version`` so the cache is rebuilt."""
    buf = TUIOutputBuffer()
    _populate(buf, ["hello world"])
    selection = TranscriptSelection()
    control = BufferedOutputControl(buf, selection_provider=lambda: selection)

    first = control.create_content(width=80, height=10)
    # Same call with no state change → cache hit, same UIContent object.
    assert control.create_content(width=80, height=10) is first

    # Begin a selection: cache must miss next time and produce a NEW
    # UIContent object whose get_line for the selected row carries the
    # highlight style.
    selection.begin(SelectionPoint(line=0, column=0))
    selection.update_head(SelectionPoint(line=0, column=5))
    after = control.create_content(width=80, height=10)
    assert after is not first
    highlighted = after.get_line(0)
    styles = "".join(str(f[0]) for f in highlighted)
    assert "class:selection" in styles


def test_extract_selection_text_single_line_partial():
    buf = TUIOutputBuffer()
    _populate(buf, ["hello world"])
    selection = TranscriptSelection()
    selection.begin(SelectionPoint(line=0, column=6))
    selection.update_head(SelectionPoint(line=0, column=11))
    assert extract_selection_text(buf, selection) == "world"


def test_extract_selection_text_across_multiple_lines():
    buf = TUIOutputBuffer()
    _populate(buf, ["alpha", "beta", "gamma"])
    selection = TranscriptSelection()
    # Start mid-line 0, end mid-line 2.
    selection.begin(SelectionPoint(line=0, column=2))
    selection.update_head(SelectionPoint(line=2, column=3))
    assert extract_selection_text(buf, selection) == "pha\nbeta\ngam"


def test_extract_selection_text_handles_fullwidth_chars():
    """Char-index coordinates round-trip CJK glyphs cleanly."""
    buf = TUIOutputBuffer()
    _populate(buf, ["你好A"])
    selection = TranscriptSelection()
    # Char indices: 你=0, 好=1, A=2. Select [0, 2) → "你好".
    selection.begin(SelectionPoint(line=0, column=0))
    selection.update_head(SelectionPoint(line=0, column=2))
    assert extract_selection_text(buf, selection) == "你好"


def test_extract_selection_text_empty_selection_returns_empty_string():
    buf = TUIOutputBuffer()
    _populate(buf, ["alpha"])
    selection = TranscriptSelection()
    assert extract_selection_text(buf, selection) == ""


def test_blank_lines_padded_with_single_space_in_rendered_get_line():
    """Empty fragment lists become ``[("", " ")]`` so prompt_toolkit's
    ``rowcol_to_yx`` gets at least one entry per visible row.

    Without this padding, clicks on a blank line cannot be resolved
    back to a buffer line and fall through to the ``Point(0, 0)``
    fallback — which during a selection drag drags the highlight up
    to the top of the buffer.
    """
    buf = TUIOutputBuffer()
    buf.write("first\n")
    buf.write("\n")  # blank line
    buf.write("third\n")
    selection = TranscriptSelection()
    control = BufferedOutputControl(buf, selection_provider=lambda: selection)
    content = control.create_content(width=80, height=10)
    # The blank line (index 1) is padded.
    assert content.get_line(1) == [("", " ")]
    # Non-blank lines are untouched (no phantom space).
    assert "".join(f[1] for f in content.get_line(0)) == "first"


def test_extract_selection_text_preserves_blank_rows_without_phantom_space():
    """Extraction reads the raw snapshot so the padded space never ends up
    on the clipboard."""
    buf = TUIOutputBuffer()
    buf.write("first\n")
    buf.write("\n")
    buf.write("third\n")
    selection = TranscriptSelection()
    selection.begin(SelectionPoint(line=0, column=0))
    selection.update_head(SelectionPoint(line=2, column=5))
    assert extract_selection_text(buf, selection) == "first\n\nthird"


def test_uicontent_cache_distinct_per_selection_range():
    """Two different selection ranges produce two different UIContent objects."""
    buf = TUIOutputBuffer()
    _populate(buf, ["alpha beta"])
    selection = TranscriptSelection()
    control = BufferedOutputControl(buf, selection_provider=lambda: selection)

    selection.begin(SelectionPoint(line=0, column=0))
    selection.update_head(SelectionPoint(line=0, column=5))
    first = control.create_content(width=80, height=10)

    selection.update_head(SelectionPoint(line=0, column=8))
    second = control.create_content(width=80, height=10)
    assert first is not second
    assert "".join(f[1] for f in first.get_line(0)) == "alpha beta"  # same plain text
    assert "".join(f[1] for f in second.get_line(0)) == "alpha beta"
