# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.tui.output_buffer.TUIOutputBuffer`.

Covers the contract Rich + prompt_toolkit care about:
- write() splits incoming text on ``\\n`` and parses each complete line
  through ``ANSI(...)`` so styles survive
- partial (no-trailing-newline) bytes are buffered and surface in tokens()
- tokens() concatenates committed history + live-state snapshot + partial
- on_change fires after every write, even for partial input
- line_count tracks committed + live + partial rows
- concurrent writes don't lose bytes or interleave inside a single
  ``write`` call (per-write atomicity)
"""

from __future__ import annotations

import threading

import pytest

from datus.cli.tui.live_display_state import LiveDisplayLine
from datus.cli.tui.output_buffer import TUIOutputBuffer


def _flatten_text(tokens):
    return "".join(text for _, text in tokens)


def test_write_splits_on_newline_and_commits_complete_lines():
    buf = TUIOutputBuffer()
    buf.write("hello\nworld\n")
    assert buf.line_count() == 2
    text = _flatten_text(buf.tokens())
    assert text == "hello\nworld"


def test_partial_line_is_buffered_until_newline_arrives():
    buf = TUIOutputBuffer()
    buf.write("partial")
    # No newline yet — line_count counts the partial as one in-flight row.
    assert buf.line_count() == 1
    assert _flatten_text(buf.tokens()) == "partial"

    buf.write(" continued\n")
    assert buf.line_count() == 1  # one committed line, no partial
    assert _flatten_text(buf.tokens()) == "partial continued"


def test_ansi_color_codes_are_preserved_in_tokens():
    buf = TUIOutputBuffer()
    buf.write("\x1b[31mred\x1b[0m\n")
    tokens = buf.tokens()
    # Every character of "red" must carry an ansired style fragment.
    red_chars = [tok for tok in tokens if tok[0] == "ansired"]
    assert len(red_chars) == 3
    assert "".join(t[1] for t in red_chars) == "red"


def test_on_change_fires_for_every_write_including_partial():
    calls = []
    buf = TUIOutputBuffer(on_change=lambda: calls.append(1))
    buf.write("partial-without-newline")
    buf.write("\n")
    buf.write("complete\n")
    assert len(calls) == 3


def test_set_on_change_replaces_callback():
    early = []
    late = []
    buf = TUIOutputBuffer(on_change=lambda: early.append(1))
    buf.write("a\n")
    buf.set_on_change(lambda: late.append(1))
    buf.write("b\n")
    assert early == [1]
    assert late == [1]


def test_live_state_lines_appear_between_committed_and_partial():
    snapshot = []
    buf = TUIOutputBuffer(live_state_snapshot_fn=lambda: list(snapshot))
    buf.write("committed-1\n")
    buf.write("partial-tail")

    # Inject a streaming live tail.
    snapshot.append(LiveDisplayLine(segments=[("class:foo", "live-1")]))
    snapshot.append(LiveDisplayLine(segments=[("class:foo", "live-2")]))

    rendered = _flatten_text(buf.tokens())
    # Order: committed → live tail → partial.
    assert rendered == "committed-1\nlive-1\nlive-2\npartial-tail"
    # line_count covers all three regions.
    assert buf.line_count() == 1 + 2 + 1


def test_empty_write_is_noop_no_callback():
    calls = []
    buf = TUIOutputBuffer(on_change=lambda: calls.append(1))
    assert buf.write("") == 0
    assert calls == []
    assert buf.line_count() == 0


def test_isatty_writable_flush_satisfy_rich_contract():
    buf = TUIOutputBuffer()
    assert buf.isatty() is False
    assert buf.writable() is True
    # Should not raise.
    buf.flush()


def test_fileno_raises_oserror_for_callers_probing_real_fd():
    buf = TUIOutputBuffer()
    with pytest.raises(OSError):
        buf.fileno()


def test_concurrent_writes_keep_all_bytes_per_write_atomic():
    """Two threads, each writing many full lines — every line must
    survive intact (no torn characters). The committed line count must
    equal the total emitted lines."""
    buf = TUIOutputBuffer()

    def producer(tag: str, n: int) -> None:
        for i in range(n):
            buf.write(f"{tag}-{i}\n")

    threads = [
        threading.Thread(target=producer, args=("A", 200)),
        threading.Thread(target=producer, args=("B", 200)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    text = _flatten_text(buf.tokens())
    lines = text.splitlines()
    assert len(lines) == 400
    # All A-* and B-* labels appear.
    a_lines = [line for line in lines if line.startswith("A-")]
    b_lines = [line for line in lines if line.startswith("B-")]
    assert len(a_lines) == 200
    assert len(b_lines) == 200
    # Each line is fully formed (no mid-line interleave).
    for line in lines:
        assert "\n" not in line


def test_trailing_newline_is_stripped_when_no_partial_or_live_tail():
    """The output Window doesn't want an empty trailing row eating
    vertical space — check that a write ending exactly at ``\\n`` does
    not leave a dangling newline token."""
    buf = TUIOutputBuffer()
    buf.write("only-line\n")
    tokens = buf.tokens()
    assert tokens[-1] != ("", "\n"), "trailing newline should be dropped when no partial/live tail follows"
    assert _flatten_text(tokens) == "only-line"


def test_clear_drops_committed_and_partial_and_invalidates():
    """Ctrl+O verbose-toggle path calls ``clear()`` to reset the
    scrollable pane before reprinting the multi-turn history in the
    new mode; the on_change callback must fire so the empty buffer is
    repainted before the reprint lands."""
    calls = []
    buf = TUIOutputBuffer(on_change=lambda: calls.append(1))
    buf.write("turn-1\nturn-2\n")
    buf.write("partial-tail")
    buf.tokens()  # populate render cache
    assert buf.line_count() == 3
    assert buf.render_line_count() == 3
    calls.clear()

    buf.clear()

    assert buf.line_count() == 0
    assert buf.render_line_count() == 0
    assert buf.tokens() == []
    # Callback fires exactly once for the non-empty-→-empty transition.
    assert len(calls) == 1


def test_clear_is_noop_on_empty_buffer():
    """Repeated clears on already-empty buffer should not fire on_change
    spuriously — otherwise the TUI repaint loop wakes for nothing on
    every key press that lands on a fresh session."""
    calls = []
    buf = TUIOutputBuffer(on_change=lambda: calls.append(1))
    buf.clear()
    buf.clear()
    assert calls == []


def test_render_line_count_matches_last_tokens_snapshot():
    """``render_line_count`` MUST stay consistent with the tokens
    rendered, even if the live state mutates between calls — that race
    is what crashes prompt_toolkit with ``IndexError: fragment_lines[i]``."""
    live = []
    buf = TUIOutputBuffer(live_state_snapshot_fn=lambda: list(live))

    # Empty: never called tokens() → cache stays 0.
    assert buf.render_line_count() == 0

    # Two committed lines, live empty.
    buf.write("a\nb\n")
    buf.tokens()  # cursor's caller fetches text first, then count
    assert buf.render_line_count() == 2

    # Simulate the race: between this tokens() call and the next
    # render_line_count(), live state grows. The cached count must
    # ignore the new live entries to stay aligned with the tokens
    # the renderer already received.
    buf.tokens()  # snapshot: live still empty → cache=2
    live.append(LiveDisplayLine(segments=[("", "X")]))
    live.append(LiveDisplayLine(segments=[("", "Y")]))
    assert buf.render_line_count() == 2
    # Live count picks up the new entries though — that's the live API.
    assert buf.line_count() == 4

    # After a fresh tokens() call, render_line_count catches up.
    buf.tokens()
    assert buf.render_line_count() == 4


def test_set_live_state_snapshot_fn_swaps_source():
    def src_a():
        return [LiveDisplayLine(segments=[("", "A")])]

    def src_b():
        return [LiveDisplayLine(segments=[("", "B")])]

    buf = TUIOutputBuffer(live_state_snapshot_fn=src_a)
    buf.write("hist\n")
    assert _flatten_text(buf.tokens()) == "hist\nA"
    buf.set_live_state_snapshot_fn(src_b)
    assert _flatten_text(buf.tokens()) == "hist\nB"
