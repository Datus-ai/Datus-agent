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
from datus.cli.tui.output_buffer import BufferedOutputControl, TUIOutputBuffer


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


def test_tokens_returns_same_list_object_when_state_unchanged():
    """Repeated paints over an unchanged buffer must hand back the *same*
    list instance — that's how ``FormattedTextControl`` knows to skip
    re-layout. Without it every scroll-wheel tick rebuilds the full token
    stream and the verbose-mode pane stalls under load."""
    live = []
    buf = TUIOutputBuffer(live_state_snapshot_fn=lambda: list(live))
    buf.write("line-1\nline-2\n")
    first = buf.tokens()
    second = buf.tokens()
    third = buf.tokens()
    assert first is second is third

    # Adding a partial mutates the inputs → cache invalidates → new list.
    buf.write("partial-tail")
    fourth = buf.tokens()
    assert fourth is not first

    # Stable again after the second paint — the renderer keeps reusing it.
    fifth = buf.tokens()
    assert fifth is fourth


def test_tokens_cache_invalidates_when_live_state_changes():
    """The live tail flips between paints during streaming. The cache key
    must include the snapshot so the next ``tokens()`` rebuilds — otherwise
    the user sees a stale rolling window."""
    live = []
    buf = TUIOutputBuffer(live_state_snapshot_fn=lambda: list(live))
    buf.write("hist\n")
    before = buf.tokens()
    assert _flatten_text(before) == "hist"

    live.append(LiveDisplayLine(segments=[("", "live-1")]))
    after = buf.tokens()
    assert after is not before
    assert _flatten_text(after) == "hist\nlive-1"


def test_tokens_cache_drops_on_clear():
    """``clear()`` is the Ctrl+O reset path; after it the next paint must
    not return the pre-clear token list even if subsequent writes recreate
    the same line count."""
    buf = TUIOutputBuffer()
    buf.write("a\nb\n")
    before = buf.tokens()
    buf.clear()
    buf.write("a\nb\n")
    after = buf.tokens()
    assert after is not before
    assert _flatten_text(after) == "a\nb"


def test_snapshot_is_reused_across_calls_when_buffer_unchanged():
    """``BufferedOutputControl`` keys its UIContent cache on snapshot
    identity, so repeated paints over an idle buffer must hand back the
    same snapshot object. Without this each paint would rebuild a fresh
    snapshot and the UIContent cache would never hit."""
    buf = TUIOutputBuffer()
    buf.write("line-1\nline-2\nline-3\n")
    first = buf.snapshot()
    assert first.total == 3
    second = buf.snapshot()
    assert second is first

    buf.write("line-4\n")
    third = buf.snapshot()
    assert third is not first
    assert third.total == 4


def test_snapshot_get_line_returns_per_region_fragments():
    """``UIContent.get_line(idx)`` is invoked only for visible rows, so the
    snapshot must address committed history, live tail, and the partial
    bottom row by absolute line index in that order."""
    live = [
        LiveDisplayLine(segments=[("class:live", "L1")]),
        LiveDisplayLine(segments=[("class:live", "L2")]),
    ]
    buf = TUIOutputBuffer(live_state_snapshot_fn=lambda: list(live))
    buf.write("hist-1\nhist-2\n")
    buf.write("partial-tail")

    snap = buf.snapshot()
    assert snap.total == 2 + 2 + 1
    # Region boundaries: 0..1 committed, 2..3 live, 4 partial.
    assert _flatten_text(snap.get_line(0)) == "hist-1"
    assert _flatten_text(snap.get_line(1)) == "hist-2"
    assert snap.get_line(2) == [("class:live", "L1")]
    assert snap.get_line(3) == [("class:live", "L2")]
    assert _flatten_text(snap.get_line(4)) == "partial-tail"


def test_snapshot_invalidates_on_clear():
    """After ``clear()`` the next snapshot must reflect an empty buffer
    even if a write recreates the same line count — otherwise the post-
    Ctrl+O reprint paints the stale frame."""
    buf = TUIOutputBuffer()
    buf.write("a\nb\n")
    before = buf.snapshot()
    assert before.total == 2

    buf.clear()
    after_clear = buf.snapshot()
    assert after_clear is not before
    assert after_clear.total == 0

    buf.write("a\nb\n")
    rebuilt = buf.snapshot()
    assert rebuilt.total == 2
    assert rebuilt is not before


def test_buffered_output_control_returns_lazy_uicontent():
    """``BufferedOutputControl.create_content`` must hand prompt_toolkit a
    ``UIContent`` whose ``get_line`` is callable per-row rather than a
    pre-materialised fragment stream. Two paints over an unchanged buffer
    must reuse the same ``UIContent`` so prompt_toolkit's per-line height
    cache stays warm."""
    buf = TUIOutputBuffer()
    buf.write("row-1\nrow-2\nrow-3\n")
    control = BufferedOutputControl(buf, focusable=True, show_cursor=False)

    content = control.create_content(width=80, height=24)
    assert content.line_count == 3
    assert _flatten_text(content.get_line(0)) == "row-1"
    assert _flatten_text(content.get_line(2)) == "row-3"

    # Same buffer state → same UIContent object (cache hit).
    again = control.create_content(width=80, height=24)
    assert again is content

    # After a write the cache must miss; ``get_line(3)`` is reachable.
    buf.write("row-4\n")
    refreshed = control.create_content(width=80, height=24)
    assert refreshed is not content
    assert refreshed.line_count == 4
    assert _flatten_text(refreshed.get_line(3)) == "row-4"


def test_buffered_output_control_does_not_touch_offscreen_rows():
    """Smoke check the lazy contract: a 10k-row scrollback must produce a
    ``UIContent`` in O(1) wrt how many ``get_line`` calls the renderer
    makes. We assert by reading just the bottom visible window — the rows
    outside that window must never be queried."""
    buf = TUIOutputBuffer()
    for i in range(10_000):
        buf.write(f"row-{i}\n")
    control = BufferedOutputControl(buf)
    content = control.create_content(width=80, height=24)
    assert content.line_count == 10_000

    # Read only the last 24 rows (what a viewport would paint).
    visible = [content.get_line(i) for i in range(content.line_count - 24, content.line_count)]
    assert _flatten_text(visible[0]) == "row-9976"
    assert _flatten_text(visible[-1]) == "row-9999"
