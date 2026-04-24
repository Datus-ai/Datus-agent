# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for :class:`datus.cli.tui.live_display_state.LiveDisplayState`."""

from __future__ import annotations

import threading

import pytest

from datus.cli.tui.live_display_state import MAX_LIVE_LINES_TOTAL, LiveDisplayLine, LiveDisplayState


@pytest.mark.ci
class TestLiveDisplayState:
    def test_default_is_inactive(self) -> None:
        state = LiveDisplayState()
        assert state.is_active() is False
        assert state.line_count() == 0
        assert state.snapshot() == []

    def test_set_lines_activates_and_invalidate_fires(self) -> None:
        hits: list[int] = []
        state = LiveDisplayState(invalidate_cb=lambda: hits.append(1))
        state.set_lines(
            [
                LiveDisplayLine(segments=[("class:a", "first")]),
                LiveDisplayLine(segments=[("class:a", "second")]),
            ]
        )
        assert state.is_active() is True
        assert state.line_count() == 2
        assert state.snapshot()[0].segments == [("class:a", "first")]
        assert state.snapshot()[1].segments == [("class:a", "second")]
        assert hits, "invalidate_cb should fire after set_lines"

    def test_set_lines_truncates_to_global_max(self) -> None:
        state = LiveDisplayState()
        lines = [LiveDisplayLine(segments=[("class:a", f"line {i}")]) for i in range(MAX_LIVE_LINES_TOTAL + 3)]
        state.set_lines(lines)
        assert state.line_count() == MAX_LIVE_LINES_TOTAL
        kept = state.snapshot()
        # Tail of the input is kept.
        assert kept[-1].segments[0][1] == f"line {MAX_LIVE_LINES_TOTAL + 2}"

    def test_dynamic_max_rows_provider_shrinks_cap(self) -> None:
        """Custom provider (e.g. terminal-height-aware) tightens the cap."""
        state = LiveDisplayState()
        state.set_max_rows_provider(lambda: 5)
        state.set_lines([LiveDisplayLine(segments=[("x", f"{i}")]) for i in range(20)])
        assert state.line_count() == 5
        # Last rows are retained (newest content is at the end).
        kept = state.snapshot()
        assert kept[0].segments[0][1] == "15"
        assert kept[-1].segments[0][1] == "19"

    def test_max_rows_provider_floor_and_ceiling(self) -> None:
        """Providers returning garbage fall through to safe bounds."""
        state = LiveDisplayState()
        state.set_max_rows_provider(lambda: -1)
        assert state.max_rows() >= 3  # _PINNED_MIN_ROWS
        state.set_max_rows_provider(lambda: 10**9)
        assert state.max_rows() == MAX_LIVE_LINES_TOTAL
        state.set_max_rows_provider(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert state.max_rows() == MAX_LIVE_LINES_TOTAL

    def test_clear_only_invalidates_when_had_content(self) -> None:
        hits: list[int] = []
        state = LiveDisplayState(invalidate_cb=lambda: hits.append(1))
        state.clear()  # already empty
        assert hits == []
        state.set_lines([LiveDisplayLine(segments=[("class:a", "x")])])
        state.clear()
        # set + clear = two invalidations
        assert len(hits) == 2

    def test_set_invalidate_replaces_callback(self) -> None:
        first_hits: list[int] = []
        second_hits: list[int] = []
        state = LiveDisplayState(invalidate_cb=lambda: first_hits.append(1))
        state.set_invalidate(lambda: second_hits.append(1))
        state.set_lines([LiveDisplayLine(segments=[("class:a", "x")])])
        assert first_hits == []
        assert second_hits == [1]

    def test_snapshot_is_a_copy(self) -> None:
        state = LiveDisplayState()
        state.set_lines([LiveDisplayLine(segments=[("class:a", "x")])])
        snap = state.snapshot()
        snap.clear()
        assert state.line_count() == 1, "external mutation of snapshot must not affect state"

    def test_concurrent_writers_do_not_lose_lock(self) -> None:
        state = LiveDisplayState()

        def writer(char: str) -> None:
            for i in range(200):
                state.set_lines([LiveDisplayLine(segments=[("class:a", f"{char}{i}")])])

        threads = [threading.Thread(target=writer, args=(c,)) for c in "abcd"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Just make sure we never crashed and the final state is consistent.
        assert 0 <= state.line_count() <= MAX_LIVE_LINES_TOTAL
