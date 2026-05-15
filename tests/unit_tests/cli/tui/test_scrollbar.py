# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.tui.scrollbar`."""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from datus.cli.tui.scrollbar import (
    ScrollbarController,
    build_scrollbar_fragments,
)


def _make_controller(*, total: int, viewport: int, scroll_state: dict):
    """Helper: bind a controller to a dict-backed scroll state for tests."""
    return ScrollbarController(
        viewport_rows_fn=lambda: viewport,
        total_rows_fn=lambda: total,
        get_scroll_fn=lambda: scroll_state["offset"],
        set_scroll_fn=lambda v: scroll_state.__setitem__("offset", v),
        invalidate_fn=lambda: scroll_state.__setitem__("invalidate_calls", scroll_state.get("invalidate_calls", 0) + 1),
    )


# ── thumb geometry ───────────────────────────────────────────────────


def test_thumb_fills_track_when_content_fits():
    state = {"offset": 0}
    ctl = _make_controller(total=10, viewport=20, scroll_state=state)
    top, height = ctl.thumb_geometry()
    assert (top, height) == (0, 20)
    assert ctl.is_overflow() is False
    assert ctl.max_scroll() == 0


def test_thumb_size_scales_with_visible_fraction():
    # total=100, viewport=20 → thumb should be 20*20/100 = 4 rows tall.
    state = {"offset": 0}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    _, height = ctl.thumb_geometry()
    assert height == 4


def test_thumb_position_tracks_scroll_offset():
    state = {"offset": 0}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    # max_scroll = 80, free = 20 - 4 = 16.
    # offset=0 → thumb_top=0
    assert ctl.thumb_geometry()[0] == 0
    # offset=max_scroll → thumb at bottom of free space (=16).
    state["offset"] = 80
    assert ctl.thumb_geometry()[0] == 16
    # offset=40 → thumb_top = round(40 * 16 / 80) = 8
    state["offset"] = 40
    assert ctl.thumb_geometry()[0] == 8


def test_thumb_height_clamped_to_minimum_one():
    # total=1_000_000 → thumb would round to <1 row, but we must clamp.
    state = {"offset": 0}
    ctl = _make_controller(total=1_000_000, viewport=20, scroll_state=state)
    _, height = ctl.thumb_geometry()
    assert height == 1


def test_thumb_position_clamped_at_bottom():
    # Even when offset somehow exceeds max_scroll, thumb does not run past
    # the end of the track.
    state = {"offset": 999}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    top, height = ctl.thumb_geometry()
    assert top + height <= 20


# ── click / drag handling ────────────────────────────────────────────


def _mouse(event_type, y, *, button=MouseButton.LEFT):
    return MouseEvent(
        position=Point(x=0, y=y),
        event_type=event_type,
        button=button,
        modifiers=frozenset(),
    )


def test_click_on_track_jumps_to_proportional_offset():
    state = {"offset": 0}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    # viewport=20, max_scroll=80. Clicking row 10 (mid-track) →
    # scroll = round(10 * 80 / 19) = round(42.10..) = 42.
    ctl.handle_event(_mouse(MouseEventType.MOUSE_DOWN, y=10))
    assert ctl.dragging is True
    assert state["offset"] == 42


def test_drag_extends_after_initial_click():
    state = {"offset": 0}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    ctl.handle_event(_mouse(MouseEventType.MOUSE_DOWN, y=0))
    assert state["offset"] == 0
    ctl.handle_event(_mouse(MouseEventType.MOUSE_MOVE, y=19))
    assert state["offset"] == 80  # max_scroll
    ctl.handle_event(_mouse(MouseEventType.MOUSE_UP, y=19))
    assert ctl.dragging is False


def test_drag_without_prior_press_is_ignored():
    """A drift MOUSE_MOVE event before MOUSE_DOWN should not move the viewport."""
    state = {"offset": 5}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    ctl.handle_event(_mouse(MouseEventType.MOUSE_MOVE, y=15))
    assert state["offset"] == 5
    assert ctl.dragging is False


def test_mouse_up_clears_drag_flag():
    state = {"offset": 0}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    ctl.handle_event(_mouse(MouseEventType.MOUSE_DOWN, y=5))
    assert ctl.dragging is True
    ctl.handle_event(_mouse(MouseEventType.MOUSE_UP, y=5))
    assert ctl.dragging is False


def test_wheel_events_on_scrollbar_adjust_offset():
    state = {"offset": 5}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    ctl.handle_event(_mouse(MouseEventType.SCROLL_DOWN, y=0, button=MouseButton.NONE))
    assert state["offset"] == 6
    ctl.handle_event(_mouse(MouseEventType.SCROLL_UP, y=0, button=MouseButton.NONE))
    assert state["offset"] == 5


def test_click_below_track_is_clamped_to_max_scroll():
    state = {"offset": 0}
    ctl = _make_controller(total=100, viewport=20, scroll_state=state)
    ctl.handle_event(_mouse(MouseEventType.MOUSE_DOWN, y=50))  # well past row 19
    # Clamping enforces row <= viewport-1, so we land on max_scroll.
    assert state["offset"] == 80


# ── fragment build ───────────────────────────────────────────────────


def test_fragments_have_one_handler_per_row_with_trailing_newlines():
    state = {"offset": 0}
    ctl = _make_controller(total=100, viewport=4, scroll_state=state)
    fragments = build_scrollbar_fragments(ctl)
    # 4 rows + 3 separators = 7 fragments.
    assert len(fragments) == 7
    # Per-row fragments are 3-tuples with a callable mouse handler.
    row_fragments = [f for f in fragments if len(f) == 3]
    assert len(row_fragments) == 4
    for f in row_fragments:
        assert callable(f[2])


def test_fragments_alternate_thumb_and_track_styles():
    state = {"offset": 0}
    ctl = _make_controller(total=200, viewport=4, scroll_state=state)
    fragments = build_scrollbar_fragments(ctl)
    styles = [f[0] for f in fragments if len(f) == 3]
    # total=200/viewport=4 → thumb_height ≈ 1; offset=0 → thumb at row 0.
    assert styles[0] == "class:scrollbar.thumb"
    assert all(s == "class:scrollbar.track" for s in styles[1:])
