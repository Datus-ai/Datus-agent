# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Integration-style tests for :class:`DatusApp`'s selection + scrollbar wiring.

These tests inject ``MouseEvent`` sequences into the public output mouse
handler and verify the selection state, scrollbar drag, and clipboard
side-effects from the outside — i.e. the contract a real terminal would
observe. They avoid running the prompt_toolkit Application loop (which
needs a TTY).
"""

from __future__ import annotations

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from datus.cli.tui import app as app_mod
from datus.cli.tui.app import DatusApp
from datus.cli.tui.output_buffer import TUIOutputBuffer


@pytest.fixture
def captured_clipboard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture clipboard writes from the in-app copy path."""
    captures: list[str] = []
    monkeypatch.setattr(app_mod, "copy_to_clipboard", lambda text: captures.append(text) or True)
    return captures


@pytest.fixture
def tui_app(monkeypatch: pytest.MonkeyPatch) -> DatusApp:
    buf = TUIOutputBuffer()
    buf.write("alpha beta\n")
    buf.write("gamma delta\n")
    buf.write("epsilon zeta\n")

    app = DatusApp(
        status_tokens_fn=lambda: [],
        dispatch_fn=lambda _text: None,
        output_buffer=buf,
        output_line_count_fn=buf.line_count,
        output_tokens_fn=buf.tokens,
    )

    # Stub viewport rows to a deterministic value so autoscroll-edge
    # detection in the mouse handler is predictable.
    monkeypatch.setattr(app, "_output_viewport_rows", lambda: 3)
    # Force the scroll callback to start at the bottom so the click→drag
    # sequence pulls the viewport back from sticky-bottom predictably.
    return app


def _click(event_type, x, y, button=MouseButton.LEFT):
    return MouseEvent(
        position=Point(x=x, y=y),
        event_type=event_type,
        button=button,
        modifiers=frozenset(),
    )


def test_mouse_down_begins_selection_and_disengages_sticky_bottom(tui_app: DatusApp):
    assert tui_app._output_at_bottom is True
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=2, y=0))
    assert tui_app._selection.dragging is True
    assert tui_app._selection.anchor is not None
    assert tui_app._selection.anchor.line == 0
    assert tui_app._selection.anchor.column == 2
    assert tui_app._output_at_bottom is False


def test_drag_extends_selection_head(tui_app: DatusApp):
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=1))
    head = tui_app._selection.head
    assert head is not None
    assert (head.line, head.column) == (1, 5)


def test_mouse_up_with_text_writes_to_clipboard(tui_app: DatusApp, captured_clipboard: list[str]):
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
    assert captured_clipboard == ["alpha"]
    assert tui_app._selection.dragging is False
    # Selection is preserved post-release so the highlight stays visible.
    assert tui_app._selection.range() is not None


def test_mouse_up_with_empty_selection_does_not_write_clipboard(tui_app: DatusApp, captured_clipboard: list[str]):
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    # Release without moving = empty selection.
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
    assert captured_clipboard == []


def test_mouse_up_shows_copied_hint(tui_app: DatusApp, captured_clipboard: list[str]):
    """Successful copy on output pane release surfaces a transient hint."""
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
    assert captured_clipboard == ["alpha"]
    assert "Copied to clipboard" in tui_app._hint_text


def test_mouse_up_with_empty_selection_does_not_show_hint(tui_app: DatusApp):
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
    assert tui_app._hint_text == ""


def test_status_bar_release_shows_copied_hint(tui_app: DatusApp, captured_clipboard: list[str]):
    """Copy via status-bar MOUSE_UP also surfaces the hint."""
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=1))
    tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
    assert captured_clipboard != []
    assert "Copied to clipboard" in tui_app._hint_text


def test_show_hint_replaces_previous_text(tui_app: DatusApp):
    """A new hint replaces an older one rather than queuing."""
    tui_app.show_hint("first", duration=0)
    tui_app.show_hint("second", duration=0)
    assert tui_app._hint_text == "second"


def test_scrollbar_drag_suppresses_mouse_events_on_output_pane(tui_app: DatusApp):
    """While the scrollbar widget is mid-drag, output-pane events are ignored."""
    tui_app._scrollbar_controller._dragging = True
    # MOUSE_DOWN on output pane: must NOT start a selection.
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    assert tui_app._selection.dragging is False
    # MOUSE_UP forwarded to scrollbar; clears the drag flag.
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
    assert tui_app._scrollbar_controller.dragging is False


def test_scrollbar_drag_follows_mouse_jitter_off_gutter(tui_app: DatusApp, monkeypatch: pytest.MonkeyPatch):
    """A horizontal jitter into the output pane during a scrollbar drag must
    keep adjusting the scroll offset — not freeze the thumb mid-drag.

    The scrollbar window is only 1 column wide, so any horizontal mouse
    movement during a vertical drag can land in the output pane. We
    forward MOUSE_MOVE / MOUSE_UP events arriving there back to the
    scrollbar controller so the offset keeps tracking the cursor until
    the user releases the button.
    """
    for _ in range(20):
        tui_app._output_buffer.write("line\n")
    monkeypatch.setattr(tui_app, "_output_viewport_rows", lambda: 10)
    tui_app._output_at_bottom = False
    tui_app._output_scroll_offset = 0

    # Start drag on the scrollbar at row 0 (top).
    tui_app._scrollbar_controller.handle_event(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    assert tui_app._scrollbar_controller.dragging is True

    # Jitter into the output pane mid-drag: position.y arrives as the
    # buffer line index (vertical_scroll + viewport row). Forwarding
    # should translate it back and update the scroll offset.
    vertical_scroll = tui_app._get_output_scroll(tui_app._output_window)
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=0, y=vertical_scroll + 9))
    # max_scroll = total(23) - viewport(10) = 13. Row 9 → 9 / 9 * 13 = 13.
    assert tui_app._output_scroll_offset == 13

    # Releasing inside the output pane must end the drag.
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=vertical_scroll + 9))
    assert tui_app._scrollbar_controller.dragging is False


def test_scrollbar_drag_release_on_status_bar_ends_drag(tui_app: DatusApp, monkeypatch: pytest.MonkeyPatch):
    """Releasing the scrollbar drag while the cursor sits on the status bar
    must still clear the dragging flag (otherwise the next click anywhere
    in the app gets misrouted)."""
    for _ in range(20):
        tui_app._output_buffer.write("line\n")
    monkeypatch.setattr(tui_app, "_output_viewport_rows", lambda: 10)
    tui_app._scrollbar_controller.handle_event(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    assert tui_app._scrollbar_controller.dragging is True
    tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
    assert tui_app._scrollbar_controller.dragging is False


def test_drag_at_bottom_row_does_not_trigger_autoscroll(tui_app: DatusApp):
    """Dragging onto the last visible row must not start auto-scroll.

    prompt_toolkit clamps ``position.y`` to the last rendered row when
    the OS reports a mouse past the viewport, so we cannot infer
    "past-the-bottom" from position.y alone. Downward autoscroll is
    instead driven by the status bar's mouse handler — this test simply
    confirms the output pane does not arm it on equality.
    """
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    # viewport=3, total=3, so bottom row idx = 2. Drag right at it.
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=4, y=2))
    assert tui_app._selection_autoscroll.direction == 0


def test_drag_to_top_row_arms_scroll_up_when_scrolled_off_zero(tui_app: DatusApp):
    """Dragging onto the topmost visible row while scrolled-down arms up-scroll."""
    for _ in range(20):
        tui_app._output_buffer.write("line\n")
    # Manual scroll: vertical_scroll > 0 so there's content above the
    # current top edge to scroll into view.
    tui_app._output_at_bottom = False
    tui_app._output_scroll_offset = 5
    top_row = tui_app._get_output_scroll(tui_app._output_window)
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=top_row + 2))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=0, y=top_row))
    assert tui_app._selection_autoscroll.direction == -1


def test_top_edge_does_not_arm_when_already_at_buffer_top(tui_app: DatusApp):
    """At vertical_scroll=0 there's nothing to scroll into view — disarmed."""
    # tui_app fixture buffer has 3 lines, viewport=3 → vertical_scroll==0.
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=2))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=0, y=0))
    assert tui_app._selection_autoscroll.direction == 0


def test_status_bar_arms_scroll_down_during_drag(tui_app: DatusApp):
    """Mouse motion onto the status bar while dragging arms downward autoscroll."""
    for _ in range(20):
        tui_app._output_buffer.write("line\n")
    # Start the drag so the status handler treats subsequent events as
    # drag-extension rather than passive decoration.
    tui_app._output_at_bottom = False
    tui_app._output_scroll_offset = 0
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=0, y=0))
    assert tui_app._selection_autoscroll.direction == 1


def test_move_back_into_body_disarms_downward_autoscroll(tui_app: DatusApp, monkeypatch: pytest.MonkeyPatch):
    """Returning the cursor to the output body must cancel a status-bar-armed
    downward autoscroll.

    Regression: a drag that wandered onto the status bar armed +1, but the
    body's MOUSE_MOVE handler only disarmed -1, so scrolling continued
    indefinitely after the user moved the pointer back over the text.
    """
    for _ in range(20):
        tui_app._output_buffer.write("line\n")
    monkeypatch.setattr(tui_app, "_output_viewport_rows", lambda: 5)
    tui_app._output_at_bottom = False
    tui_app._output_scroll_offset = 2

    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=2))
    # Status bar arms downward scroll.
    tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=0, y=0))
    assert tui_app._selection_autoscroll.direction == 1

    # Move back into the body (not on the top edge): must disarm.
    top_row = tui_app._get_output_scroll(tui_app._output_window)
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=3, y=top_row + 2))
    assert tui_app._selection_autoscroll.direction == 0


def test_status_bar_finishes_drag_on_release(tui_app: DatusApp, captured_clipboard: list[str]):
    """A MOUSE_UP delivered to the status bar must close out the selection drag.

    Without this, releasing the mouse below the output pane would never
    fire the output's MOUSE_UP and the autoscroll loop + dragging flag
    would stay live indefinitely.
    """
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=1))
    assert tui_app._selection.dragging is True
    tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
    assert tui_app._selection.dragging is False
    assert tui_app._selection_autoscroll.direction == 0
    # Selection captured text should still have been written to clipboard.
    assert captured_clipboard != []


def test_escape_clears_active_selection(tui_app: DatusApp):
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
    tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
    assert not tui_app._selection.is_empty()

    # Simulate the Escape binding's body directly — exercising the
    # registered handler would require the full key processor, which is
    # not what this unit test is about.
    tui_app._selection.clear()
    assert tui_app._selection.is_empty()


def test_scroll_wheel_still_works(tui_app: DatusApp):
    """Wheel events keep their existing semantics after the mouse-handler refactor."""
    # Push a bigger buffer so there's something to scroll past.
    for _ in range(20):
        tui_app._output_buffer.write("line\n")
    initial_at_bottom = tui_app._output_at_bottom
    tui_app._output_mouse_handler(_click(MouseEventType.SCROLL_UP, x=0, y=0, button=MouseButton.NONE))
    # Wheel-up disengages sticky-bottom.
    assert tui_app._output_at_bottom != initial_at_bottom or initial_at_bottom is False


def test_set_output_scroll_offset_disengages_then_reengages_sticky_bottom(tui_app: DatusApp):
    for _ in range(20):
        tui_app._output_buffer.write("line\n")
    # Halfway scroll: sticky-bottom OFF.
    tui_app._set_output_scroll_offset(5)
    assert tui_app._output_at_bottom is False
    assert tui_app._output_scroll_offset == 5
    # At max: sticky-bottom re-engages.
    tui_app._set_output_scroll_offset(10_000)
    assert tui_app._output_at_bottom is True
    assert tui_app._output_scroll_offset == tui_app._output_max_scroll()
