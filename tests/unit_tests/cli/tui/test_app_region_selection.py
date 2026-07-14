# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Wiring tests for per-region selection in :class:`DatusApp`.

Every visible pane — status bar, todo sidebar, queue preview, input
buffers — must be drag-copyable, with each pane's selection independent
of the others. These tests drive the public mouse handlers with synthetic
``MouseEvent`` sequences and assert the cross-region contract (mutual
exclusion, cross-pane release finalisation, clipboard side effects)
without running the prompt_toolkit Application loop.
"""

from __future__ import annotations

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from datus.cli.tui import app as app_mod
from datus.cli.tui.app import DatusApp
from datus.cli.tui.output_buffer import TUIOutputBuffer


@pytest.fixture
def captured_clipboard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captures: list[str] = []
    monkeypatch.setattr(app_mod, "copy_to_clipboard", lambda text: captures.append(text) or True)
    return captures


@pytest.fixture
def tui_app(monkeypatch: pytest.MonkeyPatch) -> DatusApp:
    buf = TUIOutputBuffer()
    buf.write("alpha beta\n")
    buf.write("gamma delta\n")

    app = DatusApp(
        status_tokens_fn=lambda: [("class:status-bar", "model: gpt-4 | tokens: 12")],
        dispatch_fn=lambda _text: None,
        output_buffer=buf,
        output_line_count_fn=buf.line_count,
        output_tokens_fn=buf.tokens,
        todo_tokens_fn=lambda: [("class:todo", "✓ task one\n· task two")],
        todo_has_items_fn=lambda: True,
    )
    monkeypatch.setattr(app, "_output_viewport_rows", lambda: 3)
    return app


def _click(event_type, x, y, button=MouseButton.LEFT):
    return MouseEvent(
        position=Point(x=x, y=y),
        event_type=event_type,
        button=button,
        modifiers=frozenset(),
    )


class TestRegionRegistry:
    def test_all_auxiliary_regions_registered(self, tui_app: DatusApp):
        assert set(tui_app._selection_regions) == {"status", "todo", "queue"}

    def test_input_and_search_buffers_registered(self, tui_app: DatusApp):
        assert set(tui_app._selection_buffers) == {"input", "search"}


class TestStatusBarSelection:
    def test_drag_across_status_bar_copies_text(self, tui_app: DatusApp, captured_clipboard: list[str]):
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
        assert captured_clipboard == ["model"]
        assert "Copied to clipboard" in tui_app._hint_text

    def test_output_drag_still_owns_status_bar_events(self, tui_app: DatusApp, captured_clipboard: list[str]):
        """During a scrollback drag the status bar keeps its autoscroll/finalise
        role — it must NOT start a status-local selection."""
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=2, y=0))
        assert tui_app._selection_regions["status"].selection.is_empty()
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
        assert tui_app._selection.dragging is False


class TestTodoSidebarSelection:
    def test_drag_across_sidebar_copies_task_text(self, tui_app: DatusApp, captured_clipboard: list[str]):
        region = tui_app._selection_regions["todo"]
        tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_MOVE, x=10, y=1))
        tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_UP, x=10, y=1))
        assert captured_clipboard == ["✓ task one\n· task two"]


class TestSelectionIndependence:
    def test_region_drag_clears_output_selection(self, tui_app: DatusApp, captured_clipboard: list[str]):
        # Finish an output-pane selection first; the highlight persists.
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
        assert not tui_app._selection.is_empty()
        # Starting a status-bar drag must drop the scrollback highlight.
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        assert tui_app._selection.is_empty()

    def test_output_drag_clears_region_selection(self, tui_app: DatusApp, captured_clipboard: list[str]):
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
        assert not tui_app._selection_regions["status"].selection.is_empty()
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        assert tui_app._selection_regions["status"].selection.is_empty()

    def test_region_drag_clears_sibling_region(self, tui_app: DatusApp, captured_clipboard: list[str]):
        todo = tui_app._selection_regions["todo"]
        tui_app._region_mouse_dispatch(todo, _click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._region_mouse_dispatch(todo, _click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        tui_app._region_mouse_dispatch(todo, _click(MouseEventType.MOUSE_UP, x=5, y=0))
        assert not todo.selection.is_empty()
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        assert todo.selection.is_empty()

    def test_region_drag_clears_input_buffer_selection(self, tui_app: DatusApp, captured_clipboard: list[str]):
        buffer = tui_app._input_area.buffer
        buffer.text = "select me"
        buffer.cursor_position = 0
        buffer.start_selection()
        buffer.cursor_position = 6
        assert app_mod._buffer_selection_text(buffer) == "select"
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        assert buffer.selection_state is None


class TestCrossRegionRelease:
    def test_region_drag_released_over_output_pane_finalises_copy(
        self, tui_app: DatusApp, captured_clipboard: list[str]
    ):
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        assert tui_app._selection_regions["status"].selection.dragging is True
        # Pointer wanders up into the scrollback and the button is released
        # there: the status drag must still close out and copy.
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=3, y=1))
        assert tui_app._selection_regions["status"].selection.dragging is False
        assert captured_clipboard == ["model"]
        # The output pane's own selection stays untouched.
        assert tui_app._selection.is_empty()

    def test_region_drag_move_over_output_pane_does_not_extend(self, tui_app: DatusApp, captured_clipboard: list[str]):
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        head_before = tui_app._selection_regions["status"].selection.head
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=3, y=1))
        assert tui_app._selection_regions["status"].selection.head == head_before
        assert tui_app._selection.is_empty()

    def test_output_drag_released_over_sidebar_finalises_copy(self, tui_app: DatusApp, captured_clipboard: list[str]):
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        todo = tui_app._selection_regions["todo"]
        tui_app._region_mouse_dispatch(todo, _click(MouseEventType.MOUSE_UP, x=0, y=0))
        assert tui_app._selection.dragging is False
        assert captured_clipboard == ["alpha"]
        assert todo.selection.is_empty()


class TestInputBufferCopy:
    def test_release_with_native_selection_copies_text(self, tui_app: DatusApp, captured_clipboard: list[str]):
        buffer = tui_app._input_area.buffer
        buffer.text = "SELECT 1 FROM t"
        buffer.cursor_position = 0
        buffer.start_selection()
        buffer.cursor_position = 8
        handler = tui_app._input_area.control.mouse_handler
        with set_app(tui_app.application):
            handler(_click(MouseEventType.MOUSE_UP, x=8, y=0))
        assert captured_clipboard == ["SELECT 1"]
        assert "Copied to clipboard" in tui_app._hint_text
        # The native highlight persists after the copy (still extractable).
        assert app_mod._buffer_selection_text(buffer) == "SELECT 1"

    def test_release_without_selection_copies_nothing(self, tui_app: DatusApp, captured_clipboard: list[str]):
        buffer = tui_app._input_area.buffer
        buffer.text = "plain text"
        handler = tui_app._input_area.control.mouse_handler
        with set_app(tui_app.application):
            handler(_click(MouseEventType.MOUSE_UP, x=2, y=0))
        assert captured_clipboard == []

    def test_search_buffer_release_copies_selection(self, tui_app: DatusApp, captured_clipboard: list[str]):
        buffer = tui_app._search_buffer
        buffer.text = "needle"
        buffer.cursor_position = 0
        buffer.start_selection()
        buffer.cursor_position = 6
        handler = tui_app._selection_buffers["search"]
        # The wrapped control handler lives on the search bar's BufferControl;
        # find it through the registered buffer to avoid poking layout internals.
        assert handler is buffer
        search_control = tui_app._search_bar.content.get_children()[1].content
        with set_app(tui_app.application):
            search_control.mouse_handler(_click(MouseEventType.MOUSE_UP, x=6, y=0))
        assert captured_clipboard == ["needle"]


class TestOffscreenRelease:
    """Releases that happen outside the terminal window must still copy.

    prompt_toolkit has no mouse capture and terminals either clamp the
    release onto an edge cell (which may be a decorative row with no
    selection logic) or drop it entirely. In the latter case the earliest
    proof of release is the next event whose button is no longer LEFT —
    a hover move (any-event tracking) or a fresh press.
    """

    def test_hover_move_after_lost_release_finalises_output_drag(
        self, tui_app: DatusApp, captured_clipboard: list[str]
    ):
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        # Pointer left the window, button released outside, pointer returns:
        # the terminal reports a buttonless hover move.
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=3, y=1, button=MouseButton.NONE))
        assert tui_app._selection.dragging is False
        assert captured_clipboard == ["alpha"]

    def test_hover_move_after_lost_release_finalises_region_drag(
        self, tui_app: DatusApp, captured_clipboard: list[str]
    ):
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        handler = tui_app._input_area.control.mouse_handler
        # Hover re-enters over the input area (a different surface).
        handler(_click(MouseEventType.MOUSE_MOVE, x=1, y=0, button=MouseButton.NONE))
        assert tui_app._selection_regions["status"].selection.dragging is False
        assert captured_clipboard == ["model"]

    def test_release_clamped_onto_separator_finalises_drag(self, tui_app: DatusApp, captured_clipboard: list[str]):
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        # A drag released below the terminal edge clamps onto the bottom
        # rows — separators and the hint line share this handler.
        result = tui_app._decoration_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=0))
        assert result is None
        assert tui_app._selection.dragging is False
        assert captured_clipboard == ["alpha"]

    def test_release_clamped_onto_scrollbar_gutter_finalises_drag(
        self, tui_app: DatusApp, captured_clipboard: list[str]
    ):
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        gutter_handler = tui_app._scrollbar_window.content.content.mouse_handler
        gutter_handler(_click(MouseEventType.MOUSE_UP, x=0, y=1))
        assert tui_app._selection.dragging is False
        assert captured_clipboard == ["alpha"]
        # The selection release must not have started a scrollbar drag.
        assert tui_app._scrollbar_controller.dragging is False

    def test_fresh_press_after_lost_release_copies_then_starts_new_drag(
        self, tui_app: DatusApp, captured_clipboard: list[str]
    ):
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        # Release was lost off-screen; the user clicks again in the output
        # pane. The dangling drag is copied first, then a new one begins.
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=2, y=1))
        assert captured_clipboard == ["alpha"]
        assert tui_app._selection.dragging is True
        anchor = tui_app._selection.anchor
        assert (anchor.line, anchor.column) == (1, 2)

    def test_hover_during_dangling_drag_on_status_bar_does_not_arm_autoscroll(
        self, tui_app: DatusApp, captured_clipboard: list[str]
    ):
        for _ in range(20):
            tui_app._output_buffer.write("line\n")
        tui_app._output_at_bottom = False
        tui_app._output_scroll_offset = 0
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=4, y=1))
        # Buttonless hover over the status bar: the release happened
        # elsewhere — finalise instead of arming downward autoscroll.
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=0, y=0, button=MouseButton.NONE))
        assert tui_app._selection.dragging is False
        assert tui_app._selection_autoscroll.direction == 0
        assert captured_clipboard != []

    def test_scrollbar_own_drag_release_still_works(self, tui_app: DatusApp, monkeypatch: pytest.MonkeyPatch):
        for _ in range(20):
            tui_app._output_buffer.write("line\n")
        monkeypatch.setattr(tui_app, "_output_viewport_rows", lambda: 10)
        tui_app._scrollbar_controller.handle_event(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        assert tui_app._scrollbar_controller.dragging is True
        gutter_handler = tui_app._scrollbar_window.content.content.mouse_handler
        # With no fragments painted (no render), the wrapped control falls
        # back to NotImplemented — but it must not intercept the event away
        # from the scrollbar while the scrollbar itself is dragging.
        gutter_handler(_click(MouseEventType.MOUSE_UP, x=0, y=5))
        # Forwarded release through the output pane still ends the drag.
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=0, y=5))
        assert tui_app._scrollbar_controller.dragging is False


class TestMultiClickApp:
    def test_output_double_click_copies_word(self, tui_app: DatusApp, captured_clipboard: list[str]):
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=2, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=2, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=2, y=0))
        assert captured_clipboard == ["alpha"]
        assert "Copied to clipboard" in tui_app._hint_text
        assert tui_app._selection.dragging is False

    def test_output_triple_click_copies_line(self, tui_app: DatusApp, captured_clipboard: list[str]):
        for _ in range(2):
            tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=2, y=1))
            tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=2, y=1))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=2, y=1))
        assert captured_clipboard == ["gamma", "gamma delta"]

    def test_status_bar_double_click_copies_word(self, tui_app: DatusApp, captured_clipboard: list[str]):
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=8, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=8, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=8, y=0))
        # Status text is "model: gpt-4 | tokens: 12"; char 8 sits in "gpt".
        assert captured_clipboard == ["gpt"]

    def test_input_triple_click_copies_whole_line(self, tui_app: DatusApp, captured_clipboard: list[str]):
        buffer = tui_app._input_area.buffer
        buffer.text = "SELECT 1 FROM t"
        buffer.cursor_position = 3
        handler = tui_app._input_area.control.mouse_handler
        with set_app(tui_app.application):
            for _ in range(3):
                handler(_click(MouseEventType.MOUSE_UP, x=3, y=0))
        assert captured_clipboard[-1] == "SELECT 1 FROM t"


def _mount_interaction_wizard(tui_app: DatusApp, **event_kwargs):
    """Mount a real InteractionApp panel (the ask_user / permission UI)."""
    import asyncio

    from datus.cli.interaction_app import InteractionApp
    from datus.schemas.interaction_event import InteractionEvent

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    app = InteractionApp([InteractionEvent(**event_kwargs)])
    panel = app.build_embedded_panel(future)
    tui_app.mount_wizard(panel)
    return loop


def _wizard_region_with_text(tui_app: DatusApp, needle: str):
    """Find the mounted wizard region whose content contains ``needle``."""
    for name in tui_app._wizard_selection_names:
        region = tui_app._selection_regions.get(name)
        if region is None:
            continue
        lines = region.lines()
        for row, line in enumerate(lines):
            text = "".join(f[1] for f in line if len(f) > 1)
            if needle in text:
                return region, row
    return None, -1


class TestWizardSelection:
    """Embedded wizards (ask_user / permission / slash-command pickers)
    must be drag-copyable like every other pane."""

    def test_interaction_content_is_copyable(self, tui_app: DatusApp, captured_clipboard: list[str]):
        loop = _mount_interaction_wizard(
            tui_app,
            title="Permission",
            content="DELETE FROM orders WHERE 1=1",
            content_type="text",
            choices={"y": "Yes", "n": "No"},
        )
        try:
            region, row = _wizard_region_with_text(tui_app, "DELETE FROM orders")
            tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_DOWN, x=0, y=row))
            tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_MOVE, x=200, y=row))
            tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_UP, x=200, y=row))
            assert captured_clipboard == ["  DELETE FROM orders WHERE 1=1"]
        finally:
            tui_app.unmount_wizard()
            loop.close()

    def test_interaction_choices_are_copyable(self, tui_app: DatusApp, captured_clipboard: list[str]):
        loop = _mount_interaction_wizard(
            tui_app,
            title="Ask",
            content="pick one",
            content_type="text",
            choices={"a": "Allow always", "d": "Deny"},
        )
        try:
            region, row = _wizard_region_with_text(tui_app, "Allow always")
            tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_DOWN, x=0, y=row))
            tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_MOVE, x=200, y=row))
            tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_UP, x=200, y=row))
            assert captured_clipboard == ["  1. Allow always"]
        finally:
            tui_app.unmount_wizard()
            loop.close()

    def test_free_text_buffer_gets_copy_wrapper(self, tui_app: DatusApp):
        loop = _mount_interaction_wizard(
            tui_app,
            title="Ask",
            content="describe it",
            content_type="text",
            allow_free_text=True,
        )
        try:
            wizard_buffers = [n for n in tui_app._selection_buffers if n.startswith("wizard:")]
            assert wizard_buffers != []
        finally:
            tui_app.unmount_wizard()
            loop.close()

    def test_unmount_cleans_wizard_registrations(self, tui_app: DatusApp):
        loop = _mount_interaction_wizard(
            tui_app,
            title="Ask",
            content="hello",
            content_type="text",
            choices={"y": "Yes"},
        )
        assert tui_app._wizard_selection_names != []
        tui_app.unmount_wizard()
        loop.close()
        assert tui_app._wizard_selection_names == []
        assert [n for n in tui_app._selection_regions if n.startswith("wizard:")] == []
        assert [n for n in tui_app._selection_buffers if n.startswith("wizard:")] == []

    def test_click_driven_widgets_are_left_alone(self, tui_app: DatusApp):
        """CheckboxList/RadioList-style controls (fragment-embedded mouse
        handlers) must keep their click behaviour — no selection wiring."""
        import asyncio

        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        from datus.cli.tui.wizard_host import EmbeddedWizard

        clicked: list[str] = []

        def _row_handler(_event):
            clicked.append("row")
            return None

        clickable = FormattedTextControl(lambda: [("", "[ ] option a", _row_handler)])
        plain = FormattedTextControl(lambda: [("", "static label")])
        container = HSplit([Window(clickable), Window(plain)])
        loop = asyncio.new_event_loop()
        panel = EmbeddedWizard(
            container=container,
            key_bindings=KeyBindings(),
            first_focus=None,
            done_future=loop.create_future(),
        )
        tui_app.mount_wizard(panel)
        try:
            # The clickable control keeps its class-level fragment dispatch…
            assert "mouse_handler" not in clickable.__dict__
            # …while the plain label got selection wiring.
            assert "mouse_handler" in plain.__dict__
            region, row = _wizard_region_with_text(tui_app, "static label")
            assert region.name.startswith("wizard:")
            assert row == 0
        finally:
            tui_app.unmount_wizard()
            loop.close()

    def test_wizard_selection_clears_output_selection(self, tui_app: DatusApp, captured_clipboard: list[str]):
        # Finished output-pane selection persists…
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        tui_app._output_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
        assert not tui_app._selection.is_empty()
        loop = _mount_interaction_wizard(
            tui_app,
            title="Ask",
            content="pick",
            content_type="text",
            choices={"y": "Yes"},
        )
        try:
            region, row = _wizard_region_with_text(tui_app, "Yes")
            # …until a drag starts inside the wizard.
            tui_app._region_mouse_dispatch(region, _click(MouseEventType.MOUSE_DOWN, x=0, y=row))
            assert tui_app._selection.is_empty()
        finally:
            tui_app.unmount_wizard()
            loop.close()


class TestEscapeClearsEverything:
    def test_any_selection_active_and_clear_all(self, tui_app: DatusApp, captured_clipboard: list[str]):
        assert tui_app._any_selection_active() is False
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        tui_app._status_mouse_handler(_click(MouseEventType.MOUSE_UP, x=5, y=0))
        assert tui_app._any_selection_active() is True
        tui_app._clear_other_selections(keep=None)
        assert tui_app._any_selection_active() is False

    def test_input_buffer_selection_counts_as_active(self, tui_app: DatusApp):
        buffer = tui_app._input_area.buffer
        buffer.text = "abc"
        buffer.cursor_position = 0
        buffer.start_selection()
        buffer.cursor_position = 2
        assert tui_app._any_selection_active() is True
        tui_app._clear_other_selections(keep=None)
        assert buffer.selection_state is None
