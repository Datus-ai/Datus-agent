# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.tui.region_selection.SelectableRegion`.

Pure-data tests: the region is driven with synthetic ``MouseEvent``
sequences and inspected from the outside (selection state, rendered
fragments, ``on_begin`` / ``on_copy`` callbacks) — no prompt_toolkit
Application is mounted.
"""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from datus.cli.tui.region_selection import SelectableRegion


def _click(event_type, x, y, button=MouseButton.LEFT):
    return MouseEvent(
        position=Point(x=x, y=y),
        event_type=event_type,
        button=button,
        modifiers=frozenset(),
    )


def _make_region(tokens, *, name="test"):
    copied: list[str] = []
    begun: list[str] = []
    region = SelectableRegion(
        name,
        lambda: tokens,
        on_begin=lambda: begun.append(name),
        on_copy=copied.append,
    )
    return region, copied, begun


class TestMouseStateMachine:
    def test_down_begins_selection_and_fires_on_begin(self):
        region, _copied, begun = _make_region([("", "hello world")])
        result = region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=2, y=0))
        assert result is None
        assert region.selection.dragging is True
        anchor = region.selection.anchor
        assert (anchor.line, anchor.column) == (0, 2)
        assert begun == ["test"]

    def test_move_extends_head_while_dragging(self):
        region, _copied, _begun = _make_region([("", "hello world")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        head = region.selection.head
        assert (head.line, head.column) == (0, 5)

    def test_up_copies_selected_text(self):
        region, copied, _begun = _make_region([("", "hello world")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=5, y=0))
        assert copied == ["hello"]
        assert region.selection.dragging is False
        # Highlight persists after the copy (cleared by Escape / next click).
        assert not region.selection.is_empty()

    def test_click_without_drag_copies_nothing(self):
        region, copied, _begun = _make_region([("", "hello world")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=3, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=3, y=0))
        assert copied == []
        assert region.selection.is_empty()

    def test_move_without_drag_is_not_handled(self):
        region, _copied, _begun = _make_region([("", "hello")])
        assert region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=1, y=0)) is NotImplemented

    def test_up_without_drag_is_not_handled(self):
        region, _copied, _begun = _make_region([("", "hello")])
        assert region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=1, y=0)) is NotImplemented

    def test_scroll_events_fall_through(self):
        region, _copied, _begun = _make_region([("", "hello")])
        event = _click(MouseEventType.SCROLL_UP, x=0, y=0, button=MouseButton.NONE)
        assert region.handle_mouse(event) is NotImplemented

    def test_right_button_down_is_not_handled(self):
        region, _copied, _begun = _make_region([("", "hello")])
        event = _click(MouseEventType.MOUSE_DOWN, x=0, y=0, button=MouseButton.RIGHT)
        assert region.handle_mouse(event) is NotImplemented

    def test_down_on_empty_region_is_not_handled(self):
        region, _copied, begun = _make_region([])
        # split_lines of an empty stream yields a single empty line, so a
        # click resolves to (0, 0) — dragging over nothing still copies
        # nothing because extraction of an empty line is empty.
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=0, y=0))
        assert region.selection.is_empty()

    def test_point_past_last_line_clamps_to_last_line(self):
        region, _copied, _begun = _make_region([("", "one\ntwo")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=3, y=99))
        assert (region.selection.head.line, region.selection.head.column) == (1, 3)


class TestMultiClick:
    def test_double_click_selects_and_copies_word(self):
        region, copied, _begun = _make_region([("", "hello world\nsecond line")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=8, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=8, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=8, y=0))
        assert copied == ["world"]
        assert region.selection.dragging is False
        rng = region.selection.range()
        assert (rng[0].column, rng[1].column) == (6, 11)

    def test_triple_click_selects_and_copies_line(self):
        region, copied, _begun = _make_region([("", "hello world\nsecond line")])
        for _ in range(2):
            region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=8, y=0))
            region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=8, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=8, y=0))
        assert copied == ["world", "hello world"]

    def test_double_click_on_blank_line_copies_nothing(self):
        region, copied, _begun = _make_region([("", "top\n\nbottom")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=1))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=0, y=1))
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=1))
        assert copied == []
        assert region.selection.is_empty()

    def test_fourth_click_starts_fresh_drag(self):
        region, copied, _begun = _make_region([("", "hello world")])
        for _ in range(3):
            region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=2, y=0))
            region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=2, y=0))
        # Word + line copies happened; the 4th click begins a normal drag.
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=2, y=0))
        assert region.selection.dragging is True
        assert copied == ["hello", "hello world"]


class TestExtraction:
    def test_multiline_extraction_joins_with_newline(self):
        region, copied, _begun = _make_region([("class:todo", "task one\ntask two\ntask three")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=8, y=2))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=8, y=2))
        assert copied == ["task one\ntask two\ntask thr"]

    def test_extraction_strips_trailing_padding_spaces(self):
        region, copied, _begun = _make_region([("", "abc   \ndef")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=3, y=1))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=3, y=1))
        assert copied == ["abc\ndef"]

    def test_cjk_selection_uses_char_indices(self):
        # position.x counts characters, not visual cells — two CJK glyphs
        # are char indices 0 and 1 even though they span 4 cells.
        region, copied, _begun = _make_region([("", "你好 world")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=2, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=2, y=0))
        assert copied == ["你好"]

    def test_tokens_fn_exception_yields_empty_content(self):
        def _boom():
            raise RuntimeError("provider failed")

        region = SelectableRegion("bad", _boom)
        assert region.lines() == [[]]
        assert region.extract_text() == ""


class TestRenderTokens:
    def test_unselected_content_passes_through(self):
        tokens = [("class:status-bar", "model: gpt"), ("", "\n"), ("class:status-bar", "ctx: 4k")]
        region, _copied, _begun = _make_region(tokens)
        rendered = region.render_tokens()
        assert ("class:status-bar", "model: gpt") in rendered
        assert ("class:status-bar", "ctx: 4k") in rendered

    def test_selected_range_gains_selection_class(self):
        region, _copied, _begun = _make_region([("class:x", "hello world")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=5, y=0))
        rendered = region.render_tokens()
        assert ("class:x class:selection", "hello") in rendered
        assert ("class:x", " world") in rendered

    def test_blank_line_padded_with_space(self):
        # Blank rows must paint at least one cell so prompt_toolkit's
        # rowcol_to_yx lookup can resolve mouse coordinates on them.
        region, _copied, _begun = _make_region([("", "top\n\nbottom")])
        rendered = region.render_tokens()
        lines: list[list] = [[]]
        for fragment in rendered:
            if fragment == ("", "\n"):
                lines.append([])
            else:
                lines[-1].append(fragment)
        assert lines[1] == [("", " ")]

    def test_padding_never_reaches_clipboard(self):
        region, copied, _begun = _make_region([("", "top\n\nbottom")])
        region.handle_mouse(_click(MouseEventType.MOUSE_DOWN, x=0, y=0))
        region.handle_mouse(_click(MouseEventType.MOUSE_MOVE, x=6, y=2))
        region.handle_mouse(_click(MouseEventType.MOUSE_UP, x=6, y=2))
        assert copied == ["top\n\nbottom"]
