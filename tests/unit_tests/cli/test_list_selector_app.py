# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.list_selector_app.ListSelectorApp``.

CI-level: no TTY, no external deps. The prompt_toolkit Application is not
run — we test the data model, index logic, and formatted-text rendering.
"""

from __future__ import annotations

import pytest

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR
from datus.cli.list_selector_app import (
    ListItem,
    ListSelection,
    ListSelectorApp,
    _clip,
    _display_width,
)

pytestmark = pytest.mark.ci


class TestListItem:
    def test_defaults(self):
        item = ListItem(key="k", primary="Hello")
        assert item.key == "k"
        assert item.primary == "Hello"
        assert item.secondary == ""
        assert item.is_current is False

    def test_with_secondary(self):
        item = ListItem(key="k", primary="Hello", secondary="meta")
        assert item.secondary == "meta"

    def test_is_current(self):
        item = ListItem(key="k", primary="Hello", is_current=True)
        assert item.is_current is True


class TestListSelection:
    def test_key(self):
        sel = ListSelection(key="chosen")
        assert sel.key == "chosen"


class TestDisplayWidth:
    def test_ascii(self):
        assert _display_width("hello") == 5

    def test_cjk(self):
        assert _display_width("\u4f60\u597d") == 4

    def test_mixed(self):
        assert _display_width("a\u4f60b") == 4


class TestClip:
    def test_no_clip_needed(self):
        assert _clip("hello", 10) == "hello"

    def test_clips_ascii(self):
        assert _clip("hello world", 5) == "hello"

    def test_clips_cjk(self):
        assert _clip("\u4f60\u597d\u4e16\u754c", 5) == "\u4f60\u597d"

    def test_empty(self):
        assert _clip("", 10) == ""


class TestListSelectorAppInit:
    def test_empty_items(self):
        app = ListSelectorApp(title="Test", items=[])
        assert app._total == 0
        assert app.run() is None

    def test_cursor_starts_at_zero(self):
        items = [ListItem(key="a", primary="A"), ListItem(key="b", primary="B")]
        app = ListSelectorApp(title="Test", items=items)
        assert app._cursor == 0

    def test_cursor_starts_at_current(self):
        items = [
            ListItem(key="a", primary="A"),
            ListItem(key="b", primary="B", is_current=True),
            ListItem(key="c", primary="C"),
        ]
        app = ListSelectorApp(title="Test", items=items)
        assert app._cursor == 1

    def test_has_secondary_detects_items(self):
        items_no_secondary = [ListItem(key="a", primary="A")]
        app = ListSelectorApp(title="Test", items=items_no_secondary)
        assert app._has_secondary() is False

        items_with_secondary = [ListItem(key="a", primary="A", secondary="meta")]
        app2 = ListSelectorApp(title="Test", items=items_with_secondary)
        assert app2._has_secondary() is True


class TestRenderHeader:
    def test_shows_title(self):
        items = [ListItem(key="a", primary="A")]
        app = ListSelectorApp(title="Pick one", items=items)
        lines = app._render_header()
        text = "".join(content for _style, content in lines)
        assert "Pick one" in text

    def test_shows_count_when_scrollable(self):
        items = [ListItem(key=str(i), primary=f"Item {i}") for i in range(30)]
        app = ListSelectorApp(title="Many items", items=items)
        lines = app._render_header()
        text = "".join(content for _style, content in lines)
        assert "30 items" in text


class TestRenderList:
    def test_empty_items(self):
        app = ListSelectorApp(title="Test", items=[])
        lines = app._render_list()
        text = "".join(content for _style, content in lines)
        assert "nothing to show" in text

    def test_selected_item_uses_cursor_style(self):
        items = [ListItem(key="a", primary="A"), ListItem(key="b", primary="B")]
        app = ListSelectorApp(title="Test", items=items)
        app._cursor = 0
        lines = app._render_list()
        styles = [style for style, _content in lines]
        assert CLR_CURSOR in styles

    def test_current_item_uses_current_style(self):
        items = [
            ListItem(key="a", primary="A", is_current=True),
            ListItem(key="b", primary="B"),
        ]
        app = ListSelectorApp(title="Test", items=items)
        app._cursor = 1
        lines = app._render_list()
        styles = [style for style, _content in lines]
        assert CLR_CURRENT in styles

    def test_current_marker_text(self):
        items = [
            ListItem(key="a", primary="A", is_current=True),
            ListItem(key="b", primary="B"),
        ]
        app = ListSelectorApp(title="Test", items=items)
        app._cursor = 1
        lines = app._render_list()
        text = "".join(content for _style, content in lines)
        assert "\u2190 current" in text

    def test_secondary_line_shown_when_present(self):
        items = [ListItem(key="a", primary="Main", secondary="Details")]
        app = ListSelectorApp(title="Test", items=items)
        lines = app._render_list()
        text = "".join(content for _style, content in lines)
        assert "Main" in text
        assert "Details" in text

    def test_no_secondary_line_when_absent(self):
        items = [ListItem(key="a", primary="Main")]
        app = ListSelectorApp(title="Test", items=items)
        lines = app._render_list()
        text_lines = [content for _style, content in lines]
        for line in text_lines:
            assert "      " not in line or "\u2192" in line or line.strip() == ""

    def test_scroll_indicator(self):
        items = [ListItem(key=str(i), primary=f"Item {i}") for i in range(30)]
        app = ListSelectorApp(title="Test", items=items)
        lines = app._render_list()
        text = "".join(content for _style, content in lines)
        assert "of 30" in text


class TestRenderFooterHint:
    def test_contains_key_hints(self):
        items = [ListItem(key="a", primary="A")]
        app = ListSelectorApp(title="Test", items=items)
        lines = app._render_footer_hint()
        text = "".join(content for _style, content in lines)
        assert "navigate" in text
        assert "select" in text
        assert "cancel" in text


class TestEnsureVisible:
    def test_cursor_below_viewport(self):
        items = [ListItem(key=str(i), primary=f"Item {i}") for i in range(30)]
        app = ListSelectorApp(title="Test", items=items)
        app._cursor = 20
        app._offset = 0
        app._ensure_visible()
        assert app._offset > 0
        assert app._cursor >= app._offset
        assert app._cursor < app._offset + app._max_visible

    def test_cursor_above_viewport(self):
        items = [ListItem(key=str(i), primary=f"Item {i}") for i in range(30)]
        app = ListSelectorApp(title="Test", items=items)
        app._offset = 10
        app._cursor = 5
        app._ensure_visible()
        assert app._offset == 5
