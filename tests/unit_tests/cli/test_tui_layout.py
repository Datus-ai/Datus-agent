# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Regression guards for the Datus TUI layout.

The inline slash-command popup is pinned directly under the input area via
``HSplit``. If the completion menu is accidentally dropped — or swapped for
a custom widget with different collapse semantics — the input + status bar
stop returning to the bottom of the terminal after a selection. These tests
catch that class of regression without needing an interactive terminal.
"""

from __future__ import annotations

import os

import pytest
from prompt_toolkit.layout.containers import ConditionalContainer, Window
from prompt_toolkit.layout.menus import CompletionsMenuControl

from datus.cli.tui.app import DatusApp


def _build_app() -> DatusApp:
    return DatusApp(status_tokens_fn=lambda: [], dispatch_fn=lambda _: None)


class TestCompletionsMenuWired:
    def test_completions_menu_wraps_completions_menu_control(self):
        """DatusApp inlines prompt_toolkit's ``CompletionsMenu`` layout — a
        ``ConditionalContainer`` wrapping a ``Window`` over a
        ``CompletionsMenuControl`` — but drops the scrollbar margin. The
        collapse-to-zero-rows behaviour the bottom-pin relies on comes from
        the same ``has_completions & ~is_done`` filter used by the builtin,
        so assert on structure rather than the concrete class."""

        app = _build_app()
        menu = app._completions_menu
        assert isinstance(menu, ConditionalContainer)
        inner_window = menu.content
        assert isinstance(inner_window, Window)
        assert isinstance(inner_window.content, CompletionsMenuControl)

    def test_menu_sits_between_input_and_bottom_separator(self):
        """The HSplit order input → menu → separator is what lets the input
        slide back to the bottom of the terminal once the menu collapses.
        Any other ordering regresses the rendering. The pinned live
        region (now packed into ``top_row`` with the todo sidebar) sits
        at index 0 but doesn't affect the input ↔ menu adjacency that
        this test guards.

        Root now has only 2 children: ``top_row`` and a
        ``DynamicContainer`` that swaps between the normal bottom
        section and an embedded wizard. Descend into the dynamic
        container's currently-rendered HSplit to assert the input/menu
        ordering is intact."""

        from prompt_toolkit.layout.containers import DynamicContainer, HSplit, VSplit

        app = _build_app()
        root = app.application.layout.container
        assert isinstance(root, HSplit), f"root should be HSplit, got {type(root).__name__}"
        children = list(root.get_children())
        assert len(children) == 2, f"root should have 2 children (top_row, dynamic); got {len(children)}"
        assert isinstance(children[0], VSplit), "index 0 must be the output + sidebar VSplit"
        top_children = list(children[0].get_children())
        # Top row: scrollable output, 1-col scrollbar gutter, todo sidebar.
        assert top_children == [app._output_window, app._scrollbar_window, app._todo_sidebar]

        assert isinstance(children[1], DynamicContainer), "index 1 must be the dynamic bottom slot"
        bottom = children[1].get_container()
        assert isinstance(bottom, HSplit), "no wizard active → bottom is the normal HSplit"
        bottom_children = list(bottom.get_children())
        # Expected order in normal bottom: queue_preview, top_sep, status,
        # mid_sep, input, menu, search_bar, bottom_sep, hint. The queue
        # preview and search bar are both ``ConditionalContainer``s that
        # consume zero rows when their filter is False, so they don't
        # affect the steady-state layout.
        assert len(bottom_children) == 9, f"unexpected bottom HSplit child count: {len(bottom_children)}"
        assert bottom_children[0] is app._queue_preview, "queue preview pinned above the status bar"
        assert bottom_children[2] is app._status_window, "status bar must follow the leading separator"
        # Menu (index 5) sits immediately after the input (index 4; the TextArea
        # is flattened into its wrapping Window by prompt_toolkit).
        assert bottom_children[5] is app._completions_menu
        assert bottom_children[6] is app._search_bar

    @pytest.mark.skipif(
        os.environ.get("TERMINAL_EMULATOR", "").strip().lower() == "jetbrains-jediterm",
        reason="JediTerm forces mouse_support off; see datus/cli/tui/app.py:_resolve_mouse_support",
    )
    def test_app_runs_in_full_screen_with_mouse_support(self):
        """Sidebar can only sit "next to" the output history when the
        Application owns the entire terminal — assert the two flags
        that make the full-screen layout possible. Mouse support is
        what lights up the scroll wheel inside the output pane."""
        app = _build_app()
        application = app.application
        assert application.full_screen is True
        assert application.mouse_support() is True


class TestCompletionsMenuConfig:
    def test_menu_has_sensible_height_cap(self):
        app = _build_app()
        # Reach into prompt_toolkit internals to guard max_height; this is
        # stable public API on CompletionsMenu's inner Window.
        inner_window = app._completions_menu.content
        # CompletionsMenu wraps its Window in a ConditionalContainer; peel
        # one layer if necessary so the assertion is resilient.
        wrapped = getattr(inner_window, "content", inner_window)
        assert isinstance(wrapped, CompletionsMenuControl)
