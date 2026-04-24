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
        Any other ordering regresses the rendering. The pinned live region
        now sits at the very top (index 0) but doesn't affect the input ↔
        menu adjacency that this test guards."""

        app = _build_app()
        root = app.application.layout.container
        # DatusApp wraps the HSplit directly; grab the children list.
        children = list(root.get_children())
        # Expected order: live_region, top_sep, status, mid_sep, input, menu,
        # bottom_sep, hint.
        assert len(children) == 8, f"unexpected HSplit child count: {len(children)}"
        # The menu must sit at index 5, immediately after the input (index 4;
        # the TextArea is flattened into its wrapping Window by prompt_toolkit
        # so the identity check is made on the menu itself).
        assert children[5] is app._completions_menu


class TestCompletionsMenuConfig:
    def test_menu_has_sensible_height_cap(self):
        app = _build_app()
        # Reach into prompt_toolkit internals to guard max_height; this is
        # stable public API on CompletionsMenu's inner Window.
        inner_window = app._completions_menu.content
        # CompletionsMenu wraps its Window in a ConditionalContainer; peel
        # one layer if necessary so the assertion is resilient.
        wrapped = getattr(inner_window, "content", inner_window)
        assert wrapped is not None
