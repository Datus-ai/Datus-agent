# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.language_app.LanguageApp``.

CI-level: no TTY, no external deps. The prompt_toolkit Application is not
run — we test the data model, index logic, and formatted-text rendering.
"""

from __future__ import annotations

import pytest

from datus.cli.language_app import (
    LANGUAGE_CHOICES,
    SCOPE_CHOICES,
    LanguageApp,
    LanguageSelection,
    _Phase,
)

pytestmark = pytest.mark.ci


class TestLanguageSelection:
    def test_defaults(self):
        sel = LanguageSelection(code="zh")
        assert sel.code == "zh"
        assert sel.scope == "project"

    def test_explicit_scope(self):
        sel = LanguageSelection(code="en", scope="global")
        assert sel.scope == "global"


class TestLanguageChoices:
    def test_auto_is_first(self):
        keys = list(LANGUAGE_CHOICES.keys())
        assert keys[0] == "auto"

    def test_common_codes_present(self):
        for code in ("en", "zh", "ja", "ko", "es", "fr", "de", "pt", "ru", "it"):
            assert code in LANGUAGE_CHOICES


class TestScopeChoices:
    def test_project_and_global(self):
        assert "project" in SCOPE_CHOICES
        assert "global" in SCOPE_CHOICES


class TestLanguageAppInit:
    def test_default_index_matches_current(self):
        app = LanguageApp(console=None, current_language="zh")
        assert app._lang_keys[app._lang_idx] == "zh"

    def test_default_index_falls_back_to_zero(self):
        app = LanguageApp(console=None, current_language="unknown-code")
        assert app._lang_idx == 0

    def test_initial_phase_is_language(self):
        app = LanguageApp(console=None)
        assert app._phase == _Phase.LANGUAGE

    def test_scope_index_starts_at_zero(self):
        app = LanguageApp(console=None)
        assert app._scope_idx == 0


class TestRenderHeader:
    def test_language_phase_shows_current(self):
        app = LanguageApp(console=None, current_language="en", current_source="global")
        lines = app._render_header()
        text = "".join(content for _style, content in lines)
        assert "en" in text
        assert "English" in text
        assert "global" in text

    def test_language_phase_shows_not_set(self):
        app = LanguageApp(console=None, current_language="", current_source="not set")
        lines = app._render_header()
        text = "".join(content for _style, content in lines)
        assert "not set" in text

    def test_scope_phase_shows_selected_code(self):
        app = LanguageApp(console=None, scope_only="zh")
        lines = app._render_header()
        text = "".join(content for _style, content in lines)
        assert "zh" in text
        assert "Save" in text


class TestRenderList:
    def test_language_phase_lists_all_choices(self):
        app = LanguageApp(console=None, current_language="en", current_source="global")
        lines = app._render_list()
        text = "".join(content for _style, content in lines)
        for code in LANGUAGE_CHOICES:
            assert code in text

    def test_selected_item_uses_cursor_style(self):
        from datus.cli.cli_styles import CLR_CURSOR

        app = LanguageApp(console=None)
        app._lang_idx = 1
        lines = app._render_list()
        styles = [style for style, _content in lines]
        assert styles[1] == CLR_CURSOR

    def test_current_language_shows_arrow_marker(self):
        app = LanguageApp(console=None, current_language="zh")
        lines = app._render_list()
        text = "".join(content for _style, content in lines)
        assert "\u2190 current" in text

    def test_scope_phase_shows_project_and_global(self):
        app = LanguageApp(console=None, current_language="zh", scope_only="zh")
        lines = app._render_list()
        text = "".join(content for _style, content in lines)
        assert "project" in text
        assert "global" in text


class TestRenderFooterHint:
    def test_contains_navigate_and_select(self):
        app = LanguageApp(console=None)
        lines = app._render_footer_hint()
        text = "".join(content for _style, content in lines)
        assert "navigate" in text
        assert "select" in text
        assert "cancel" in text


class TestScopeOnlyInit:
    def test_scope_only_starts_in_scope_phase(self):
        app = LanguageApp(console=None, scope_only="zh")
        assert app._phase == _Phase.SCOPE
        assert app._selected_code == "zh"

    def test_scope_only_none_starts_in_language_phase(self):
        app = LanguageApp(console=None)
        assert app._phase == _Phase.LANGUAGE
        assert app._selected_code == ""


# ─────────────────────────────────────────────────────────────────────
# Dual-mode finish hook + embedded panel + key bindings (Enter cycles
# from LANGUAGE phase to SCOPE phase; second Enter resolves the
# future with a :class:`LanguageSelection`).
# ─────────────────────────────────────────────────────────────────────


import asyncio  # noqa: E402

from prompt_toolkit.keys import Keys  # noqa: E402

from datus.cli.tui.wizard_host import EmbeddedWizard  # noqa: E402


def _make_future():
    loop = asyncio.new_event_loop()
    return loop, loop.create_future()


def _evt():
    """Stub key binding event whose ``.app.invalidate()`` is a no-op."""
    return type("E", (), {"app": type("A", (), {"invalidate": lambda self: None})()})()


class TestEmbeddedPanel:
    def test_build_embedded_panel_returns_wizard(self):
        app = LanguageApp(console=None, current_language="en")
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            assert isinstance(panel, EmbeddedWizard)
            assert panel.done_future is fut
            assert app._on_done is not None
        finally:
            loop.close()

    def test_enter_in_language_phase_advances_to_scope_phase(self):
        """First Enter only stores the selection and switches phase —
        the future MUST stay pending until the second Enter (scope)."""
        app = LanguageApp(console=None, current_language="en")
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            enter = [b for b in panel.key_bindings.bindings if Keys.Enter in b.keys][0]
            enter.handler(_evt())
            assert not fut.done()
            assert app._phase == _Phase.SCOPE
            assert app._selected_code == "en"
        finally:
            loop.close()

    def test_enter_in_scope_phase_resolves_with_selection(self):
        app = LanguageApp(console=None, scope_only="zh")
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            enter = [b for b in panel.key_bindings.bindings if Keys.Enter in b.keys][0]
            enter.handler(_evt())
            assert fut.done()
            result = fut.result()
            assert isinstance(result, LanguageSelection)
            assert result.code == "zh"
            assert result.scope == "project"
        finally:
            loop.close()

    def test_escape_resolves_with_none(self):
        app = LanguageApp(console=None)
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            esc = [b for b in panel.key_bindings.bindings if Keys.Escape in b.keys][0]
            esc.handler(_evt())
            assert fut.done() and fut.result() is None
        finally:
            loop.close()

    def test_ctrl_c_resolves_with_none(self):
        app = LanguageApp(console=None)
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            ctrl_c = [b for b in panel.key_bindings.bindings if Keys.ControlC in b.keys][0]
            ctrl_c.handler(_evt())
            assert fut.done() and fut.result() is None
        finally:
            loop.close()


class TestKeyBindingNavigation:
    def test_up_in_language_phase_wraps_when_at_top(self):
        app = LanguageApp(console=None)
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            up = [b for b in panel.key_bindings.bindings if b.keys == ("up",)][0]
            assert app._lang_idx == 0
            up.handler(_evt())
            # Wraps to last entry.
            assert app._lang_idx == len(app._lang_keys) - 1
            # And the offset slides to make the last entry visible.
            assert app._lang_offset == max(0, len(app._lang_keys) - app._max_visible)
        finally:
            loop.close()

    def test_down_in_language_phase_advances(self):
        app = LanguageApp(console=None)
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            down = [b for b in panel.key_bindings.bindings if b.keys == ("down",)][0]
            assert app._lang_idx == 0
            down.handler(_evt())
            assert app._lang_idx == 1
        finally:
            loop.close()

    def test_down_in_scope_phase_clamps_at_end(self):
        app = LanguageApp(console=None, scope_only="zh")
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            down = [b for b in panel.key_bindings.bindings if b.keys == ("down",)][0]
            # Two SCOPE_CHOICES → idx clamps at 1.
            down.handler(_evt())
            down.handler(_evt())
            assert app._scope_idx == len(app._scope_keys) - 1
        finally:
            loop.close()

    def test_up_in_scope_phase_clamps_at_zero(self):
        app = LanguageApp(console=None, scope_only="zh")
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            up = [b for b in panel.key_bindings.bindings if b.keys == ("up",)][0]
            up.handler(_evt())
            assert app._scope_idx == 0
        finally:
            loop.close()

    def test_pageup_in_language_phase(self):
        app = LanguageApp(console=None)
        app._lang_idx = 5
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            pgup = [b for b in panel.key_bindings.bindings if b.keys == ("pageup",)][0]
            pgup.handler(_evt())
            assert app._lang_idx == max(0, 5 - app._max_visible)
        finally:
            loop.close()

    def test_pagedown_in_language_phase(self):
        app = LanguageApp(console=None)
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            pgdn = [b for b in panel.key_bindings.bindings if b.keys == ("pagedown",)][0]
            pgdn.handler(_evt())
            assert app._lang_idx == min(len(app._lang_keys) - 1, app._max_visible)
        finally:
            loop.close()


class TestFinishHook:
    def test_finish_without_on_done_is_noop(self):
        app = LanguageApp(console=None)
        assert app._on_done is None
        app._finish(None)  # No raise.

    def test_finish_invokes_on_done(self):
        captured: list = []
        app = LanguageApp(console=None)
        app._on_done = lambda r: captured.append(r)
        sel = LanguageSelection(code="zh", scope="global")
        app._finish(sel)
        assert captured == [sel]


class TestRunStandalone:
    def test_run_returns_selection(self):
        from unittest.mock import MagicMock, patch

        app = LanguageApp(console=None)
        sel = LanguageSelection(code="en", scope="project")
        fake_app = MagicMock()
        fake_app.run.return_value = sel
        with patch("datus.cli.language_app.Application", return_value=fake_app):
            assert app.run() is sel
        assert app._on_done is None

    def test_run_keyboard_interrupt_returns_none(self):
        from unittest.mock import MagicMock, patch

        app = LanguageApp(console=None)
        fake_app = MagicMock()
        fake_app.run.side_effect = KeyboardInterrupt
        with patch("datus.cli.language_app.Application", return_value=fake_app):
            assert app.run() is None

    def test_run_unexpected_exception_returns_none(self):
        from unittest.mock import MagicMock, patch

        from rich.console import Console

        app = LanguageApp(console=Console(no_color=True))
        fake_app = MagicMock()
        fake_app.run.side_effect = RuntimeError("boom")
        with patch("datus.cli.language_app.Application", return_value=fake_app):
            assert app.run() is None
