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

    def test_selected_item_uses_reverse_style(self):
        app = LanguageApp(console=None)
        app._lang_idx = 1
        lines = app._render_list()
        styles = [style for style, _content in lines]
        assert "reverse" in styles[1]

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
