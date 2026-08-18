# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/utils/language_utils.py — CI level, zero external deps."""

import pytest

from datus.utils.language_utils import (
    LANGUAGE_NAME_MAP,
    NATIVE_DIRECTIVE_MAP,
    build_fallback_directive,
    ensure_native_directive,
    resolve_language_name,
    resolve_native_directive,
)


class TestResolveLanguageName:
    @pytest.mark.parametrize(
        "code,expected",
        [
            ("en", "English"),
            ("zh", "Chinese"),
            ("zh-cn", "Chinese"),
            ("zh-tw", "Traditional Chinese"),
            ("ja", "Japanese"),
        ],
    )
    def test_known_codes(self, code, expected):
        assert resolve_language_name(code) == expected

    @pytest.mark.parametrize("code", ["EN", "ZH-CN", " zh ", "Ja"])
    def test_case_and_whitespace_insensitive(self, code):
        assert resolve_language_name(code) == LANGUAGE_NAME_MAP[code.strip().lower()]

    def test_unknown_code_returned_as_is(self):
        assert resolve_language_name("xx-yy") == "xx-yy"

    @pytest.mark.parametrize("code", ["", None])
    def test_empty_defaults_to_english(self, code):
        assert resolve_language_name(code) == "English"


class TestResolveNativeDirective:
    def test_every_named_language_has_a_native_directive(self):
        """A code the name map knows but the directive map doesn't would ship a
        bare English meta-instruction — the exact weak form this map exists to
        replace."""
        assert set(LANGUAGE_NAME_MAP) == set(NATIVE_DIRECTIVE_MAP)

    @pytest.mark.parametrize(
        "code,fragment",
        [
            ("zh", "简体中文"),
            ("zh-CN", "简体中文"),
            ("zh-tw", "繁體中文"),
            ("ja", "日本語"),
            ("en", "in English"),
        ],
    )
    def test_directive_is_written_in_the_target_language(self, code, fragment):
        assert fragment in resolve_native_directive(code)

    @pytest.mark.parametrize("code", ["xx-yy", "", None])
    def test_unknown_or_empty_has_no_directive(self, code):
        """No translation to offer — better a name-only line than a sentence in
        the wrong language."""
        assert resolve_native_directive(code) == ""


class TestBuildFallbackDirective:
    def test_includes_name_and_native_sentence(self):
        out = build_fallback_directive("zh")
        assert out.startswith("# Response Language")
        assert "Chinese (zh)" in out
        assert "简体中文" in out

    def test_unknown_code_keeps_name_line_only(self):
        out = build_fallback_directive("xx")
        assert out == "# Response Language\n- Use: xx (xx)"


class TestEnsureNativeDirective:
    """A home bootstrapped before this change keeps serving the old
    ``~/.datus/template`` copy (``ensure_templates`` uses ``replace=False``),
    so the native line has to survive an outdated render.
    """

    OLD_SECTION = "# Response Language\n- Use: Chinese (zh)\n- Exclude: code, SQL"

    def test_appends_to_a_stale_render(self):
        out = ensure_native_directive(self.OLD_SECTION, "zh")
        assert out.startswith(self.OLD_SECTION)
        assert out.endswith(NATIVE_DIRECTIVE_MAP["zh"])

    def test_does_not_duplicate_when_template_already_rendered_it(self):
        section = f"{self.OLD_SECTION}\n- {NATIVE_DIRECTIVE_MAP['zh']}"
        assert ensure_native_directive(section, "zh") == section

    @pytest.mark.parametrize("code", ["xx-yy", "", None])
    def test_untranslated_code_left_alone(self, code):
        assert ensure_native_directive(self.OLD_SECTION, code) == self.OLD_SECTION

    def test_empty_section_left_alone(self):
        """An empty render means "no directive"; a bare native line without the
        header would be worse than nothing."""
        assert ensure_native_directive("", "zh") == ""
