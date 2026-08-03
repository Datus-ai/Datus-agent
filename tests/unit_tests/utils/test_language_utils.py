# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/utils/language_utils.py — CI level, zero external deps."""

import pytest

from datus.utils.language_utils import LANGUAGE_NAME_MAP, resolve_language_name


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
