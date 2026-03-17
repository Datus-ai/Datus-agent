# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import pytest

from datus.utils.text_utils import clean_text


class TestCleanText:
    def test_returns_empty_string_for_empty_input(self):
        assert clean_text("") == ""

    def test_returns_none_for_none_input(self):
        assert clean_text(None) is None

    def test_returns_non_string_unchanged(self):
        assert clean_text(42) == 42

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_normalizes_nfkc(self):
        # NFKC normalization: fullwidth chars -> ASCII
        result = clean_text("\uff41")  # fullwidth 'a'
        assert result == "a"

    def test_replaces_nbsp_with_space(self):
        result = clean_text("hello\u00a0world")
        assert result == "hello world"

    def test_removes_zero_width_space(self):
        result = clean_text("hello\u200bworld")
        assert result == "helloworld"

    def test_removes_bom(self):
        result = clean_text("\ufeffhello")
        assert result == "hello"

    def test_removes_control_characters(self):
        # \x00 through \x08 should be removed
        result = clean_text("hello\x00world")
        assert result == "helloworld"

    def test_preserves_newline(self):
        result = clean_text("line1\nline2")
        assert result == "line1\nline2"

    def test_preserves_tab(self):
        result = clean_text("col1\tcol2")
        assert result == "col1\tcol2"

    def test_normalizes_crlf_to_lf(self):
        result = clean_text("line1\r\nline2")
        assert result == "line1\nline2"

    def test_standalone_cr_removed_by_control_char_regex(self):
        # \r is \x0D which is in range \x0B-\x1F — removed by control char regex
        # before the line-break normalization step runs
        result = clean_text("line1\rline2")
        assert result == "line1line2"

    def test_removes_x0b_control_char(self):
        # \x0B (vertical tab) should be removed
        result = clean_text("hello\x0bworld")
        assert result == "helloworld"

    def test_normal_text_unchanged(self):
        text = "Hello, World! 123"
        assert clean_text(text) == text

    def test_unicode_chinese_preserved(self):
        text = "你好世界"
        assert clean_text(text) == text

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            ("  spaces  ", "spaces"),
            ("\nhello\n", "hello"),
            ("\thello\t", "hello"),
        ],
    )
    def test_strip_various_whitespace(self, input_text, expected):
        assert clean_text(input_text) == expected
