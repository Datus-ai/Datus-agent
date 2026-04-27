# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import pytest

from datus.utils.text_utils import (
    LITELLM_EMPTY_PLACEHOLDER,
    LitellmPlaceholderStreamFilter,
    clean_text,
    redact_uri,
    strip_litellm_placeholder,
)


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


class TestStripLitellmPlaceholder:
    def test_exact_placeholder_returns_empty(self):
        assert strip_litellm_placeholder(LITELLM_EMPTY_PLACEHOLDER) == ""

    def test_placeholder_with_surrounding_whitespace_returns_empty(self):
        assert strip_litellm_placeholder(f"  {LITELLM_EMPTY_PLACEHOLDER}  ") == ""

    def test_normal_text_unchanged(self):
        assert strip_litellm_placeholder("Hello, world!") == "Hello, world!"

    def test_text_containing_placeholder_as_substring_unchanged(self):
        text = f"Error: {LITELLM_EMPTY_PLACEHOLDER} was returned"
        assert strip_litellm_placeholder(text) == text

    def test_empty_string_returns_empty(self):
        assert strip_litellm_placeholder("") == ""

    def test_none_returns_none(self):
        assert strip_litellm_placeholder(None) is None

    def test_non_string_returns_unchanged(self):
        assert strip_litellm_placeholder(42) == 42


class TestLitellmPlaceholderStreamFilter:
    @staticmethod
    def _drive(filter_obj: LitellmPlaceholderStreamFilter, chunks: list[str]) -> str:
        emitted = "".join(filter_obj.feed(chunk) for chunk in chunks)
        emitted += filter_obj.finalize()
        return emitted

    def test_normal_text_passes_through_unchanged(self):
        f = LitellmPlaceholderStreamFilter()
        assert self._drive(f, ["Hello, ", "world!"]) == "Hello, world!"

    def test_normal_text_starting_with_bracket_passes_through(self):
        f = LitellmPlaceholderStreamFilter()
        assert self._drive(f, ["[", "INFO", "] ready"]) == "[INFO] ready"

    def test_exact_placeholder_token_by_token_dropped(self):
        f = LitellmPlaceholderStreamFilter()
        chunks = list(LITELLM_EMPTY_PLACEHOLDER)
        assert self._drive(f, chunks) == ""

    def test_exact_placeholder_single_chunk_dropped(self):
        f = LitellmPlaceholderStreamFilter()
        assert self._drive(f, [LITELLM_EMPTY_PLACEHOLDER]) == ""

    def test_placeholder_chunked_typical_split_dropped(self):
        f = LitellmPlaceholderStreamFilter()
        chunks = ["[System", ": Empty ", "message content ", "sanitised to satisfy ", "protocol]"]
        assert "".join(chunks) == LITELLM_EMPTY_PLACEHOLDER
        assert self._drive(f, chunks) == ""

    def test_placeholder_followed_by_real_text_emits_only_tail(self):
        f = LitellmPlaceholderStreamFilter()
        chunks = [LITELLM_EMPTY_PLACEHOLDER, " continuing"]
        assert self._drive(f, chunks) == " continuing"

    def test_placeholder_split_then_followed_by_text(self):
        f = LitellmPlaceholderStreamFilter()
        chunks = [LITELLM_EMPTY_PLACEHOLDER[:10], LITELLM_EMPTY_PLACEHOLDER[10:] + " tail"]
        assert self._drive(f, chunks) == " tail"

    def test_prefix_that_diverges_is_flushed(self):
        f = LitellmPlaceholderStreamFilter()
        # "[Sys" is a prefix of placeholder; "tem error" diverges from "tem: ..."
        chunks = ["[Sys", "tem error]"]
        assert self._drive(f, chunks) == "[System error]"

    def test_partial_prefix_then_stream_ends_returns_buffer(self):
        f = LitellmPlaceholderStreamFilter()
        # Stream ends mid-placeholder-prefix — return what we got
        assert self._drive(f, ["[Sys"]) == "[Sys"

    def test_empty_and_none_chunks_ignored(self):
        f = LitellmPlaceholderStreamFilter()
        assert f.feed("") == ""
        assert f.feed(None) == ""  # type: ignore[arg-type]
        assert f.finalize() == ""

    def test_passthrough_state_preserved_across_chunks(self):
        f = LitellmPlaceholderStreamFilter()
        # First chunk diverges → passthrough activated
        assert f.feed("hi ") == "hi "
        # Subsequent chunks always pass through directly
        assert f.feed("[System: Empty") == "[System: Empty"
        assert f.feed(" tail") == " tail"

    def test_reset_allows_reuse(self):
        f = LitellmPlaceholderStreamFilter()
        assert self._drive(f, [LITELLM_EMPTY_PLACEHOLDER]) == ""
        # After finalize, the filter is reset and reusable
        assert self._drive(f, ["normal text"]) == "normal text"

    def test_finalize_returns_empty_when_buffer_exactly_placeholder(self):
        f = LitellmPlaceholderStreamFilter()
        # Feed the placeholder split into pieces so the buffer holds it at finalize
        for ch in LITELLM_EMPTY_PLACEHOLDER:
            f.feed(ch)
        assert f.finalize() == ""


class TestRedactUri:
    def test_empty_string_returns_empty(self):
        assert redact_uri("") == ""

    def test_none_returns_none(self):
        assert redact_uri(None) is None

    def test_uri_with_user_and_password_redacted(self):
        assert redact_uri("mysql://alice:secret@db.example.com:3306/sales") == (
            "mysql://alice:***@db.example.com:3306/sales"
        )

    def test_uri_with_password_only_redacted(self):
        assert redact_uri("postgresql://:secret@host/db") == "postgresql://***@host/db"

    def test_uri_without_password_unchanged(self):
        uri = "postgresql://alice@db.example.com:5432/sales"
        assert redact_uri(uri) == uri

    def test_uri_without_credentials_unchanged(self):
        uri = "sqlite:///tmp/local.db"
        assert redact_uri(uri) == uri

    def test_query_and_fragment_preserved(self):
        assert redact_uri("postgresql://u:p@host:5432/db?sslmode=require#frag") == (
            "postgresql://u:***@host:5432/db?sslmode=require#frag"
        )

    def test_invalid_uri_without_password_returned_as_is(self):
        # No password → early return, never touches the malformed port.
        weird = "http://[::1]:bad/path"
        assert redact_uri(weird) == weird

    def test_ipv6_host_brackets_preserved(self):
        assert redact_uri("postgresql://u:p@[::1]:5432/db") == "postgresql://u:***@[::1]:5432/db"

    def test_ipv6_host_without_port_brackets_preserved(self):
        assert redact_uri("postgresql://u:p@[::1]/db") == "postgresql://u:***@[::1]/db"

    def test_malformed_port_with_password_omits_port(self):
        # parts.port raises ValueError; we must not propagate, and the port
        # is dropped from the redacted netloc rather than crashing.
        assert redact_uri("http://u:p@example.com:bad/path") == "http://u:***@example.com/path"
