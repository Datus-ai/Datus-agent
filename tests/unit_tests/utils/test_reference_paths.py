# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import pytest

from datus.utils.reference_paths import normalize_reference_path, quote_path_segment, split_reference_path


class TestNormalizeReferencePath:
    def test_empty_string_returns_empty(self):
        assert normalize_reference_path("") == ""

    def test_none_like_falsy_returns_empty(self):
        assert normalize_reference_path("") == ""

    def test_simple_path_unchanged(self):
        assert normalize_reference_path("domain.table") == "domain.table"

    def test_strips_whitespace(self):
        assert normalize_reference_path("  domain.table  ") == "domain.table"

    def test_strips_trailing_punctuation_no_quotes(self):
        assert normalize_reference_path("domain.table,") == "domain.table"
        assert normalize_reference_path("domain.table.") == "domain.table"
        assert normalize_reference_path("domain.table;") == "domain.table"

    def test_quoted_final_segment_unquoted(self):
        result = normalize_reference_path('domain."my table"')
        assert result == "domain.my table"

    def test_quoted_segment_with_spaces_stops_at_space_outside_quotes(self):
        result = normalize_reference_path('"my domain".table extra')
        # Should stop at the space after 'table' (outside quotes)
        assert "extra" not in result

    def test_path_with_no_quotes_simple(self):
        result = normalize_reference_path("a.b.c")
        assert result == "a.b.c"

    def test_trailing_punctuation_with_multiple_chars(self):
        result = normalize_reference_path("schema.table)]}")
        assert result == "schema.table"

    def test_empty_after_strip_returns_empty(self):
        result = normalize_reference_path("   ")
        assert result == ""

    def test_only_quoted_segment(self):
        result = normalize_reference_path('"my table"')
        assert result == "my table"

    def test_three_level_path(self):
        result = normalize_reference_path("db.schema.table")
        assert result == "db.schema.table"

    def test_quoted_middle_segment_preserved(self):
        result = normalize_reference_path('"my schema".table')
        assert "my schema" in result
        assert "table" in result


class TestSplitReferencePath:
    def test_empty_string_returns_empty_list(self):
        assert split_reference_path("") == []

    def test_simple_path_splits_on_dots(self):
        assert split_reference_path("a.b.c") == ["a", "b", "c"]

    def test_strips_trailing_punctuation(self):
        assert split_reference_path("a.b.c,") == ["a", "b", "c"]

    def test_quoted_segment_unquoted(self):
        result = split_reference_path('domain."my table"')
        assert result == ["domain", "my table"]

    def test_all_quoted_segments_unquoted(self):
        result = split_reference_path('"my domain"."my table"')
        assert result == ["my domain", "my table"]

    def test_double_dot_skips_empty_segment(self):
        result = split_reference_path("domain..name")
        assert result == ["domain", "name"]

    def test_no_quotes_splits_on_dots_only(self):
        # Without quotes, split_reference_path splits only on '.' not whitespace
        result = split_reference_path("domain.table")
        assert "domain" in result
        assert "table" in result

    def test_whitespace_only_returns_empty(self):
        assert split_reference_path("   ") == []

    def test_single_component(self):
        assert split_reference_path("table") == ["table"]

    def test_three_level_unquoted(self):
        result = split_reference_path("db.schema.table")
        assert result == ["db", "schema", "table"]

    def test_quoted_segment_with_spaces_in_path(self):
        result = split_reference_path('domain.layer1."name with spaces"')
        assert result == ["domain", "layer1", "name with spaces"]

    def test_trailing_dot_ignored(self):
        result = split_reference_path("domain.table.")
        assert result == ["domain", "table"]


class TestQuotePathSegment:
    def test_empty_string_returns_empty(self):
        assert quote_path_segment("") == ""

    def test_simple_segment_unchanged(self):
        assert quote_path_segment("table") == "table"

    def test_segment_with_space_gets_quoted(self):
        result = quote_path_segment("my table")
        assert result == '"my table"'

    def test_segment_with_dot_gets_quoted(self):
        result = quote_path_segment("schema.name")
        assert result == '"schema.name"'

    def test_segment_with_at_sign_gets_quoted(self):
        result = quote_path_segment("user@domain")
        assert result == '"user@domain"'

    def test_existing_quotes_stripped_before_requoting(self):
        result = quote_path_segment('"my table"')
        assert result == '"my table"'

    def test_whitespace_only_returns_empty(self):
        assert quote_path_segment("   ") == ""

    def test_internal_quotes_removed(self):
        result = quote_path_segment('tab"le')
        # Internal quotes are removed, result should not have internal quotes
        assert '"' not in result.strip('"')

    def test_no_special_chars_no_quotes_added(self):
        result = quote_path_segment("simple_table")
        assert result == "simple_table"

    @pytest.mark.parametrize(
        "segment, should_be_quoted",
        [
            ("normal", False),
            ("with space", True),
            ("with.dot", True),
            ("with@at", True),
            ("with\ttab", True),
        ],
    )
    def test_quoting_decision(self, segment, should_be_quoted):
        result = quote_path_segment(segment)
        if should_be_quoted:
            assert result.startswith('"') and result.endswith('"')
        else:
            assert not (result.startswith('"') and result.endswith('"'))
