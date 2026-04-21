# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli._render_utils``."""

from __future__ import annotations

from rich.table import Table

from datus.cli._render_utils import build_row_table, format_cell


class TestFormatCell:
    def test_none_becomes_empty_string(self):
        assert format_cell(None) == ""

    def test_bool_uses_lowercase_words(self):
        assert format_cell(True) == "true"
        assert format_cell(False) == "false"

    def test_dict_is_inline_json(self):
        assert format_cell({"k": "v"}) == '{"k": "v"}'

    def test_list_is_inline_json(self):
        assert format_cell([1, 2, 3]) == "[1, 2, 3]"

    def test_primitive_str(self):
        assert format_cell(42) == "42"
        assert format_cell("hi") == "hi"


class TestBuildRowTableInference:
    def test_returns_none_for_empty_list(self):
        assert build_row_table([]) is None

    def test_returns_none_for_non_list(self):
        assert build_row_table({"k": "v"}) is None
        assert build_row_table("a,b,c") is None

    def test_returns_none_for_list_of_primitives(self):
        assert build_row_table([1, 2, 3]) is None

    def test_infers_columns_from_first_row(self):
        t = build_row_table([{"id": 1, "name": "a"}])
        assert isinstance(t, Table)
        labels = [str(c.header) for c in t.columns]
        assert labels == ["id", "name"]

    def test_column_order_is_first_appearance_across_rows(self):
        t = build_row_table(
            [
                {"id": 1, "name": "a"},
                {"id": 2, "extra": "x"},
            ]
        )
        assert isinstance(t, Table)
        labels = [str(c.header) for c in t.columns]
        assert labels == ["id", "name", "extra"]

    def test_missing_cells_render_blank(self):
        t = build_row_table(
            [
                {"id": 1, "name": "a"},
                {"id": 2},  # missing 'name'
            ]
        )
        name_col = next(c for c in t.columns if str(c.header) == "name")
        assert list(name_col.cells) == ["a", ""]

    def test_nested_values_inline_json(self):
        t = build_row_table([{"k": {"nested": 1}}])
        col = next(c for c in t.columns if str(c.header) == "k")
        assert list(col.cells) == ['{"nested": 1}']


class TestBuildRowTableExplicitColumns:
    def test_columns_select_and_relabel(self):
        rows = [{"logic_name": "x", "name": "y", "uri": "u", "ignored": "z"}]
        t = build_row_table(
            rows,
            columns=[("logic_name", "Logic Name"), ("uri", "URI")],
        )
        labels = [str(c.header) for c in t.columns]
        assert labels == ["Logic Name", "URI"]
        # Columns not listed are dropped from the output.
        logic_col = next(c for c in t.columns if str(c.header) == "Logic Name")
        uri_col = next(c for c in t.columns if str(c.header) == "URI")
        assert list(logic_col.cells) == ["x"]
        assert list(uri_col.cells) == ["u"]

    def test_title_is_passed_through(self):
        t = build_row_table([{"x": 1}], title="My Table")
        assert "My Table" in str(t.title)

    def test_empty_columns_list_returns_none(self):
        assert build_row_table([{"x": 1}], columns=[]) is None
