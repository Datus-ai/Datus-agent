# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli._render_utils``."""

from __future__ import annotations

from rich.table import Table

from datus.cli._render_utils import build_kv_table, build_row_table, format_cell


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


class TestHideEmptyColumns:
    """Pruning all-empty columns in the inference path.

    Motivation: BI adapters (Superset, Grafana) return placeholder fields
    like ``chart_ids: []`` and ``extra: {}`` from ``list_dashboards``
    because those platforms populate them only in the per-dashboard
    detail endpoint. Without this trim, ``.superset.list_dashboards``
    showed two always-empty columns that contributed nothing.
    """

    def test_inference_default_drops_all_empty_columns(self):
        rows = [
            {"id": 1, "name": "a", "chart_ids": [], "extra": {}},
            {"id": 2, "name": "b", "chart_ids": [], "extra": {}},
        ]
        table = build_row_table(rows)
        labels = [str(c.header) for c in table.columns]
        assert labels == ["id", "name"]

    def test_inference_keeps_column_with_any_nonempty_row(self):
        """A column with a single non-empty row is real data, not noise."""
        rows = [
            {"id": 1, "description": None},
            {"id": 2, "description": "has desc"},
            {"id": 3, "description": ""},
        ]
        table = build_row_table(rows)
        labels = [str(c.header) for c in table.columns]
        assert "description" in labels

    def test_inference_returns_none_when_all_columns_empty(self):
        rows = [{"a": None, "b": ""}, {"a": [], "b": {}}]
        assert build_row_table(rows) is None

    def test_explicit_columns_do_not_prune_by_default(self):
        """Explicitly listed columns stay — the caller hand-picked them."""
        rows = [{"x": "v", "empty": []}]
        table = build_row_table(rows, columns=[("x", "X"), ("empty", "E")])
        labels = [str(c.header) for c in table.columns]
        assert labels == ["X", "E"]

    def test_explicit_columns_honors_opt_in(self):
        rows = [{"x": "v", "empty": []}]
        table = build_row_table(rows, columns=[("x", "X"), ("empty", "E")], hide_empty_columns=True)
        labels = [str(c.header) for c in table.columns]
        assert labels == ["X"]

    def test_inference_honors_opt_out(self):
        rows = [{"x": "v", "empty": []}]
        table = build_row_table(rows, hide_empty_columns=False)
        labels = [str(c.header) for c in table.columns]
        assert labels == ["x", "empty"]

    def test_zero_and_false_are_not_treated_as_empty(self):
        """``0`` / ``False`` / ``0.0`` are meaningful values, keep them."""
        rows = [{"count": 0, "enabled": False, "ratio": 0.0}]
        table = build_row_table(rows)
        labels = [str(c.header) for c in table.columns]
        assert set(labels) == {"count", "enabled", "ratio"}


class TestFormatCellTruncation:
    def test_no_truncate_when_under_limit(self):
        assert format_cell("hello", max_width=100) == "hello"

    def test_middle_truncate_when_over_limit(self):
        text = "x" * 200
        out = format_cell(text, max_width=50)
        assert out != text
        assert "..." in out
        assert len(out) <= 50

    def test_truncate_applies_after_json_serialisation(self):
        """Nested dict is compacted to JSON *then* truncated."""
        value = {"a": "x" * 500}
        out = format_cell(value, max_width=80)
        assert "..." in out
        assert len(out) <= 80


class TestBuildKvTable:
    def test_single_dict_renders_as_two_column_table(self):
        table = build_kv_table({"id": 42, "name": "solo"})
        assert isinstance(table, Table)
        headers = [str(c.header) for c in table.columns]
        assert headers == ["Field", "Value"]
        field_cells = list(table.columns[0].cells)
        value_cells = list(table.columns[1].cells)
        assert field_cells == ["id", "name"]
        assert value_cells == ["42", "solo"]

    def test_preserves_insertion_order(self):
        table = build_kv_table({"z": 1, "a": 2, "m": 3})
        assert list(table.columns[0].cells) == ["z", "a", "m"]

    def test_nested_values_inline_json(self):
        table = build_kv_table({"extra": {"k": "v"}})
        value = list(table.columns[1].cells)[0]
        assert value == '{"k": "v"}'

    def test_long_value_truncated(self):
        big = {"stuff": "y" * 400}
        table = build_kv_table({"extra": big}, max_cell_width=80)
        value = list(table.columns[1].cells)[0]
        assert "..." in value
        assert len(value) <= 80

    def test_non_dict_returns_none(self):
        assert build_kv_table([{"id": 1}]) is None
        assert build_kv_table("not a dict") is None
        assert build_kv_table(42) is None

    def test_empty_dict_returns_none(self):
        assert build_kv_table({}) is None

    def test_title_passed_through(self):
        table = build_kv_table({"x": 1}, title="My Record")
        assert "My Record" in str(table.title)
