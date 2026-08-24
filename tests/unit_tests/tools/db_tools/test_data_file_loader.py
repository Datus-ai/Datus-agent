# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.tools.db_tools.data_file_loader`.

Scope is deliberately everything that needs no DuckDB extension: naming, the SQL
guard, header detection, and the CSV/JSON/Parquet readers (all statically linked
into the wheel). Spreadsheet loading needs the autoloadable ``excel`` extension,
which is a real external dependency, so those live in
``tests/integration/tools/db_tools/test_data_file_loader_excel.py``.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from datus.tools.db_tools.data_file_loader import (
    DataFileError,
    build_table_name,
    column_letter,
    detect_header_row,
    find_file_reading_functions,
    inspect_file,
    load_file,
    ownership_tag,
    registered_objects,
    sanitize_identifier,
)


@pytest.fixture
def connection(tmp_path):
    con = duckdb.connect(str(tmp_path / "catalog.duckdb"))
    try:
        yield con
    finally:
        con.close()


# --------------------------------------------------------------- identifiers


class TestSanitizeIdentifier:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("orders", "orders"),
            ("Order Detail", "order_detail"),
            ("2024 Sales Data", "2024_sales_data"),
            ("Café-Sales.v2", "cafe_sales_v2"),
            ("__weird__name__", "weird_name"),
            ("a---b", "a_b"),
            ("Q3明细", "q3"),
            ("a b", "a_b"),
            ("x/y\\z", "x_y_z"),
            ("tab\tsep", "tab_sep"),
            ("1-2-3", "1_2_3"),
        ],
    )
    def test_reduces_to_lower_snake_ascii(self, raw, expected):
        assert sanitize_identifier(raw) == expected

    @pytest.mark.parametrize("raw", ["明细", "売上", "!!!", "", "。、；"])
    def test_returns_empty_when_nothing_ascii_survives(self, raw):
        """Callers depend on '' to trigger the positional/hash fallback."""
        assert sanitize_identifier(raw) == ""


class TestBuildTableName:
    def test_file_and_sheet_are_combined(self):
        assert build_table_name(rel_path="jeffshop.xlsx", sheet="Orders", sheet_index=0, taken=[]) == "jeffshop_orders"

    def test_non_ascii_sheet_falls_back_to_position(self):
        assert build_table_name(rel_path="jeffshop.xlsx", sheet="汇总", sheet_index=1, taken=[]) == "jeffshop_s2"

    def test_non_ascii_filename_falls_back_to_hash(self):
        name = build_table_name(rel_path="销售明细.xlsx", sheet=None, sheet_index=0, taken=[])
        assert name.startswith("t_")
        assert name.replace("_", "").isalnum()

    def test_leading_digit_is_prefixed(self):
        """A bare SQL identifier may not start with a digit."""
        name = build_table_name(rel_path="2024report.csv", sheet=None, sheet_index=0, taken=[])
        assert name == "t_2024report"

    def test_sheet_starting_with_a_digit_is_prefixed(self):
        name = build_table_name(rel_path="book.xlsx", sheet="2024", sheet_index=0, taken=[])
        assert name == "book_s2024"

    def test_is_deterministic(self):
        """Reloads must resolve to the same name or the catalog grows forever."""
        args = dict(rel_path="a/b/orders.xlsx", sheet="明细", sheet_index=2)
        assert build_table_name(**args, taken=[]) == build_table_name(**args, taken=[])

    def test_collision_gets_a_stable_suffix(self):
        first = build_table_name(rel_path="a/data.csv", sheet=None, sheet_index=0, taken=[])
        second = build_table_name(rel_path="b/data.csv", sheet=None, sheet_index=0, taken=[first])
        third = build_table_name(rel_path="b/data.csv", sheet=None, sheet_index=0, taken=[first])
        assert second != first
        assert second == third

    def test_result_is_always_a_legal_bare_identifier(self):
        for rel_path, sheet in [
            ("明细.xlsx", "汇总"),
            ("2024 报表.xlsx", None),
            ("a b/c d.csv", None),
            ("!!!.csv", "!!!"),
        ]:
            name = build_table_name(rel_path=rel_path, sheet=sheet, sheet_index=0, taken=[])
            assert name.replace("_", "").isalnum()
            assert not name[0].isdigit()
            assert name == name.lower()


class TestColumnLetter:
    @pytest.mark.parametrize("index,expected", [(1, "A"), (4, "D"), (26, "Z"), (27, "AA"), (52, "AZ"), (53, "BA")])
    def test_maps_index_to_spreadsheet_letters(self, index, expected):
        assert column_letter(index) == expected

    @pytest.mark.parametrize("index", [0, -1])
    def test_rejects_non_positive(self, index):
        with pytest.raises(ValueError):
            column_letter(index)


class TestOwnershipTag:
    def test_distinguishes_sheets_of_one_file(self):
        assert ownership_tag("a.xlsx", "S1") != ownership_tag("a.xlsx", "S2")

    def test_distinguishes_same_sheet_name_in_different_files(self):
        assert ownership_tag("a.xlsx", "S1") != ownership_tag("b.xlsx", "S1")

    def test_sheetless_differs_from_empty_sheet_name(self):
        assert ownership_tag("a.csv", None) != ownership_tag("a.csv", "")

    def test_is_stable_for_the_same_source(self):
        assert ownership_tag("a/b.xlsx", "S") == ownership_tag("a/b.xlsx", "S")


# ------------------------------------------------------------------ SQL guard


class TestFindFileReadingFunctions:
    @pytest.mark.parametrize(
        "sql,expected",
        [
            ("SELECT * FROM read_csv_auto('/etc/passwd')", ["read_csv_auto"]),
            ("select * from READ_XLSX('x.xlsx')", ["read_xlsx"]),
            ("SELECT * FROM read_parquet ( 'x' )", ["read_parquet"]),
            ("SELECT * FROM glob('/**')", ["glob"]),
            ("SELECT read_text('/etc/hosts')", ["read_text"]),
            ("SELECT read_blob('/x')", ["read_blob"]),
            ("SELECT * FROM parquet_scan('/x')", ["parquet_scan"]),
            ("SELECT * FROM read_json_auto('/x')", ["read_json_auto"]),
        ],
    )
    def test_detects_file_readers(self, sql, expected):
        assert find_file_reading_functions(sql) == expected

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT category, sum(amount) FROM orders GROUP BY 1",
            "SELECT count(*) FROM t",
            "SELECT date_trunc('month', d) FROM t",
            "SELECT * FROM my_read_csv_helper",
        ],
    )
    def test_leaves_ordinary_sql_alone(self, sql):
        assert find_file_reading_functions(sql) == []

    def test_reports_each_offender_once(self):
        sql = "SELECT * FROM read_csv_auto('a') UNION ALL SELECT * FROM read_csv_auto('b')"
        assert find_file_reading_functions(sql) == ["read_csv_auto"]

    def test_detects_reader_nested_in_a_subquery(self):
        """A bypass would otherwise only need one level of nesting."""
        sql = "SELECT * FROM (SELECT * FROM read_csv_auto('/etc/passwd')) x WHERE 1=1"
        assert find_file_reading_functions(sql) == ["read_csv_auto"]

    def test_detects_reader_inside_a_cte(self):
        sql = "WITH x AS (SELECT * FROM read_parquet('/secret')) SELECT * FROM x"
        assert find_file_reading_functions(sql) == ["read_parquet"]


# --------------------------------------------------------------- header probe


class TestDetectHeaderRow:
    def test_skips_title_and_blank_rows(self):
        grid = [
            ("2024 Sales Detail", None, None),
            (None, None, None),
            ("region", "qty", "amount"),
            ("east", "3", "10.5"),
        ]
        assert detect_header_row(grid) == 3

    def test_first_row_when_already_a_header(self):
        assert detect_header_row([("a", "b"), ("1", "2")]) == 1

    def test_single_column_sheet_falls_back_to_first_non_empty(self):
        assert detect_header_row([(None,), ("only",), ("1",)]) == 2

    def test_returns_none_for_a_fully_empty_grid(self):
        assert detect_header_row([(None, None), (None, None)]) is None

    def test_returns_none_for_no_rows(self):
        assert detect_header_row([]) is None

    def test_ignores_a_wide_row_followed_by_a_narrow_one(self):
        """A two-cell note above the real header must not win."""
        grid = [
            ("note:", "draft", None),
            (None, None, None),
            ("region", "qty", "amount"),
            ("east", "3", "10.5"),
        ]
        assert detect_header_row(grid) == 3

    def test_whitespace_only_cells_count_as_empty(self):
        grid = [("   ", "  "), ("region", "qty"), ("east", "3")]
        assert detect_header_row(grid) == 2


# -------------------------------------------------------------------- loading


class TestLoadCsv:
    def test_creates_a_queryable_view(self, tmp_path, connection):
        source = tmp_path / "traffic.csv"
        source.write_text("day,visits\n2024-07-01,120\n2024-07-02,88\n")

        loaded, skipped = load_file(source, "traffic.csv", connection=connection, conversion_cache_dir=tmp_path)

        assert skipped == []
        assert len(loaded) == 1
        table = loaded[0]
        assert table.table == "traffic"
        assert table.row_count == 2
        assert table.preview_columns == ["day", "visits"]
        assert connection.execute('SELECT sum(visits) FROM "traffic"').fetchone()[0] == 208

    def test_profile_reports_per_column_stats(self, tmp_path, connection):
        source = tmp_path / "t.csv"
        source.write_text("n\n1\n2\n3\n")
        loaded, _ = load_file(source, "t.csv", connection=connection, conversion_cache_dir=tmp_path)

        profile = {item["column_name"]: item for item in loaded[0].columns}
        assert profile["n"]["column_type"] == "BIGINT"
        assert float(profile["n"]["null_percentage"]) == 0.0
        assert profile["n"]["count"] == 3

    def test_tsv_uses_tab_delimiter(self, tmp_path, connection):
        source = tmp_path / "t.tsv"
        source.write_text("a\tb\n1\t2\n")
        loaded, _ = load_file(source, "t.tsv", connection=connection, conversion_cache_dir=tmp_path)
        assert loaded[0].preview_columns == ["a", "b"]

    def test_view_tracks_later_edits(self, tmp_path, connection):
        """The lazy view is the whole point: no reload needed for a CSV edit."""
        source = tmp_path / "t.csv"
        source.write_text("n\n1\n")
        load_file(source, "t.csv", connection=connection, conversion_cache_dir=tmp_path)
        assert connection.execute('SELECT count(*) FROM "t"').fetchone()[0] == 1

        source.write_text("n\n1\n2\n3\n")
        assert connection.execute('SELECT count(*) FROM "t"').fetchone()[0] == 3

    def test_materialize_snapshots_instead(self, tmp_path, connection):
        source = tmp_path / "t.csv"
        source.write_text("n\n1\n")
        loaded, _ = load_file(source, "t.csv", connection=connection, conversion_cache_dir=tmp_path, materialize=True)
        assert loaded[0].materialized is True

        source.write_text("n\n1\n2\n")
        assert connection.execute('SELECT count(*) FROM "t"').fetchone()[0] == 1

    def test_reload_can_flip_materialize_back_to_view(self, tmp_path, connection):
        """CREATE OR REPLACE cannot change object kind, so the drop must handle it."""
        source = tmp_path / "t.csv"
        source.write_text("n\n1\n")
        kwargs = dict(connection=connection, conversion_cache_dir=tmp_path)
        load_file(source, "t.csv", materialize=True, **kwargs)
        load_file(source, "t.csv", materialize=False, existing_objects=registered_objects(connection), **kwargs)

        views = connection.execute("SELECT view_name FROM duckdb_views() WHERE database_name != 'system'").fetchall()
        assert ("t",) in views

    def test_reload_can_flip_view_to_materialized(self, tmp_path, connection):
        source = tmp_path / "t.csv"
        source.write_text("n\n1\n")
        kwargs = dict(connection=connection, conversion_cache_dir=tmp_path)
        load_file(source, "t.csv", materialize=False, **kwargs)
        load_file(source, "t.csv", materialize=True, existing_objects=registered_objects(connection), **kwargs)

        tables = connection.execute("SELECT table_name FROM duckdb_tables() WHERE database_name != 'system'").fetchall()
        assert ("t",) in tables


class TestLoadJsonAndParquet:
    def test_json_array(self, tmp_path, connection):
        source = tmp_path / "d.json"
        source.write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')
        loaded, _ = load_file(source, "d.json", connection=connection, conversion_cache_dir=tmp_path)
        assert loaded[0].row_count == 2
        assert loaded[0].preview_columns == ["a", "b"]

    def test_jsonl(self, tmp_path, connection):
        source = tmp_path / "d.jsonl"
        source.write_text('{"a": 1}\n{"a": 2}\n')
        loaded, _ = load_file(source, "d.jsonl", connection=connection, conversion_cache_dir=tmp_path)
        assert loaded[0].row_count == 2

    def test_parquet(self, tmp_path, connection):
        source = tmp_path / "d.parquet"
        pd.DataFrame({"m": [1, 2, 3]}).to_parquet(source, index=False)
        loaded, _ = load_file(source, "d.parquet", connection=connection, conversion_cache_dir=tmp_path)
        assert loaded[0].row_count == 3


class TestIdempotency:
    def _load(self, connection, path, rel_path, tmp_path):
        return load_file(
            path,
            rel_path,
            connection=connection,
            conversion_cache_dir=tmp_path,
            existing_objects=registered_objects(connection),
        )

    def test_repeated_loads_do_not_grow_the_catalog(self, tmp_path, connection):
        source = tmp_path / "t.csv"
        source.write_text("n\n1\n")

        first, _ = self._load(connection, source, "t.csv", tmp_path)
        second, _ = self._load(connection, source, "t.csv", tmp_path)
        third, _ = self._load(connection, source, "t.csv", tmp_path)

        names = [item.table for item in first]
        assert [item.table for item in second] == names
        assert [item.table for item in third] == names
        assert sorted(registered_objects(connection)) == sorted(names)

    def test_distinct_files_with_the_same_stem_coexist(self, tmp_path, connection):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a/data.csv").write_text("x\n1\n")
        (tmp_path / "b/data.csv").write_text("y\n2\n3\n")

        first, _ = self._load(connection, tmp_path / "a/data.csv", "a/data.csv", tmp_path)
        second, _ = self._load(connection, tmp_path / "b/data.csv", "b/data.csv", tmp_path)

        assert first[0].table != second[0].table
        assert connection.execute(f'SELECT count(*) FROM "{first[0].table}"').fetchone()[0] == 1
        assert connection.execute(f'SELECT count(*) FROM "{second[0].table}"').fetchone()[0] == 2

    def test_reloading_one_of_two_colliding_files_keeps_its_name(self, tmp_path, connection):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a/data.csv").write_text("x\n1\n")
        (tmp_path / "b/data.csv").write_text("y\n2\n")

        first, _ = self._load(connection, tmp_path / "a/data.csv", "a/data.csv", tmp_path)
        self._load(connection, tmp_path / "b/data.csv", "b/data.csv", tmp_path)
        again, _ = self._load(connection, tmp_path / "a/data.csv", "a/data.csv", tmp_path)

        assert again[0].table == first[0].table

    def test_ownership_comment_records_provenance(self, tmp_path, connection):
        source = tmp_path / "t.csv"
        source.write_text("n\n1\n")
        load_file(source, "sub/t.csv", connection=connection, conversion_cache_dir=tmp_path)
        assert registered_objects(connection)["t"] == ownership_tag("sub/t.csv", None)


class TestUnsupportedFormats:
    @pytest.mark.parametrize("name", ["book.ods", "book.xlsb"])
    def test_names_a_convertible_alternative(self, tmp_path, connection, name):
        source = tmp_path / name
        source.write_bytes(b"x")
        with pytest.raises(DataFileError) as excinfo:
            load_file(source, name, connection=connection, conversion_cache_dir=tmp_path)
        assert ".xlsx" in str(excinfo.value)

    def test_unknown_extension_lists_what_is_supported(self, tmp_path, connection):
        source = tmp_path / "notes.txt"
        source.write_text("hi")
        with pytest.raises(DataFileError) as excinfo:
            load_file(source, "notes.txt", connection=connection, conversion_cache_dir=tmp_path)
        assert ".csv" in str(excinfo.value)

    def test_extensionless_file_is_rejected_clearly(self, tmp_path, connection):
        source = tmp_path / "README"
        source.write_text("hi")
        with pytest.raises(DataFileError) as excinfo:
            load_file(source, "README", connection=connection, conversion_cache_dir=tmp_path)
        assert "README" in str(excinfo.value)

    def test_corrupt_legacy_xls_reports_cleanly(self, tmp_path, connection):
        """Reached without the excel extension: .xls never goes through DuckDB."""
        source = tmp_path / "legacy.xls"
        source.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 600)
        with pytest.raises(DataFileError) as excinfo:
            load_file(source, "legacy.xls", connection=connection, conversion_cache_dir=tmp_path)
        assert "legacy.xls" in str(excinfo.value)


class TestInspectFile:
    def test_csv_preview_creates_nothing(self, tmp_path, connection):
        source = tmp_path / "t.csv"
        source.write_text("a,b\n1,2\n")

        details = inspect_file(source, connection=connection)

        assert details["preview_columns"] == ["a", "b"]
        assert details["preview_rows"] == [[1, 2]]
        assert registered_objects(connection) == {}

    def test_legacy_xls_explains_the_conversion(self, tmp_path, connection):
        source = tmp_path / "old.xls"
        source.write_bytes(b"\xd0\xcf\x11\xe0")
        details = inspect_file(source, connection=connection)
        assert "Parquet" in details["note"]

    def test_unsupported_format_raises(self, tmp_path, connection):
        source = tmp_path / "x.ods"
        source.write_bytes(b"x")
        with pytest.raises(DataFileError):
            inspect_file(source, connection=connection)
