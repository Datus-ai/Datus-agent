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

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from datus.tools.db_tools.data_file_loader import (
    DataFileError,
    build_table_name,
    column_letter,
    detect_csv_encoding,
    detect_header_row,
    find_file_reading_functions,
    inspect_file,
    load_file,
    ownership_tag,
    registered_objects,
    sanitize_identifier,
    summarize_view,
    unresolved_table_references,
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


class TestUnresolvedTableReferences:
    """The uploads guard is a whitelist because a blacklist cannot be completed.

    DuckDB's replacement scan reads a bare path written where a table goes —
    verified against 1.5.2, absolute paths and globs both — with no function call
    anywhere and a statement that parses as a plain SELECT. Requiring every
    reference to resolve to a registered object closes that, closes every
    file-reading function at once, and closes whatever DuckDB adds next.
    """

    KNOWN = ["jeffshop_q3", "sales"]

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM '/data/tenants/other/x.parquet'",
            "SELECT * FROM '/data/tenants/*/**/*.csv'",
            "SELECT a.k FROM sales a JOIN '/tmp/y.parquet' b ON a.k = b.k",
            "WITH x AS (SELECT * FROM '/tmp/z.csv') SELECT * FROM x",
        ],
    )
    def test_rejects_a_path_used_as_a_table(self, sql):
        assert unresolved_table_references(sql, self.KNOWN)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM read_csv_auto('/etc/passwd')",
            "SELECT * FROM read_parquet('/x')",
            "SELECT * FROM duckdb_views()",
            "SELECT * FROM some_reader_duckdb_adds_later('/etc/passwd')",
        ],
    )
    def test_rejects_table_functions_including_unknown_ones(self, sql):
        """A table function carries its name on the inner node, not the Table, so
        treating it as unnamed would exempt exactly what this must catch."""
        assert unresolved_table_references(sql, self.KNOWN)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM sales",
            "SELECT * FROM jeffshop_q3",
            'SELECT * FROM local_files.main."jeffshop_q3"',
            "SELECT store_name, sum(amount) FROM jeffshop_q3 GROUP BY 1",
            "WITH x AS (SELECT 1 AS a) SELECT * FROM x",
            "SELECT count(*) FROM sales WHERE k IN (SELECT k FROM jeffshop_q3)",
            "SELECT 1",
        ],
    )
    def test_allows_registered_tables_and_ctes(self, sql):
        assert unresolved_table_references(sql, self.KNOWN) == []

    def test_qualified_name_resolves_on_its_table_part(self):
        assert unresolved_table_references("SELECT * FROM other_db.other_schema.sales", self.KNOWN) == []

    def test_unparseable_sql_returns_a_sentinel(self):
        """The caller fails closed on this rather than treating it as clean."""
        assert unresolved_table_references("this is not sql ((", self.KNOWN) == ["<unparseable>"]

    def test_matching_is_case_insensitive(self):
        assert unresolved_table_references("SELECT * FROM SALES", self.KNOWN) == []


class TestConversionCacheLocation:
    def test_falls_back_to_a_temp_dir_when_the_catalog_has_no_directory(self):
        """An in-memory or unnamed catalog has nothing to sit beside, and the
        process working directory is not ours to write into."""
        import tempfile

        from datus.tools.db_tools.data_file_loader import default_conversion_cache_dir

        resolved = default_conversion_cache_dir("")
        assert resolved.is_absolute()
        assert str(resolved).startswith(tempfile.gettempdir())

    def test_sits_beside_a_file_backed_catalog(self):
        from datus.tools.db_tools.data_file_loader import default_conversion_cache_dir

        assert default_conversion_cache_dir("/var/x/local_files.duckdb") == Path("/var/x/_conversions")


class TestCsvEncoding:
    """DuckDB's CSV reader assumes UTF-8 and fails the whole read on anything
    else — a GB18030 file errors with ``CSV Error on Line: 1``. Excel's "Save as
    CSV" on a Chinese Windows writes GB18030, so preserving the bytes through
    upload only gets them as far as a reader that cannot decode them.
    """

    ROWS = "门店,品类,金额\n布鲁克林,鞋类,299.5\n费城,配件,88.0\n"

    def _write(self, tmp_path, name, data):
        target = tmp_path / name
        target.write_bytes(data)
        return target

    @pytest.mark.parametrize(
        "name,codec,expected",
        [
            ("utf8.csv", "utf-8", "utf-8"),
            ("u16.csv", "utf-16", "utf-16"),
            ("gb.csv", "gb18030", "gb18030"),
        ],
    )
    def test_detects_the_encoding(self, tmp_path, name, codec, expected):
        assert detect_csv_encoding(self._write(tmp_path, name, self.ROWS.encode(codec))) == expected

    def test_a_utf8_bom_is_decisive(self, tmp_path):
        source = self._write(tmp_path, "bom.csv", b"\xef\xbb\xbf" + self.ROWS.encode("utf-8"))
        assert detect_csv_encoding(source) == "utf-8"

    def test_plain_ascii_is_utf8(self, tmp_path):
        assert detect_csv_encoding(self._write(tmp_path, "a.csv", b"a,b\n1,2\n")) == "utf-8"

    @pytest.mark.parametrize("codec", ["utf-8", "utf-16", "gb18030"])
    def test_round_trips_through_the_loader(self, tmp_path, connection, codec):
        source = self._write(tmp_path, f"t_{codec}.csv", self.ROWS.encode(codec))

        loaded, _ = load_file(source, source.name, connection=connection, conversion_cache_dir=tmp_path)

        assert loaded[0].preview_columns == ["门店", "品类", "金额"]
        assert connection.execute(f'SELECT count(*) FROM "{loaded[0].table}"').fetchone()[0] == 2

    def test_a_gb18030_tsv_keeps_its_tab_delimiter(self, tmp_path, connection):
        """Delimiter and encoding are passed together, so setting one must not
        disturb the other."""
        source = self._write(tmp_path, "t.tsv", self.ROWS.replace(",", "\t").encode("gb18030"))

        loaded, _ = load_file(source, "t.tsv", connection=connection, conversion_cache_dir=tmp_path)

        assert loaded[0].preview_columns == ["门店", "品类", "金额"]

    def test_the_chosen_encoding_is_reported(self, tmp_path, connection):
        """A heuristic that cannot be seen cannot be corrected."""
        source = self._write(tmp_path, "gb.csv", self.ROWS.encode("gb18030"))
        loaded, _ = load_file(source, "gb.csv", connection=connection, conversion_cache_dir=tmp_path)
        assert loaded[0].encoding == "gb18030"

    def test_an_explicit_encoding_overrides_detection(self, tmp_path, connection):
        source = self._write(tmp_path, "gb.csv", self.ROWS.encode("gb18030"))

        loaded, _ = load_file(
            source, "gb.csv", connection=connection, conversion_cache_dir=tmp_path, encoding="gb18030"
        )

        assert loaded[0].encoding == "gb18030"
        assert loaded[0].preview_columns == ["门店", "品类", "金额"]

    def test_big5_is_refused_rather_than_corrupted(self, tmp_path, connection):
        """DuckDB accepts ``encoding='big5'`` and then loses the row boundary
        after a trailing byte in the ASCII range: two rows come back as one
        header. Python decodes the same bytes correctly, so this is the reader —
        and mojibake with no error is the one outcome worse than a refusal.
        """
        source = self._write(tmp_path, "big5.csv", "門店,金額\n台北,299.5\n".encode("big5"))

        with pytest.raises(DataFileError) as excinfo:
            load_file(source, "big5.csv", connection=connection, conversion_cache_dir=tmp_path)

        assert "big5" in str(excinfo.value)
        assert "UTF-8" in str(excinfo.value)

    def test_latin1_is_never_auto_selected(self, tmp_path):
        """It decodes any byte sequence, so offering it would mask every real
        answer behind mojibake."""
        source = self._write(tmp_path, "gb.csv", self.ROWS.encode("gb18030"))
        assert detect_csv_encoding(source) != "latin-1"

    def test_a_truncated_sample_does_not_defeat_detection(self, tmp_path):
        """A fixed-size sample almost always ends mid-character; a one-shot
        decode would report that as failure for the correct encoding."""
        payload = ("门店,金额\n" + "布鲁克林,1\n" * 400).encode("gb18030")
        source = self._write(tmp_path, "big.csv", payload)
        assert detect_csv_encoding(source, sample_bytes=1001) == "gb18030"

    def test_utf8_survives_every_truncation_point(self, tmp_path):
        """The UTF-8 short-circuit has to tolerate a cut mid-character too.

        With a strict one-shot decode there it reported "not UTF-8" and the file
        fell through to the CJK candidates, where gb18030 accepts the bytes — so
        a plain UTF-8 CSV was read as gb18030, and at some cut points refused
        outright as Big5. Reachable at the default sample size: any UTF-8 file
        over 256 KiB whose boundary lands inside a multi-byte character.
        """
        payload = "店,值\n".encode("utf-8") + "布鲁,1\n".encode("utf-8") * 40
        source = self._write(tmp_path, "u.csv", payload)

        misdetected = []
        for cut in range(1, len(payload) + 1):
            try:
                chosen = detect_csv_encoding(source, sample_bytes=cut)
            except DataFileError as exc:  # a wrong refusal counts as a failure
                chosen = f"refused: {exc}"
            if chosen != "utf-8":
                misdetected.append((cut, chosen))

        assert misdetected == []

    def test_truncation_does_not_loosen_non_utf8_detection(self, tmp_path):
        """The tolerant short-circuit must not start claiming everything is UTF-8.

        Cut at a range of sizes, since the incremental decoder's tolerance is
        what makes the short-circuit safe and a too-tolerant one would swallow
        these bytes as UTF-8.
        """
        source = self._write(tmp_path, "gb.csv", self.ROWS.encode("gb18030"))

        for cut in (16, 32, 64, 128):
            assert detect_csv_encoding(source, sample_bytes=cut) != "utf-8"

    def test_a_tiny_sample_does_not_trigger_the_big5_refusal(self, tmp_path):
        """The refusal guards against silent corruption of real data. On a few
        bytes the detector will label almost anything, and refusing a valid file
        on that basis is the worse error — such a file has no rows to corrupt."""
        source = self._write(tmp_path, "gb.csv", self.ROWS.encode("gb18030"))

        # No exception; the value itself is not the contract here.
        assert detect_csv_encoding(source, sample_bytes=8)

    def test_an_unreadable_file_does_not_raise(self, tmp_path):
        assert detect_csv_encoding(tmp_path / "absent.csv") == "utf-8"


# ------------------------------------------------------------------- catalog


class TestRegisteredObjects:
    def test_empty_catalog_is_an_empty_mapping(self, connection):
        assert registered_objects(connection) == {}

    def test_a_failed_read_is_not_an_empty_catalog(self):
        """The two must not collapse into the same value.

        Callers treat "no names" as "nothing is registered": the SQL guard turns
        it into a refusal naming the table the model just registered. Swallowing
        the error made an unreadable catalog indistinguishable from an empty one,
        which sent the model re-registering and re-querying in a loop.
        """

        class Broken:
            def execute(self, *_args, **_kwargs):
                raise RuntimeError("Connection Error: connection has been closed")

        with pytest.raises(RuntimeError):
            registered_objects(Broken())


# ----------------------------------------------------------- text date hints


class TestTextDateHints:
    """A spreadsheet's dates typed as text stay VARCHAR, and every date function
    then fails to bind. The profile says which cast to use, verified per column."""

    def _profile(self, connection, values, *, literal_type="VARCHAR"):
        rows = ", ".join("(NULL)" if value is None else f"('{value}')" for value in values)
        connection.execute(f"CREATE VIEW v AS SELECT CAST(d AS {literal_type}) AS d FROM (VALUES {rows}) t(d)")
        return {column["column_name"]: column for column in summarize_view(connection, "v")}["d"]

    def test_all_iso_dates_are_flagged_for_a_date_cast(self, connection):
        column = self._profile(connection, ["2017-07-01", "2017-08-31"])
        assert column["column_type"] == "VARCHAR"
        assert column["cast_hint"] == 'text dates: CAST("d" AS DATE) to use date functions'

    def test_a_clock_component_asks_for_a_timestamp_cast(self, connection):
        column = self._profile(connection, ["2017-07-01 09:30:00", "2017-07-02 00:00:00"])
        assert column["cast_hint"] == 'text timestamps: CAST("d" AS TIMESTAMP) to use date functions'

    def test_nulls_do_not_block_the_hint(self, connection):
        column = self._profile(connection, ["2017-07-01", None, "2017-08-31"])
        assert "cast_hint" in column

    def test_one_unparseable_value_withholds_the_hint(self, connection):
        """SUMMARIZE reports the *lexical* min/max, so a mixed column can show ISO
        ends. A hint whose cast NULLs rows is worse than no hint, so the check is
        every value, not the extremes."""
        column = self._profile(connection, ["2017-07-01", "n/a", "2017-08-31"])
        assert "cast_hint" not in column

    def test_ordinary_text_is_not_flagged(self, connection):
        column = self._profile(connection, ["Brooklyn", "Philadelphia"])
        assert "cast_hint" not in column

    def test_an_all_null_column_is_not_flagged(self, connection):
        column = self._profile(connection, [None, None])
        assert "cast_hint" not in column

    def test_a_column_already_typed_as_a_date_gets_no_hint(self, connection):
        column = self._profile(connection, ["2017-07-01"], literal_type="DATE")
        assert column["column_type"] == "DATE"
        assert "cast_hint" not in column

    def test_the_hint_is_runnable_for_a_header_with_a_space(self, connection):
        """The hint exists to be copied into a query, and spreadsheet headers have
        spaces in them. Unquoted, ``CAST(order date AS DATE)`` is a parser error."""
        connection.execute("""CREATE VIEW v AS SELECT * FROM (VALUES ('2017-07-01')) t("order date")""")
        hint = {column["column_name"]: column for column in summarize_view(connection, "v")}["order date"]["cast_hint"]

        expression = hint.split(": ", 1)[1].rsplit(" to use", 1)[0]
        assert expression == 'CAST("order date" AS DATE)'
        # Runs, rather than merely looking quoted.
        assert connection.execute(f"SELECT strftime({expression}, '%Y-%m') FROM v").fetchone()[0] == "2017-07"

    def test_a_bare_year_is_not_a_date(self, connection):
        """``TRY_CAST('2017' AS DATE)`` is NULL, so a year column stays un-hinted."""
        column = self._profile(connection, ["2017", "2018"])
        assert "cast_hint" not in column
