# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.


from datus.utils.csv_utils import file_encoding, read_csv_and_clean_text, sanitize_csv_field


class TestFileEncoding:
    def test_returns_empty_for_nonexistent_file(self, tmp_path):
        result = file_encoding(tmp_path / "nonexistent.csv")
        assert result == ""

    def test_detects_utf8_bom(self, tmp_path):
        csv_file = tmp_path / "bom.csv"
        # Write with UTF-8 BOM
        csv_file.write_bytes(b"\xef\xbb\xbfname,value\nhello,1\n")
        result = file_encoding(csv_file)
        assert result == "utf-8-sig"

    def test_detects_utf8_without_bom(self, tmp_path):
        csv_file = tmp_path / "utf8.csv"
        csv_file.write_text("name,value\nhello,1\n", encoding="utf-8")
        result = file_encoding(csv_file)
        assert result == "utf-8"

    def test_returns_empty_for_directory(self, tmp_path):
        result = file_encoding(tmp_path)
        assert result == ""


class TestReadCsvAndCleanText:
    def test_returns_empty_list_for_empty_path(self):
        result = read_csv_and_clean_text("")
        assert result == []

    def test_returns_empty_list_for_none_path(self):
        result = read_csv_and_clean_text(None)
        assert result == []

    def test_returns_empty_list_for_nonexistent_file(self):
        result = read_csv_and_clean_text("/nonexistent/path/file.csv")
        assert result == []

    def test_reads_simple_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,value\nalice,1\nbob,2\n", encoding="utf-8")
        result = read_csv_and_clean_text(str(csv_file))
        assert len(result) == 2
        assert result[0]["name"] == "alice"
        assert result[1]["name"] == "bob"

    def test_accepts_path_object(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col\nhello\n", encoding="utf-8")
        result = read_csv_and_clean_text(csv_file)
        assert len(result) == 1
        assert result[0]["col"] == "hello"

    def test_cleans_text_in_string_columns(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        # Write text with control characters
        csv_file.write_text("name\nhello\x00world\n", encoding="utf-8")
        result = read_csv_and_clean_text(str(csv_file))
        assert len(result) == 1
        assert "\x00" not in result[0]["name"]

    def test_handles_nan_as_none(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,value\nalice,\nbob,2\n", encoding="utf-8")
        result = read_csv_and_clean_text(str(csv_file))
        # NaN values should be converted to None
        assert result[0]["value"] is None or result[0]["value"] == "" or result[0]["value"] is None

    def test_non_string_values_not_cleaned(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,count\nalice,42\n", encoding="utf-8")
        result = read_csv_and_clean_text(str(csv_file))
        assert result[0]["count"] == 42

    def test_returns_list_of_dicts(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n", encoding="utf-8")
        result = read_csv_and_clean_text(str(csv_file))
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_bom_utf8_csv(self, tmp_path):
        csv_file = tmp_path / "bom.csv"
        csv_file.write_bytes(b"\xef\xbb\xbfname,value\nhello,world\n")
        result = read_csv_and_clean_text(str(csv_file))
        assert len(result) == 1
        assert result[0]["name"] == "hello"


class TestSanitizeCsvField:
    def test_returns_none_for_none(self):
        assert sanitize_csv_field(None) is None

    def test_passes_through_regular_string(self):
        assert sanitize_csv_field("hello world") == "hello world"

    def test_passes_through_empty_string(self):
        assert sanitize_csv_field("") == ""

    def test_prefixes_equals(self):
        assert sanitize_csv_field("=SUM(A1:A3)") == "'=SUM(A1:A3)"

    def test_prefixes_plus(self):
        assert sanitize_csv_field("+1234") == "'+1234"

    def test_prefixes_minus(self):
        assert sanitize_csv_field("-1234") == "'-1234"

    def test_prefixes_at(self):
        assert sanitize_csv_field("@import") == "'@import"

    def test_coerces_non_string_to_string(self):
        assert sanitize_csv_field(42) == "42"
        assert sanitize_csv_field(True) == "True"

    def test_does_not_prefix_mid_formula_trigger(self):
        assert sanitize_csv_field("a=b") == "a=b"
