"""Tests for cross-dialect type mapping (datus/tools/migration/type_mapping.py)."""

import pytest

from datus.tools.migration.type_mapping import (
    UnsupportedTypeError,
    map_columns_between_dialects,
)


class TestMapColumnsDuckDBToGreenplum:
    """Test DuckDB -> Greenplum type mapping for all supported types."""

    @pytest.mark.parametrize(
        "source_type,expected_type",
        [
            ("VARCHAR", "VARCHAR"),
            ("VARCHAR(255)", "VARCHAR(255)"),
            ("TEXT", "TEXT"),
            ("INTEGER", "INTEGER"),
            ("INT", "INTEGER"),
            ("BIGINT", "BIGINT"),
            ("SMALLINT", "SMALLINT"),
            ("DOUBLE", "DOUBLE PRECISION"),
            ("FLOAT", "REAL"),
            ("DECIMAL(10,2)", "NUMERIC(10,2)"),
            ("DECIMAL", "NUMERIC"),
            ("BOOLEAN", "BOOLEAN"),
            ("DATE", "DATE"),
            ("TIMESTAMP", "TIMESTAMP"),
            ("TIMESTAMP WITH TIME ZONE", "TIMESTAMP WITH TIME ZONE"),
            # alias coverage (L-2)
            ("BOOL", "BOOLEAN"),
            ("TINYINT", "SMALLINT"),
            ("REAL", "REAL"),
            ("NUMERIC", "NUMERIC"),
            ("UUID", "UUID"),
        ],
    )
    def test_supported_types(self, source_type, expected_type):
        columns = [{"name": "col1", "type": source_type, "nullable": True}]
        result = map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert result[0]["target_type"] == expected_type

    def test_multiple_columns(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR(100)", "nullable": True},
            {"name": "amount", "type": "DECIMAL(10,2)", "nullable": True},
        ]
        result = map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert len(result) == 3
        assert result[0]["target_type"] == "INTEGER"
        assert result[1]["target_type"] == "VARCHAR(100)"
        assert result[2]["target_type"] == "NUMERIC(10,2)"

    def test_preserves_original_fields(self):
        columns = [{"name": "id", "type": "INTEGER", "nullable": False}]
        result = map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert result[0]["name"] == "id"
        assert result[0]["type"] == "INTEGER"
        assert result[0]["nullable"] is False
        assert "target_type" in result[0]


class TestMapColumnsDuckDBToStarRocks:
    """Test DuckDB -> StarRocks type mapping for all supported types."""

    @pytest.mark.parametrize(
        "source_type,expected_type",
        [
            ("VARCHAR", "VARCHAR(65533)"),
            ("VARCHAR(255)", "VARCHAR(255)"),
            ("TEXT", "STRING"),
            ("INTEGER", "INT"),
            ("INT", "INT"),
            ("BIGINT", "BIGINT"),
            ("SMALLINT", "SMALLINT"),
            ("DOUBLE", "DOUBLE"),
            ("FLOAT", "FLOAT"),
            ("DECIMAL(10,2)", "DECIMAL(10,2)"),
            ("DECIMAL", "DECIMAL(38,9)"),
            ("BOOLEAN", "BOOLEAN"),
            ("DATE", "DATE"),
            ("TIMESTAMP", "DATETIME"),
            ("TIMESTAMP WITH TIME ZONE", "DATETIME"),
            # alias coverage (L-3)
            ("BOOL", "BOOLEAN"),
            ("TINYINT", "TINYINT"),
            ("REAL", "FLOAT"),
            ("STRING", "STRING"),
            ("UUID", "VARCHAR(36)"),
        ],
    )
    def test_supported_types(self, source_type, expected_type):
        columns = [{"name": "col1", "type": source_type, "nullable": True}]
        result = map_columns_between_dialects(columns, "duckdb", "starrocks")
        assert result[0]["target_type"] == expected_type


class TestUnsupportedTypes:
    """Test that unsupported types raise UnsupportedTypeError."""

    @pytest.mark.parametrize(
        "source_type",
        [
            "LIST",
            "STRUCT",
            "MAP",
            "UNION",
            "BLOB",
            "BYTEA",
            # L-4: spatial types
            "GEOMETRY",
            "POINT",
            "LINESTRING",
            "POLYGON",
        ],
    )
    def test_unsupported_type_raises_error(self, source_type):
        columns = [{"name": "bad_col", "type": source_type, "nullable": True}]
        with pytest.raises(UnsupportedTypeError) as exc_info:
            map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert exc_info.value.column_name == "bad_col"
        assert exc_info.value.source_type == source_type

    @pytest.mark.parametrize(
        "source_type",
        [
            "LIST",
            "STRUCT",
            "MAP",
            "UNION",
            "BLOB",
            "BYTEA",
        ],
    )
    def test_unsupported_type_in_starrocks(self, source_type):
        columns = [{"name": "data", "type": source_type, "nullable": True}]
        with pytest.raises(UnsupportedTypeError):
            map_columns_between_dialects(columns, "duckdb", "starrocks")

    def test_unmapped_type_raises_error(self):
        """Type not in unsupported set but also not in dialect map should raise."""
        columns = [{"name": "data", "type": "JSONB", "nullable": True}]
        with pytest.raises(UnsupportedTypeError):
            map_columns_between_dialects(columns, "duckdb", "greenplum")


class TestHugeintHandling:
    """Test HUGEINT mapping behavior."""

    def test_hugeint_maps_to_bigint_for_greenplum(self):
        columns = [{"name": "big_id", "type": "HUGEINT", "nullable": False}]
        result = map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert result[0]["target_type"] == "NUMERIC(38,0)"

    def test_hugeint_maps_to_largeint_for_starrocks(self):
        columns = [{"name": "big_id", "type": "HUGEINT", "nullable": False}]
        result = map_columns_between_dialects(columns, "duckdb", "starrocks")
        assert result[0]["target_type"] == "LARGEINT"


class TestCaseInsensitivity:
    """Test that type matching is case-insensitive."""

    @pytest.mark.parametrize(
        "source_type,expected_type",
        [
            ("varchar", "VARCHAR"),
            ("Varchar", "VARCHAR"),
            ("VARCHAR", "VARCHAR"),
            ("integer", "INTEGER"),
            ("Integer", "INTEGER"),
            ("boolean", "BOOLEAN"),
            ("Boolean", "BOOLEAN"),
        ],
    )
    def test_mixed_case_types(self, source_type, expected_type):
        columns = [{"name": "col", "type": source_type, "nullable": True}]
        result = map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert result[0]["target_type"] == expected_type


class TestParameterizedTypes:
    """Test types with parameters like DECIMAL(10,2) or VARCHAR(255)."""

    def test_decimal_with_precision(self):
        columns = [{"name": "price", "type": "DECIMAL(18,4)", "nullable": True}]
        result = map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert result[0]["target_type"] == "NUMERIC(18,4)"

    def test_varchar_with_length(self):
        columns = [{"name": "code", "type": "VARCHAR(50)", "nullable": True}]
        result = map_columns_between_dialects(columns, "duckdb", "starrocks")
        assert result[0]["target_type"] == "VARCHAR(50)"

    def test_decimal_with_precision_starrocks(self):
        columns = [{"name": "price", "type": "DECIMAL(18,4)", "nullable": True}]
        result = map_columns_between_dialects(columns, "duckdb", "starrocks")
        assert result[0]["target_type"] == "DECIMAL(18,4)"


class TestUnsupportedDialect:
    """Test behavior with unsupported source/target dialects."""

    def test_unsupported_source_dialect(self):
        columns = [{"name": "col", "type": "INTEGER", "nullable": True}]
        with pytest.raises(ValueError, match="Unsupported.*dialect"):
            map_columns_between_dialects(columns, "oracle", "greenplum")

    def test_unsupported_target_dialect(self):
        columns = [{"name": "col", "type": "INTEGER", "nullable": True}]
        with pytest.raises(ValueError, match="Unsupported.*dialect"):
            map_columns_between_dialects(columns, "duckdb", "oracle")

    def test_both_dialects_unsupported(self):
        columns = [{"name": "col", "type": "INTEGER", "nullable": True}]
        with pytest.raises(ValueError, match="Unsupported.*dialect"):
            map_columns_between_dialects(columns, "oracle", "oracle")


class TestEmptyInput:
    """Test edge cases."""

    def test_empty_columns_list(self):
        result = map_columns_between_dialects([], "duckdb", "greenplum")
        assert result == []

    @pytest.mark.parametrize(
        "raw_type",
        [
            "  VARCHAR(100)  ",
            "\tVARCHAR(100)\t",
            "\nVARCHAR(100)\n",
        ],
    )
    def test_whitespace_in_type(self, raw_type):
        columns = [{"name": "col", "type": raw_type, "nullable": True}]
        result = map_columns_between_dialects(columns, "duckdb", "greenplum")
        assert result[0]["target_type"] == "VARCHAR(100)"


class TestPostgresqlDialect:
    """Test that postgresql is aliased to Greenplum mapping (L-9)."""

    def test_postgresql_integer_maps_correctly(self):
        columns = [{"name": "id", "type": "INTEGER", "nullable": False}]
        result = map_columns_between_dialects(columns, "duckdb", "postgresql")
        assert result[0]["target_type"] == "INTEGER"
