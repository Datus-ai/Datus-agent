# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.schema_metadata.init_utils."""

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from datus.storage.schema_metadata.init_utils import exists_table_value
from datus.utils.exceptions import DatusException


def _make_schema_table(identifiers: list[str], definitions: list[str]) -> pa.Table:
    """Build a pyarrow Table that mimics the schema store's search_all_schemas result."""
    return pa.table({"identifier": identifiers, "definition": definitions})


def _make_value_table(identifiers: list[str]) -> pa.Table:
    """Build a pyarrow Table that mimics the value store's search_all_value result."""
    return pa.table({"identifier": identifiers})


# ---------------------------------------------------------------------------
# overwrite mode
# ---------------------------------------------------------------------------


class TestExistsTableValueOverwrite:
    """When build_mode='overwrite', the function should return empty results immediately."""

    @pytest.mark.ci
    def test_overwrite_returns_empty(self):
        """build_mode='overwrite' should return empty dict and empty set without querying storage."""
        storage = MagicMock()
        schema_tables, value_tables = exists_table_value(storage, build_mode="overwrite")
        assert schema_tables == {}
        assert value_tables == set()
        # Should NOT have called any search methods
        storage.search_all_schemas.assert_not_called()
        storage.search_all_value.assert_not_called()


# ---------------------------------------------------------------------------
# incremental mode -- empty results
# ---------------------------------------------------------------------------


class TestExistsTableValueIncrementalEmpty:
    """Incremental mode with zero rows from storage."""

    @pytest.mark.ci
    def test_incremental_empty_results(self):
        """When storage has no data, should return empty dict and set."""
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table([], [])
        storage.search_all_value.return_value = _make_value_table([])

        schema_tables, value_tables = exists_table_value(storage, build_mode="incremental")
        assert schema_tables == {}
        assert value_tables == set()


# ---------------------------------------------------------------------------
# incremental mode -- with data
# ---------------------------------------------------------------------------


class TestExistsTableValueIncrementalWithData:
    """Incremental mode with actual rows from storage."""

    @pytest.mark.ci
    def test_incremental_returns_schemas_and_values(self):
        """Should populate dict from schema rows and set from value rows."""
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table(
            ["tbl_a", "tbl_b"],
            ["CREATE TABLE a ...", "CREATE TABLE b ..."],
        )
        storage.search_all_value.return_value = _make_value_table(["tbl_a"])

        schema_tables, value_tables = exists_table_value(storage, build_mode="incremental")
        assert schema_tables == {"tbl_a": "CREATE TABLE a ...", "tbl_b": "CREATE TABLE b ..."}
        assert value_tables == {"tbl_a"}

    @pytest.mark.ci
    def test_incremental_passes_filter_params(self):
        """Should forward database_name, catalog_name, schema_name, table_type to storage."""
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table([], [])
        storage.search_all_value.return_value = _make_value_table([])

        exists_table_value(
            storage,
            database_name="mydb",
            catalog_name="mycat",
            schema_name="myschema",
            table_type="view",
            build_mode="incremental",
        )

        storage.search_all_schemas.assert_called_once_with(
            catalog_name="mycat",
            database_name="mydb",
            schema_name="myschema",
            table_type="view",
        )
        storage.search_all_value.assert_called_once_with(
            catalog_name="mycat",
            database_name="mydb",
            schema_name="myschema",
            table_type="view",
        )

    @pytest.mark.ci
    def test_incremental_only_values(self):
        """Schema store empty but value store has rows."""
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table([], [])
        storage.search_all_value.return_value = _make_value_table(["val_1", "val_2"])

        schema_tables, value_tables = exists_table_value(storage, build_mode="incremental")
        assert schema_tables == {}
        assert value_tables == {"val_1", "val_2"}


# ---------------------------------------------------------------------------
# batch processing (> 500 rows)
# ---------------------------------------------------------------------------


class TestExistsTableValueBatching:
    """Verify that batching logic works for > 500 rows."""

    @pytest.mark.ci
    def test_large_schema_batch(self):
        """Should handle > 500 schema rows via internal batching."""
        n = 750
        ids = [f"tbl_{i}" for i in range(n)]
        defs = [f"CREATE TABLE tbl_{i} ..." for i in range(n)]
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table(ids, defs)
        storage.search_all_value.return_value = _make_value_table([])

        schema_tables, value_tables = exists_table_value(storage, build_mode="incremental")
        assert len(schema_tables) == n
        assert schema_tables["tbl_0"] == "CREATE TABLE tbl_0 ..."
        assert schema_tables[f"tbl_{n - 1}"] == f"CREATE TABLE tbl_{n - 1} ..."

    @pytest.mark.ci
    def test_large_value_batch(self):
        """Should handle > 500 value rows via internal batching."""
        n = 1200
        ids = [f"val_{i}" for i in range(n)]
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table([], [])
        storage.search_all_value.return_value = _make_value_table(ids)

        schema_tables, value_tables = exists_table_value(storage, build_mode="incremental")
        assert len(value_tables) == n

    @pytest.mark.ci
    def test_exactly_500_items(self):
        """Edge case: exactly 500 rows (single batch, no remainder)."""
        n = 500
        ids = [f"tbl_{i}" for i in range(n)]
        defs = [f"def_{i}" for i in range(n)]
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table(ids, defs)
        storage.search_all_value.return_value = _make_value_table(ids)

        schema_tables, value_tables = exists_table_value(storage, build_mode="incremental")
        assert len(schema_tables) == n
        assert len(value_tables) == n


# ---------------------------------------------------------------------------
# exception handling
# ---------------------------------------------------------------------------


class TestExistsTableValueExceptionHandling:
    """Exception from storage should be wrapped in DatusException."""

    @pytest.mark.ci
    def test_search_all_schemas_error_raises_datus_exception(self):
        """When search_all_schemas raises, should wrap in DatusException."""
        storage = MagicMock()
        storage.search_all_schemas.side_effect = RuntimeError("connection lost")

        with pytest.raises(DatusException) as exc_info:
            exists_table_value(storage, build_mode="incremental")
        assert "Failed to load already existing metadata" in str(exc_info.value)

    @pytest.mark.ci
    def test_search_all_value_error_raises_datus_exception(self):
        """When search_all_value raises, should wrap in DatusException."""
        storage = MagicMock()
        storage.search_all_schemas.return_value = _make_schema_table([], [])
        storage.search_all_value.side_effect = ValueError("bad data")

        with pytest.raises(DatusException) as exc_info:
            exists_table_value(storage, build_mode="incremental")
        assert "Failed to load already existing metadata" in str(exc_info.value)
