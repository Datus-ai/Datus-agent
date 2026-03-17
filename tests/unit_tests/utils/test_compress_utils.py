from typing import Dict, List
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.tools.func_tool import DBFuncTool
from datus.utils.compress_utils import (
    DataCompressor,
    _compress_pyarrow_table,
    _format_as_csv,
    _format_as_table,
    _get_data_dimensions,
    _get_row_count_fast,
    _identify_id_time_columns,
    _to_dataframe_efficient,
)
from tests.conftest import load_acceptance_config


def test_compress_mock():
    import numpy as np

    # Initialize compressor with CSV output format
    compressor_csv = DataCompressor(model_name="gpt-3.5-turbo", token_threshold=1024, output_format="csv")

    # Initialize compressor with table output format
    compressor_table = DataCompressor(model_name="gpt-3.5-turbo", token_threshold=1024, output_format="table")

    # Generate sample data with many rows
    sample_data = []
    for i in range(100):
        sample_data.append(
            {
                "user_id": i + 1,
                "name": f"User_{i + 1}",
                "email": f"user{i + 1}@example.com",
                "created_time": f"2024-01-{(i % 30) + 1:02d}",
                "score": np.random.randint(60, 100),
                "department": f"Dept_{(i % 5) + 1}",
                "salary": np.random.randint(3000, 10000),
                "age": np.random.randint(22, 60),
            }
        )

    # Test CSV format compression
    print("=" * 50)
    print("CSV Format Compression:")
    print("=" * 50)
    result_csv = compressor_csv.compress(sample_data)
    print(f"Original rows: {result_csv['original_rows']}")
    print(f"Original columns: {result_csv['original_columns']}")  # Now shows list of column names
    assert len(result_csv["original_columns"]) > 0
    print(f"Is compressed: {result_csv['is_compressed']}")
    print(f"Compression type: {result_csv['compression_type']}")
    print(f"Removed columns: {result_csv['removed_columns']}")
    print(f"\nCompressed data (CSV format):\n{result_csv['compressed_data']}")

    print("\n" + "=" * 50)
    print("Table Format Compression:")
    print("=" * 50)
    result_table = compressor_table.compress(sample_data[:30])  # Use fewer rows for table format demo
    print(f"Original rows: {result_table['original_rows']}")
    print(f"Original columns: {result_table['original_columns']}")  # Now shows list of column names
    print(f"Is compressed: {result_table['is_compressed']}")
    print(f"Compression type: {result_table['compression_type']}")
    print(f"Removed columns: {result_table['removed_columns']}")
    print(f"\nCompressed data (Table format):\n{result_table['compressed_data']}")

    # Test with pandas DataFrame
    print("\n" + "=" * 50)
    print("Testing with pandas DataFrame:")
    print("=" * 50)
    df = pd.DataFrame(sample_data[:50])
    result_df = compressor_csv.compress(df)
    print(f"Original columns: {result_df['original_columns']}")  # Shows actual column names
    print(f"Compression type: {result_df['compression_type']}")
    print(f"\nCompressed DataFrame:\n{result_df['compressed_data']}")

    # Test with small data that doesn't need compression
    print("\n" + "=" * 50)
    print("Testing with small data (no compression needed):")
    print("=" * 50)
    small_data = sample_data[:5]
    result_small = compressor_csv.compress(small_data)
    print(f"Is compressed: {result_small['is_compressed']}")
    print(f"Original columns: {result_small['original_columns']}")
    print(f"\nData:\n{result_small['compressed_data']}")

    # Test quick compress class method
    print("\n" + "=" * 50)
    print("Testing quick_compress class method:")
    print("=" * 50)
    quick_result = DataCompressor.quick_compress(sample_data[:40], output_format="csv")
    print(f"Quick compressed data:\n{quick_result}")

    # Test with very large list to show performance improvement
    print("\n" + "=" * 50)
    print("Testing with large dataset (10000 rows):")
    print("=" * 50)
    large_data = sample_data * 100  # 10000 rows
    import time

    start_time = time.time()
    result_large = compressor_csv.compress(large_data)
    end_time = time.time()

    print(f"Original rows: {result_large['original_rows']}")
    print(f"Compression type: {result_large['compression_type']}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")
    print("After compressed", result_large["compressed_data"])


@pytest.fixture
def agent_config():
    return load_acceptance_config()


@pytest.fixture
def db_manager(agent_config: AgentConfig) -> DBManager:
    return db_manager_instance(agent_config.namespaces)


def test_compress(db_manager: DBManager):
    sql = """SELECT
    name,
    setCode,
    rarity,
    type,
    manaCost,
    cardKingdomId,
    cardKingdomFoilId
FROM cards
WHERE cardKingdomFoilId IS NOT NULL
    AND cardKingdomId IS NOT NULL
ORDER BY
    CASE rarity
        WHEN 'mythic' THEN 1
        WHEN 'rare' THEN 2
        WHEN 'uncommon' THEN 3
        ELSE 4
    END,
    name;"""
    connector: BaseSqlConnector = db_manager.get_conn("bird_sqlite", "card_games")
    tool = DBFuncTool(connector)
    result = tool.read_query(sql)

    assert result.success == 1
    assert result.result
    query_result = result.result
    assert query_result["is_compressed"]
    assert query_result["original_rows"] > 10
    print(query_result["compressed_data"])

    assert len(query_result["original_columns"]) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_token_counter(model, text):
    """Return a token count based on text length so tests are deterministic."""
    return len(text) // 4


def _make_list(n: int) -> List[Dict]:
    return [{"id": i, "name": f"user_{i}", "score": i * 2} for i in range(n)]


# ---------------------------------------------------------------------------
# _identify_id_time_columns (lines 24-38)
# ---------------------------------------------------------------------------


class TestIdentifyIdTimeColumns:
    def test_id_columns_detected(self):
        id_cols, time_cols = _identify_id_time_columns(["user_id", "order_key", "name"])
        assert "user_id" in id_cols
        assert "order_key" in id_cols
        assert "name" not in id_cols

    def test_time_columns_detected(self):
        id_cols, time_cols = _identify_id_time_columns(["created_at", "updated_time", "value"])
        assert "created_at" in time_cols
        assert "updated_time" in time_cols
        assert "value" not in time_cols

    def test_id_takes_priority_over_time(self):
        """Column matching both id and time patterns → id bucket wins (searched first)."""
        id_cols, time_cols = _identify_id_time_columns(["id_timestamp"])
        assert "id_timestamp" in id_cols
        assert "id_timestamp" not in time_cols

    def test_empty_columns(self):
        id_cols, time_cols = _identify_id_time_columns([])
        assert id_cols == []
        assert time_cols == []

    def test_timestamp_column(self):
        _, time_cols = _identify_id_time_columns(["ts_timestamp"])
        assert "ts_timestamp" in time_cols

    def test_date_column(self):
        _, time_cols = _identify_id_time_columns(["birth_date"])
        assert "birth_date" in time_cols


# ---------------------------------------------------------------------------
# _compress_pyarrow_table (lines 48-69)
# ---------------------------------------------------------------------------


class TestCompressPyarrowTable:
    def test_small_table_unchanged(self):
        table = pa.table({"a": list(range(10)), "b": list(range(10))})
        compressed, head_idx, tail_idx = _compress_pyarrow_table(table)
        assert len(compressed) == 10
        assert head_idx == list(range(10))
        assert tail_idx == []

    def test_exactly_20_rows_unchanged(self):
        table = pa.table({"x": list(range(20))})
        compressed, head_idx, tail_idx = _compress_pyarrow_table(table)
        assert len(compressed) == 20
        assert tail_idx == []

    def test_large_table_compressed_to_21_rows(self):
        """21 head + ellipsis + tail = 21 rows in result."""
        # Use string column so ellipsis row ("...") has same dtype
        table = pa.table({"v": [str(i) for i in range(50)]})
        compressed, head_idx, tail_idx = _compress_pyarrow_table(table)
        # 10 head + 1 ellipsis + 10 tail = 21
        assert len(compressed) == 21
        assert head_idx == list(range(10))
        assert tail_idx == list(range(40, 50))

    def test_ellipsis_row_contains_dots(self):
        # Use string column so ellipsis row ("...") has same dtype
        table = pa.table({"col": [str(i) for i in range(30)]})
        compressed, _, _ = _compress_pyarrow_table(table)
        # Row at index 10 should be the ellipsis row
        df = compressed.to_pandas()
        assert df.iloc[10]["col"] == "..."


# ---------------------------------------------------------------------------
# _get_row_count_fast (line 75, 81)
# ---------------------------------------------------------------------------


class TestGetRowCountFast:
    def test_pyarrow_table(self):
        t = pa.table({"a": [1, 2, 3]})
        assert _get_row_count_fast(t) == 3

    def test_dataframe(self):
        df = pd.DataFrame({"a": range(5)})
        assert _get_row_count_fast(df) == 5

    def test_list(self):
        assert _get_row_count_fast([1, 2, 3, 4]) == 4

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            _get_row_count_fast("not_supported")


# ---------------------------------------------------------------------------
# _format_as_csv (lines 84-133)
# ---------------------------------------------------------------------------


class TestFormatAsCsv:
    def test_normal_dataframe_has_index_header(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv = _format_as_csv(df)
        assert csv.startswith("index,")
        assert "a" in csv
        assert "b" in csv

    def test_compressed_csv_has_ellipsis_row(self):
        df = pd.DataFrame({"a": list(range(21)), "b": list(range(21))})
        head_idx = list(range(10))
        tail_idx = list(range(11, 21))
        csv = _format_as_csv(df, compressed_indices=(head_idx, tail_idx))
        assert "..." in csv

    def test_no_compression_indices_uses_standard_csv(self):
        df = pd.DataFrame({"x": [1]})
        csv = _format_as_csv(df, compressed_indices=None)
        assert "x" in csv


# ---------------------------------------------------------------------------
# _format_as_table (lines 136-178)
# ---------------------------------------------------------------------------


class TestFormatAsTable:
    def test_normal_dataframe_as_string(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = _format_as_table(df)
        assert "col" in result

    def test_compressed_table_has_ellipsis(self):
        df = pd.DataFrame({"a": list(range(21))})
        head_idx = list(range(10))
        tail_idx = list(range(11, 21))
        result = _format_as_table(df, compressed_indices=(head_idx, tail_idx))
        assert "..." in result


# ---------------------------------------------------------------------------
# _to_dataframe_efficient (lines 181-190)
# ---------------------------------------------------------------------------


class TestToDataframeEfficient:
    def test_passthrough_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        result = _to_dataframe_efficient(df)
        assert result is df

    def test_pyarrow_to_pandas(self):
        t = pa.table({"x": [1, 2, 3]})
        result = _to_dataframe_efficient(t)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_list_of_dicts_to_dataframe(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = _to_dataframe_efficient(data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            _to_dataframe_efficient("bad")


# ---------------------------------------------------------------------------
# _is_empty_data (lines 193-204)
# ---------------------------------------------------------------------------


class TestIsEmptyData:
    def test_none_is_empty(self):
        from datus.utils.compress_utils import _is_empty_data

        assert _is_empty_data(None) is True

    def test_empty_list_is_empty(self):
        from datus.utils.compress_utils import _is_empty_data

        assert _is_empty_data([]) is True

    def test_non_empty_list_is_not_empty(self):
        from datus.utils.compress_utils import _is_empty_data

        assert _is_empty_data([1]) is False

    def test_empty_dataframe_is_empty(self):
        from datus.utils.compress_utils import _is_empty_data

        assert _is_empty_data(pd.DataFrame()) is True

    def test_non_empty_dataframe(self):
        from datus.utils.compress_utils import _is_empty_data

        assert _is_empty_data(pd.DataFrame({"a": [1]})) is False

    def test_empty_pyarrow_table(self):
        from datus.utils.compress_utils import _is_empty_data

        assert _is_empty_data(pa.table({"a": []})) is True

    def test_non_empty_pyarrow_table(self):
        from datus.utils.compress_utils import _is_empty_data

        assert _is_empty_data(pa.table({"a": [1]})) is False


# ---------------------------------------------------------------------------
# _get_data_dimensions (lines 207-225)
# ---------------------------------------------------------------------------


class TestGetDataDimensions:
    def test_pyarrow_dimensions(self):
        t = pa.table({"a": [1, 2], "b": [3, 4]})
        rows, cols = _get_data_dimensions(t)
        assert rows == 2
        assert cols == ["a", "b"]

    def test_dataframe_dimensions(self):
        df = pd.DataFrame({"x": range(5), "y": range(5)})
        rows, cols = _get_data_dimensions(df)
        assert rows == 5
        assert cols == ["x", "y"]

    def test_list_of_dicts_dimensions(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        rows, cols = _get_data_dimensions(data)
        assert rows == 2
        assert cols == ["a", "b"]

    def test_empty_list_dimensions(self):
        rows, cols = _get_data_dimensions([])
        assert rows == 0
        assert cols == []


# ---------------------------------------------------------------------------
# DataCompressor.compress — various paths (lines 265-452)
# ---------------------------------------------------------------------------


class TestDataCompressorCompress:
    """All token counts are mocked for determinism."""

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_empty_list_returns_none_result(self, _mock):
        dc = DataCompressor()
        result = dc.compress([])
        assert result["is_compressed"] is False
        assert result["compressed_data"] == "Empty dataset"
        assert result["original_rows"] == 0

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_empty_dataframe(self, _mock):
        dc = DataCompressor()
        result = dc.compress(pd.DataFrame())
        assert result["is_compressed"] is False

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_small_list_no_compression(self, _mock):
        dc = DataCompressor(token_threshold=100000)
        result = dc.compress(_make_list(5))
        assert result["is_compressed"] is False
        assert result["compression_type"] == "none"
        assert result["original_rows"] == 5

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_large_list_row_compression(self, _mock):
        dc = DataCompressor(token_threshold=100000)
        result = dc.compress(_make_list(50))
        assert result["is_compressed"] is True
        assert result["compression_type"] == "rows"
        assert result["original_rows"] == 50

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_large_dataframe_row_compression(self, _mock):
        dc = DataCompressor(token_threshold=100000)
        df = pd.DataFrame(_make_list(30))
        result = dc.compress(df)
        assert result["is_compressed"] is True
        assert result["compression_type"] == "rows"

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_large_pyarrow_table_row_compression(self, _mock):
        dc = DataCompressor(token_threshold=100000)
        # Use all-string columns so the ellipsis row ("...") matches schema
        data = [{"id": str(i), "name": f"user_{i}"} for i in range(30)]
        table = pa.table(pd.DataFrame(data))
        result = dc.compress(table)
        assert result["is_compressed"] is True
        assert result["compression_type"] == "rows"

    @patch("datus.utils.compress_utils.litellm.token_counter")
    def test_small_data_column_compression_when_over_threshold(self, mock_tc):
        """Force column compression on small data by returning high token count."""
        # Return huge count first (triggers compression), then small count
        mock_tc.side_effect = [99999, 100]
        dc = DataCompressor(token_threshold=10)
        data = [{"id": i, "col1": "x" * 100, "col2": "y" * 100, "col3": "z" * 100} for i in range(5)]
        result = dc.compress(data)
        # is_compressed should be True because column compression was triggered
        assert result["is_compressed"] is True
        assert result["compression_type"] == "columns"

    @patch("datus.utils.compress_utils.litellm.token_counter")
    def test_rows_and_columns_compression(self, mock_tc):
        """Large data that still exceeds threshold after row compression → rows_and_columns."""
        # First call (after row compression): still too many tokens
        # Second call (after column compression): within threshold
        mock_tc.side_effect = [99999, 100]
        dc = DataCompressor(token_threshold=10)
        data = [{"id": i, "col1": "x" * 100, "col2": "y" * 100, "col3": "z"} for i in range(30)]
        result = dc.compress(data)
        assert result["compression_type"] == "rows_and_columns"

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_compress_output_format_table(self, _mock):
        dc = DataCompressor(output_format="table", token_threshold=100000)
        result = dc.compress(_make_list(5))
        assert isinstance(result["compressed_data"], str)

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_compress_with_pyarrow_none_data(self, _mock):
        dc = DataCompressor()
        result = dc.compress(None)
        assert result["is_compressed"] is False

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_result_has_all_keys(self, _mock):
        dc = DataCompressor(token_threshold=100000)
        result = dc.compress(_make_list(3))
        assert set(result.keys()) == {
            "original_rows",
            "original_columns",
            "is_compressed",
            "compressed_data",
            "removed_columns",
            "compression_type",
        }

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_original_columns_is_list(self, _mock):
        dc = DataCompressor(token_threshold=100000)
        result = dc.compress(_make_list(3))
        assert isinstance(result["original_columns"], list)

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_compress_large_list_table_format(self, _mock):
        dc = DataCompressor(output_format="table", token_threshold=100000)
        result = dc.compress(_make_list(30))
        assert "..." in result["compressed_data"]

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_quick_compress_returns_string(self, _mock):
        result = DataCompressor.quick_compress(_make_list(5), token_threshold=100000)
        assert isinstance(result, str)

    @patch("datus.utils.compress_utils.litellm.token_counter", side_effect=_mock_token_counter)
    def test_count_tokens_fallback_on_exception(self, _mock):
        """When litellm raises, fallback estimation is used."""
        _mock.side_effect = Exception("model not found")
        dc = DataCompressor()
        count = dc.count_tokens("hello world test token")
        assert count == len("hello world test token") // 4


# ---------------------------------------------------------------------------
# DataCompressor._compress_columns (lines 269-327)
# ---------------------------------------------------------------------------


class TestCompressColumns:
    @patch("datus.utils.compress_utils.litellm.token_counter")
    def test_no_compressible_columns_returns_unchanged(self, mock_tc):
        """If all columns are id/time columns there's nothing to remove."""
        mock_tc.return_value = 9999
        dc = DataCompressor(token_threshold=10)
        df = pd.DataFrame({"user_id": [1, 2], "created_time": ["a", "b"]})
        compressed_df, removed = dc._compress_columns(df)
        assert removed == []
        assert list(compressed_df.columns) == list(df.columns)

    @patch("datus.utils.compress_utils.litellm.token_counter")
    def test_removes_middle_columns(self, mock_tc):
        """Columns are removed from the middle outward when over threshold."""
        # First call: over threshold; subsequent calls: under
        mock_tc.side_effect = [9999, 1]
        dc = DataCompressor(token_threshold=10)
        df = pd.DataFrame({"id": [1], "col_a": ["x"], "col_b": ["y"], "col_c": ["z"], "col_d": ["w"]})
        compressed_df, removed = dc._compress_columns(df)
        assert len(removed) >= 1
