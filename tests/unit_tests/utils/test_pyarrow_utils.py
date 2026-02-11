from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from datus.utils.pyarrow_utils import concat_columns, concat_columns_with_cleaning


@pytest.fixture
def sample_table():
    return pa.table(
        {
            "domain": ["Health Care", "Finance/Banking", "Tech"],
            "layer1": ["Cat 1", "Cat2", "Cat 3"],
            "layer2": ["Sub/A", "Sub B", "Sub/C"],
            "search_text": ["Term 1", "Term/2", "Term3"],
        }
    )


# Test concat_columns_with_cleaning with replacements
def test_concat_columns_with_cleaning(sample_table):
    replacements = {" ": "_", "/": "_"}
    result = concat_columns_with_cleaning(
        sample_table,
        columns=["domain", "layer1", "layer2", "search_text"],
        separator="_",
        replacements=replacements,
    )
    expected_result = ["Health_Care_Cat_1_Sub_A_Term_1", "Finance_Banking_Cat2_Sub_B_Term_2", "Tech_Cat_3_Sub_C_Term3"]
    after_result = result.to_pylist()
    assert after_result == expected_result


# Test concat_columns without cleaning
def test_concat_columns(sample_table):
    result = concat_columns(sample_table, columns=["domain", "layer1", "layer2", "search_text"], separator="_")
    expected_result = ["Health Care_Cat 1_Sub/A_Term 1", "Finance/Banking_Cat2_Sub B_Term/2", "Tech_Cat 3_Sub/C_Term3"]
    assert result.to_pylist() == expected_result


# Test concat_columns_with_cleaning with null handling
def test_null_handling(sample_table):
    # Add a column with None values
    table_with_nulls = sample_table.append_column("null_column", pa.array([None, "Not Null", None]))

    result = concat_columns_with_cleaning(
        table_with_nulls,
        columns=["domain", "layer1", "layer2", "search_text", "null_column"],
        separator="_",
        null_handling="replace",
        null_replacement="NULL",
    )
    expected_result = [
        "Health Care_Cat 1_Sub/A_Term 1_NULL",
        "Finance/Banking_Cat2_Sub B_Term/2_Not Null",
        "Tech_Cat 3_Sub/C_Term3_NULL",
    ]
    assert result.to_pylist() == expected_result


# Test empty columns
def test_empty_columns():
    table = pa.table({"col1": ["a", "b", "c"], "col2": ["", "", ""]})
    result = concat_columns(table, columns=["col1", "col2"], separator="_")
    assert result.to_pylist() == ["a_", "b_", "c_"]


# Test single column concat
def test_single_column_concat(sample_table):
    result = concat_columns(sample_table, columns=["domain"], separator="_")
    assert result.to_pylist() == ["Health Care", "Finance/Banking", "Tech"]


class TestPyArrowUtilsExtended:
    """Extended test cases for PyArrow utilities."""

    @pytest.fixture
    def large_table(self):
        """Create a large table for performance testing."""
        size = 10000
        return pa.table(
            {
                "col1": [f"value_{i}" for i in range(size)],
                "col2": [f"data_{i}" for i in range(size)],
                "col3": [f"item_{i}" for i in range(size)],
            }
        )

    @pytest.fixture
    def nested_data_table(self):
        """Create table with nested/complex data."""
        return pa.table(
            {
                "json_like": ['{"key": "value"}', '{"id": 123}', '{"name": "test"}'],
                "array_like": ["[1,2,3]", "[a,b,c]", "[x,y,z]"],
                "nested_path": ["a/b/c/d", "x/y/z", "p/q/r/s/t"],
            }
        )

    def test_performance_large_dataset(self, large_table):
        """Test performance with large datasets."""
        import time

        start_time = time.time()
        result = concat_columns(large_table, columns=["col1", "col2", "col3"], separator="_")
        end_time = time.time()

        # Should complete within reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 5.0  # 5 seconds threshold
        assert len(result) == 10000

        # Verify sample results
        result_list = result.to_pylist()
        assert result_list[0] == "value_0_data_0_item_0"
        assert result_list[-1] == "value_9999_data_9999_item_9999"

    def test_memory_usage_large_dataset(self, large_table):
        """Test memory efficiency with large datasets."""
        # Test that we can process large data without excessive memory usage
        result = concat_columns_with_cleaning(
            large_table, columns=["col1", "col2"], separator="|", replacements={"_": "-"}
        )

        # Check that result maintains expected structure
        assert len(result) == 10000
        sample_result = result.slice(0, 1).to_pylist()[0]
        assert sample_result == "value-0|data-0"

    def test_complex_nested_data(self, nested_data_table):
        """Test handling of complex/nested-like data."""
        result = concat_columns_with_cleaning(
            nested_data_table,
            columns=["json_like", "array_like", "nested_path"],
            separator=" | ",
            replacements={"{": "[", "}": "]", '"': "'"},
        )

        expected_results = [
            "['key': 'value'] | [1,2,3] | a/b/c/d",
            "['id': 123] | [a,b,c] | x/y/z",
            "['name': 'test'] | [x,y,z] | p/q/r/s/t",
        ]
        assert result.to_pylist() == expected_results

    def test_concurrent_operations(self):
        """Test thread safety of PyArrow operations."""

        def worker(_thread_id, size=1000):
            table = pa.table(
                {
                    "id": [f"thread_{_thread_id}_id_{i}" for i in range(size)],
                    "data": [f"thread_{_thread_id}_data_{i}" for i in range(size)],
                }
            )

            result = concat_columns(table, columns=["id", "data"], separator=f"_sep_{_thread_id}_")
            return _thread_id, result.to_pylist()

        # Run multiple threads concurrently
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]

            for future in as_completed(futures):
                thread_id, result_list = future.result()
                results[thread_id] = result_list

        # Verify all threads completed successfully
        assert len(results) == 5

        # Verify thread-specific results
        for thread_id, result_list in results.items():
            assert len(result_list) == 1000
            assert result_list[0] == f"thread_{thread_id}_id_0_sep_{thread_id}_thread_{thread_id}_data_0"

    def test_null_value_edge_cases(self):
        """Test various null value scenarios."""
        # All nulls
        all_null_table = pa.table(
            {
                "col1": pa.array([None, None, None]),
                "col2": pa.array([None, None, None]),
            }
        )

        result = concat_columns_with_cleaning(
            all_null_table, columns=["col1", "col2"], separator="_", null_handling="replace", null_replacement="NULL"
        )
        assert result.to_pylist() == ["NULL_NULL", "NULL_NULL", "NULL_NULL"]

        # Mixed nulls and empty strings
        mixed_table = pa.table(
            {
                "col1": [None, "", "value"],
                "col2": ["", None, "data"],
            }
        )

        result = concat_columns_with_cleaning(
            mixed_table, columns=["col1", "col2"], separator="|", null_handling="replace", null_replacement="MISSING"
        )
        assert result.to_pylist() == ["MISSING|", "|MISSING", "value|data"]

    def test_extreme_separator_cases(self):
        """Test with unusual separators."""
        table = pa.table(
            {
                "col1": ["a", "b", "c"],
                "col2": ["x", "y", "z"],
            }
        )

        # Multi-character separator
        result = concat_columns(table, columns=["col1", "col2"], separator="<->")
        assert result.to_pylist() == ["a<->x", "b<->y", "c<->z"]

        # Empty separator
        result = concat_columns(table, columns=["col1", "col2"], separator="")
        assert result.to_pylist() == ["ax", "by", "cz"]

        # Unicode separator
        result = concat_columns(table, columns=["col1", "col2"], separator="🔗")
        assert result.to_pylist() == ["a🔗x", "b🔗y", "c🔗z"]

    def test_replacement_edge_cases(self):
        """Test edge cases in character replacement."""
        table = pa.table(
            {
                "col1": ["abc def", "xyz/123", "test@example.com"],
                "col2": ["hello world", "path/to/file", "user@domain.org"],
            }
        )

        # Multiple overlapping replacements
        result = concat_columns_with_cleaning(
            table,
            columns=["col1", "col2"],
            separator="|",
            replacements={" ": "_", "/": "_SLASH_", "@": "_AT_", "test": "TEST", "example": "EXAMPLE"},
        )

        expected_results = [
            "abc_def|hello_world",
            "xyz_SLASH_123|path_SLASH_to_SLASH_file",
            "TEST_AT_EXAMPLE.com|user_AT_domain.org",
        ]
        assert result.to_pylist() == expected_results

    def test_chunked_array_handling(self):
        """Test proper handling of chunked arrays."""
        # Create chunked arrays manually
        chunk1 = pa.array(["a1", "b1"])
        chunk2 = pa.array(["a2", "b2"])
        chunked_col1 = pa.chunked_array([chunk1, chunk2])

        chunk3 = pa.array(["x1", "y1"])
        chunk4 = pa.array(["x2", "y2"])
        chunked_col2 = pa.chunked_array([chunk3, chunk4])

        chunked_table = pa.table({"col1": chunked_col1, "col2": chunked_col2})

        result = concat_columns(chunked_table, columns=["col1", "col2"], separator="-")
        assert result.to_pylist() == ["a1-x1", "b1-y1", "a2-x2", "b2-y2"]

    def test_data_type_coercion(self):
        """Test automatic data type coercion to string."""
        mixed_types_table = pa.table(
            {
                "strings": ["text1", "text2", "text3"],
                "integers": [1, 2, 3],
                "floats": [1.1, 2.2, 3.3],
                "booleans": [True, False, True],
                "dates": pc.strptime(["2023-01-01", "2023-01-02", "2023-01-03"], format="%Y-%m-%d", unit="s"),
            }
        )

        result = concat_columns(mixed_types_table, columns=["strings", "integers", "floats", "booleans"], separator=":")

        expected_results = ["text1:1:1.1:true", "text2:2:2.2:false", "text3:3:3.3:true"]
        assert result.to_pylist() == expected_results

    def test_empty_table_edge_case(self):
        """Test behavior with empty tables."""
        empty_table = pa.table({"col1": pa.array([]), "col2": pa.array([])})

        result = concat_columns(empty_table, columns=["col1", "col2"], separator="_")
        assert len(result) == 0
        assert isinstance(result, (pa.Array, pa.ChunkedArray))

    def test_single_value_table(self):
        """Test with single-value tables."""
        single_table = pa.table({"col1": ["only_value"], "col2": ["another_value"]})

        result = concat_columns_with_cleaning(
            single_table, columns=["col1", "col2"], separator="<>", replacements={"only": "SINGLE", "_": "-"}
        )

        assert result.to_pylist() == ["SINGLE-value<>another-value"]

    def test_error_recovery_and_validation(self):
        """Test error conditions and recovery."""
        table = pa.table({"col1": ["test1", "test2"], "col2": ["value1", "value2"]})

        # Test with invalid column name
        with pytest.raises(Exception, match="does not exist in schema"):
            concat_columns(table, columns=["nonexistent_col"], separator="_")
        concat_columns(table, columns=[], separator="_")

    def test_integration_with_pyarrow_compute(self):
        """Test integration with other PyArrow compute functions."""
        table = pa.table(
            {"category": ["A", "B", "A", "C"], "subcategory": ["X", "Y", "Z", "X"], "value": ["v1", "v2", "v3", "v4"]}
        )

        # Concatenate columns
        concatenated = concat_columns(table, columns=["category", "subcategory"], separator="_")

        # Add back to table
        enriched_table = table.append_column("combined", concatenated)

        # Use PyArrow compute for filtering
        filtered = enriched_table.filter(pc.equal(enriched_table["category"], "A"))

        combined_values = filtered["combined"].to_pylist()
        assert combined_values == ["A_X", "A_Z"]

        # Test sorting by concatenated column
        sorted_table = enriched_table.sort_by([("combined", "ascending")])
        sorted_combined = sorted_table["combined"].to_pylist()
        assert sorted_combined == ["A_X", "A_Z", "B_Y", "C_X"]

    def test_memory_optimization_patterns(self):
        """Test memory optimization patterns."""
        # Create table that could potentially use a lot of memory
        size = 50000
        table = pa.table(
            {
                "long_text_1": [f"this_is_a_very_long_text_field_with_lots_of_content_{i}" for i in range(size)],
                "long_text_2": [f"another_long_text_field_with_different_content_{i}" for i in range(size)],
            }
        )

        # Test that concatenation doesn't cause memory issues
        result = concat_columns(table, columns=["long_text_1", "long_text_2"], separator="|||")

        # Verify result without loading all into memory at once
        assert len(result) == size

        # Test with slicing to avoid full materialization
        first_chunk = result.slice(0, 100)
        assert len(first_chunk) == 100

        sample_result = first_chunk.slice(0, 1).to_pylist()[0]
        expected_start = (
            "this_is_a_very_long_text_field_with_lots_of_content_0|||another_long_text_field_with_different_content_0"
        )
        assert sample_result == expected_start
