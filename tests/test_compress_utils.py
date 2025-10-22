import pandas as pd
import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.tools.func_tool import DBFuncTool
from datus.utils.compress_utils import DataCompressor
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
