# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
from io import StringIO
from typing import Dict, List, Literal, Optional, Tuple, Union

import litellm
import pandas as pd
import pyarrow as pa


# Static utility functions outside of class
def _identify_id_time_columns(columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Identify ID and time columns

    Returns:
        Tuple[List[str], List[str]]: (id_columns, time_columns)
    """
    # Use single combined regex pattern for each type to reduce matching operations
    # ID pattern: contains 'id' or 'key' anywhere in the column name
    id_pattern = re.compile(r"(id|key)", re.IGNORECASE)
    # Time pattern: contains common time-related keywords
    time_pattern = re.compile(r"(time|date|created|updated|timestamp)", re.IGNORECASE)

    id_columns = []
    time_columns = []

    for col in columns:
        # Single search operation per column instead of multiple matches
        if id_pattern.search(col):
            id_columns.append(col)
        elif time_pattern.search(col):
            time_columns.append(col)

    return id_columns, time_columns


def _compress_pyarrow_table(table: pa.Table) -> Tuple[pa.Table, List[int], List[int]]:
    """
    Compress PyArrow Table by rows (take first 10 and last 10 rows)

    Returns:
        Tuple[pa.Table, List[int], List[int]]: (compressed_table, head_indices, tail_indices)
    """
    total_rows = len(table)
    if total_rows <= 20:
        return table, list(range(total_rows)), []

    # Take first 10 rows and last 10 rows
    head_table = table.slice(0, 10)
    tail_table = table.slice(total_rows - 10, 10)

    # Create ellipsis row
    ellipsis_data = {}
    for col_name in table.column_names:
        ellipsis_data[col_name] = ["..."]
    ellipsis_table = pa.table(ellipsis_data)

    # Merge tables
    compressed_table = pa.concat_tables([head_table, ellipsis_table, tail_table])

    # Return indices for proper row numbering
    head_indices = list(range(10))
    tail_indices = list(range(total_rows - 10, total_rows))

    return compressed_table, head_indices, tail_indices


def _get_row_count_fast(data: Union[List[Dict], pd.DataFrame, pa.Table]) -> int:
    """Get row count efficiently"""
    if isinstance(data, pa.Table):
        return len(data)
    elif isinstance(data, pd.DataFrame):
        return len(data)
    elif isinstance(data, list):
        return len(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def _format_as_csv(df: pd.DataFrame, compressed_indices: Optional[Tuple[List[int], List[int]]] = None) -> str:
    """
    Format DataFrame as CSV with proper row indices

    Args:
        df: DataFrame to format
        compressed_indices: If provided, tuple of (head_indices, tail_indices) for proper row numbering

    Returns:
        CSV formatted string with row indices
    """
    if compressed_indices and compressed_indices[0] and compressed_indices[1]:
        # Handle compressed data with ellipsis
        head_indices, tail_indices = compressed_indices

        # Split the dataframe
        head_df = df.iloc[: len(head_indices)].copy()
        tail_df = df.iloc[len(head_indices) + 1 :].copy()

        # Set proper indices
        head_df.index = head_indices
        tail_df.index = tail_indices

        # Format each part
        result_parts = []

        # Header
        header = "index," + ",".join(df.columns)
        result_parts.append(header)

        # Head rows
        for idx, row in head_df.iterrows():
            row_str = str(idx) + "," + ",".join(str(val) for val in row.values)
            result_parts.append(row_str)

        # Ellipsis row
        ellipsis_str = "...," + ",".join("..." for _ in df.columns)
        result_parts.append(ellipsis_str)

        # Tail rows
        for idx, row in tail_df.iterrows():
            row_str = str(idx) + "," + ",".join(str(val) for val in row.values)
            result_parts.append(row_str)

        return "\n".join(result_parts)
    else:
        # Normal formatting with index
        output = StringIO()
        df.to_csv(output, index=True, index_label="index")
        return output.getvalue().strip()


def _format_as_table(df: pd.DataFrame, compressed_indices: Optional[Tuple[List[int], List[int]]] = None) -> str:
    """
    Format DataFrame as table with proper row indices

    Args:
        df: DataFrame to format
        compressed_indices: If provided, tuple of (head_indices, tail_indices) for proper row numbering

    Returns:
        Table formatted string with row indices
    """
    if compressed_indices and compressed_indices[0] and compressed_indices[1]:
        # Handle compressed data with ellipsis
        head_indices, tail_indices = compressed_indices

        # Split the dataframe
        head_df = df.iloc[: len(head_indices)].copy()
        ellipsis_df = df.iloc[len(head_indices) : len(head_indices) + 1].copy()
        tail_df = df.iloc[len(head_indices) + 1 :].copy()

        # Set proper indices
        head_df.index = head_indices
        tail_df.index = tail_indices

        # Format each part
        head_str = head_df.to_string()
        tail_str = tail_df.to_string()

        # Get the ellipsis row formatted properly
        ellipsis_df.index = ["..."]
        ellipsis_str = ellipsis_df.to_string()

        # Combine with proper formatting
        head_lines = head_str.split("\n")
        tail_lines = tail_str.split("\n")
        ellipsis_lines = ellipsis_str.split("\n")

        # Keep header from head, add ellipsis, then tail without header
        result = "\n".join(head_lines) + "\n" + "\n".join(ellipsis_lines[1:]) + "\n" + "\n".join(tail_lines[1:])

        return result
    else:
        return df.to_string()


def _to_dataframe_efficient(data: Union[List[Dict], pd.DataFrame, pa.Table]) -> pd.DataFrame:
    """Convert to DataFrame efficiently"""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pa.Table):
        return data.to_pandas()
    elif isinstance(data, list):
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def _is_empty_data(data: Union[List[Dict], pd.DataFrame, pa.Table]) -> bool:
    """Check if data is empty based on its type"""
    if data is None:
        return True
    elif isinstance(data, list):
        return len(data) == 0
    elif isinstance(data, pd.DataFrame):
        return data.empty
    elif isinstance(data, pa.Table):
        return len(data) == 0
    else:
        return not bool(data)


def _get_data_dimensions(data: Union[List[Dict], pd.DataFrame, pa.Table]) -> Tuple[int, List[str]]:
    """
    Get row count and column names of data

    Returns:
        Tuple[int, List[str]]: (row_count, column_names)
    """
    row_count = _get_row_count_fast(data)

    if isinstance(data, pa.Table):
        column_names = data.column_names
    elif isinstance(data, pd.DataFrame):
        column_names = data.columns.tolist()
    elif isinstance(data, list) and data:
        column_names = list(data[0].keys()) if isinstance(data[0], dict) else []
    else:
        column_names = []

    return row_count, column_names


class DataCompressor:
    """
    Data compressor for NL2SQL Agent query results
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        token_threshold: int = 1024,
        tolerance_ratio: float = 0.1,
        output_format: Literal["table", "csv"] = "csv",
    ):
        """
        Initialize data compressor

        Args:
            model_name: Model name for token calculation. Supports all models via LiteLLM:
                - OpenAI: "gpt-4", "gpt-3.5-turbo", "gpt-4o", "o1", "o3", etc.
                - Anthropic: "anthropic/claude-3-sonnet", "anthropic/claude-sonnet-4", etc.
                - DeepSeek: "deepseek/deepseek-chat", "deepseek/deepseek-r1", etc.
                - Moonshot: "moonshot/kimi-k2.5", etc.
                - Gemini: "gemini/gemini-2.5-flash", "gemini/gemini-3-pro", etc.
            token_threshold: Token threshold
            tolerance_ratio: Tolerance ratio, no compression if exceeding within this ratio
            output_format: Output format, "csv" or "table"
        """
        self.model_name = model_name
        self.token_threshold = token_threshold
        self.tolerance_ratio = tolerance_ratio
        self.max_tolerable_tokens = int(token_threshold * (1 + tolerance_ratio))
        self.output_format = output_format

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in text using LiteLLM"""
        try:
            # Use LiteLLM's unified token counter (supports 100+ models)
            return litellm.token_counter(model=self.model_name, text=text)
        except Exception:
            # Fallback: rough estimation (1 token ≈ 4 characters for English text)
            return len(text) // 4

    def _compress_columns(
        self, df: pd.DataFrame, compressed_indices: Optional[Tuple[List[int], List[int]]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Compress columns, keep ID and time columns, compress middle columns of others

        Returns:
            Tuple[pd.DataFrame, List[str]]: (compressed_df, removed_columns)
        """
        columns = df.columns.tolist()
        id_columns, time_columns = _identify_id_time_columns(columns)

        # Columns that must be kept
        keep_columns = list(set(id_columns + time_columns))

        # Compressible columns (except those that must be kept)
        compressible_columns = [col for col in columns if col not in keep_columns]

        if not compressible_columns:
            return df, []  # If no compressible columns, return original data

        # Start compressing columns from the middle
        mid_index = len(compressible_columns) // 2

        # Gradually remove middle columns until token count meets requirements
        compressed_df = df.copy()
        removed_columns = []
        removed_count = 0

        while True:
            # Try current compression result
            if self.output_format == "csv":
                current_text = _format_as_csv(compressed_df, compressed_indices)
            else:
                current_text = _format_as_table(compressed_df, compressed_indices)

            current_tokens = self.count_tokens(current_text)

            if current_tokens <= self.max_tolerable_tokens or removed_count >= len(compressible_columns):
                break

            # Determine which column index to remove (expand from middle to both sides)
            if removed_count % 2 == 0:
                # Even times, remove to the right
                remove_index = mid_index + removed_count // 2
            else:
                # Odd times, remove to the left
                remove_index = mid_index - (removed_count + 1) // 2

            # Ensure valid index
            if 0 <= remove_index < len(compressible_columns):
                col_to_remove = compressible_columns[remove_index]
                if col_to_remove in compressed_df.columns:
                    compressed_df = compressed_df.drop(columns=[col_to_remove])
                    removed_columns.append(col_to_remove)

            removed_count += 1

        return compressed_df, removed_columns

    def compress(self, data: Union[List[Dict], pd.DataFrame, pa.Table]) -> Dict:
        """
        Compress data and return result with metadata

        Args:
            data: Input data, supports List[Dict], pandas.DataFrame, pyarrow.Table

        Returns:
            Dict: Contains original_rows, original_columns (list of column names), is_compressed,
                  compressed_data, removed_columns (if any), and compression_type
        """
        if _is_empty_data(data):
            return {
                "original_rows": 0,
                "original_columns": [],
                "is_compressed": False,
                "compressed_data": "Empty dataset",
                "removed_columns": [],
                "compression_type": "none",
            }

        # Get row count and column names
        original_rows, original_columns = _get_data_dimensions(data)

        is_compressed = False
        compression_type = "none"
        removed_columns = []
        compressed_indices = None
        df = None  # Will be created only when needed

        # If data is small, convert directly and check tokens
        if original_rows <= 20:
            df = _to_dataframe_efficient(data)

            if self.output_format == "csv":
                text = _format_as_csv(df)
            else:
                text = _format_as_table(df)

            tokens = self.count_tokens(text)

            if tokens <= self.max_tolerable_tokens:
                compressed_data = text
            else:
                # Small data but many tokens, compress columns only
                compressed_df, removed_columns = self._compress_columns(df)

                if self.output_format == "csv":
                    compressed_data = _format_as_csv(compressed_df)
                else:
                    compressed_data = _format_as_table(compressed_df)

                is_compressed = True
                compression_type = "columns"
        else:
            # Large data, needs row compression
            is_compressed = True
            compression_type = "rows"

            # Handle different data types efficiently without unnecessary conversion
            if isinstance(data, pa.Table):
                # PyArrow: use native slicing
                compressed_table, head_indices, tail_indices = _compress_pyarrow_table(data)
                df = compressed_table.to_pandas()
                compressed_indices = (head_indices, tail_indices)
            elif isinstance(data, pd.DataFrame):
                # Pandas DataFrame: use native head/tail methods
                head_df = data.head(10)
                tail_df = data.tail(10)

                # Store original indices
                head_indices = list(data.index[:10])
                tail_indices = list(data.index[-10:])
                compressed_indices = (head_indices, tail_indices)

                # Create ellipsis row
                ellipsis_row = pd.DataFrame([["..."] * len(data.columns)], columns=data.columns)

                # Merge
                df = pd.concat([head_df, ellipsis_row, tail_df], ignore_index=True)
            elif isinstance(data, list):
                # List[Dict]: compress directly without converting entire list to DataFrame
                head_data = data[:10]
                tail_data = data[-10:] if len(data) > 10 else []

                # Store indices
                head_indices = list(range(10))
                tail_indices = list(range(len(data) - 10, len(data)))
                compressed_indices = (head_indices, tail_indices)

                # Create compressed list with ellipsis
                if head_data and isinstance(head_data[0], dict):
                    columns = list(head_data[0].keys())
                    ellipsis_dict = {col: "..." for col in columns}
                    compressed_list = head_data + [ellipsis_dict] + tail_data

                    # Convert only the compressed data to DataFrame
                    df = pd.DataFrame(compressed_list)
                else:
                    # Fallback for non-dict lists
                    df = _to_dataframe_efficient(data)
                    head_df = df.head(10)
                    tail_df = df.tail(10)
                    ellipsis_row = pd.DataFrame([["..."] * len(df.columns)], columns=df.columns)
                    df = pd.concat([head_df, ellipsis_row, tail_df], ignore_index=True)

            # Check token count
            if self.output_format == "csv":
                text = _format_as_csv(df, compressed_indices)
            else:
                text = _format_as_table(df, compressed_indices)

            tokens = self.count_tokens(text)

            # If still exceeding threshold too much, compress columns
            if tokens > self.max_tolerable_tokens:
                df, removed_columns = self._compress_columns(df, compressed_indices)

                if self.output_format == "csv":
                    text = _format_as_csv(df, compressed_indices)
                else:
                    text = _format_as_table(df, compressed_indices)

                compression_type = "rows_and_columns"

            compressed_data = text

        return {
            "original_rows": original_rows,
            "original_columns": original_columns,
            "is_compressed": is_compressed,
            "compressed_data": compressed_data,
            "removed_columns": removed_columns,
            "compression_type": compression_type,
        }

    @classmethod
    def quick_compress(
        cls,
        data: Union[List[Dict], pd.DataFrame, pa.Table],
        model_name: str = "gpt-3.5-turbo",
        token_threshold: int = 1024,
        output_format: Literal["table", "csv"] = "csv",
    ) -> str:
        """
        Quick compress method for one-time use

        Args:
            data: Input data
            model_name: Model name for token calculation
            token_threshold: Token threshold
            output_format: Output format, "csv" or "table"

        Returns:
            str: Compressed data string
        """
        compressor = cls(model_name=model_name, token_threshold=token_threshold, output_format=output_format)
        result = compressor.compress(data)
        return result["compressed_data"]
