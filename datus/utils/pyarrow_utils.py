# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc


def concat_columns_with_cleaning(
    table: pa.Table,
    columns: List[str],
    separator: str = "_",
    replacements: Optional[Dict[str, str]] = None,
    null_handling: str = "emit_null",
    null_replacement: str = "",
) -> pa.Array:
    """
    Clean specified columns (if replacement rules are provided) and concatenate them
    into a single string column, separated by the given separator.

    Parameters
    ----------
    table : pa.Table
        The input PyArrow table.
    columns : list[str]
        List of column names to concatenate.
    separator : str
        The separator to insert between column values.
    replacements : dict({old: new})
        A dictionary of replacement rules for each column (e.g., {"domain": {" ": "_", "/": "_"}}).
    null_handling : str
        Options for handling null values ('emit_null', 'skip', 'replace').
    null_replacement : str
        The replacement string for null values (if `null_handling='replace'`).

    Returns
    -------
    pa.Array
        The concatenated result as a PyArrow Array or ChunkedArray.
    """

    array_list = []
    for col in columns:
        arr = table[col]
        # First cast to string if not already
        arr = pc.cast(arr, pa.string())

        # Apply all replacements for the column
        if replacements:
            for old, new in replacements.items():
                arr = pc.replace_substring(arr, old, new)

        array_list.append(arr)

    # Prepare JoinOptions to control null handling
    join_opts = pc.JoinOptions(null_handling=null_handling, null_replacement=null_replacement)

    # Use binary_join_element_wise to concatenate the columns with the separator
    args = list(array_list) + [separator]
    joined = pc.binary_join_element_wise(*args, options=join_opts)

    return joined


def concat_columns(table: pa.Table, columns: List[str], separator: str = "_") -> pa.Array:
    """
    Concatenate specified columns into a single string column, separated by the given separator.

    Parameters
    ----------
    table : pa.Table
        The input PyArrow table.
    columns : list[str]
        List of column names to concatenate.
    separator : str
        The separator to insert between column values.

    Returns
    -------
    pa.Array
        The concatenated result as a PyArrow Array or ChunkedArray.
    """
    array_list = [pc.cast(table[col], pa.string()) for col in columns]

    # Use binary_join_element_wise to concatenate the columns with the separator
    array_list = array_list + [separator]
    joined = pc.binary_join_element_wise(*array_list)

    return joined
