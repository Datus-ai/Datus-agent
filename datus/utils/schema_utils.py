# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List

from datus.utils.json_utils import to_str
from datus.utils.sql_utils import parse_metadata_from_ddl


def table_metadata_struct(table_metadata: List[Dict[str, Any]]) -> str:
    """
    Prase table metadata to hierarchical structure.
    Args:
        table_metadata: List of table metadata dictionaries with the following structure:
            [
                {
                    "schema_name": "log_events",
                    "table_name": "log_events_202404",
                    "schema_text": "create table ABC(id int,...);"
                },
                ...
            ]
    Returns:
        Dict[str, Any]: Hierarchical structure of table metadata
            [
                "schema_name.table":[
                    "comment": "",//optional
                    "columns": [{
                        "name": "col1",
                        "type": "bigit",
                        "comment": "col_comment"//optional
                    }]
                ]
            ]
    """
    result = {}
    for table in table_metadata:
        parsed_table_data = parse_metadata_from_ddl(table["schema_text"])
        table_name = f"{table['schema_name']}.{table['table_name']}"
        struct_table = {"columns": parsed_table_data["columns"]}
        if "comment" in parsed_table_data["table"]:
            struct_table["comment"] = parsed_table_data["table"]["comment"]

        result[table_name] = struct_table
    return to_str(result)


def table_metadata2markdown(table_metadata: List[Dict[str, Any]]) -> str:
    """
    Convert table metadata to formatted markdown tables.
    Args:
        table_metadata: List of table metadata dictionaries with the following structure:
            [
                {
                    "schema_name": "log_events",
                    "table_name": "log_events_202404",
                    "schema_text": "create table ABC(id int,...);"
                },
                ...
            ]

    Returns:
        str: Formatted markdown string with table descriptions and column information
    """
    if not table_metadata:
        return ""

    markdown_output = []

    for table in table_metadata:
        # Add table header with name and description
        table_names = table["table_name"].split(".")
        table_metadata = parse_metadata_from_ddl(table["schema_text"])
        table_header = f"**Table: `{table['schema_name']}.{table_names[-1]}`"
        # parse description, column, column_data_type, column_desc
        if "comment" in table_metadata["table"]:
            table_header += f" ({table_metadata['table']['comment']})"
        table_header += "**"
        markdown_output.append(table_header)

        # Prepare column data for tabulate
        headers = ["Column Name", "Data Type", "Description"]
        table_data = []
        for col in table_metadata["columns"]:
            table_data.append([col["name"], col["type"], col.get("comment", "")])

        # Add table using tabulate
        try:
            from tabulate import tabulate

            table_markdown = tabulate(table_data, headers=headers, tablefmt="pipe")
            markdown_output.append(table_markdown)
        except ImportError:
            raise ImportError("Please install tabulate: pip install tabulate")

        # Add spacing between tables
        markdown_output.append("")

    return "\n".join(markdown_output)
