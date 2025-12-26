# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import csv
import os
import re
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Union

import pyarrow as pa
from pydantic import BaseModel, Field, field_validator

from datus.schemas.base import TABLE_TYPE, BaseInput, BaseResult
from datus.schemas.doc_search_node_models import DocSearchResult
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MAX_SQL_RESULT_LENGTH = int(os.getenv("MAX_SQL_RESULT_LENGTH", 2000))


class SqlTask(BaseModel):
    """
    Input model for SQL task.
    Validates the input parameters for SQL task processing.
    """

    id: str = Field(default="", description="The id of the task")
    database_type: str = Field(default="", description="Type of the database (e.g., sqlite, duckdb, snowflake)")
    task: str = Field(default="", description="The SQL task description or query")
    catalog_name: str = Field(default="", description="Catalog name for context")
    database_name: str = Field(default="", description="Name of the database for context")
    schema_name: str = Field(default="", description="Schema name for context")
    output_dir: str = Field(default="output", description="Output directory path")
    external_knowledge: str = Field(default="", description="External knowledge for the input")
    tables: Optional[List[str]] = Field(default=[], description="List of table names to use")
    schema_linking_type: TABLE_TYPE = Field(default="table", description="Schema linking type for the task")

    current_date: Optional[str] = Field(
        default=None, description="Current date reference for relative time expressions"
    )
    date_ranges: str = Field(default="", description="Parsed date ranges context from date parser for SQL generation")

    # Metrics relative part - subject path
    subject_path: Optional[List[str]] = Field(
        default=None, description="Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])"
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with an optional default value."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary representation."""
        return self.model_dump()

    def to_str(self) -> str:
        """Convert the task to a JSON string representation."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SqlTask":
        """Create SqlTask instance from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_str(cls, json_str: str) -> "SqlTask":
        """Convert the task to a JSON string representation."""
        return cls.model_validate_json(json_str)

    @field_validator("task")
    def validate_task(cls, v):
        if not v.strip():
            raise ValueError("'task' must not be empty")
        return v


class BaseTableSchema(BaseModel):
    identifier: str = Field(default="", description="Unique identifier of the table", init=True)
    catalog_name: str = Field(default="", description="Name of the catalog", init=True)
    table_name: str = Field(..., description="Name of the table", init=True)
    database_name: str = Field(..., description="Name of the database", init=True)
    schema_name: str = Field(default="", description="Name of the schema", init=True)


class TableSchema(BaseTableSchema):
    """
    Model for table schema information returned by schema linking.
    """

    definition: str = Field(..., description="DDL schema text of the table")
    table_type: str = Field("table", description="Type of the schema")

    def to_prompt(self, dialect: str = "snowflake") -> str:
        """
        Convert the schema to a concise string representation for LLM prompt.
        Simplifies the schema by:
        - Removing redundant whitespace and newlines
        - Converting verbose type definitions to simple types (e.g., VARCHAR(16777216) -> VARCHAR)
        - Removing redundant SQL keywords
        - Formatting as: database.schema.table: TABLE definition

        Returns:
            A simplified string representation of the table schema
        """

        schema_text = " ".join(self.definition.split())
        # TODO: improve the schema compact for all databases
        return schema_text.replace("VARCHAR(16777216)", "VARCHAR")

    @classmethod
    def table_names_to_prompt(cls, schemas: List[TableSchema]) -> str:
        if not schemas:
            return ""
        return "\n".join([schema.table_name for schema in schemas])

    @classmethod
    def list_to_prompt(cls, schemas: List[TableSchema], dialect: str = "snowflake") -> str:
        if not schemas:
            return ""
        return "\n\n".join([schema.to_prompt(dialect) for schema in schemas])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableSchema:
        """Create TableSchema instance from dictionary."""
        return cls(
            identifier=data.get("identifier", ""),
            catalog_name=data.get("catalog_name", ""),
            table_name=data["table_name"],
            database_name=data.get("database_name", ""),
            schema_name=data.get("schema_name", ""),
            definition=data["definition"],
            table_type=data.get("table_type", "table"),
        )

    @classmethod
    def from_arrow(cls, table: pa.Table) -> List[TableSchema]:
        result = []
        for index in range(table.num_rows):
            result.append(
                cls(
                    identifier=table["identifier"][index].as_py(),
                    catalog_name=table["catalog_name"][index].as_py(),
                    table_name=table["table_name"][index].as_py(),
                    database_name=table["database_name"][index].as_py(),
                    schema_name=table["schema_name"][index].as_py(),
                    definition=table["definition"][index].as_py(),
                    table_type=table["table_type"][index].as_py(),
                )
            )
        return result

    def to_dict(self):
        return self.model_dump()


class TableValue(BaseTableSchema):
    """Model for table value information returned by schema linking."""

    table_values: str = Field(..., description="Sample values from the table")
    table_type: str = Field("table", description="Type of the schema")

    def to_prompt(
        self,
        dialect: str = "snowflake",
        max_value_length: int = 500,
        max_text_mark_length: int = 16,
        processed_schemas: str = "",
    ) -> str:
        """
        Convert table values to a concise string representation for LLM prompt.
        If processed_schemas is provided, will replace TEXT column values longer than 16 characters with <TEXT>.
        """

        values_str = str(self.table_values)

        if processed_schemas and dialect == DBType.SQLITE:
            table_schema = self._parse_table_schema(processed_schemas)
            if table_schema:
                values_str = self._process_text_columns(values_str, max_text_mark_length, table_schema)

        if len(values_str) > max_value_length:
            logger.warning("table value is too long, truncating to %s characters" % max_value_length)
            values_str = values_str[:max_value_length] + "...(truncated)"
        full_name = self.table_name if dialect == DBType.SQLITE else self.identifier
        return f"{full_name} values: \n{values_str}"

    def _parse_table_schema(self, processed_schemas: str) -> Dict[str, str]:
        """
        Parse the table schema from processed_schemas string to extract column types.
        Returns a dictionary mapping column names to their types.
        """

        # Find the schema for this table
        # Pattern to match table definition: table_name: CREATE TABLE ...
        # The table definition ends with ); (closing parenthesis and semicolon)
        pattern = rf"{re.escape(self.table_name)}:\s*CREATE\s+TABLE[^(]*\(([^)]+)\)\s*;"
        match = re.search(pattern, processed_schemas, re.IGNORECASE | re.DOTALL)

        if not match:
            # Try alternative pattern without semicolon
            pattern = rf"{re.escape(self.table_name)}:\s*CREATE\s+TABLE[^(]*\(([^)]+)\)"
            match = re.search(pattern, processed_schemas, re.IGNORECASE | re.DOTALL)

        if not match:
            return {}

        table_def = match.group(1)
        column_info = {}

        # Use a different approach: find all column definitions with regex
        # Pattern: `column_name` TYPE followed by optional comment
        pattern = r"`([^`]+)`\s+(\w+)(?:\s*,\s*\/\*[^*]*\*\/)?"
        matches = re.findall(pattern, table_def)

        for col_name, col_type in matches:
            column_info[col_name] = col_type.upper()

        # If the regex approach didn't work, try a simpler approach
        if not column_info:
            parts = table_def.split(",")
            for part in parts:
                part = part.strip()
                if part.startswith("`"):
                    col_match = re.match(r"`([^`]+)`\s+(\w+)", part)
                    if col_match:
                        col_name = col_match.group(1)
                        col_type = col_match.group(2).upper()
                        column_info[col_name] = col_type
                elif part.strip().startswith("PRIMARY KEY") or part.strip().startswith("FOREIGN KEY"):
                    break

        return column_info

    def _process_text_columns(self, values_str: str, max_text_mark_length, table_schema: Dict[str, str]) -> str:
        """
        Process CSV data to replace long TEXT column values with <TEXT>.
        """

        try:
            csv_input = StringIO(values_str.strip())
            csv_reader = csv.reader(csv_input)

            headers = next(csv_reader)

            processed_rows = [headers]
            for row in csv_reader:
                processed_row = []
                for i, value in enumerate(row):
                    if i < len(headers):
                        col_name = headers[i]
                        col_type = table_schema.get(col_name, "")

                        if col_type == "TEXT" and len(str(value)) > max_text_mark_length:
                            processed_row.append("<TEXT>")
                        else:
                            processed_row.append(value)
                    else:
                        processed_row.append(value)
                processed_rows.append(processed_row)

            output = StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerows(processed_rows)
            return output.getvalue().strip()

        except Exception as e:
            logger.warning(f"Failed to process TEXT columns: {e}")
            return values_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableValue:
        """Create TableValue instance from dictionary."""
        return cls(
            identifier=data.get("identifier", ""),
            catalog_name=data.get("catalog_name", ""),
            table_name=data["table_name"],
            database_name=data.get("database_name", ""),
            schema_name=data.get("schema_name", ""),
            table_values=data["table_values"] if "table_values" in data else data["sample_rows"],
            table_type=data.get("table_type", "table"),
        )

    @classmethod
    def from_arrow(cls, table: pa.Table) -> List[TableValue]:
        result = []
        for index in range(table.num_rows):
            result.append(
                cls(
                    identifier=table["identifier"][index].as_py(),
                    catalog_name=table["catalog_name"][index].as_py(),
                    table_name=table["table_name"][index].as_py(),
                    database_name=table["database_name"][index].as_py(),
                    schema_name=table["schema_name"][index].as_py(),
                    table_values=table["sample_rows"][index].as_py(),
                    table_type=table["table_type"][index].as_py(),
                )
            )
        return result

    def to_dict(self):
        return self.model_dump()


class Metric(BaseModel):
    """
    Model for metrics information used in SQL generation.
    """

    name: str = Field(..., description="Name of the metric")
    description: str = Field(default="", description="Description of the metric")

    def to_prompt(self, dialect: str = "snowflake") -> str:
        return self.description if self.description else f"Metric: {self.name}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Metric:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
        )


class ReferenceSql(BaseModel):
    name: str = Field(..., description="Name of the reference SQL table")
    sql: str = Field(..., description="SQL query of the reference table")
    comment: str = Field(default="", description="Comment of the reference SQL table")
    summary: str = Field(default="", description="Summary of the reference SQL table")
    tags: str = Field(default="", description="Tags of the reference SQL table")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReferenceSql:
        return cls(
            name=data.get("name", ""),
            sql=data.get("sql", ""),
            comment=data.get("comment", ""),
            summary=data.get("summary", ""),
            tags=data.get("tags", ""),
        )


class GenerateSQLInput(BaseInput):
    """
    Input model for SQL generation node.
    Validates the input parameters for SQL query generation.
    """

    database_type: Optional[str] = Field(None, description="Type of the database")
    table_schemas: Union[List[TableSchema], str] = Field(..., description="List of table schemas to use")
    data_details: Optional[List[TableValue]] = Field(None, description="Optional sample data from tables")
    metrics: Optional[List[Metric]] = Field(None, description="Optional metrics for query generation")
    sql_task: SqlTask = Field(..., description="The SQL task to generate SQL from")
    contexts: Optional[List[SQLContext]] = Field(default=[], description="Optional context information for the input")
    external_knowledge: str = Field(default="", description="External knowledge for the input")
    prompt_version: str = Field(default="1.0", description="Version for prompt")
    max_table_schemas_length: int = Field(default=4000, description="Max table schemas length")
    max_data_details_length: int = Field(default=2000, description="Max data details length")
    max_context_length: int = Field(default=8000, description="Max context length")
    max_value_length: int = Field(default=500, description="Max value length")
    max_text_mark_length: int = Field(default=16, description="Max text mark length")
    database_docs: Optional[str] = Field(default="", description="Database documentation")


class GenerateSQLResult(BaseResult):
    """
    Result model for SQL generation node.
    Contains the generated SQL query and related information.
    """

    sql_query: str = Field(..., description="The generated SQL query")
    tables: List[str] = Field(default_factory=list, description="List of tables used in the query")
    explanation: Optional[str] = Field(None, description="Explanation of the SQL query")


class ExecuteSQLInput(BaseInput):
    """
    Input model for SQL execution node.
    Validates the SQL query to be executed.
    """

    database_name: str = Field(default="", description="The name of the database")
    sql_query: str = Field(..., description="The SQL query to execute")
    result_format: str = Field(default="csv", description="Format of the result: 'csv' or 'arrow' or 'list'")


class ExecuteSQLResult(BaseResult):
    """
    Result model for SQL execution node.
    Contains the execution results.
    """

    sql_query: Optional[str] = Field("", description="The SQL query to execute")
    row_count: Optional[int] = Field(None, description="The number of rows returned")
    sql_return: Any = Field(  # TODO: change to Union[str, ArrowTable, List[Reuslt]]
        default=None, description="The result of SQL execution (string or Arrow data)"
    )
    result_format: str = Field(default="", description="Format of the result: 'csv' or 'arrow' or 'pandas' or 'list'")

    class Config:
        arbitrary_types_allowed = True

    def compact_result(self) -> str:
        """
        Returns a compact string representation of the execution result.
        Only includes row count and truncated sql return (max length defined by DATUS_MAX_RESULT_LENGTH).
        Returns:
            str: Formatted string with row count and truncated result
        """
        sql_result = ""
        if hasattr(self.sql_return, "to_csv"):
            sql_result = self.sql_return.to_csv(index=False)
        else:
            sql_result = str(self.sql_return)
        truncated_return = (
            (sql_result[:MAX_SQL_RESULT_LENGTH] + "...")
            if sql_result and len(sql_result) > MAX_SQL_RESULT_LENGTH
            else sql_result
        )

        # errors = f"Error: {self.error}\n" if not self.success else ""
        return f"Error: {self.error}\nRows: {self.row_count}\nResult: {truncated_return}"


class SQLContext(BaseModel):
    # sql_id: str = Field(..., description="The id of the SQL")
    sql_query: str = Field(..., description="The generated SQL query")
    explanation: Optional[str] = Field("", description="Explanation of the SQL query")
    # TODO: modify str to List[Result] with arrow format
    sql_return: Any = Field(default="", description="The result of SQL execution")
    sql_error: Optional[str] = Field("", description="The error of SQL execution")
    row_count: Optional[int] = Field(0, description="The number of rows returned")
    reflection_strategy: Optional[str] = Field("", description="The reflection strategy")
    reflection_explanation: Optional[str] = Field("", description="The reflection explanation")

    def to_dict(self):
        return self.model_dump()

    def to_str(self, max_sql_return_length: int = 4294967296):
        sql_return_str = ""
        if self.sql_return is not None:
            if hasattr(self.sql_return, "to_csv"):  # Check if it's a DataFrame
                if self.sql_return.empty:
                    sql_return_str = "Empty result set"
                else:
                    sql_return_str = self.sql_return.to_csv(index=False)
            else:
                sql_return_str = str(self.sql_return)

        if len(sql_return_str) > max_sql_return_length:
            logger.warning("Sql return is too long, truncating to %s characters" % max_sql_return_length)
            sql_return_str = sql_return_str[:max_sql_return_length] + "\n... (truncated)"

        return (
            f"SQL: {self.sql_query}\n"
            f"Explanation: {self.explanation}\n"
            f"Result: {sql_return_str}\n"
            f"Reflection Strategy: {self.reflection_strategy}\n"
            f"Reflection Explanation: {self.reflection_explanation}"
        )

    def to_sample_str(self):
        sql_return_str = ""
        if self.sql_return is not None:
            if hasattr(self.sql_return, "to_csv"):  # Check if it's a DataFrame
                if self.sql_return.empty:
                    sql_return_str = "Empty result set"
                else:
                    sql_return_str = self.sql_return.to_csv(index=False)
            else:
                sql_return_str = str(self.sql_return)

        return (
            f"SQL: {self.sql_query}\n"
            f"Explanation: {self.explanation}\n"
            f"Result: {sql_return_str}\n"
            f"Reflection Strategy: {self.reflection_strategy}\n"
            f"Reflection Explanation: {self.reflection_explanation}"
        )

    def compact_result(self) -> str:
        # TODO: implement it for large result
        pass


class Context(BaseModel):
    """
    Model for context information used in SQL generation.
    """

    sql_contexts: List[SQLContext] = Field(default_factory=list, description="The SQL contexts")
    table_schemas: List[TableSchema] = Field(default_factory=list, description="The table schemas")
    table_values: List[TableValue] = Field(default_factory=list, description="The table values")
    metrics: List[Metric] = Field(default_factory=list, description="The metrics")
    doc_search_keywords: List[str] = Field(default_factory=list, description="The document search keywords")
    document_result: Optional[DocSearchResult] = Field(default=None, description="The document result")
    parallel_results: Optional[Dict[str, Any]] = Field(default=None, description="Results from parallel node execution")
    last_selected_result: Optional[Any] = Field(
        default=None, description="The last selected result from selection node"
    )
    selection_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata about selection process")

    def update_schema_and_values(self, table_schemas: List[TableSchema], table_values: List[TableValue]):
        """
        Update the table schema and sample data to be used in the context.
        ⚠️⚠️⚠️Special note: If this method is called, it will have the following effects:
            1. `schema_linking_node` will always return the updated value and will not match it in the vector library.
            2. It will be expanded and spliced into user questions in `agentic_node`
        """
        self.table_schemas = table_schemas
        self.table_values = table_values

    def update_last_sql_context(self, sql_context: SQLContext):
        self.sql_contexts[-1] = sql_context

    def update_metrics(self, metrics: List[Metric]):
        self.metrics = metrics

    def update_document_result(self, document_result: DocSearchResult):
        self.document_result = document_result

    def update_doc_search_keywords(self, doc_search_keywords: List[str]):
        self.doc_search_keywords = doc_search_keywords

    def update_parallel_results(self, parallel_results: Dict[str, Any]):
        self.parallel_results = parallel_results

    def update_selection_result(self, selected_result: Any, metadata: Dict[str, Any]):
        self.last_selected_result = selected_result
        self.selection_metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Context":
        return cls.model_validate(data)

    def to_str(self) -> str:
        table_names = [schema.table_name for schema in self.table_schemas]
        table_values_names = [value.table_name for value in self.table_values]
        sql_contexts = [
            f"sql: {context.sql_query} explain:{context.explanation}"
            f"return:{context.row_count} reflection:{context.reflection_explanation}"
            for context in self.sql_contexts
        ]
        parallel_info = f"Parallel results: {len(self.parallel_results) if self.parallel_results else 0}"
        selection_info = f"Selection made: {bool(self.last_selected_result)}"

        return (
            f"\nTable Schemas: {table_names}\n"
            f"Table Values: {table_values_names}\n"
            f"SQL Contexts: {sql_contexts}\n"
            f"Metrics Count: {len(self.metrics)}\n"
            f"{parallel_info}\n"
            f"{selection_info}"
        )


class OutputInput(BaseInput):
    """
    Input model for output node.
    Validates the output result.
    """

    finished: bool = Field(True, description="Whether the task is finished")
    error: Optional[str] = Field(None, description="The error message")
    task_id: str = Field(..., description="The id of the task")
    task: str = Field(..., description="The task description")
    database_name: str = Field(..., description="The name of the database")
    output_dir: str = Field(..., description="The target directory to save the output")
    gen_sql: str = Field(..., description="The generated SQL")
    sql_result: Optional[str] = Field(None, description="The result of SQL execution")
    row_count: Optional[int] = Field(None, description="The number of rows returned")
    table_schemas: List[TableSchema] = Field([], description="The schemas of the tables")
    metrics: List[Metric] = Field(default=[], description="The metrics")
    external_knowledge: str = Field(default="", description="The external knowledge")
    prompt_version: str = Field(default="1.0", description="Version for prompt")
    check_result: bool = Field(default=False, description="Whether to check the result of the previous step")
    file_type: Literal["csv", "sql", "json", "all"] = Field(default="all", description="The output file type")


class OutputResult(BaseResult):
    """
    Result model for output node.
    Contains the output result.
    """

    output: str = Field(..., description="The output result")
    sql_query: str = Field(default="", description="First generated SQL")
    sql_result: str = Field(default="", description="The final result of SQL execution")
    sql_query_final: str = Field(default="", description="The final SQL")
    sql_result_final: str = Field(default="", description="The final result of SQL execution")


class ReflectionInput(BaseInput):
    """
    Input model for reflection node.
    Validates input for execution analysis.
    """

    task_description: SqlTask = Field(..., description="Task description containing task details")
    sql_context: List[SQLContext] = Field(..., description="Result and explanation of last execution step")
    prompt_version: str = Field(default="2.1", description="Version for prompt")
    sql_return_sample_line: int = Field(
        default=10,
        description="In SQL, the number of rows in the sample data returned" ", where -1 means return all rows.",
    )
    # sql_return: str = Field(..., description="The SQL execution result to analyze")
    # row_count: int = Field(..., description="Number of rows returned")
    # error: Optional[str] = Field("", description="Error returned")


class StrategyType(str, Enum):
    SUCCESS = "SUCCESS"
    DOC_SEARCH = "DOC_SEARCH"
    SIMPLE_REGENERATE = "SIMPLE_REGENERATE"
    SCHEMA_LINKING = "SCHEMA_LINKING"
    REASONING = "REASONING"
    COLUMN_EXPLORATION = "COLUMN_EXPLORATION"
    UNKNOWN = "UNKNOWN"


STRATEGY_LIST = [strategy.value for strategy in StrategyType]


class ReflectionResult(BaseResult):
    """
    Result model for reflection node.
    Contains analysis results and optimization strategy.
    """

    strategy: Optional[StrategyType] = Field(None, description="Suggested strategy for workflow changes")
    details: Dict[str, Union[str, List[str], Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Detailed analysis information, can contain strings, lists or nested dictionaries",
    )

    class Config:
        use_enum_values = True
