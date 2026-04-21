# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from datus.schemas.base import TABLE_TYPE, BaseInput, BaseResult
from datus.schemas.node_models import SQLContext, SqlTask, TableSchema, TableValue
from datus.utils.constants import DBType


class SchemaLinkingInput(BaseInput):
    """
    Input model for schema linking node.
    Validates the input query text for schema analysis.
    """

    input_text: str = Field(..., description="The query text to analyze for schema linking")
    database_type: str = Field(DBType.SQLITE, description="Database type: sqlite, duckdb snowflake, etc ")
    catalog_name: str = Field("", description="Catalog name for context")
    database_name: str = Field("", description="Database name for context")
    schema_name: str = Field("", description="Schema name for context")

    matching_rate: Literal["fast", "medium", "slow", "from_llm"] = Field(
        "fast",
        description="Match rates of the schema linking, allowed values: fast, medium, slow, from_llm",
    )
    sql_context: Optional[SQLContext] = Field(None, description="The SQL context")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt")
    top_n: int = Field(default=5, description="Number of top tables to return")
    table_type: TABLE_TYPE = Field(default="table", description="Table type for the task")

    def top_n_by_rate(self) -> int:
        if self.matching_rate == "fast":
            return 5
        elif self.matching_rate == "medium":
            return 10
        return 20

    @field_validator("matching_rate")
    def validate_matching_rate(cls, v):
        if v not in ["fast", "medium", "slow", "from_llm"]:
            raise ValueError("'matching_rate' must be one of: fast, medium, slow, from_llm")
        return v

    @field_validator("top_n")
    def validate_top_n(cls, v):
        if v <= 0:
            raise ValueError("'top_n' must be greater than 0")
        return v

    @classmethod
    def from_sql_task(
        cls,
        sql_task: SqlTask,
        matching_rate: str = "fast",
    ) -> SchemaLinkingInput:
        """Create SchemaLinkingInput instance from SqlTask."""
        return cls(
            input_text=sql_task.task,
            catalog_name=sql_task.catalog_name,
            database_name=sql_task.database_name,
            database_type=sql_task.database_type,
            schema_name=sql_task.schema_name,
            matching_rate=matching_rate,
            table_type=sql_task.schema_linking_type,
        )


class SchemaLinkingResult(BaseResult):
    """
    Result model for schema linking node.
    Validates the output of schema analysis including table schemas and values.
    """

    table_schemas: List[TableSchema] = Field(default_factory=list, description="List of related table schemas")
    schema_count: int = Field(..., description="Number of schemas found")
    table_values: List[TableValue] = Field(default_factory=list, description="List of related table values")
    value_count: int = Field(..., description="Number of table values found")

    @model_validator(mode="after")
    def _validate_counts(self) -> "SchemaLinkingResult":
        if len(self.table_schemas) != self.schema_count:
            raise ValueError("'schema_count' must match the length of 'table_schemas'")
        if len(self.table_values) != self.value_count:
            raise ValueError("'value_count' must match the length of 'table_values'")
        return self

    def compact_result(self) -> str:
        """Return a compact string representation of schema linking results"""
        schema_tables = ", ".join([schema.table_name for schema in self.table_schemas])
        value_tables = ", ".join([value.table_name for value in self.table_values])
        return f"Schema Tables: [{schema_tables}]\nValue Tables: [{value_tables}]"
