"""Data models for Table and SemanticModel API endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field

# ========== Table Detail Models ==========


class ColumnInfo(BaseModel):
    """Column information."""

    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    nullable: bool = Field(..., description="Whether column is nullable")
    default_value: Optional[str] = Field(None, description="Default value")
    pk: bool = Field(default=False, description="Whether column is primary key")


class IndexInfo(BaseModel):
    """Index information."""

    name: str = Field(..., description="Index name")
    columns: List[str] = Field(..., description="Column names in the index")
    type: str = Field(..., description="Index type (unique/string)")


class TableDetailData(BaseModel):
    """Table detail data."""

    name: str = Field(..., description="Table name")
    description: Optional[str] = Field(None, description="Table description")
    rows: Optional[int] = Field(None, description="Number of rows in the table")
    columns: List[ColumnInfo] = Field(..., description="Column information")
    indexes: List[IndexInfo] = Field(..., description="Index information")


class GetTableDetailInput(BaseModel):
    """Get table detail input."""

    table: str = Field(
        ...,
        description="Full table name e.g. 'production_db.public.frpm' or 'db.schema.table'",
    )


class GetTableDetailData(BaseModel):
    """Get table detail result data."""

    table: TableDetailData


class GetTablesColumnsInput(BaseModel):
    """Batch table-columns input (autocomplete prefetch)."""

    tables: list[str] = Field(
        ...,
        description="Full table names, e.g. ['db.schema.orders', 'db.schema.users']",
    )


class TableColumnBrief(BaseModel):
    """Slim column shape for autocomplete — no default_value (unused there)."""

    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    nullable: bool = Field(..., description="Whether column is nullable")
    pk: bool = Field(default=False, description="Whether column is primary key")


class TableColumns(BaseModel):
    """Columns for a single table."""

    table: str = Field(..., description="Full table name as requested")
    columns: list[TableColumnBrief] = Field(..., description="Column information")


class GetTablesColumnsData(BaseModel):
    """Batch table-columns result. Tables that fail to resolve are omitted."""

    tables: list[TableColumns] = Field(..., description="Per-table columns")


# ========== SemanticModel Models ==========


class GetSemanticModelData(BaseModel):
    """Get semantic model result data."""

    yaml: str = Field(..., description="SemanticModel YAML content")


class SemanticModelInput(BaseModel):
    """Save semantic model input."""

    table: str = Field(..., description="Full table name")
    yaml: str = Field(..., description="SemanticModel YAML content")
    catalog: Optional[str] = Field(None, description="Current catalog context")
    database: Optional[str] = Field(None, description="Current database context")
    db_schema: Optional[str] = Field(None, description="Current schema context")
    semantic_model_name: Optional[str] = Field(None, description="Semantic model owning a shared physical table")


class ValidateSemanticModelData(BaseModel):
    """Validate semantic model result data."""

    valid: bool = Field(..., description="Whether YAML is valid")
    invalid_message: Optional[List[str]] = Field(None, description="Error message if invalid")
