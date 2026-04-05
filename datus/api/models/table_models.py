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


# ========== SemanticModel Models ==========


class GetSemanticModelData(BaseModel):
    """Get semantic model result data."""

    yaml: str = Field(..., description="SemanticModel YAML content")


class SemanticModelInput(BaseModel):
    """Save semantic model input."""

    table: str = Field(..., description="Full table name")
    yaml: str = Field(..., description="SemanticModel YAML content")


class ValidateSemanticModelData(BaseModel):
    """Validate semantic model result data."""

    valid: bool = Field(..., description="Whether YAML is valid")
    invalid_message: Optional[List[str]] = Field(None, description="Error message if invalid")
