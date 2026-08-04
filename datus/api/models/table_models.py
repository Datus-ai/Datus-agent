"""Data models for Table and SemanticModel API endpoints."""

from typing import Any, Dict, List, Literal, Optional

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


SEMANTIC_MODEL_FILE_DESCRIPTION = (
    "Project-relative semantic model file, e.g. subject/semantic_models/<datasource>/<name>.yml"
)


class GetSemanticModelData(BaseModel):
    """Get semantic model result data."""

    yaml: str = Field(..., description="SemanticModel YAML content")
    semantic_model_name: Optional[str] = Field(None, description="Semantic model name declared by the YAML")
    semantic_model_file: str = Field(..., description=SEMANTIC_MODEL_FILE_DESCRIPTION)
    revision: str = Field(..., description="SHA-256 revision of the YAML content")


class ValidateSemanticModelInput(BaseModel):
    """Validate semantic model input."""

    semantic_model_file: str = Field(..., description=SEMANTIC_MODEL_FILE_DESCRIPTION)
    yaml: str = Field(..., description="SemanticModel YAML content")
    semantic_model_name: Optional[str] = Field(
        None,
        description="Optional assertion that the YAML declares this semantic model",
    )


class SaveSemanticModelInput(ValidateSemanticModelInput):
    """Save semantic model input."""

    expected_revision: Optional[str] = Field(
        None,
        description=(
            "SHA-256 revision returned by the last GET. Omit to skip the concurrency "
            "check; supply it to reject overwrites of a file changed since it loaded."
        ),
    )


class ValidateSemanticModelData(BaseModel):
    """Validate semantic model result data."""

    valid: bool = Field(..., description="Whether YAML is valid")
    invalid_message: Optional[List[str]] = Field(None, description="Error message if invalid")


class SaveSemanticModelData(BaseModel):
    """Outcome of saving and reconciling one semantic model artifact."""

    status: Literal["synced", "saved_not_synced", "validation_failed", "conflict"]
    yaml_saved: bool
    kb_synced: bool
    semantic_model_name: Optional[str] = None
    semantic_model_file: str
    revision: str
    retryable: bool = False
    failed_stage: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    sync: Optional[Dict[str, Any]] = None
