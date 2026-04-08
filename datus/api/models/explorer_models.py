"""Data models for Explorer API endpoints."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ========== Catalog Models ==========


class DatabaseCatalogInfo(BaseModel):
    """Database catalog information."""

    name: str = Field(..., description="Database name")
    type: str = Field(..., description="Database type (snowflake, mysql, postgresql, starrocks)")
    uri: str = Field(..., description="Database connection URI")
    current: bool = Field(..., description="Whether this is the current database")
    catalog_name: Optional[str] = Field(None, description="Catalog name for databases that support catalogs")
    schema_name: Optional[str] = Field(None, description="Schema name")
    connection_status: str = Field(..., description="Connection status (connected/disconnected)")
    tables_count: int = Field(..., description="Number of tables")
    last_accessed: str = Field(..., description="Last accessed timestamp")
    schemas: Optional[Dict[str, List[str]]] = Field(
        None, description="Schema -> tables mapping (for databases with schemas)"
    )
    tables: Optional[List[str]] = Field(None, description="List of tables (for databases without schemas)")


class CatalogListData(BaseModel):
    """Catalog list data."""

    databases: List[DatabaseCatalogInfo]


# ========== Subject Tree Models ==========


class SubjectNodeType(str, Enum):
    """Subject node type."""

    DIRECTORY = "directory"
    METRIC = "metric"
    REFERENCE_SQL = "reference_sql"
    KNOWLEDGE = "knowledge"


class SubjectNode(BaseModel):
    """Subject tree node."""

    name: str = Field(..., description="Node name")
    type: Optional[SubjectNodeType] = Field(None, description="Node type")
    subject_path: List[str] = Field(default_factory=list, description="Full path from root")
    children: Optional[List["SubjectNode"]] = Field(None, description="Child nodes")


# Enable forward reference
SubjectNode.model_rebuild()


class SubjectListData(BaseModel):
    """Subject list data."""

    subjects: List[SubjectNode]


# ========== Create Operations ==========


class CreateDirectoryInput(BaseModel):
    """Create directory input."""

    subject_path: List[str] = Field(..., description="Parent path where directory will be created")


class CreateMetricInput(BaseModel):
    """Create metric input."""

    subject_path: List[str] = Field(..., description="Parent path where metric will be created")
    name: str = Field(..., min_length=1, description="Metric name")


# ========== Rename/Delete Operations ==========


class RenameSubjectInput(BaseModel):
    """Rename/move subject input."""

    type: SubjectNodeType = Field(..., description="Type of subject to rename")
    subject_path: List[str] = Field(..., description="Current path of the subject")
    new_subject_path: List[str] = Field(..., description="New path for the subject")


class DeleteSubjectInput(BaseModel):
    """Delete subject input."""

    type: SubjectNodeType = Field(..., description="Type of subject to delete")
    subject_path: List[str] = Field(..., description="Path of the subject to delete")


# ========== Metric Get/Edit ==========


class MetricInfo(BaseModel):
    """Metric information."""

    name: str = Field(..., description="Metric name")
    yaml: str = Field(..., description="Metric YAML content")


class EditMetricInput(BaseModel):
    """Edit metric input."""

    subject_path: List[str] = Field(..., description="Path to the metric")
    yaml: str = Field(..., description="Updated YAML content")


class EditSemanticModelInput(BaseModel):
    """Edit semantic model entry (table or column)."""

    entry_id: str = Field(..., description="Entry ID (e.g., 'table:orders', 'column:orders.amount')")
    update_values: Dict[str, Any] = Field(..., description="Fields to update (e.g., {'description': 'new desc'})")


# ========== Reference SQL Create/Get/Edit ==========


class ReferenceSQLInfo(BaseModel):
    """Reference SQL information."""

    name: str = Field(..., description="Reference SQL name")
    sql: str = Field(..., description="SQL query")
    summary: str = Field(..., description="SQL summary")
    search_text: str = Field(..., description="Text for vector search")


class ReferenceSQLInput(ReferenceSQLInfo):
    """Create reference SQL input."""

    subject_path: List[str] = Field(..., description="Parent path where reference SQL will be created")


class SubjectPathInput(BaseModel):
    """Subject path input."""

    subject_path: List[str] = Field(..., description="Subject path name")


# ========== Knowledge Create/Get/Edit ==========


class CreateKnowledgeInput(BaseModel):
    """Create knowledge input."""

    subject_path: List[str] = Field(..., description="Subject path for the knowledge")
    name: str = Field(..., min_length=1, description="Knowledge name")
    search_text: str = Field(..., min_length=1, description="Search text for vector retrieval")
    explanation: str = Field(..., min_length=1, description="Knowledge explanation/details")


class KnowledgeInfo(BaseModel):
    """Knowledge information for get response."""

    name: str = Field(..., description="Knowledge name")
    search_text: str = Field(..., description="Search text for vector retrieval")
    explanation: str = Field(..., description="Knowledge explanation/details")


class EditKnowledgeInput(BaseModel):
    """Edit knowledge input."""

    subject_path: List[str] = Field(..., description="Subject path to the knowledge (parent_path + name)")
    search_text: str = Field(..., description="Updated search text")
    explanation: str = Field(..., description="Updated explanation")
