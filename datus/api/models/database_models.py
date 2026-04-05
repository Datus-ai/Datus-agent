"""Pydantic models for Database Management API endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


# Database listing models
class DatabaseInfo(BaseModel):
    """Information about a database connection."""

    name: str = Field(..., description="Database name")
    uri: str = Field(..., description="Database connection URI")
    type: str = Field(..., description="Database type (sqlite, duckdb, postgresql, etc.)")
    current: bool = Field(..., description="Whether this is the current database")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    connection_status: str = Field(..., description="Connection status (connected, disconnected)")
    tables_count: Optional[int] = Field(None, description="Number of tables in the database")
    last_accessed: Optional[str] = Field(None, description="Last access timestamp")
    tables: Optional[List[str]] = Field(None, description="List of table names")


class ListDatabasesInput(BaseModel):
    """Input model for listing databases."""

    datasource_id: str = Field("", description="The id of datasource to list databases from")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: str = Field("", description="Database name")
    schema_name: str = Field("", description="Schema name")
    include_sys_schemas: bool = Field(False, description="Include system schemas when listing databases")


class ListDatabasesData(BaseModel):
    """Data for listing databases."""

    databases: List[DatabaseInfo] = Field(..., description="List of databases")
    total_count: int = Field(..., description="Total number of databases")
    current_database: Optional[str] = Field(None, description="Current database name")


class DatabasesData(BaseModel):
    """Data for database list."""

    databases: List[DatabaseInfo] = Field(..., description="List of databases data")
