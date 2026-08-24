"""Pydantic models for Database Management API endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


# Database listing models
class DatabaseInfo(BaseModel):
    """Information about a database connection."""

    name: str = Field(..., description="Database name")
    # Which configured datasource this database came from. A project can bind
    # several, and a client rendering them in one tree cannot address a table
    # without knowing which connection profile to reach it through.
    datasource: str = Field("", description="Name of the datasource (connection profile) serving this database")
    uri: str = Field(..., description="Database connection URI")
    type: str = Field(..., description="Database type (sqlite, duckdb, postgresql, etc.)")
    current: bool = Field(..., description="Whether this is the current database")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    connection_status: str = Field(..., description="Connection status (connected, disconnected)")
    tables_count: Optional[int] = Field(None, description="Number of tables in the database")
    last_accessed: Optional[str] = Field(None, description="Last access timestamp")
    tables: Optional[List[str]] = Field(None, description="List of table names")
    error: Optional[str] = Field(
        None, description="Why the object listing is unavailable, when the connection itself is usable"
    )


class ListDatabasesInput(BaseModel):
    """Input model for listing databases."""

    datasource_id: str = Field(
        "", description="Name of the datasource to list databases from; empty means the current one"
    )
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: str = Field("", description="Database name")
    schema_name: str = Field("", description="Schema name")
    include_sys_schemas: bool = Field(False, description="Include system schemas when listing databases")


class ListDatabasesData(BaseModel):
    """Data for listing databases."""

    databases: List[DatabaseInfo] = Field(..., description="List of databases")
    total_count: int = Field(..., description="Total number of databases")
    current_database: Optional[str] = Field(None, description="Current database name")
    current_datasource: Optional[str] = Field(None, description="Datasource these databases were listed from")


class DatabasesData(BaseModel):
    """Data for database list."""

    databases: List[DatabaseInfo] = Field(..., description="List of databases data")
