# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Optional

from pydantic import BaseModel, Field


class ConnectionConfig(BaseModel):
    """Base connection configuration for all database connectors."""

    timeout_seconds: int = Field(default=30, description="Connection timeout in seconds")

    class Config:
        extra = "forbid"  # Reject unknown fields to catch typos


class FileConnectionConfig(ConnectionConfig):
    """Configuration for file-based databases (SQLite, DuckDB)."""

    db_path: str = Field(..., description="Path to the database file")
    read_only: bool = Field(default=False, description="Whether to open database in read-only mode")


class SQLiteConfig(FileConnectionConfig):
    """SQLite-specific configuration."""

    check_same_thread: bool = Field(
        default=False, description="Check that connection is used in the same thread it was created"
    )
    database_name: Optional[str] = Field(default=None, description="Optional database name override")


class DuckDBConfig(FileConnectionConfig):
    """DuckDB-specific configuration."""

    enable_external_access: bool = Field(default=True, description="Enable external file access")
    memory_limit: Optional[str] = Field(default=None, description="Memory limit (e.g., '2GB')")
    database_name: Optional[str] = Field(default=None, description="Optional database name override")
