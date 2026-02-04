# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from datus.schemas.base import BaseInput, BaseResult

# =============================================================================
# Document Search Models
# =============================================================================


class DocSearchInput(BaseInput):
    """Input model for document search."""

    platform: str = Field(..., description="Platform name (e.g., snowflake, duckdb, postgresql)")
    keywords: List[str] = Field(..., description="Keywords to search for in documents")
    version: Optional[str] = Field(None, description="Filter by version (optional)")
    top_n: int = Field(5, description="Number of documents to return per keyword")

    @field_validator("top_n")
    def validate_top_n(cls, v):
        if v <= 0:
            raise ValueError("'top_n' must be a positive integer")
        return v


class DocSearchResult(BaseResult):
    """Result model for document search."""

    docs: Any = Field(
        default_factory=dict,
        description="Retrieved documents for each keyword, with full metadata",
    )
    doc_count: int = Field(0, description="Total number of documents found")


# =============================================================================
# Document Navigation Models
# =============================================================================


class DocNavInput(BaseInput):
    """Input model for listing document navigation structure."""

    platform: str = Field(..., description="Platform name (e.g., snowflake, duckdb, postgresql)")
    version: Optional[str] = Field(None, description="Filter by version (optional)")


class DocNavResult(BaseResult):
    """Result model for document navigation listing."""

    platform: str = Field(..., description="Platform name")
    version: Optional[str] = Field(None, description="Version filter applied")
    nav_tree: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Navigation tree with titles and hierarchy",
    )
    total_docs: int = Field(0, description="Total number of unique documents")


# =============================================================================
# Get Document Models
# =============================================================================


class GetDocInput(BaseInput):
    """Input model for getting document by titles."""

    platform: str = Field(..., description="Platform name (e.g., snowflake, duckdb, postgresql)")
    titles: List[str] = Field(..., description="List of titles to match in hierarchy")
    version: Optional[str] = Field(None, description="Filter by version (optional)")


class GetDocResult(BaseResult):
    """Result model for getting document content."""

    platform: str = Field(..., description="Platform name")
    version: Optional[str] = Field(None, description="Version of document")
    title: str = Field("", description="Matched document title")
    hierarchy: str = Field("", description="Full hierarchy path")
    chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Document chunks in order",
    )
    chunk_count: int = Field(0, description="Number of chunks")
