# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Abstract base classes for vector database backends.

Three-layer abstraction:
  BaseVectorBackend  (lifecycle: initialize, connect, close)
      |
      +-- connect(namespace) -> VectorDatabase  (db-level: table operations)
                                    |
                                    +-- open_table(name) -> VectorTable  (table-level: CRUD + search)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from datus.storage.conditions import WhereExpr

# ---------------------------------------------------------------------------
# Embedding function ABC (backend-independent)
# ---------------------------------------------------------------------------


class EmbeddingFunction(ABC):
    """Backend-independent embedding function base class.

    All embedding implementations (FastEmbedEmbeddings, OpenAIEmbeddings, etc.)
    must inherit from this class.  Vector backends receive instances of this
    type and adapt them internally to their own native representation.
    """

    name: str

    @abstractmethod
    def ndims(self) -> int:
        """Return the dimensionality of the embedding vectors."""

    @abstractmethod
    def generate_embeddings(self, texts: Union[List[str], np.ndarray], *args, **kwargs) -> List[List[float]]:
        """Generate embedding vectors for the given texts."""


# ---------------------------------------------------------------------------
# Table-level interface
# ---------------------------------------------------------------------------


class VectorTable(ABC):
    """Abstract interface for a single vector table."""

    # -- Write operations --

    @abstractmethod
    def add(self, data: pd.DataFrame) -> None:
        """Add rows to the table."""

    @abstractmethod
    def merge_insert(self, data: pd.DataFrame, on_column: str) -> None:
        """Upsert rows using merge insert."""

    @abstractmethod
    def delete(self, where: WhereExpr) -> None:
        """Delete rows matching a where clause."""

    @abstractmethod
    def update(self, where: WhereExpr, values: Dict[str, Any]) -> None:
        """Update rows matching a where clause."""

    # -- Search operations --

    @abstractmethod
    def search_vector(
        self,
        query_text: str,
        vector_column: str,
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Perform a vector similarity search."""

    @abstractmethod
    def search_hybrid(
        self,
        query_text: str,
        vector_source_column: str,
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Perform a hybrid (vector + FTS) search with reranking."""

    @abstractmethod
    def search_all(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        """Retrieve all rows with optional filtering."""

    @abstractmethod
    def count_rows(self, where: WhereExpr = None) -> int:
        """Count rows with optional filter."""

    # -- Index operations --

    @abstractmethod
    def create_vector_index(self, column: str, metric: str = "cosine", **kwargs) -> None:
        """Create a vector index."""

    @abstractmethod
    def create_fts_index(self, field_names: Union[str, List[str]]) -> None:
        """Create a full-text search index."""

    @abstractmethod
    def create_scalar_index(self, column: str) -> None:
        """Create a scalar index."""

    # -- Maintenance operations (no-op defaults) --

    def compact_files(self) -> None:
        """Compact table files. Not all backends support this."""

    def cleanup_old_versions(self) -> None:
        """Clean up old table versions. Not all backends support this."""


# ---------------------------------------------------------------------------
# Database-level interface
# ---------------------------------------------------------------------------


class VectorDatabase(ABC):
    """Abstract interface for a vector database connection (namespace)."""

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""

    @abstractmethod
    def table_names(self, limit: int = 100) -> List[str]:
        """List table names in the database."""

    @abstractmethod
    def create_table(
        self,
        table_name: str,
        schema: Optional[pa.Schema] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
        exist_ok: bool = True,
        unique_columns: Optional[List[str]] = None,
    ) -> VectorTable:
        """Create a table and return a VectorTable handle.

        Args:
            table_name: Name of the table.
            schema: Optional PyArrow schema for column definitions.
            embedding_function: Optional embedding function for vector generation.
            vector_column: Name of the vector column.
            source_column: Name of the source text column.
            exist_ok: If True, do not raise if the table already exists.
            unique_columns: Optional list of column names that should have
                unique constraints.  Backends that support upsert via
                ``ON CONFLICT`` (e.g. PostgreSQL) will create the necessary
                unique indexes.  Backends that do not need them (e.g. LanceDB)
                may safely ignore this parameter.
        """

    @abstractmethod
    def open_table(
        self,
        table_name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
    ) -> VectorTable:
        """Open an existing table and return a VectorTable handle.

        Args:
            table_name: Name of the table to open.
            embedding_function: Optional embedding function. Backends that
                persist embedding configuration (e.g. LanceDB) may ignore this.
            vector_column: Name of the vector column (used by some backends).
            source_column: Name of the source text column (used by some backends).
        """

    @abstractmethod
    def drop_table(self, table_name: str, ignore_missing: bool = False) -> None:
        """Drop a table from the database."""

    def refresh_table(
        self,
        table_name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
    ) -> VectorTable:
        """Refresh table handle (for retry). Default: re-open."""
        return self.open_table(table_name, embedding_function, vector_column, source_column)

    def close(self) -> None:
        """Release resources held by this database connection."""


# ---------------------------------------------------------------------------
# Backend-level interface (lifecycle only)
# ---------------------------------------------------------------------------


class BaseVectorBackend(ABC):
    """Abstract base for vector database backends.

    Responsible only for lifecycle management.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the backend with configuration."""

    @abstractmethod
    def connect(self, namespace: str) -> VectorDatabase:
        """Return a VectorDatabase for the given namespace.

        Args:
            namespace: Logical namespace for data isolation. Must not be empty.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
