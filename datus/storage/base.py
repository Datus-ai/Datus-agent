# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
from datus_storage_base.conditions import WhereExpr
from datus_storage_base.vector.base import VectorDatabase, VectorTable

from datus.storage.embedding_models import EmbeddingModel
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _SharedTableState:
    """Mutable state shared between a singleton storage and its scoped views.

    Using a dedicated object (instead of a plain bool) ensures that
    ``copy.copy()`` preserves the reference — when the singleton sets
    ``initialized = True`` or updates ``table``, all scoped views see
    the change immediately.
    """

    __slots__ = ("initialized", "table")

    def __init__(self):
        self.initialized: bool = False
        self.table: Optional[VectorTable] = None


class StorageBase:
    """Base class for all storage components using a vector backend."""

    def __init__(self, db: Optional[VectorDatabase] = None):
        """Initialize the storage base.

        Args:
            db: Optional pre-created VectorDatabase connection.
                If provided, it is used directly instead of the global
                namespace connection.  This allows stores like DocumentStore
                to use per-platform isolated databases.
        """
        if db is not None:
            self.db: VectorDatabase = db
        else:
            from datus.storage.backend_holder import create_vector_connection

            self.db = create_vector_connection()

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format (UTC)."""
        return datetime.now(timezone.utc).isoformat()


class BaseEmbeddingStore(StorageBase):
    """Base class for all embedding stores using a vector backend.
    table_name: the name of the table to store the embedding
    embedding_field: the field name of the embedding
    """

    def __init__(
        self,
        table_name: str,
        embedding_model: EmbeddingModel,
        on_duplicate_columns: str = "vector",
        schema: Optional[pa.Schema] = None,
        vector_source_name: str = "definition",
        vector_column_name: str = "vector",
        unique_columns: Optional[List[str]] = None,
        db: Optional[VectorDatabase] = None,
        table_prefix: str = "",
        extra_fields: Optional[List[pa.Field]] = None,
        default_values: Optional[Dict[str, Any]] = None,
        scope_indices: Optional[List[str]] = None,
    ):
        super().__init__(db=db)
        self.model = embedding_model
        self.batch_size = embedding_model.batch_size
        self.table_name = f"{table_prefix}{table_name}" if table_prefix else table_name
        self.vector_source_name = vector_source_name
        self.vector_column_name = vector_column_name
        self.on_duplicate_columns = on_duplicate_columns
        # Append extra fields to schema if provided
        if schema is not None and extra_fields:
            schema = pa.schema(list(schema) + extra_fields)
        # Ensure datasource_id field exists for subject-tree-based stores
        # (needed for RDB subject tree lookups even in PHYSICAL isolation mode)
        if schema is not None:
            existing_names = {f.name for f in schema}
            if "datasource_id" not in existing_names:
                schema = pa.schema(list(schema) + [pa.field("datasource_id", pa.string())])
        self._schema = schema
        self._unique_columns = unique_columns
        self._default_values: Dict[str, Any] = dict(default_values) if default_values else {}
        self._scope_indices: List[str] = list(scope_indices or [])
        # Delay table initialization until first use.
        self._shared = _SharedTableState()
        self._table_lock = Lock()
        self._write_lock = Lock()

    @property
    def table(self) -> Optional[VectorTable]:
        return self._shared.table

    @table.setter
    def table(self, value: Optional[VectorTable]):
        self._shared.table = value

    def _ensure_table_ready(self):
        """Ensure table is ready for operations, with proper error handling."""
        if self._shared.initialized:
            return

        with self._table_lock:
            if self._shared.initialized:
                return

            # First check if embedding model is available
            self._check_embedding_model_ready()
            # Initialize table with embedding function
            self._ensure_table(self._schema)
            self._shared.initialized = True
            # Auto-create scalar indices for scope fields (e.g. workspace_id)
            for col in self._scope_indices:
                self._create_scalar_index(col)
            logger.debug(f"Table {self.table_name} initialized successfully with embedding function")

    def _apply_default_values(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill in default values for rows that are missing them."""
        if not self._default_values:
            return data
        for row in data:
            for k, v in self._default_values.items():
                row.setdefault(k, v)
        return data

    def _search_all(
        self, where: WhereExpr = None, select_fields: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> pa.Table:
        self._ensure_table_ready()
        if limit:
            row_limit = limit
        else:
            row_limit = self.table.count_rows(where) if where else self.table.count_rows()
        result = self.table.search_all(where=where, select_fields=select_fields, limit=row_limit)
        if self.vector_column_name in result.column_names:
            result = result.drop([self.vector_column_name])
        return result

    def _check_embedding_model_ready(self):
        """Check if embedding model is ready for use."""
        # Check if model has failed before
        if self.model.is_model_failed:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=(
                    f"Embedding model '{self.model.model_name}' is not available: {self.model.model_error_message}"
                ),
            )

        # Try to access the model (this will trigger lazy loading)
        try:
            _ = self.model.model
        except DatusException as e:
            # Re-raise DatusException directly to avoid nesting
            raise e
        except Exception as e:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=f"Embedding model '{self.model.model_name}' initialization failed: {str(e)}",
            ) from e

    def truncate(self) -> None:
        """Drop the entire table and reset state (admin operation)."""
        with self._table_lock:
            self.db.drop_table(self.table_name, ignore_missing=True)
            self._shared.table = None
            self._shared.initialized = False

    def truncate_scoped(self) -> None:
        """Delete all rows visible to the current connection.

        In ``IsolationType.LOGICAL`` mode the backend automatically scopes
        the delete to the current ``datasource_id``.  In ``PHYSICAL`` mode
        this drops the entire table (equivalent to ``truncate()``).
        """
        from datus.storage.backend_holder import get_isolation_type

        self._ensure_table_ready()
        if get_isolation_type() == "logical":
            # LOGICAL mode — delete scoped rows via backend's _ds_where()
            self.table.delete(None)
        else:
            # PHYSICAL mode — drop the whole table
            self.truncate()

    def _ensure_table(self, schema: Optional[pa.Schema] = None):
        if self.db.table_exists(self.table_name):
            self.table = self.db.open_table(
                self.table_name,
                embedding_function=self.model.model,
                vector_column=self.vector_column_name,
                source_column=self.vector_source_name,
            )
        else:
            try:
                self.table = self.db.create_table(
                    self.table_name,
                    schema=schema,
                    embedding_function=self.model.model,
                    vector_column=self.vector_column_name,
                    source_column=self.vector_source_name,
                    exist_ok=True,
                    unique_columns=self._unique_columns,
                )
            except Exception as e:
                raise DatusException(
                    ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                    message_args={"operation": "create_table", "table_name": self.table_name, "error_message": str(e)},
                ) from e

    def create_vector_index(
        self,
        metric: str = "cosine",
    ):
        """
        Create a vector index (IVF_PQ or IVF_FLAT) for the table to optimize vector search.

        Args:
            metric (str): Distance metric for vector search ('cosine', 'l2', or 'dot').
                Default: 'cosine'.
        """
        self._ensure_table_ready()
        if not self._supports_runtime_indexing():
            return
        try:
            row_count = self.table.count_rows()
            logger.debug(f"Creating vector index for {self.table_name} with {row_count} rows")

            # Determine index type based on dataset size
            index_type = "IVF_PQ" if row_count >= 5000 else "IVF_FLAT"
            logger.debug(f"Selected index type: {index_type}")

            # Calculate number of partitions (IVF)
            num_partitions = max(1, min(1024, int(row_count**0.5)))
            if row_count < 1000:
                num_partitions = max(1, row_count // 10)
            elif row_count < 5000:
                num_partitions = max(1, row_count // 20)
            logger.debug(f"Number of partitions: {num_partitions}")

            # Calculate number of sub-vectors (PQ, only for IVF_PQ)
            num_sub_vectors = 32
            if index_type == "IVF_PQ":
                vector_dim = self.model.dim_size
                if row_count < 1000:
                    num_sub_vectors = min(16, max(8, vector_dim // 64))
                elif row_count < 5000:
                    num_sub_vectors = min(32, max(16, vector_dim // 32))
                else:
                    num_sub_vectors = min(96, max(32, vector_dim // 16))
                logger.debug(f"Number of sub-vectors: {num_sub_vectors}")

            index_params = {
                "index_type": index_type,
                "num_partitions": num_partitions,
                "replace": True,
            }
            if index_type == "IVF_PQ":
                index_params["num_sub_vectors"] = num_sub_vectors
            accelerator = self.model.device
            if accelerator and accelerator == "cuda" or accelerator == "mps":
                index_params["accelerator"] = accelerator

            self.table.create_vector_index(self.vector_column_name, metric=metric, **index_params)
            logger.debug(f"Successfully created {index_type} index for {self.table_name}")

        except Exception as e:
            logger.warning(f"Failed to create vector index for {self.table_name}: {str(e)}")

    def create_fts_index(self, field_names: Union[str, List[str]]):
        """Create a full-text search index (LanceDB only)."""
        self._ensure_table_ready()
        if not self._supports_runtime_indexing():
            return
        try:
            self.table.create_fts_index(field_names)
        except Exception as e:
            logger.warning(f"Failed to create fts index for {self.table_name} table: {str(e)}")

    def store_batch(self, data: List[Dict[str, Any]]):
        """
        Store a batch of data in the database. The following steps are performed:

            1. Encode the vector field
            2. Merge insert the data into the table

        Args:
            data: List[Dict[str, Any]] the data to store
        """
        if not data:
            return
        data = self._apply_default_values(data)
        # Ensure table is ready before storing data
        self._ensure_table_ready()

        try:
            with self._write_lock:
                if len(data) <= self.batch_size:
                    self._add_with_retry(pd.DataFrame(data))
                    return
                # split the data into batches and store them
                for i in range(0, len(data), self.batch_size):
                    batch = data[i : i + self.batch_size]
                    self._add_with_retry(pd.DataFrame(batch))
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def store(self, data: List[Dict[str, Any]]):
        if not data:
            return
        data = self._apply_default_values(data)
        # Ensure table is ready before storing data
        self._ensure_table_ready()
        try:
            with self._write_lock:
                self._add_with_retry(pd.DataFrame(data))
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def upsert_batch(self, data: List[Dict[str, Any]], on_column: str = "id"):
        """
        Upsert a batch of data using merge_insert (update if exists, insert if not).

        Args:
            data: List of dictionaries to upsert
            on_column: Column name to match for deduplication (default: "id")
        """
        if not data:
            return
        data = self._apply_default_values(data)
        self._ensure_table_ready()

        # Deduplicate input data by on_column, keeping the last occurrence
        # This prevents duplicates when the same id appears multiple times in the input batch
        df = pd.DataFrame(data)
        if on_column in df.columns:
            original_count = len(df)
            df = df.drop_duplicates(subset=[on_column], keep="last")
            if len(df) < original_count:
                logger.debug(
                    f"Deduplicated {original_count - len(df)} records with duplicate '{on_column}' before upsert"
                )
        data = df.to_dict("records")

        try:
            with self._write_lock:
                if len(data) <= self.batch_size:
                    self._upsert_with_retry(pd.DataFrame(data), on_column)
                    return
                # Split the data into batches and upsert them
                for i in range(0, len(data), self.batch_size):
                    batch = data[i : i + self.batch_size]
                    self._upsert_with_retry(pd.DataFrame(batch), on_column)
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def _upsert_with_retry(
        self, frame: pd.DataFrame, on_column: str, max_attempts: int = 3, initial_delay: float = 0.05
    ) -> None:
        """Upsert a DataFrame with simple retry/backoff on commit conflicts."""
        if self.table is None:
            raise DatusException(
                ErrorCode.STORAGE_SAVE_FAILED,
                message_args={"error_message": "Table is not initialized"},
            )

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self.table.merge_insert(frame, on_column)
                return
            except Exception as err:
                error_message = str(err)
                if "Commit conflict" not in error_message:
                    raise err

                last_error = err
                delay = initial_delay * (attempt + 1)
                logger.warning(
                    f"Commit conflict detected when upserting to table '{self.table_name}' "
                    f"(attempt {attempt + 1}/{max_attempts}). Retrying after {delay:.2f}s."
                )
                self.table = self.db.refresh_table(
                    self.table_name,
                    embedding_function=self.model.model,
                    vector_column=self.vector_column_name,
                    source_column=self.vector_source_name,
                )
                time.sleep(delay)

        assert last_error is not None  # for type checkers
        raise last_error

    def _add_with_retry(self, frame: pd.DataFrame, max_attempts: int = 3, initial_delay: float = 0.05) -> None:
        """Insert a DataFrame with simple retry/backoff on commit conflicts."""
        if self.table is None:
            raise DatusException(
                ErrorCode.STORAGE_SAVE_FAILED,
                message_args={"error_message": "Table is not initialized"},
            )

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self.table.add(frame)
                return
            except Exception as err:
                error_message = str(err)
                if "Commit conflict" not in error_message:
                    raise err

                last_error = err
                delay = initial_delay * (attempt + 1)
                logger.warning(
                    f"Commit conflict detected when writing to table '{self.table_name}' "
                    f"(attempt {attempt + 1}/{max_attempts}). Retrying after {delay:.2f}s."
                )
                self.table = self.db.refresh_table(
                    self.table_name,
                    embedding_function=self.model.model,
                    vector_column=self.vector_column_name,
                    source_column=self.vector_source_name,
                )
                time.sleep(delay)

        assert last_error is not None  # for type checkers
        raise last_error

    def search(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
        query_type: str = "vector",
    ) -> pa.Table:
        self._ensure_table_ready()

        if query_type == "hybrid":
            search_result = self._search_hybrid(query_txt, select_fields, top_n, where)
        else:
            search_result = self._search_vector(query_txt, select_fields, top_n, where)
        if self.vector_column_name in search_result.column_names:
            search_result = search_result.drop([self.vector_column_name])
        return search_result

    def _search_hybrid(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
    ) -> pa.Table:
        try:
            if not top_n:
                top_n = self.table.count_rows(where) if where else self.table.count_rows()
            results = self.table.search_hybrid(
                query_txt,
                self.vector_source_name,
                top_n,
                where=where,
                select_fields=select_fields,
            )
            if len(results) > top_n:
                results = results[:top_n]
            return results
        except Exception as e:
            logger.warning(f"Failed to search hybrid: {str(e)}, use vector search instead")
            return self._search_vector(query_txt, select_fields, top_n, where)

    def _search_vector(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
    ) -> pa.Table:
        try:
            if not top_n:
                top_n = self.table.count_rows(where) if where else self.table.count_rows()
            return self.table.search_vector(
                query_txt,
                self.vector_column_name,
                top_n,
                where=where,
                select_fields=select_fields,
            )
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_SEARCH_FAILED,
                message_args={
                    "error_message": str(e),
                    "query": query_txt,
                    "where_clause": str(where) if where else "(none)",
                    "top_n": str(top_n or "all"),
                },
            ) from e

    def table_size(self) -> int:
        self._ensure_table_ready()
        return self.table.count_rows()

    def update(self, where: WhereExpr, update_values: Dict[str, Any], unique_filter: Optional[WhereExpr] = None):
        self._ensure_table_ready()
        if not update_values:
            return
        if not where:
            return
        if unique_filter:
            existing = self.table.count_rows(unique_filter)
            if existing:
                raise DatusException(
                    ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                    message_args={
                        "operation": "update",
                        "table_name": self.table_name,
                        "error_message": f"Conflicting rows already match {unique_filter}",
                    },
                )

        # Re-embed when the vector source column is updated.
        # Only overwrite the vector when we got a non-None embedding back; on failure or empty
        # source text, leave the existing vector untouched so the row stays usable.
        if self.vector_source_name in update_values:
            new_text = update_values[self.vector_source_name]
            if new_text:
                try:
                    vectors = self.model.model.generate_embeddings([str(new_text)])
                    if vectors and len(vectors) > 0 and vectors[0] is not None:
                        update_values[self.vector_column_name] = vectors[0]
                except Exception as e:
                    logger.warning(f"Failed to re-embed on update for {self.table_name}: {e}")

        self.table.update(where=where, values=update_values)

    # -- Convenience methods for subclasses --

    def _supports_runtime_indexing(self) -> bool:
        """Check if the backend supports runtime index creation.

        LanceDB requires explicit index creation after data insertion.
        Other backends (e.g. pgvector) handle indexing at DDL level
        and should skip runtime index calls.
        """
        return hasattr(self.table, "create_scalar_index") and type(self.table).__name__.startswith("Lance")

    def _create_scalar_index(self, column: str) -> None:
        """Create a scalar index on the given column (LanceDB only)."""
        self._ensure_table_ready()
        if not self._supports_runtime_indexing():
            return
        try:
            self.table.create_scalar_index(column)
        except Exception as e:
            logger.warning(f"Failed to create scalar index on '{column}' for {self.table_name}: {str(e)}")

    def _delete_rows(self, where: WhereExpr) -> None:
        """Delete rows matching the where clause."""
        self._ensure_table_ready()
        if where:
            self.table.delete(where)

    def _count_rows(self, where: WhereExpr = None) -> int:
        """Count rows with optional filter."""
        self._ensure_table_ready()
        return self.table.count_rows(where)

    def query_with_filter(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        """Query rows with filter, field selection, and optional limit."""
        self._ensure_table_ready()
        if limit is None:
            limit = self.table.count_rows(where)
        return self.table.search_all(
            where=where,
            select_fields=select_fields,
            limit=limit,
        )
