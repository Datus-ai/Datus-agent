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

from datus.storage.datasource_scope import (
    DATASOURCE_ID_COLUMN,
    STORAGE_KEY_COLUMN,
    build_storage_key,
    datasource_condition,
)
from datus.storage.embedding_models import EmbeddingModel
from datus.storage.fts import FtsField, FtsIndexStatus, FtsSpec
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
                If provided, it is used directly instead of opening a new
                connection bound to the active project. Stores that need
                a composite-project scope (e.g. ``DocumentStore``) pass
                their own ``db`` built from the desired project.
        """
        if db is not None:
            self.db: VectorDatabase = db
        else:
            from datus.storage.backend_holder import create_vector_connection
            from datus.utils.path_manager import get_path_manager

            self.db = create_vector_connection(get_path_manager().project_name)

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
        datasource_scoped: bool = False,
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
        # Datasource-scoped shared tables keep the physical namespace at the
        # project level and isolate rows by datasource_id/storage_key.
        if datasource_scoped and schema is not None:
            existing_names = {f.name for f in schema}
            extra_scope_fields = []
            if DATASOURCE_ID_COLUMN not in existing_names:
                extra_scope_fields.append(pa.field(DATASOURCE_ID_COLUMN, pa.string()))
            if "id" in existing_names and STORAGE_KEY_COLUMN not in existing_names:
                extra_scope_fields.append(pa.field(STORAGE_KEY_COLUMN, pa.string()))
            if extra_scope_fields:
                schema = pa.schema(list(schema) + extra_scope_fields)
        self._schema = schema
        self._unique_columns = unique_columns
        self._default_values: Dict[str, Any] = dict(default_values) if default_values else {}
        self._scope_indices: List[str] = list(scope_indices or [])
        self._datasource_scoped = datasource_scoped
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

    def _open_existing_table_for_read(self) -> Optional[VectorTable]:
        """Open an existing vector table for read-only access without embeddings."""
        if self.table is not None:
            return self.table
        if not self.db.table_exists(self.table_name):
            return None
        self.table = self.db.open_table(self.table_name)
        self._ensure_persisted_scope_columns()
        return self.table

    def _empty_result(self, select_fields: Optional[List[str]] = None) -> pa.Table:
        """Return an empty result table shaped like this store's schema."""
        schema = self._schema or pa.schema([])
        if select_fields is None:
            fields = [field for field in schema if field.name != self.vector_column_name]
        else:
            fields = []
            for field_name in select_fields:
                if field_name == self.vector_column_name:
                    continue
                if field_name in schema.names:
                    fields.append(schema.field(field_name))
                else:
                    fields.append(pa.field(field_name, pa.null()))
        result_schema = pa.schema(fields)
        arrays = [pa.array([], type=field.type) for field in fields]
        return pa.Table.from_arrays(arrays, schema=result_schema)

    def _apply_default_values(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill in default values for rows that are missing them."""
        schema_names = set(self._schema.names) if self._datasource_scoped and self._schema is not None else set()
        for row in data:
            for k, v in self._default_values.items():
                row.setdefault(k, v)
            if DATASOURCE_ID_COLUMN in schema_names:
                row.setdefault(DATASOURCE_ID_COLUMN, "")
            if STORAGE_KEY_COLUMN in schema_names and row.get("id") not in (None, ""):
                row.setdefault(STORAGE_KEY_COLUMN, build_storage_key(row.get(DATASOURCE_ID_COLUMN, ""), row["id"]))
        return data

    def _scope_column_migration_exprs(self) -> Dict[str, str]:
        """Return default SQL expressions for missing row-scope columns."""

        if not self._datasource_scoped or self._schema is None:
            return {}
        schema_names = set(self._schema.names)
        exprs: Dict[str, str] = {}
        if DATASOURCE_ID_COLUMN in schema_names:
            exprs[DATASOURCE_ID_COLUMN] = "''"
        if STORAGE_KEY_COLUMN in schema_names and "id" in schema_names:
            exprs[STORAGE_KEY_COLUMN] = "'legacy:' || id"
        return exprs

    def _ensure_persisted_scope_columns(self) -> None:
        """Best-effort migration for existing vector tables missing scope columns."""

        if self.table is None:
            return
        ensure_columns = getattr(self.table, "ensure_columns", None)
        if ensure_columns is None:
            return
        exprs = self._scope_column_migration_exprs()
        if not exprs:
            return
        try:
            ensure_columns(exprs)
        except Exception as exc:
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message_args={
                    "operation": "ensure_scope_columns",
                    "table_name": self.table_name,
                    "error_message": str(exc),
                },
            ) from exc

    def _search_all(
        self, where: WhereExpr = None, select_fields: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> pa.Table:
        table = self._open_existing_table_for_read()
        if table is None:
            return self._empty_result(select_fields)
        if limit is not None:
            row_limit = limit
        else:
            row_limit = table.count_rows(where) if where else table.count_rows()
        if row_limit == 0:
            return self._empty_result(select_fields)
        result = table.search_all(where=where, select_fields=select_fields, limit=row_limit)
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
            model = self.model.model
            if model is None:
                raise DatusException(
                    ErrorCode.MODEL_EMBEDDING_ERROR,
                    message=f"Embedding model '{self.model.model_name}' initialization produced no model",
                )
        except DatusException as e:
            # Re-raise DatusException directly to avoid nesting
            raise e
        except Exception as e:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=f"Embedding model '{self.model.model_name}' initialization failed: {str(e)}",
            ) from e

    def _ensure_embedding_cache_ready_for_search(self):
        """Avoid on-demand downloads for read-time vector search."""
        try:
            has_snapshot = self.model.has_local_fastembed_snapshot()
        except DatusException:
            raise
        except Exception as e:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=f"Embedding cache readiness check failed for '{self.table_name}': {e}",
            ) from e

        if not has_snapshot:
            raise DatusException(
                ErrorCode.MODEL_EMBEDDING_ERROR,
                message=f"Embedding model cache is missing for vector search on '{self.table_name}'",
            )

    def truncate(self) -> None:
        """Drop the entire table and reset state (admin operation)."""
        with self._table_lock:
            self.db.drop_table(self.table_name, ignore_missing=True)
            self._shared.table = None
            self._shared.initialized = False

    def truncate_scoped(self) -> None:
        """Legacy full-table truncate alias.

        Datasource-scoped overwrite paths must call ``delete_datasource_rows``
        instead of this method.
        """
        self.truncate()

    def delete_datasource_rows(self, datasource_id: str) -> None:
        """Delete rows for one datasource without dropping the project table."""

        if not self._datasource_scoped:
            raise DatusException(
                ErrorCode.STORAGE_INVALID_ARGUMENT,
                message_args={
                    "error_message": f"table '{self.table_name}' is not datasource-scoped",
                },
            )
        self._delete_rows(datasource_condition(datasource_id))

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
        self._ensure_persisted_scope_columns()

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

    def create_fts_index(self, fields: Union[str, List[str], FtsSpec]):
        """Create and verify one native FTS index per configured field."""
        self._ensure_table_ready()
        if not self._supports_runtime_indexing():
            return
        if isinstance(fields, FtsSpec):
            spec = fields
        elif isinstance(fields, str):
            spec = FtsSpec((FtsField(fields),))
        else:
            spec = FtsSpec.from_names(fields)

        remove_legacy = getattr(self.table, "remove_legacy_fts_index", None)
        if remove_legacy is not None and remove_legacy():
            logger.info("Removed legacy Tantivy FTS index for %s before rebuilding", self.table_name)

        try:
            for field in spec.fields:
                self.table.create_fts_index(field)
            status_fn = getattr(self.table, "fts_index_status", None)
            if status_fn is not None:
                status = status_fn(spec)
                if status != FtsIndexStatus.READY:
                    raise RuntimeError(f"FTS index verification returned {status}")
        except Exception as exc:
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message_args={
                    "operation": "create_fts_index",
                    "table_name": self.table_name,
                    "error_message": str(exc),
                },
            ) from exc

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
        allow_hybrid_fallback: bool = True,
    ) -> pa.Table:
        table = self._open_existing_table_for_read()
        if table is None:
            return self._empty_result(select_fields)
        row_count = table.count_rows(where) if where else table.count_rows()
        if row_count == 0:
            return self._empty_result(select_fields)

        self._ensure_embedding_cache_ready_for_search()
        self._ensure_table_ready()

        if query_type == "hybrid":
            search_result = self._search_hybrid(
                query_txt, select_fields, top_n, where, allow_fallback=allow_hybrid_fallback
            )
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
        allow_fallback: bool = True,
    ) -> pa.Table:
        try:
            if not top_n:
                top_n = self.table.count_rows(where) if where else self.table.count_rows()
            results = self.table.search_hybrid(
                query_txt,
                self.vector_column_name,
                top_n,
                where=where,
                select_fields=select_fields,
            )
            if len(results) > top_n:
                results = results[:top_n]
            return results
        except Exception as e:
            if allow_fallback:
                logger.warning("Hybrid search failed for %s; falling back to vector search: %s", self.table_name, e)
                return self._search_vector(query_txt, select_fields, top_n, where)
            raise DatusException(
                ErrorCode.STORAGE_SEARCH_FAILED,
                message_args={
                    "error_message": str(e),
                    "query": query_txt,
                    "where_clause": str(where) if where else "(none)",
                    "top_n": str(top_n or "all"),
                },
            ) from e

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
        return self._count_rows()

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
        table = self._open_existing_table_for_read()
        if table is None:
            return 0
        return table.count_rows(where)

    def query_with_filter(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        """Query rows with filter, field selection, and optional limit."""
        table = self._open_existing_table_for_read()
        if table is None:
            return self._empty_result(select_fields)
        if limit is None:
            limit = table.count_rows(where)
        if limit == 0:
            return self._empty_result(select_fields)
        return table.search_all(
            where=where,
            select_fields=select_fields,
            limit=limit,
        )
