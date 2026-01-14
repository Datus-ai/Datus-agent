# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import time
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import lancedb
import pandas as pd
import pyarrow as pa
from lancedb.embeddings import EmbeddingFunctionConfig
from lancedb.pydantic import LanceModel
from lancedb.query import LanceQueryBuilder
from lancedb.rerankers import Reranker
from lancedb.table import Table
from pydantic import Field

from datus.storage.embedding_models import EmbeddingModel
from datus.storage.lancedb_conditions import WhereExpr, build_where
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class StorageBase:
    """Base class for all storage components using LanceDB."""

    def __init__(self, db_path: str):
        """Initialize the storage base.

        Args:
            db_path: Path to the LanceDB database directory
        """
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        # self._ensure_tables()

    def _ensure_tables(self):
        """Ensure all required tables exist in LanceDB."""
        self._ensure_success_story_table()

    def _ensure_success_story_table(self):
        """Ensure the success story table exists in LanceDB."""
        try:
            if "success_story" not in self.db.table_names():
                # Create table schema using PyArrow
                schema = pa.schema(
                    [
                        pa.field("sql", pa.string()),
                        pa.field("user_name", pa.string()),
                        pa.field("type", pa.string()),
                        pa.field("bi_tool", pa.string()),
                        pa.field("description", pa.string()),
                        pa.field("created_at", pa.string()),
                        pa.field("embedding", pa.list_(pa.float64(), list_size=384)),
                    ]
                )
                self.db.create_table("success_story", schema=schema)
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message_args={"operation": "create_table", "table_name": "success_story", "error_message": str(e)},
            ) from e

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()


class BaseModelData(LanceModel):
    created_at: str = Field(init=True, default="")

    class Config:
        arbitrary_types_allowed = True


class BaseEmbeddingStore(StorageBase):
    """Base class for all embedding stores using LanceDB.
    table_name: the name of the table to store the embedding
    embedding_field: the field name of the embedding
    """

    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedding_model: EmbeddingModel,
        on_duplicate_columns: str = "vector",
        schema: Optional[Union[pa.Schema, LanceModel]] = None,
        vector_source_name: str = "definition",
        vector_column_name: str = "vector",
    ):
        super().__init__(db_path)
        self.model = embedding_model
        self.batch_size = embedding_model.batch_size
        self.table_name = table_name
        self.vector_source_name = vector_source_name
        self.vector_column_name = vector_column_name
        self.on_duplicate_columns = on_duplicate_columns
        self._schema = schema
        # Delay table initialization until first use
        self.table: Optional[Table] = None
        self._table_initialized = False
        self._table_lock = Lock()
        self._write_lock = Lock()

    def _ensure_table_ready(self):
        """Ensure table is ready for operations, with proper error handling."""
        if self._table_initialized:
            return

        with self._table_lock:
            if self._table_initialized:
                return

            # First check if embedding model is available
            self._check_embedding_model_ready()
            # Initialize table with embedding function
            self._ensure_table(self._schema)
            self._table_initialized = True
            logger.debug(f"Table {self.table_name} initialized successfully with embedding function")

    def _search_all(
        self, where: WhereExpr = None, select_fields: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> pa.Table:
        self._ensure_table_ready()
        where_clause = build_where(where)
        query_builder = self.table.search()
        if where_clause:
            query_builder = query_builder.where(where_clause)
        if select_fields:
            query_builder = query_builder.select(select_fields)
        if limit:
            row_limit = limit
        else:
            row_limit = self.table.count_rows(where_clause) if where_clause else self.table.count_rows()
        result = query_builder.limit(row_limit).to_arrow()
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
                    f"Embedding model '{self.model.model_name}' is not available:" f" {self.model.model_error_message}"
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

    def _ensure_table(self, schema: Optional[Union[pa.Schema, LanceModel]] = None):
        if self.table_name in self.db.table_names(limit=100):
            self.table = self.db.open_table(self.table_name)
        else:
            try:
                self.table: Table = self.db.create_table(
                    self.table_name,
                    schema=schema,
                    embedding_functions=[
                        EmbeddingFunctionConfig(
                            vector_column=self.vector_column_name,
                            source_column=self.vector_source_name,
                            function=self.model.model,
                        )
                    ],
                    exist_ok=True,
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
            accelerator (str): Optional accelerator ('cuda' for GPU, 'mps' for MPS, None for CPU).
                Default: none.
        """
        self._ensure_table_ready()
        try:
            row_count = self.table.count_rows()
            logger.debug(f"Creating vector index for {self.table_name} with {row_count} rows")

            # Determine index type based on dataset size
            index_type = "IVF_PQ" if row_count >= 5000 else "IVF_FLAT"
            logger.debug(f"Selected index type: {index_type}")

            # Calculate number of partitions (IVF)
            # Rule: ~sqrt(n) for large datasets, minimum 1, capped at 1024
            num_partitions = max(1, min(1024, int(row_count**0.5)))
            if row_count < 1000:
                num_partitions = max(1, row_count // 10)  # Small datasets: 10 vectors per partition
            elif row_count < 5000:
                num_partitions = max(1, row_count // 20)  # Medium datasets: 20 vectors per partition
            logger.debug(f"Number of partitions: {num_partitions}")

            # Calculate number of sub-vectors (PQ, only for IVF_PQ)
            # Rule: 8-96, based on vector dimension and dataset size
            num_sub_vectors = 32  # Default for medium datasets
            if index_type == "IVF_PQ":
                # Get vector dimension (e.g., 1024 for bge-large-en-v1.5)
                vector_dim = self.model.dim_size

                if row_count < 1000:
                    num_sub_vectors = min(16, max(8, vector_dim // 64))  # Small datasets: fewer sub-vectors
                elif row_count < 5000:
                    num_sub_vectors = min(32, max(16, vector_dim // 32))  # Medium datasets
                else:
                    num_sub_vectors = min(96, max(32, vector_dim // 16))  # Large datasets: more sub-vectors
                logger.debug(f"Number of sub-vectors: {num_sub_vectors}")

            # Create index with calculated parameters
            index_params = {
                "metric": metric,
                "vector_column_name": self.vector_column_name,
                "index_type": index_type,
                "num_partitions": num_partitions,
                "replace": True,  # Replace existing index if any
            }
            if index_type == "IVF_PQ":
                index_params["num_sub_vectors"] = num_sub_vectors
            accelerator = self.model.device
            if accelerator and accelerator == "cuda" or accelerator == "mps":
                index_params["accelerator"] = accelerator

            self.table.create_index(**index_params)
            logger.debug(f"Successfully created {index_type} index for {self.table_name}")

        except Exception as e:
            # Does not affect usage, so no exception is thrown.
            logger.warning(f"Failed to create vector index for {self.table_name}: {str(e)}")

    def create_fts_index(self, field_names: Union[str, List[str]]):
        self._ensure_table_ready()
        try:
            self.table.create_fts_index(field_names=field_names, replace=True)
        except Exception as e:
            # Does not affect usage, so no exception is thrown.
            logger.warning(f"Failed to create fts index for {self.table_name} table: {str(e)}")

    def store_batch(self, data: List[Dict[str, Any]]):
        """
        Store a batch of data in the database. The following steps are performed:

            1. Encode the vector field
            2. Merge insert the data into the table

        Args:
            data: List[BaseModelData] the data to store
            on_columns: List[str] the columns to merge on duplicate
        """
        if not data:
            return
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
        self._ensure_table_ready()

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
        """Upsert a DataFrame into LanceDB with simple retry/backoff on commit conflicts."""
        if self.table is None:
            raise DatusException(
                ErrorCode.STORAGE_SAVE_FAILED,
                message_args={"error_message": "Lance table is not initialized"},
            )

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self.table.merge_insert(on_column).when_matched_update_all().when_not_matched_insert_all().execute(
                    frame
                )
                return
            except Exception as err:
                error_message = str(err)
                if "Commit conflict" not in error_message:
                    raise err

                last_error = err
                delay = initial_delay * (attempt + 1)
                logger.warning(
                    f"Commit conflict detected when upserting to LanceDB table '{self.table_name}' "
                    f"(attempt {attempt + 1}/{max_attempts}). Retrying after {delay:.2f}s."
                )
                # Refresh table handle so subsequent attempts see the latest version
                self.table = self.db.open_table(self.table_name)
                time.sleep(delay)

        assert last_error is not None  # for type checkers
        raise last_error

    def _add_with_retry(self, frame: pd.DataFrame, max_attempts: int = 3, initial_delay: float = 0.05) -> None:
        """Insert a DataFrame into LanceDB with simple retry/backoff on commit conflicts."""
        if self.table is None:
            raise DatusException(
                ErrorCode.STORAGE_SAVE_FAILED,
                message_args={"error_message": "Lance table is not initialized"},
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
                    f"Commit conflict detected when writing to LanceDB table '{self.table_name}' "
                    f"(attempt {attempt + 1}/{max_attempts}). Retrying after {delay:.2f}s."
                )
                # Refresh table handle so subsequent attempts see the latest version
                self.table = self.db.open_table(self.table_name)
                time.sleep(delay)

        assert last_error is not None  # for type checkers
        raise last_error

    def search(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        # Ensure table is ready before searching
        self._ensure_table_ready()

        if reranker:
            search_result = self._search_hybrid(query_txt, reranker, select_fields, top_n, where)
        else:
            search_result = self._search_vector(query_txt, select_fields, top_n, where)
        if self.vector_column_name in search_result.column_names:
            search_result = search_result.drop([self.vector_column_name])
        return search_result

    def _search_hybrid(
        self,
        query_txt: str,
        reranker: Reranker,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        where: WhereExpr = None,
    ) -> pa.Table:
        where_clause = build_where(where)
        try:
            query_builder = self.table.search(
                query=query_txt, query_type="hybrid", vector_column_name=self.vector_source_name
            )
            query_builder = BaseEmbeddingStore._fill_query(query_builder, select_fields, where_clause)
            if not top_n:
                top_n = self.table.count_rows(where_clause) if where_clause else self.table.count_rows()
            results = query_builder.limit(top_n * 2).rerank(reranker).to_arrow()
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
        where_clause = build_where(where)
        try:
            query_builder = self.table.search(
                query=query_txt, query_type="vector", vector_column_name=self.vector_column_name
            )
            query_builder = BaseEmbeddingStore._fill_query(query_builder, select_fields, where_clause)
            if not top_n:
                top_n = self.table.count_rows(where_clause) if where_clause else self.table.count_rows()
            return query_builder.limit(top_n).to_arrow()
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_SEARCH_FAILED,
                message_args={
                    "error_message": str(e),
                    "query": query_txt,
                    "where_clause": where_clause if where_clause else "(none)",
                    "top_n": str(top_n or "all"),
                },
            ) from e

    def table_size(self) -> int:
        # Ensure table is ready before checking size
        self._ensure_table_ready()
        return self.table.count_rows()

    @classmethod
    def _fill_query(
        cls,
        query_builder: LanceQueryBuilder,
        select_fields: Optional[List[str]] = None,
        where: Optional[str] = None,
    ) -> LanceQueryBuilder:
        if where:
            query_builder = query_builder.where(where, True)

        if select_fields and len(select_fields) > 0:
            query_builder = query_builder.select(select_fields)
        return query_builder

    def update(self, where: WhereExpr, update_values: Dict[str, Any], unique_filter: Optional[WhereExpr] = None):
        self._ensure_table_ready()
        if not update_values:
            return
        final_where = build_where(where)
        if not final_where:
            return
        unique_where = build_where(unique_filter)
        if unique_where:
            existing = self.table.count_rows(unique_where)
            if existing:
                raise DatusException(
                    ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                    message_args={
                        "operation": "update",
                        "table_name": self.table_name,
                        "error_message": f"Conflicting rows already match {unique_where}",
                    },
                )
        self.table.update(where=final_where, values=update_values)
