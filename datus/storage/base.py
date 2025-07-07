from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import lancedb
import pandas as pd
import pyarrow as pa
from lancedb.embeddings import EmbeddingFunctionConfig
from lancedb.pydantic import LanceModel
from lancedb.query import LanceQueryBuilder
from lancedb.rerankers import Reranker
from pydantic import Field

from datus.storage.embedding_models import EmbeddingModel
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger("sql_agent")


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
        self._ensure_document_table()
        self._ensure_metrics_table()
        self._ensure_success_story_table()

    def _ensure_document_table(self):
        """Ensure the document table exists in LanceDB."""
        try:
            if "document" not in self.db.table_names():
                # Create table schema using PyArrow
                schema = pa.schema(
                    [
                        pa.field("title", pa.string()),
                        pa.field("hierarchy", pa.string()),  # Hierarchical structure
                        pa.field("keywords", pa.list_(pa.string())),
                        pa.field("language", pa.string()),
                        pa.field("chunk_text", pa.string()),
                        pa.field("created_at", pa.string()),
                        pa.field("embedding", pa.list_(pa.float64(), list_size=384)),
                    ]
                )
                self.db.create_table("document", schema=schema)
        except Exception as e:
            raise Exception(f"Failed to create document table: {str(e)}")

    def _ensure_metrics_table(self):
        """Ensure the metrics table exists in LanceDB."""
        try:
            if "metrics" not in self.db.table_names():
                # Create table schema using PyArrow
                schema = pa.schema(
                    [
                        pa.field("metric_name", pa.string()),
                        pa.field("metric_sql", pa.string()),
                        pa.field("dimensions", pa.list_(pa.string())),
                        pa.field("description", pa.string()),
                        pa.field("created_at", pa.string()),
                        pa.field("embedding", pa.list_(pa.float64(), list_size=384)),
                    ]
                )
                self.db.create_table("metrics", schema=schema)
        except Exception as e:
            raise Exception(f"Failed to create metrics table: {str(e)}")

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
            raise Exception(f"Failed to create success_story table: {str(e)}")

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
        self._ensure_table(schema)

    def _ensure_table(self, schema: Optional[Union[pa.Schema, LanceModel]] = None):
        try:
            self.table = self.db.create_table(
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
                ErrorCode.TOOL_STORE_FAILED,
                message=f"Failed to create LanceDB table named {self.table_name} because {str(e)}",
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
        try:
            self.table.create_fts_index(field_names=field_names, replace=True)
        except Exception as e:
            # Does not affect usage, so no exception is thrown.
            logger.warning(f"Failed to create fts index for {self.table_name} table: {str(e)}")

    def store_batch(self, data: List[Dict[str, Any]], mode: str = "append"):
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
        try:
            if len(data) <= self.batch_size:
                self.table.add(pd.DataFrame(data), mode=mode)
                return
            # split the data into batches and store them
            for i in range(0, len(data), self.batch_size):
                batch = data[i : i + self.batch_size]
                self.table.add(pd.DataFrame(batch), mode=mode)
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED,
                message=f"Failed to store batch because {str(e)}",
            ) from e

    def store(self, data: List[Dict[str, Any]], mode: str = "append"):
        self.table.add(pd.DataFrame(data), mode=mode)

    def search(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: int = 5,
        where: str = "",
        reranker: Optional[Reranker] = None,
    ) -> List[dict]:
        if reranker:
            return self._search_hybird(query_txt, reranker, select_fields, top_n, where)
        else:
            return self._search_vector(query_txt, select_fields, top_n, where)

    def _search_hybird(
        self,
        query_txt: str,
        reranker: Reranker,
        select_fields: Optional[List[str]] = None,
        top_n: int = 5,
        where: str = "",
    ) -> List[dict]:
        try:
            query_builder = self.table.search(
                query=query_txt, query_type="hybrid", vector_column_name=self.vector_source_name
            )
            query_builder = BaseEmbeddingStore._fill_query(query_builder, select_fields, where)

            results = query_builder.limit(1000).rerank(reranker).to_arrow()
            results = results.to_pylist()
            if len(results) > top_n:
                results = results[:top_n]
            return results
        except Exception as e:
            logger.warning(f"Failed to search hybird: {str(e)}, use vector search instead")
            return self._search_vector(query_txt, select_fields, top_n, where)

    def _search_vector(
        self,
        query_txt: str,
        select_fields: Optional[List[str]] = None,
        top_n: int = 5,
        where: str = "",
    ) -> List[dict]:
        query_builder = self.table.search(
            query=query_txt, query_type="vector", vector_column_name=self.vector_column_name
        )
        query_builder = BaseEmbeddingStore._fill_query(query_builder, select_fields, where)
        return query_builder.limit(top_n).to_list()

    def table_size(self) -> int:
        return self.table.count_rows()

    @classmethod
    def _fill_query(
        cls,
        query_builder: LanceQueryBuilder,
        select_fields: Optional[List[str]] = None,
        where: str = "",
    ) -> LanceQueryBuilder:
        if len(where) > 0:
            query_builder = query_builder.where(where, True)

        if select_fields and len(select_fields) > 0:
            query_builder = query_builder.select(select_fields)
        return query_builder
