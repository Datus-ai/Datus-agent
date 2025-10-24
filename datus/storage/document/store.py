# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from functools import lru_cache
from typing import List, Optional

import pandas as pd
import pyarrow as pa

from datus.storage import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel, get_document_embedding_model
from datus.utils.exceptions import DatusException, ErrorCode


class DocumentStore(BaseEmbeddingStore):
    """Store and manage document data in LanceDB."""

    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """Initialize the document store.

        Args:
            db_path: Path to the LanceDB database directory
        """
        super().__init__(
            db_path=db_path,
            table_name="document",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("title", pa.string()),
                    pa.field("hierarchy", pa.string()),
                    pa.field("keywords", pa.list_(pa.string())),
                    pa.field("language", pa.string()),
                    pa.field("chunk_text", pa.string()),
                    pa.field(
                        "vector",
                        pa.list_(pa.float64(), list_size=embedding_model.dim_size),
                    ),
                ]
            ),
            vector_column_name="vector",
            vector_source_name="chunk_text",
        )

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()
        self.create_fts_index(["chunk_text"])

    def store_document(
        self,
        title: str,
        hierarchy: str,
        keywords: List[str],
        language: str,
        chunk_text: str,
    ):
        """Store a document chunk in the database.

        Args:
            title: Document title
            hierarchy: Hierarchical structure path
            keywords: List of relevant keywords
            language: Document language
            chunk_text: The document chunk text
            embedding: Vector embedding of the chunk text

        Raises:
            Exception: If document storage fails
        """
        # Ensure table is ready before storing
        self._ensure_table_ready()
        try:
            # Add new document record
            self.table.add(
                pd.DataFrame(
                    [
                        {
                            "title": title,
                            "hierarchy": hierarchy,
                            "keywords": keywords,
                            "language": language,
                            "chunk_text": chunk_text,
                        }
                    ]
                )
            )
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def search_similar_documents(
        self, query_text: str, select_fields: Optional[List[str]] = None, top_n: int = 5
    ) -> pa.Table:
        """Search for similar documents using vector similarity.

        Args:
            query_text: Vector embedding of the query text
            select_fields: List of fields to select
            top_n: Number of similar documents to return

        Returns:
            List of similar documents with their metadata
        """
        # Ensure table is ready before searching
        self._ensure_table_ready()
        try:
            return self._search_vector(query_text, top_n=top_n, select_fields=select_fields)
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_SEARCH_FAILED,
                message_args={
                    "error_message": str(e),
                    "query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                    "where_clause": "(none)",
                    "top_n": str(top_n),
                },
            ) from e


@lru_cache(maxsize=1)
def document_store(storage_path: str) -> DocumentStore:
    # FIXME Adapt sub-agent
    return DocumentStore(storage_path, get_document_embedding_model())
