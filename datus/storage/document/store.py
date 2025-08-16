from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa

from datus.storage import BaseEmbeddingStore
from datus.storage.embedding_models import get_document_embedding_model
from datus.utils.exceptions import DatusException, ErrorCode


class DocumentStore(BaseEmbeddingStore):
    """Store and manage document data in LanceDB."""

    def __init__(self, db_path: str):
        """Initialize the document store.

        Args:
            db_path: Path to the LanceDB database directory
        """
        embedding_model = get_document_embedding_model()
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
                        "embedding",
                        pa.list_(pa.float64(), list_size=embedding_model.dim_size),
                    ),
                ]
            ),
            vector_column_name="vector",
            vector_source_name="chunk_text",
        )
        self.create_vector_index()
        self.create_fts_index("chunk_text")

    def store_document(
        self,
        title: str,
        hierarchy: str,
        keywords: List[str],
        language: str,
        chunk_text: str,
        embedding: List[float],
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
                            "embedding": embedding,
                        }
                    ]
                )
            )
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_SAVE_FAILED, message_args={"error_message": str(e)}) from e

    def search_similar_documents(self, query_embedding: List[float], top_n: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity.

        Args:
            query_embedding: Vector embedding of the query text
            top_n: Number of similar documents to return

        Returns:
            List of similar documents with their metadata
        """
        # Ensure table is ready before searching
        self._ensure_table_ready()
        try:
            results = (
                self.table.search(query_embedding)
                .limit(top_n)
                .select(["title", "hierarchy", "keywords", "language", "chunk_text", "created_at"])
                .to_list()
            )
            return results
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_SEARCH_FAILED,
                message_args={
                    "error_message": str(e),
                    "query": str(query_embedding)[:50] + "..."
                    if len(str(query_embedding)) > 50
                    else str(query_embedding),
                    "where_clause": "(none)",
                    "top_n": str(top_n),
                },
            ) from e
