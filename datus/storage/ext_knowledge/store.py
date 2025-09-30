import logging
from typing import Dict, List, Optional

import pyarrow as pa

from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel, get_document_embedding_model
from datus.storage.lancedb_conditions import and_, build_where, eq

logger = logging.getLogger(__name__)


class ExtKnowledgeStore(BaseEmbeddingStore):
    """Store and manage external business knowledge in LanceDB."""

    def __init__(self, db_path: str, embedding_model: Optional[EmbeddingModel] = None):
        """Initialize the external knowledge store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model to use, defaults to document embedding model
        """
        if embedding_model is None:
            embedding_model = get_document_embedding_model()

        super().__init__(
            db_path=db_path,
            table_name="ext_knowledge",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("domain", pa.string()),
                    pa.field("layer1", pa.string()),
                    pa.field("layer2", pa.string()),
                    pa.field("terminology", pa.string()),
                    pa.field("explanation", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="explanation",
        )

    def create_indices(self):
        """Create scalar and FTS indices for better search performance."""
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        try:
            self.table.create_scalar_index("domain", replace=True)
            self.table.create_scalar_index("layer1", replace=True)
            self.table.create_scalar_index("layer2", replace=True)
        except Exception as e:
            logger.warning(f"Failed to create scalar index for {self.table_name} table: {str(e)}")

        self.create_fts_index(["domain", "layer1", "layer2", "terminology", "explanation"])

    def store_knowledge(
        self,
        domain: str,
        layer1: str,
        layer2: str,
        terminology: str,
        explanation: str,
    ):
        """Store a single knowledge entry.

        Args:
            domain: Business domain
            layer1: First layer categorization
            layer2: Second layer categorization
            terminology: Business terminology/concept
            explanation: Detailed explanation
        """
        data = [
            {
                "domain": domain,
                "layer1": layer1,
                "layer2": layer2,
                "terminology": terminology,
                "explanation": explanation,
                "created_at": self._get_current_timestamp(),
            }
        ]
        self.store_batch(data)

    def search_knowledge(
        self,
        query_text: str,
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
        top_n: int = 5,
    ) -> pa.Table:
        """Search for similar knowledge entries.

        Args:
            query_text: Query text to search for
            domain: Filter by domain (optional)
            layer1: Filter by layer1 (optional)
            layer2: Filter by layer2 (optional)
            top_n: Number of results to return

        Returns:
            List of matching knowledge entries
        """
        conditions = []
        if domain:
            conditions.append(eq("domain", domain))
        if layer1:
            conditions.append(eq("layer1", layer1))
        if layer2:
            conditions.append(eq("layer2", layer2))

        if not conditions:
            where_condition = None
        elif len(conditions) == 1:
            where_condition = conditions[0]
        else:
            where_condition = and_(*conditions)

        results = self.search(
            query_txt=query_text,
            select_fields=["domain", "layer1", "layer2", "terminology", "explanation", "created_at"],
            top_n=top_n,
            where=where_condition,
        )

        return results

    def search_all_knowledge(
        self,
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
    ) -> pa.Table:
        """Get all knowledge entries with optional filtering.

        Args:
            domain: Filter by domain (optional)
            layer1: Filter by layer1 (optional)
            layer2: Filter by layer2 (optional)

        Returns:
            List of all matching knowledge entries
        """
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        conditions = []
        if domain:
            conditions.append(eq("domain", domain))
        if layer1:
            conditions.append(eq("layer1", layer1))
        if layer2:
            conditions.append(eq("layer2", layer2))

        if not conditions:
            where_condition = None
        elif len(conditions) == 1:
            where_condition = conditions[0]
        else:
            where_condition = and_(*conditions)

        return self._search_all(
            where=where_condition,
            select_fields=["domain", "layer1", "layer2", "terminology", "explanation", "created_at"],
        )

    def get_domains(self) -> List[str]:
        """Get all unique domains."""
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        search_result = self.table.search().select(["domain"]).limit(100000).to_list()
        return list(set(result["domain"] for result in search_result))

    def get_layers_by_domain(self, domain: str) -> List[Dict[str, str]]:
        """Get all layer1/layer2 combinations for a domain."""
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        where_clause = build_where(eq("domain", domain))
        search_result = self.table.search().where(where_clause).select(["layer1", "layer2"]).limit(100000).to_list()

        unique_layers = set()
        for result in search_result:
            unique_layers.add((result["layer1"], result["layer2"]))

        return [{"layer1": layer1, "layer2": layer2} for layer1, layer2 in unique_layers]

    def after_init(self):
        """After initialization, create indices for the table."""
        self.create_indices()


def rag_by_configuration(agent_config) -> ExtKnowledgeStore:
    """Create ExtKnowledgeStore from agent configuration."""
    return ExtKnowledgeStore(agent_config.rag_storage_path())
