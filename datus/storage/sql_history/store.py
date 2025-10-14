import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.embedding_models import get_metric_embedding_model
from datus.storage.lancedb_conditions import And, build_where, eq, like
from datus.utils.exceptions import DatusException, ErrorCode

logger = logging.getLogger(__file__)


class SqlHistoryStorage(BaseEmbeddingStore):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """Initialize the SQL history store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model for vector search
        """
        super().__init__(
            db_path=db_path,
            table_name="sql_history",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("name", pa.string()),
                    pa.field("sql", pa.string()),
                    pa.field("comment", pa.string()),
                    pa.field("summary", pa.string()),
                    pa.field("filepath", pa.string()),
                    pa.field("domain", pa.string()),
                    pa.field("layer1", pa.string()),
                    pa.field("layer2", pa.string()),
                    pa.field("tags", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="summary",
        )
        self.reranker = None

    def create_indices(self):
        """Create scalar and full-text search indices."""
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        # Create scalar indices
        self.table.create_scalar_index("id", replace=True)
        self.table.create_scalar_index("name", replace=True)
        self.table.create_scalar_index("domain", replace=True)
        self.table.create_scalar_index("layer1", replace=True)
        self.table.create_scalar_index("layer2", replace=True)
        self.table.create_scalar_index("filepath", replace=True)

        # Create full-text search index
        self.create_fts_index(["sql", "name", "comment", "summary", "tags"])

    def search_all(self, domain: str = "", selected_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search all SQL history entries for a given domain."""

        if not selected_fields:
            selected_fields = [
                "id",
                "name",
                "sql",
                "comment",
                "summary",
                "filepath",
                "domain",
                "layer1",
                "layer2",
                "tags",
            ]
        search_result = self._search_all(
            where=None if not domain else eq("domain", domain),
            select_fields=selected_fields,
        )
        result = []
        for i in range(search_result.num_rows):
            item_data = {}
            for field in selected_fields:
                item_data[field] = search_result[field][i].as_py()
            result.append(item_data)
        return result

    def filter_by_id(self, id: str) -> List[Dict[str, Any]]:
        """Filter SQL history by ID."""
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        where_clause = build_where(eq("id", id))
        search_result = self.table.search().where(where_clause).limit(100).to_list()
        return search_result

    def filter_by_domain_layers(self, domain: str = "", layer1: str = "", layer2: str = "") -> List[Dict[str, Any]]:
        """Filter SQL history by domain and layers."""
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
            search_result = self.table.search().limit(1000).to_list()
        else:
            where_condition = conditions[0] if len(conditions) == 1 else And(conditions)
            where_clause = build_where(where_condition)
            search_result = self.table.search().where(where_clause).limit(1000).to_list()

        return search_result

    def get_existing_taxonomy(self) -> Dict[str, Any]:
        """Get existing taxonomy from stored SQL history items.

        Returns:
            Dict containing existing domains, layer1_categories, layer2_categories, and common_tags
        """
        logger.info("Extracting existing taxonomy from stored SQL history")

        # Ensure table is ready
        self._ensure_table_ready()

        # Get all existing taxonomy data
        search_result = self._search_all(select_fields=["domain", "layer1", "layer2", "tags"])

        if not search_result or search_result.num_rows <= 0:
            logger.info("No existing taxonomy found in database")
            return {"domains": [], "layer1_categories": [], "layer2_categories": [], "common_tags": []}

        # Extract unique values
        layer1_categories = set()
        layer2_categories = set()
        tags = set()
        domain_column = search_result["domain"]
        layer1_column = search_result["layer1"]
        layer2_column = search_result["layer2"]
        tags_column = search_result["tags"]
        domains = set(domain_column.unique().to_pylist())
        for i in range(search_result.num_rows):
            layer1 = layer1_column[i].as_py()
            if layer1:
                layer1_categories.add((layer1, domain_column[i].as_py() or ""))

            if layer2 := layer2_column[i].as_py():
                layer2_categories.add((layer2, layer1 or ""))
            if row_tags := tags_column[i].as_py():
                # Split tags by comma if they are stored as comma-separated string
                item_tags = [tag.strip() for tag in str(row_tags).split(",") if tag.strip()]
                tags.update(item_tags)

        # Format into taxonomy structure
        taxonomy = {
            "domains": [{"name": domain, "description": "Existing business domain"} for domain in sorted(domains)],
            "layer1_categories": [
                {"name": layer1, "domain": domain, "description": "Existing primary category"}
                for layer1, domain in sorted(layer1_categories)
            ],
            "layer2_categories": [
                {"name": layer2, "layer1": layer1, "description": "Existing secondary category"}
                for layer2, layer1 in sorted(layer2_categories)
            ],
            "common_tags": [{"tag": tag, "description": "Existing tag"} for tag in sorted(tags)],
        }

        logger.info(
            f"Extracted existing taxonomy: {len(taxonomy['domains'])} domains, "
            f"{len(taxonomy['layer1_categories'])} layer1 categories, "
            f"{len(taxonomy['layer2_categories'])} layer2 categories, "
            f"{len(taxonomy['common_tags'])} tags"
        )

        return taxonomy

    def search_by_filepath(self, filepath_pattern: str) -> List[Dict[str, Any]]:
        """Search SQL history by filepath pattern."""
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        where_clause = build_where(like("filepath", f"%{filepath_pattern}%"))
        search_result = self.table.search().where(where_clause).limit(1000).to_list()
        return search_result


class SqlHistoryRAG:
    def __init__(self, db_path: str):
        self.db_path = db_path
        embedding_model = get_metric_embedding_model()
        self.sql_history_storage = SqlHistoryStorage(db_path, embedding_model)

    def store_batch(self, sql_history_items: List[Dict[str, Any]]):
        """Store batch of SQL history items."""
        logger.info(f"store sql history items: {len(sql_history_items)} items")
        self.sql_history_storage.store_batch(sql_history_items)

    def search_all_sql_history(
        self, domain: str = "", selected_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search all SQL history items."""
        return self.sql_history_storage.search_all(domain, selected_fields)

    def after_init(self):
        """Initialize indices after data loading."""
        self.sql_history_storage.create_indices()

    def get_sql_history_size(self):
        """Get total number of SQL history entries."""
        return self.sql_history_storage.table_size()

    def get_domains(self) -> List[str]:
        """Get all unique domains from sql_history.

        Returns:
            List of unique domain names
        """
        self.sql_history_storage._ensure_table_ready()

        MAX_DOMAINS = 1000
        search_result = self.sql_history_storage.table.search().select(["domain"]).limit(MAX_DOMAINS).to_list()

        if len(search_result) >= MAX_DOMAINS:
            logger.warning(
                f"Retrieved {MAX_DOMAINS} domain records (may be truncated). "
                "Consider using filters to reduce result set."
            )

        domains = list(set(result["domain"] for result in search_result if result.get("domain")))
        logger.debug(f"Found {len(domains)} unique domains from {len(search_result)} records")

        return sorted(domains)

    def get_layers_by_domain(self, domain: str = "") -> List[tuple]:
        """Get unique (layer1, layer2) combinations from sql_history.

        Args:
            domain: Domain name to filter. Leave empty to get all layers.

        Returns:
            List of (layer1, layer2) tuples. Maximum 1000 combinations.
        """
        MAX_LAYERS = 1000

        self.sql_history_storage._ensure_table_ready()

        where_clause = build_where(eq("domain", domain)) if domain else None
        search_result = self.sql_history_storage.table.search()
        if where_clause:
            search_result = search_result.where(where_clause)

        results = search_result.select(["layer1", "layer2"]).limit(MAX_LAYERS).to_list()

        if len(results) >= MAX_LAYERS:
            logger.warning(
                f"Retrieved {MAX_LAYERS} layer records (may be truncated). "
                f"Consider filtering by domain to reduce result set."
            )

        unique_layers = set((r["layer1"], r["layer2"]) for r in results if r.get("layer1") and r.get("layer2"))
        logger.debug(f"Found {len(unique_layers)} unique layers from {len(results)} records")

        return list(unique_layers)

    def get_sql_history_names(self, domain: str, layer1: str, layer2: str) -> List[Dict[str, str]]:
        """Get sql_history names with summaries in a specific layer.

        Args:
            domain: Domain name
            layer1: Primary layer name
            layer2: Secondary layer name

        Returns:
            List of dicts with 'name' and 'summary' fields
        """
        MAX_SQL_HISTORY = 1000

        conditions = [eq("domain", domain), eq("layer1", layer1), eq("layer2", layer2)]
        query_result = self.sql_history_storage._search_all(
            And(conditions),
            select_fields=["name", "summary"],
        )

        if query_result is None or query_result.num_rows == 0:
            logger.debug(f"No sql_history found for domain={domain}, layer1={layer1}, layer2={layer2}")
            return []

        if query_result.num_rows >= MAX_SQL_HISTORY:
            logger.warning(
                f"Retrieved {query_result.num_rows} sql_history items (may be truncated at {MAX_SQL_HISTORY}). "
                "Result set is very large."
            )

        result_list = query_result.to_pylist()
        logger.debug(f"Found {len(result_list)} sql_history items in {domain}/{layer1}/{layer2}")

        return result_list

    def search_sql_history_by_summary(
        self, query_text: str, domain: str = "", layer1: str = "", layer2: str = "", top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Search SQL history by summary using vector search."""
        conditions = []
        if domain:
            conditions.append(eq("domain", domain))
        if layer1:
            conditions.append(eq("layer1", layer1))
        if layer2:
            conditions.append(eq("layer2", layer2))

        if not conditions:
            where_condition = None
            where_clause = ""
        elif len(conditions) == 1:
            where_condition = conditions[0]
            where_clause = build_where(where_condition)
        else:
            where_condition = And(conditions)
            where_clause = build_where(where_condition)

        logger.info(f"Searching SQL history by summary: {query_text}, where: {where_clause}")
        search_results = self.sql_history_storage.search(
            query_text,
            top_n=top_n,
            where=where_condition,
        )

        if search_results:
            result_list = search_results.select(
                ["name", "sql", "comment", "summary", "filepath", "domain", "layer1", "layer2", "tags"]
            ).to_pylist()
            logger.info(f"Found {len(result_list)} SQL history results for query: {query_text}")
            return result_list
        else:
            logger.info(f"No SQL history results found for query: {query_text}")
            return []

    def get_sql_history_detail(self, domain: str, layer1: str, layer2: str, name: str) -> List[Dict[str, Any]]:
        return self.sql_history_storage._search_all(
            And([eq("domain", domain), eq("layer1", layer1), eq("layer2", layer2), eq("name", name)]),
            ["name", "summary", "comment", "tags", "sql"],
        ).to_pylist()

    def update(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        """
        Currently, only two update scenarios are supported:
            - Update domain, layer 1, layer 2, and name
            - Update detail fields
        """
        if "name" in update_values:
            unique_filter = And(
                [
                    eq("domain", update_values.get("domain", old_values.get("domain"))),
                    eq("layer1", update_values.get("layer1", old_values.get("layer1"))),
                    eq("layer2", update_values.get("layer2", old_values.get("layer2"))),
                    eq("name", update_values["name"]),
                ]
            )
        else:
            unique_filter = None
        where_conditions = []
        for k in ("domain", "layer1", "layer2", "name"):
            if k in old_values:
                where_conditions.append(eq(k, old_values[k]))
        if not where_conditions:
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message_args={
                    "operation": "update",
                    "table_name": self.sql_history_storage.table_name,
                    "error_message": "Missing WHERE for metrics update",
                },
            )
        where = And(where_conditions)
        self.sql_history_storage.update(where, update_values, unique_filter=unique_filter)


def sql_history_rag_by_configuration(agent_config: AgentConfig):
    """Create SqlHistoryRAG instance from agent configuration."""
    return SqlHistoryRAG(agent_config.rag_storage_path())
