# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.doc_search_node_models import DocNavResult, DocSearchInput, DocSearchResult, GetDocResult
from datus.storage.document.store import DocumentStore, document_store
from datus.storage.lancedb_conditions import And, Condition, WhereExpr, eq, like
from datus.tools.base import BaseTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SearchTool(BaseTool):
    """Tool for searching platform documentation.

    Provides three main methods:
    - list_document_nav: List navigation structure (titles/hierarchy) for a platform
    - get_document: Get document chunks by titles/hierarchy
    - search_document: Search documents by keywords with semantic similarity
    """

    tool_name = "search"
    tool_description = "Search for platform documentation using vector store"

    def __init__(self, agent_config: AgentConfig, **kwargs):
        """Initialize with agent configuration."""
        super().__init__(**kwargs)
        self.agent_config = agent_config
        self._document_store = None

    @property
    def document_store(self) -> DocumentStore:
        """Lazy initialize document store."""
        if self._document_store is None:
            self._document_store = document_store(self.agent_config.rag_storage_path())
        return self._document_store

    def execute(self, input_data: DocSearchInput) -> DocSearchResult:
        """Execute document search (default entry point).

        Args:
            input_data: Search input with platform, keywords, version, top_n

        Returns:
            DocSearchResult with matched documents
        """
        return self.search_document(
            platform=input_data.platform,
            keywords=input_data.keywords,
            version=input_data.version,
            top_n=input_data.top_n,
        )

    def list_document_nav(
        self,
        platform: str,
        version: Optional[str] = None,
    ) -> DocNavResult:
        """List navigation structure for a platform's documentation.

        Returns a hierarchical tree grouped by nav_path, with documents as leaves.

        When **version is specified**, returns a flat tree::

            [
                {"name": "SQL Reference", "children": [...], "docs": ["CREATE TABLE", ...]},
                ...
            ]

        When **version is empty** (multi-version), returns top-level version grouping::

            [
                {"version": "v3.4.0", "tree": [{"name": "SQL Reference", ...}]},
                {"version": "v3.3.0", "tree": [...]},
            ]

        Leaf naming rule:
        - If nav_path != titles -> leaf title comes from the document title
        - If nav_path == titles -> leaf title is the last element of nav_path

        Args:
            platform: Platform name (e.g., snowflake, duckdb, postgresql)
            version: Filter by version (optional)

        Returns:
            DocNavResult with hierarchical navigation tree
        """
        try:
            # Build where condition
            conditions: List[Condition] = [eq("platform", platform)]
            if version:
                conditions.append(eq("version", version))

            where: WhereExpr = And(conditions) if len(conditions) > 1 else conditions[0]

            # Get all chunks with navigation fields
            rows = self.document_store.get_all_rows(
                where=where,
                select_fields=["title", "titles", "nav_path", "version", "doc_path"],
            )

            if not rows:
                return DocNavResult(
                    success=True,
                    platform=platform,
                    version=version,
                    nav_tree=[],
                    total_docs=0,
                )

            # Group by doc_path to get one representative row per document
            doc_map: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                doc_path = row.get("doc_path", "")
                if doc_path and doc_path not in doc_map:
                    doc_map[doc_path] = row

            # Collect distinct versions
            versions = sorted({row.get("version", "") for row in doc_map.values()}, reverse=True)

            if version or len(versions) <= 1:
                # Single version → flat tree
                nav_tree = self._build_nav_tree(doc_map)
            else:
                # Multiple versions → group by version at the top level
                nav_tree = []
                for ver in versions:
                    ver_doc_map = {k: v for k, v in doc_map.items() if v.get("version", "") == ver}
                    if ver_doc_map:
                        nav_tree.append(
                            {
                                "version": ver,
                                "tree": self._build_nav_tree(ver_doc_map),
                            }
                        )

            logger.debug(f"Found {len(doc_map)} documents for platform '{platform}'")

            return DocNavResult(
                success=True,
                platform=platform,
                version=version,
                nav_tree=nav_tree,
                total_docs=len(doc_map),
            )

        except Exception as e:
            logger.error(f"Failed to list document navigation: {e}")
            return DocNavResult(
                success=False,
                error=str(e),
                platform=platform,
                version=version,
                nav_tree=[],
                total_docs=0,
            )

    @staticmethod
    def _normalize_list_field(value) -> List[str]:
        """Normalize a field that may be stored as a list or a delimited string."""
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            return [s.strip() for s in value.split(">")]
        return []

    def _build_nav_tree(self, doc_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a lightweight hierarchical navigation tree for LLM browsing.

        Groups documents by nav_path to form internal tree nodes.
        Leaf docs are just title strings — enough for the LLM to drill into
        via ``get_document(platform, titles=[...])``.

        Output format::

            [
                {
                    "name": "Administration",
                    "children": [
                        {"name": "Cluster Management", "children": [], "docs": ["Cluster Snapshot", "Scale"]}
                    ],
                    "docs": []
                }
            ]

        Args:
            doc_map: Mapping of doc_path -> first chunk row dict

        Returns:
            List of root-level tree nodes
        """
        # Internal tree node: {"children": {name: node}, "docs": [title_str, ...]}
        root: Dict[str, Any] = {"children": {}, "docs": []}

        for doc_path, doc_info in doc_map.items():
            nav_path = self._normalize_list_field(doc_info.get("nav_path", []))
            titles = self._normalize_list_field(doc_info.get("titles", []))
            title = doc_info.get("title", "")

            # Determine leaf name:
            #   nav_path == titles  -> last element of nav_path
            #   nav_path != titles  -> document title
            if nav_path and nav_path == titles:
                leaf_name = nav_path[-1]
            else:
                leaf_name = title or (titles[0] if titles else doc_path.rsplit("/", 1)[-1])

            # Walk the tree along nav_path, creating intermediate nodes as needed
            node = root
            for segment in nav_path:
                if segment not in node["children"]:
                    node["children"][segment] = {"children": {}, "docs": []}
                node = node["children"][segment]

            # Attach document title as a leaf
            if leaf_name not in node["docs"]:
                node["docs"].append(leaf_name)

        # Convert the nested dict into a sorted list of tree nodes
        def _to_list(node: Dict[str, Any]) -> List[Dict[str, Any]]:
            result = []
            for name in sorted(node["children"]):
                child = node["children"][name]
                result.append(
                    {
                        "name": name,
                        "children": _to_list(child),
                        "docs": sorted(child["docs"]),
                    }
                )
            return result

        tree = _to_list(root)

        # Root-level docs (documents with empty nav_path) become top-level entries
        if root["docs"]:
            for title in sorted(root["docs"]):
                tree.append(
                    {
                        "name": title,
                        "children": [],
                        "docs": [title],
                    }
                )

        return tree

    def get_document(
        self,
        platform: str,
        titles: List[str],
        version: Optional[str] = None,
    ) -> GetDocResult:
        """Get document chunks by matching a hierarchy path.

        All elements in ``titles`` are AND-matched against the hierarchy field,
        so they must all appear in the same document. Use this to locate ONE
        document by its parent group(s) + document title.

        Args:
            platform: Platform name (e.g., snowflake, duckdb, postgresql)
            titles: Hierarchy path to one document (e.g., ["DDL", "CREATE TABLE"])
            version: Filter by version (optional)

        Returns:
            GetDocResult with document chunks
        """
        try:
            # Build where condition for platform
            conditions: List[Condition] = [eq("platform", platform)]
            if version:
                conditions.append(eq("version", version))

            # Add hierarchy LIKE condition for each title
            for title in titles:
                conditions.append(like("hierarchy", f"%{title}%"))

            where: WhereExpr = And(conditions) if len(conditions) > 1 else conditions[0]

            # Get matching documents
            rows = self.document_store.get_all_rows(
                where=where,
                select_fields=[
                    "chunk_id",
                    "chunk_index",
                    "chunk_text",
                    "title",
                    "titles",
                    "hierarchy",
                    "nav_path",
                    "doc_path",
                    "version",
                    "keywords",
                ],
            )

            if not rows:
                return GetDocResult(
                    success=True,
                    platform=platform,
                    version=version,
                    title="",
                    hierarchy="",
                    chunks=[],
                    chunk_count=0,
                )

            # Sort chunks by chunk_index
            rows.sort(key=lambda x: x.get("chunk_index", 0))

            # Get metadata from first chunk
            first_chunk = rows[0]

            logger.info(f"Found {len(rows)} chunks for titles {titles} in platform '{platform}'")

            return GetDocResult(
                success=True,
                platform=platform,
                version=first_chunk.get("version"),
                title=first_chunk.get("title", ""),
                hierarchy=first_chunk.get("hierarchy", ""),
                chunks=rows,
                chunk_count=len(rows),
            )

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return GetDocResult(
                success=False,
                error=str(e),
                platform=platform,
                version=version,
                title="",
                hierarchy="",
                chunks=[],
                chunk_count=0,
            )

    def search_document(
        self,
        platform: str,
        keywords: List[str],
        version: Optional[str] = None,
        top_n: int = 5,
    ) -> DocSearchResult:
        """Search documents by keywords using semantic similarity.

        Args:
            platform: Platform name (e.g., snowflake, duckdb, postgresql)
            keywords: List of keywords/queries to search
            version: Filter by version (optional)
            top_n: Maximum results per keyword (default: 5)

        Returns:
            DocSearchResult with matched documents for each keyword
        """
        try:
            docs: Dict[str, List[Dict[str, Any]]] = {}
            total_count = 0

            for keyword in keywords:
                try:
                    results = self.document_store.search_docs(
                        query=keyword,
                        platform=platform,
                        version=version,
                        top_n=top_n,
                        select_fields=[
                            "chunk_id",
                            "chunk_text",
                            "title",
                            "titles",
                            "hierarchy",
                            "nav_path",
                            "doc_path",
                            "version",
                            "keywords",
                        ],
                    )

                    docs[keyword] = results
                    total_count += len(results)

                except Exception as e:
                    logger.error(f"Error searching for keyword '{keyword}': {e}")
                    docs[keyword] = []

            logger.info(f"Found {total_count} documents for {len(keywords)} keywords in platform '{platform}'")

            return DocSearchResult(
                success=True,
                docs=docs,
                doc_count=total_count,
            )

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return DocSearchResult(
                success=False,
                error=str(e),
                docs={},
                doc_count=0,
            )
