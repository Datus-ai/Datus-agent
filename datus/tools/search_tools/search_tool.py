# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any, Dict, List, Literal, Optional

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

    def _get_document_store(self, platform: str) -> Optional[DocumentStore]:
        """Get document store for a specific platform.

        Per-platform store is resolved via ``agent_config.document_storage_path(platform)``,
        which returns ``{home}/data/document/{platform}/``.

        Args:
            platform: Platform name (e.g., snowflake, starrocks, duckdb)

        Returns:
            DocumentStore instance if the platform directory exists, None otherwise.
        """
        store_path = self.agent_config.document_storage_path(platform)
        if not os.path.exists(store_path):
            return None
        return document_store(store_path)

    @staticmethod
    def _version_sort_key(version_str: str):
        """Generate a sort key that handles semantic versioning correctly.

        Splits version string into numeric and non-numeric parts so that
        ``v3.10`` sorts after ``v3.9`` (unlike plain lexicographic order).
        Non-numeric prefixes (e.g. ``v``) are compared as strings.
        """
        import re

        parts = re.split(r"(\d+)", version_str)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    @staticmethod
    def _resolve_latest_version(store: DocumentStore) -> Optional[str]:
        """Return the latest (max) version string from the store, or None if empty."""
        versions_info = store.list_versions()
        if not versions_info:
            return None
        return max(
            (v["version"] for v in versions_info),
            key=SearchTool._version_sort_key,
        )

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

        Returns a pure hierarchical tree where internal nodes are navigation
        categories and leaf nodes (empty ``children``) are document titles::

            [
                {"name": "SQL Reference", "children": [
                    {"name": "DDL", "children": [
                        {"name": "CREATE TABLE", "children": []},
                    ]},
                ]},
            ]

        Args:
            platform: Platform name (e.g., snowflake, duckdb, postgresql)
            version: Filter by version (optional)

        Returns:
            DocNavResult with hierarchical navigation tree
        """
        try:
            store = self._get_document_store(platform)
            if store is None:
                return DocNavResult(
                    success=True,
                    platform=platform,
                    version=version,
                    nav_tree=[],
                    total_docs=0,
                )

            # Default to latest version when not specified
            if not version:
                version = self._resolve_latest_version(store)

            where: WhereExpr = eq("version", version) if version else None

            rows = store.get_all_rows(
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

            # Group by (version, doc_path) to avoid collapsing across versions
            versioned_doc_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for row in rows:
                doc_path = row.get("doc_path", "")
                ver = row.get("version", "")
                if not doc_path:
                    continue
                versioned_doc_map.setdefault(ver, {})
                if doc_path not in versioned_doc_map[ver]:
                    versioned_doc_map[ver][doc_path] = row

            versions = sorted(
                versioned_doc_map.keys(),
                key=self._version_sort_key,
                reverse=True,
            )

            if version or len(versions) <= 1:
                # Single version → flat tree
                flat_map = versioned_doc_map.get(version or (versions[0] if versions else ""), {})
                nav_tree = self._build_nav_tree(flat_map)
            else:
                # Multiple versions → group by version at the top level
                nav_tree = []
                for ver in versions:
                    nav_tree.append({"version": ver, "tree": self._build_nav_tree(versioned_doc_map[ver])})

            total_docs = sum(len(v) for v in versioned_doc_map.values())
            logger.debug(f"Found {total_docs} documents for platform '{platform}'")

            return DocNavResult(
                success=True,
                platform=platform,
                version=version,
                nav_tree=nav_tree,
                total_docs=total_docs,
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
        """Build a pure hierarchical navigation tree for LLM browsing.

        Internal nodes are created from ``nav_path`` segments.
        Leaf nodes are document titles (h1) with empty ``children``.

        Output format::

            [
                {
                    "name": "Administration",
                    "children": [
                        {"name": "Cluster Management", "children": [
                            {"name": "Cluster Snapshot", "children": []},
                            {"name": "Scale", "children": []},
                        ]}
                    ]
                }
            ]

        Args:
            doc_map: Mapping of doc_path -> first chunk row dict

        Returns:
            List of root-level tree nodes
        """
        root: Dict[str, Any] = {"children": {}}

        for doc_path, doc_info in doc_map.items():
            nav_path = self._normalize_list_field(doc_info.get("nav_path", []))
            title = doc_info.get("title", "") or doc_path.rsplit("/", 1)[-1]

            # Walk the tree along nav_path, creating intermediate nodes as needed
            node = root
            for segment in nav_path:
                if segment not in node["children"]:
                    node["children"][segment] = {"children": {}}
                node = node["children"][segment]

            # Insert document title as a leaf (skip if same as last nav_path segment)
            if not nav_path or title != nav_path[-1]:
                if title not in node["children"]:
                    node["children"][title] = {"children": {}}

        # Convert the nested dict into a sorted list of tree nodes
        def _to_list(tree_node: Dict[str, Any]) -> List[Dict[str, Any]]:
            result = []
            for name in sorted(tree_node["children"]):
                child = tree_node["children"][name]
                result.append({"name": name, "children": _to_list(child)})
            return result

        return _to_list(root)

    def get_document(
        self,
        platform: str,
        titles: List[str],
        version: Optional[str] = None,
    ) -> GetDocResult:
        """Get document chunks by matching a hierarchy prefix path.

        The ``titles`` list is joined with ``" > "`` to form a hierarchy
        prefix that is matched against the stored ``hierarchy`` field.
        This directly maps to the navigation tree returned by
        ``list_document_nav``.

        Example::

            titles=["DDL", "CREATE TABLE"]
            → matches hierarchy "... > DDL > CREATE TABLE > ..."

        Args:
            platform: Platform name (e.g., snowflake, duckdb, postgresql)
            titles: Hierarchy path to one document (e.g., ["DDL", "CREATE TABLE"])
            version: Filter by version (optional)

        Returns:
            GetDocResult with document chunks
        """
        if not platform:
            return GetDocResult(success=False, error="platform can't be empty", platform=platform)
        if not titles:
            return GetDocResult(success=False, error="titles can't be empty", platform=platform)
        try:
            store = self._get_document_store(platform)

            # Default to latest version when not specified
            if not version and store is not None:
                version = self._resolve_latest_version(store)

            # Build where conditions
            conditions: List[Condition] = []
            if version:
                conditions.append(eq("version", version))

            # Join titles into a single hierarchy prefix for precise matching
            if titles:
                hierarchy_prefix = " > ".join(titles)
                conditions.append(like("hierarchy", f"%{hierarchy_prefix}%"))

            where: WhereExpr = None
            if len(conditions) > 1:
                where = And(conditions)
            elif len(conditions) == 1:
                where = conditions[0]

            # Get matching documents
            rows = (
                []
                if store is None
                else store.get_all_rows(
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

            # Group by doc_path — prefix matching may still hit multiple documents
            doc_groups: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                dp = row.get("doc_path", "")
                doc_groups.setdefault(dp, []).append(row)

            # Pick the best matching document (fewest extra hierarchy segments)
            best_doc_path = min(
                doc_groups,
                key=lambda dp: len(doc_groups[dp][0].get("hierarchy", "").split(">")),
            )
            best_rows = doc_groups[best_doc_path]

            # Sort chunks by chunk_index
            best_rows.sort(key=lambda x: x.get("chunk_index", 0))

            first_chunk = best_rows[0]

            logger.info(f"Found {len(best_rows)} chunks for titles {titles} in platform '{platform}'")

            return GetDocResult(
                success=True,
                platform=platform,
                version=first_chunk.get("version"),
                title=first_chunk.get("title", ""),
                hierarchy=first_chunk.get("hierarchy", ""),
                chunks=best_rows,
                chunk_count=len(best_rows),
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
        if not platform:
            return DocSearchResult(success=False, error="platform can't be empty")
        if not keywords:
            return DocSearchResult(success=False, error="keywords can't be empty")
        try:
            docs: Dict[str, List[Dict[str, Any]]] = {}
            total_count = 0

            store = self._get_document_store(platform)
            if store is not None:
                # Default to latest version when not specified
                if not version:
                    version = self._resolve_latest_version(store)

                for keyword in keywords:
                    try:
                        results = store.search_docs(
                            query=keyword,
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


def search_by_tavily(
    keywords: List[str],
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "advanced",
    include_answer: Literal[False, "basic", "advanced"] = False,
    include_raw_content: Literal[False, "text", "markdown"] = False,
    include_domains: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> DocSearchResult:
    """Search external documents using the Tavily Search API.

    Sends all keywords as a single query (tab-joined) and returns results.
    See https://docs.tavily.com/documentation/api-reference/endpoint/search

    Args:
        keywords: Keywords to search for (joined into a single query)
        max_results: Maximum number of search results to return, 0-20 (default: 5)
        search_depth: Controls latency vs. relevance tradeoff.
            "basic" — fast, 1 credit per request.
            "advanced" — slower but more relevant, 2 credits per request.
        include_answer: Whether to include an LLM-generated answer summary.
            False — no answer (default).
            "basic" — quick short answer.
            "advanced" — detailed long-form answer (3 extra credits).
        include_raw_content: Whether to include cleaned HTML content per result.
            False — only snippet content (default).
            "text" — plain text.
            "markdown" — markdown formatted.
        include_domains: Restrict search to specific domains (max 300),
            e.g. ["docs.snowflake.com", "stackoverflow.com"].
        api_key: Tavily API key. Falls back to TAVILY_API_KEY env var if not provided.

    Returns:
        DocSearchResult with matched documents.
        If include_answer is enabled, the answer is prepended to the results list.
    """
    api_key = api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return DocSearchResult(
            success=False,
            error="TAVILY_API_KEY not configured. Please set the TAVILY_API_KEY environment variable.",
            docs={},
            doc_count=0,
        )
    if not keywords:
        return DocSearchResult(success=True, docs={}, doc_count=0)
    import requests

    try:
        url = "https://api.tavily.com/search"

        query = "\t".join(keywords)
        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }
        if include_answer:
            payload["include_answer"] = include_answer
        if include_raw_content:
            payload["include_raw_content"] = include_raw_content
        if include_domains:
            payload["include_domains"] = include_domains

        response = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        texts = [(item.get("raw_content") or item.get("content") or "") for item in result.get("results", [])]

        # Prepend the generated answer if available
        answer = result.get("answer")
        if answer:
            texts.insert(0, answer)

        # DocSearchResult.docs expects Dict[str, List], key by the joined query
        docs = {query: texts}
        return DocSearchResult(success=True, docs=docs, doc_count=len(texts))
    except requests.HTTPError as e:
        return DocSearchResult(
            success=False,
            error=f"Tavily HTTP {e.response.status_code}: {e.response.text[:300]}",
            docs={},
            doc_count=0,
        )
    except Exception as e:
        logger.error(f"External search failed: {e}")
        return DocSearchResult(success=False, error=f"External search failed: {str(e)}", docs={}, doc_count=0)
