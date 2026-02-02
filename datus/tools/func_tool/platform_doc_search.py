# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.document.store import DocumentStore, document_store
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_NAME = "platform_doc_search_tools"
_NAME_LIST_NAV = "platform_doc_search_tools.list_document_nav"
_NAME_GET_DOC = "platform_doc_search_tools.get_document"
_NAME_SEARCH_DOC = "platform_doc_search_tools.search_document"


class PlatformDocSearchTool:
    """Function-call tool for platform documentation search.

    Exposes three LLM-callable functions:
    - list_document_nav: Browse the documentation navigation tree
    - get_document: Retrieve document chunks by title
    - search_document: Semantic search across documentation
    """

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self._document_store: Optional[DocumentStore] = None
        self._has_document = False

    @property
    def document_store(self) -> DocumentStore:
        """Lazy initialize document store."""
        if self._document_store is None:
            self._document_store = document_store(self.agent_config.rag_storage_path())
            self._has_document = self._document_store.table.count_rows() > 0
        return self._document_store

    @staticmethod
    def all_tools_name() -> List[str]:
        return ["list_document_nav", "get_document", "search_document"]

    def available_tools(self) -> List[Tool]:
        """Return all platform doc search tools for LLM function calling."""
        if not self._has_document:
            return []
        return [
            trans_to_function_tool(self.list_document_nav),
            trans_to_function_tool(self.get_document),
            trans_to_function_tool(self.search_document),
        ]

    def list_document_nav(
        self,
        platform: str,
        version: Optional[str] = None,
    ) -> FuncToolResult:
        """
        Browse the documentation navigation tree for a platform.

        Use this tool FIRST to discover what documentation is available,
        then use `get_document` to drill into specific documents.

        When version is specified, returns a flat tree.
        When version is omitted and multiple versions exist, returns grouped by version.

        Args:
            platform: Platform name (e.g., snowflake, duckdb, starrocks, postgresql)
            version: Filter by specific version (optional, omit to see all versions)

        Returns:
            FuncToolResult with navigation tree structure:
            - Each node has: name, children (sub-groups), docs (document title strings)
            - Use document titles from "docs" to call `get_document`
        """
        try:
            from datus.tools.search_tools.search_tool import SearchTool

            tool = SearchTool(agent_config=self.agent_config)
            tool._document_store = self.document_store
            result = tool.list_document_nav(platform=platform, version=version)

            if not result.success:
                return FuncToolResult(success=0, error=result.error)

            return FuncToolResult(
                success=1,
                result={
                    "platform": result.platform,
                    "version": result.version,
                    "nav_tree": result.nav_tree,
                    "total_docs": result.total_docs,
                },
            )
        except Exception as e:
            logger.error(f"Failed to list document nav for '{platform}': {e}")
            return FuncToolResult(success=0, error=str(e))

    def get_document(
        self,
        platform: str,
        titles: List[str],
        version: Optional[str] = None,
    ) -> FuncToolResult:
        """
        Get document content by matching a hierarchy path.

        Use the navigation tree from `list_document_nav` to build the hierarchy path.
        The `titles` parameter represents ONE document's path from parent group to
        document title. All elements are AND-matched, so they must all appear in the
        same document's hierarchy.

        IMPORTANT: To retrieve ONE document, pass its parent group(s) + document title.
        To retrieve MULTIPLE documents, call this tool multiple times.

        Examples:
            - Get "CREATE TABLE" under "DDL": titles=["DDL", "CREATE TABLE"]
            - Get "ALTER TABLE" under "DDL": titles=["DDL", "ALTER TABLE"]
            - WRONG: titles=["DDL", "CREATE TABLE", "ALTER TABLE"] → returns nothing
              because no single document matches all three

        Args:
            platform: Platform name (e.g., snowflake, duckdb, starrocks, postgresql)
            titles: Hierarchy path to ONE document (e.g., ["DDL", "CREATE TABLE"])
            version: Filter by specific version (optional)

        Returns:
            FuncToolResult with document chunks ordered by position, each containing:
            - chunk_text: The document content
            - title: Section title
            - hierarchy: Full hierarchy path
        """
        try:
            from datus.tools.search_tools.search_tool import SearchTool

            tool = SearchTool(agent_config=self.agent_config)
            tool._document_store = self.document_store
            result = tool.get_document(platform=platform, titles=titles, version=version)

            if not result.success:
                return FuncToolResult(success=0, error=result.error)

            return FuncToolResult(
                success=1,
                result={
                    "platform": result.platform,
                    "version": result.version,
                    "title": result.title,
                    "hierarchy": result.hierarchy,
                    "chunk_count": result.chunk_count,
                    "chunks": result.chunks,
                },
            )
        except Exception as e:
            logger.error(f"Failed to get document for titles {titles}: {e}")
            return FuncToolResult(success=0, error=str(e))

    def search_document(
        self,
        platform: str,
        keywords: List[str],
        version: Optional[str] = None,
        top_n: int = 5,
    ) -> FuncToolResult:
        """
        Search platform documentation using semantic similarity.

        Use this when you know what you're looking for but don't know the exact title.
        Each keyword is searched independently; results are grouped by keyword.

        Args:
            platform: Platform name (e.g., snowflake, duckdb, starrocks, postgresql)
            keywords: List of search queries (e.g., ["CREATE TABLE syntax", "data types"])
            version: Filter by specific version (optional)
            top_n: Maximum results per keyword (default 5)

        Returns:
            FuncToolResult with matched documents grouped by keyword, each containing:
            - chunk_text: Matched content
            - title: Section title
            - hierarchy: Full hierarchy path
            - doc_path: Source document path
        """
        try:
            from datus.tools.search_tools.search_tool import SearchTool

            tool = SearchTool(agent_config=self.agent_config)
            tool._document_store = self.document_store
            result = tool.search_document(platform=platform, keywords=keywords, version=version, top_n=top_n)

            if not result.success:
                return FuncToolResult(success=0, error=result.error)

            return FuncToolResult(
                success=1,
                result={
                    "docs": result.docs,
                    "doc_count": result.doc_count,
                },
            )
        except Exception as e:
            logger.error(f"Failed to search documents for keywords {keywords}: {e}")
            return FuncToolResult(success=0, error=str(e))
