# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import List

from datus.configuration.agent_config import AgentConfig
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.storage.document.store import DocumentStore, document_store
from datus.tools.base import BaseTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SearchTool(BaseTool):
    """Tool for searching documents using various methods"""

    tool_name = "search"
    tool_description = "Search for documents using various methods (internal, external, llm)"

    def __init__(self, agent_config: AgentConfig, **kwargs):
        """Initialize with a document store path"""
        super().__init__(**kwargs)
        self.agent_config = agent_config
        self._document_store = None

    @property
    def document_store(self) -> DocumentStore:
        """Lazy initialize document store"""
        if self._document_store is None:
            self._document_store = document_store(self.agent_config.rag_storage_path())
        return self._document_store

    def execute(self, input_data: DocSearchInput) -> DocSearchResult:
        """Execute document search based on method"""
        if input_data.method == "internal":
            return self._search_internal(input_data)
        elif input_data.method == "external":
            return search_by_tavily(input_data.keywords, input_data.top_n)
        elif input_data.method == "llm":
            return DocSearchResult(success=False, error="LLM search method not implemented yet", docs={}, doc_count=0)
        else:
            return DocSearchResult(
                success=False,
                error=f"Unknown search method: {input_data.method}",
                docs={},
                doc_count=0,
            )

    def _search_internal(self, input_data: DocSearchInput) -> DocSearchResult:
        """Search internal documents using DocumentStore"""
        try:
            docs = {}
            total_docs = 0

            for keyword in input_data.keywords:
                try:
                    results = self.document_store.search(
                        query_txt=keyword,
                        select_fields=[
                            "title",
                            "hierarchy",
                            "keywords",
                            "language",
                            "chunk_text",
                        ],
                        top_n=input_data.top_n,
                    ).to_pylist()

                    text_results = [result["chunk_text"] for result in results]
                    docs[keyword] = text_results
                    total_docs += len(text_results)
                except Exception as e:
                    logger.error(f"Error searching for keyword '{keyword}': {str(e)}")
                    docs[keyword] = []

            logger.info(f"Found {total_docs} documents for keywords: {input_data.keywords}")
            return DocSearchResult(success=True, docs=docs, doc_count=total_docs)
        except Exception as e:
            logger.error(f"Internal search failed: {str(e)}")
            return DocSearchResult(success=False, error=f"Internal search failed: {str(e)}", docs={}, doc_count=0)


def search_by_tavily(keywords: List[str], top_n: int) -> DocSearchResult:
    """
    Search external documents using TAVILY API
    :param keywords:
    :param top_n:
    :return:
    """
    api_key = os.environ.get("TAVILY_API_KEY")
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
        docs = {}
        total_docs = 0
        for keyword in keywords:
            payload = {
                "api_key": api_key,
                "query": keyword,
                "search_depth": "advanced",
                "max_results": top_n,
                "include_raw_content": True,
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            texts = [(item.get("raw_content") or item.get("content") or "") for item in result.get("results", [])]
            docs[keyword] = texts
            total_docs += len(texts)

        return DocSearchResult(success=True, docs=docs, doc_count=total_docs)
    except requests.HTTPError as e:
        return DocSearchResult(
            success=False, error=f"Tavily HTTP {e.response.status_code}: {e.response.text[:300]}", docs={}, doc_count=0
        )
    except Exception as e:
        logger.error(f"External search failed: {e}")
        return DocSearchResult(success=False, error=f"External search failed: {str(e)}", docs={}, doc_count=0)
