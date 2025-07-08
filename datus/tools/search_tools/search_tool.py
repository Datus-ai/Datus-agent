import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests

from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.storage.document.store import DocumentStore
from datus.tools.base import BaseTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SearchTool(BaseTool):
    """Tool for searching documents using various methods"""

    tool_name = "search"
    tool_description = "Search for documents using various methods (internal, external, llm)"

    def __init__(self, document_store_path: str = "data/datus_db", **kwargs):
        """Initialize with a document store path"""
        super().__init__(**kwargs)
        self.document_store_path = document_store_path
        self._document_store = None

    @property
    def document_store(self) -> DocumentStore:
        """Lazy initialize document store"""
        if self._document_store is None:
            self._document_store = DocumentStore(self.document_store_path)
        return self._document_store

    def execute(self, input_data: DocSearchInput) -> DocSearchResult:
        """Execute document search based on method"""
        if input_data.method == "internal":
            return self._search_internal(input_data)
        elif input_data.method == "external":
            return self._search_external(input_data)
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
                    results = (
                        self.document_store.table.search(keyword, query_type="fts")
                        .limit(input_data.top_n)
                        .select(
                            [
                                "title",
                                "hierarchy",
                                "keywords",
                                "language",
                                "chunk_text",
                                "created_at",
                            ]
                        )
                        .to_list()
                    )

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

    def _search_external(self, input_data: DocSearchInput) -> DocSearchResult:
        """Search external documents using TAVILY_API"""
        try:
            tavily_api_key = os.environ.get("TAVILY_API_KEY")
            if not tavily_api_key:
                return DocSearchResult(
                    success=False,
                    error="TAVILY_API key not configured. Please set the TAVILY_API_KEY environment variable.",
                    docs={},
                    doc_count=0,
                )

            url = "https://api.tavily.com/search"

            params = {
                "api_key": tavily_api_key,
                "query": " ".join(input_data.keywords),
                "search_depth": "advanced",
                "max_results": 3,
                "include_raw_content": True,
            }

            docs = {}
            total_docs = 0
            for keyword in input_data.keywords:
                response = requests.post(url, json=params)
                response.raise_for_status()

                result = response.json()
                raw_contents = [result["content"] for result in result.get("results", [])]
                docs[keyword] = raw_contents
                total_docs += len(raw_contents)

            return DocSearchResult(success=True, docs=docs, doc_count=total_docs)
        except Exception as e:
            logger.error(f"External search failed: {str(e)}")
            return DocSearchResult(success=False, error=f"External search failed: {str(e)}", docs={}, doc_count=0)

    def import_documents(self, directory_path: str) -> Tuple[int, List[str]]:
        """
        Import markdown documents from a directory into the document store.

        Args:
            directory_path: Path to the directory containing markdown files

        Returns:
            Tuple containing (number of documents imported, list of document titles)
        """
        try:
            document_path = Path(directory_path)
            if not document_path.exists() or not document_path.is_dir():
                logger.error(f"Directory not found: {directory_path}")
                return 0, []

            markdown_files = list(document_path.glob("*.md"))
            if not markdown_files:
                logger.error(f"No markdown files found in {directory_path}")
                return 0, []

            imported_count = 0
            imported_titles = []

            for md_file in markdown_files:
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    title = md_file.stem.replace("_", " ")
                    content_lines = content.strip().split("\n")
                    if content_lines and content_lines[0].startswith("# "):
                        title = content_lines[0].replace("# ", "")

                    self.document_store.table.add(
                        pd.DataFrame(
                            [
                                {
                                    "title": title,
                                    "hierarchy": "",
                                    "keywords": [],
                                    "language": "en",
                                    "chunk_text": content,
                                    "created_at": datetime.now().isoformat(),
                                }
                            ]
                        )
                    )

                    imported_titles.append(title)
                    imported_count += 1

                except Exception as e:
                    logger.error(f"Failed to process document {md_file}: {str(e)}")

            logger.info(f"Imported {imported_count} documents")
            return imported_count, imported_titles

        except Exception as e:
            logger.error(f"Document import failed: {str(e)}")
            return 0, []
