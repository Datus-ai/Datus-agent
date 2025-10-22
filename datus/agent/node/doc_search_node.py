# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.schemas.search_metrics_node_models import SearchMetricsInput
from datus.storage.document import DocumentStore
from datus.storage.document.store import document_store
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DocSearchNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: SearchMetricsInput = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
        )
        self._document_store = None

    @property
    def document_store(self) -> DocumentStore:
        """Lazy initialize document store"""
        if self._document_store is None:
            self._document_store = document_store(self.agent_config.rag_storage_path())
        return self._document_store

    def execute(self):
        self.result = self._execute_document()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute document search with streaming support."""
        async for action in self._doc_search_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = DocSearchInput(keywords=workflow.context.doc_search_keywords, top_n=3, method="internal")
        self.input = next_input
        return {"success": True, "message": "Document appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update document search results to workflow context."""
        result = self.result
        try:
            logger.info(f"Updating document search context: {result}")
            workflow.context.document_result = result
            return {"success": True, "message": "Updated document search context"}
        except Exception as e:
            logger.error(f"Failed to update document search context: {str(e)}")
            return {"success": False, "message": f"Document search context update failed: {str(e)}"}

    def _execute_document(self) -> DocSearchResult:
        """Execute document search based on method"""
        if self.input.method == "internal":
            return self._search_internal()
        elif self.input.method == "external":
            return self._search_external()
        elif self.input.method == "llm":
            return DocSearchResult(success=False, error="LLM search method not implemented yet", docs={}, doc_count=0)
        else:
            return DocSearchResult(
                success=False,
                error=f"Unknown search method: {self.input.method}",
                docs={},
                doc_count=0,
            )

    def _search_internal(self) -> DocSearchResult:
        """Search internal documents using DocumentStore"""
        try:
            docs = {}
            total_docs = 0

            for keyword in self.input.keywords:
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
                        top_n=self.input.top_n,
                    ).to_pylist()

                    text_results = [result["chunk_text"] for result in results]
                    docs[keyword] = text_results
                    total_docs += len(text_results)
                except Exception as e:
                    logger.error(f"Error searching for keyword '{keyword}': {str(e)}")
                    docs[keyword] = []

            logger.info(f"Found {total_docs} documents for keywords: {self.input.keywords}")
            return DocSearchResult(success=True, docs=docs, doc_count=total_docs)
        except Exception as e:
            logger.error(f"Internal search failed: {str(e)}")
            return DocSearchResult(success=False, error=f"Internal search failed: {str(e)}", docs={}, doc_count=0)

    def _search_external(self) -> DocSearchResult:
        """Search external documents using TAVILY_API"""
        try:
            import os

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
                "query": " ".join(self.input.keywords),
                "search_depth": "advanced",
                "max_results": 3,
                "include_raw_content": True,
            }

            docs = {}
            total_docs = 0
            for keyword in self.input.keywords:
                response = self.input.post(url, json=params)
                response.raise_for_status()

                result = response.json()
                raw_contents = [result["content"] for result in result.get("results", [])]
                docs[keyword] = raw_contents
                total_docs += len(raw_contents)

            return DocSearchResult(success=True, docs=docs, doc_count=total_docs)
        except Exception as e:
            logger.error(f"External search failed: {str(e)}")
            return DocSearchResult(success=False, error=f"External search failed: {str(e)}", docs={}, doc_count=0)

    async def _doc_search_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute document search with streaming support and action history tracking."""
        try:
            # Document search action
            search_action = ActionHistory(
                action_id="document_search",
                role=ActionRole.WORKFLOW,
                messages="Searching for relevant documentation",
                action_type="document_search",
                input={
                    "keywords": getattr(self.input, "keywords", []),
                    "top_n": getattr(self.input, "top_n", 3),
                },
                status=ActionStatus.PROCESSING,
            )
            yield search_action

            # Execute document search
            result = self._execute_document()

            search_action.status = ActionStatus.SUCCESS if result.success else ActionStatus.FAILED
            search_action.output = {
                "success": result.success,
                "documents_found": len(result.docs) if result.docs else 0,
            }

            # Store result for later use
            self.result = result

            # Yield the updated action with final status
            yield search_action

        except Exception as e:
            logger.error(f"Document search streaming error: {str(e)}")
            raise
