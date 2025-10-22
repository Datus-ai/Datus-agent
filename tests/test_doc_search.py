import os
from pathlib import Path
from typing import Any, Dict

import pytest

from datus.agent.node import Node
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.storage.document.doc_init import import_documents
from datus.storage.document.store import document_store
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)


@pytest.fixture
def document_input() -> Dict[str, Any]:
    """Test data for document search"""
    return {"keywords": ["basketball", "NCAA"], "top_n": 3, "method": "internal"}


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_acceptance_config(namespace="snowflake")


class TestDocumentNode:
    """Test suite for DocumentNode class"""

    def test_node_initialization(self):
        """Test node initialization"""
        input_data = DocSearchInput(keywords=["basketball"], top_n=5, method="internal")

        node = Node.new_instance(
            node_id="doc_search",
            description="Document Search",
            node_type=NodeType.TYPE_DOC_SEARCH,
            input_data=input_data,
        )

        assert node.id == "doc_search"
        assert node.description == "Document Search"
        assert node.type == NodeType.TYPE_DOC_SEARCH
        assert isinstance(node, Node)
        assert isinstance(node.input, DocSearchInput)

    def test_internal_search(self, document_input, agent_config: AgentConfig):
        """Test internal document search functionality"""
        doc_dir = Path(__file__).parent.parent / "benchmark" / "spider2" / "spider2-snow" / "resource" / "documents"
        if not doc_dir.exists():
            pytest.skip(f"Documents directory not found: {doc_dir}")

        import_documents(document_store(agent_config.rag_storage_path()), str(doc_dir))

        input_data = DocSearchInput(**document_input)
        node = Node.new_instance(
            node_id="doc_search",
            description="Document Search",
            node_type=NodeType.TYPE_DOC_SEARCH,
            input_data=input_data,
            agent_config=agent_config,
        )

        node.run()
        result = node.result
        assert isinstance(result, DocSearchResult)
        assert (
            result.success
        ), f"Internal search failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        assert len(result.docs) > 0, "No documents were found"

        for keyword, docs in result.docs.items():
            logger.info(f"Found {len(docs)} documents for keyword '{keyword}'")
            if docs:
                logger.info(f"First document sample: {docs[0][:200]}...")

    def test_external_search(self, agent_config: AgentConfig):
        """Test external document search using TAVILY_API"""
        if not os.environ.get("TAVILY_API_KEY"):
            pytest.skip("TAVILY_API_KEY environment variable not set")

        input_data = DocSearchInput(keywords=["Snowflake official documentation"], top_n=3, method="external")

        node = Node.new_instance(
            node_id="doc_search_external",
            description="External Document Search",
            node_type=NodeType.TYPE_DOC_SEARCH,
            input_data=input_data,
            agent_config=agent_config,
        )

        node.run()
        result = node.result
        assert isinstance(result, DocSearchResult)
        assert (
            result.success
        ), f"External search failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        assert len(result.docs) > 0, "No documents were found"
        for keyword, docs in result.docs.items():
            logger.info(f"Found {len(docs)} documents for keyword '{keyword}'")
            if docs:
                logger.info(f"First document sample: {docs[0][:200]}...")
