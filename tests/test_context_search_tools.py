"""
Test cases for ContextSearchTools class in datus/tools/tools.py
"""

from unittest.mock import Mock, patch

import pyarrow as pa
import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.context_search import ContextSearchTools
from datus.tools.tools import FuncToolResult


@pytest.fixture
def mock_agent_config():
    """Create a mock agent configuration."""
    config = Mock(spec=AgentConfig)
    config.rag_storage_path.return_value = "/tmp/test_rag_storage"
    return config


@pytest.fixture
def mock_schema_rag():
    """Create a mock schema RAG instance."""
    mock_rag = Mock()
    mock_rag.search_similar.return_value = (Mock(), Mock())
    mock_rag.schema_store = Mock()
    return mock_rag


@pytest.fixture
def mock_metric_rag():
    """Create a mock metric RAG instance."""
    mock_rag = Mock()
    mock_rag.search_hybrid_metrics.return_value = [
        {
            "name": "monthly_revenue",
            "description": "Monthly revenue metric",
            "sql_query": "SELECT SUM(amount) FROM sales",
        }
    ]
    return mock_rag


@pytest.fixture
def mock_doc_rag():
    """Create a mock document RAG instance."""
    mock_rag = Mock()
    mock_table = Mock()
    mock_table.to_pylist.return_value = [
        {"title": "API Documentation", "hierarchy": "Engineering/API", "chunk_text": "API endpoints documentation"}
    ]
    mock_rag.search_similar_documents.return_value = mock_table
    return mock_rag


@pytest.fixture
def mock_ext_knowledge_rag():
    """Create a mock external knowledge RAG instance."""
    mock_rag = Mock()

    mock_rag.search_knowledge.return_value = pa.Table.from_pylist(
        [{"terminology": "CLV", "explanation": "Customer Lifetime Value", "domain": "Marketing"}]
    )
    return mock_rag


@pytest.fixture
def mock_sql_history_rag():
    """Create a mock external knowledge RAG instance."""
    mock_rag = Mock()
    mock_rag.sql_history_storage.return_value = [
        {
            "id": "sql_history1",
            "name": "test sql",
            "domain": "Marketing",
            "layer1": "layer1",
            "layer2": "layer2",
            "sql": "SELECT SUM(amount) FROM sales",
            "comment": "test sql",
            "summary": "test sql",
            "filepath": "/tmp/test_rag_storage",
            "tags": "test, sql",
        },
    ]
    return mock_rag


@pytest.fixture
def context_search_tools(
    mock_agent_config, mock_schema_rag, mock_metric_rag, mock_doc_rag, mock_ext_knowledge_rag, mock_sql_history_rag
):
    """Create ContextSearchTools instance with mocked dependencies."""
    with patch(
        "datus.tools.context_search.schema_metadata_by_configuration", return_value=mock_schema_rag
    ) as mock_schema, patch(
        "datus.tools.context_search.metrics_rag_by_configuration", return_value=mock_metric_rag
    ) as mock_metric, patch(
        "datus.tools.context_search.DocumentStore", return_value=mock_doc_rag
    ) as mock_doc_store, patch(
        "datus.tools.context_search.ext_knowledge_by_configuration", return_value=mock_ext_knowledge_rag
    ) as mock_ext, patch(
        "datus.tools.context_search.sql_history_rag_by_configuration", return_value=mock_sql_history_rag
    ) as mock_sql_history:
        tools = ContextSearchTools(mock_agent_config)

        # Verify all RAG instances were created
        assert tools.schema_rag == mock_schema_rag
        assert tools.metric_rag == mock_metric_rag
        assert tools.doc_rag == mock_doc_rag
        assert tools.ext_knowledge_rag == mock_ext_knowledge_rag
        assert tools.sql_history_store == mock_sql_history_rag

        mock_schema.assert_called_once_with(mock_agent_config)
        mock_metric.assert_called_once_with(mock_agent_config)
        mock_doc_store.assert_called_once_with("/tmp/test_rag_storage")
        mock_ext.assert_called_once_with(mock_agent_config)
        mock_sql_history.assert_called_once_with(mock_agent_config)

        return tools


class TestContextSearchTools:
    """Test cases for ContextSearchTools class."""

    def test_initialization(self, context_search_tools, mock_agent_config):
        """Test that ContextSearchTools initializes correctly with all RAG components."""
        assert context_search_tools is not None
        assert hasattr(context_search_tools, "schema_rag")
        assert hasattr(context_search_tools, "metric_rag")
        assert hasattr(context_search_tools, "doc_rag")
        assert hasattr(context_search_tools, "ext_knowledge_rag")

    def test_available_tools(self, context_search_tools):
        """Test that available_tools returns the correct list of tools."""
        tools = context_search_tools.available_tools()

        assert len(tools) == 5  # Should have 5 tools
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "search_table_metadata",
            "search_metrics",
            "search_documents",
            "search_external_knowledge",
            "search_historical_sql",
        ]

        for expected_tool in expected_tools:
            assert any(expected_tool in name for name in tool_names)

    def test_search_table_metadata_success(self, context_search_tools, mock_schema_rag):
        """Test successful search_table_metadata execution."""
        # Setup mock return values
        mock_metadata = Mock()
        mock_sample_values = Mock()

        mock_metadata.select.return_value.to_pylist.return_value = [
            {"table_name": "users", "table_type": "table", "definition": "CREATE TABLE users (id INT)"}
        ]
        mock_sample_values.select.return_value.to_pylist.return_value = [
            {"identifier": "users", "table_type": "table", "sample_rows": [{"id": 1, "name": "test"}]}
        ]

        mock_schema_rag.search_similar.return_value = (mock_metadata, mock_sample_values)

        # Execute the method
        result = context_search_tools.search_table_metadata(
            query_text="user data",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="test_schema",
            top_n=5,
        )

        # Verify the result
        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert result.result is not None

        # Verify the search was called with correct parameters
        mock_schema_rag.search_similar.assert_called_once_with(
            "user data",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="test_schema",
            table_type="full",
            top_n=5,
        )

    def test_search_table_metadata_failure(self, context_search_tools, mock_schema_rag):
        """Test search_table_metadata with exception."""
        mock_schema_rag.search_similar.side_effect = Exception("Search failed")

        result = context_search_tools.search_table_metadata(query_text="user data", top_n=5)

        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert "Search failed" in result.error
        assert result.result is None

    def test_search_metrics_success(self, context_search_tools, mock_metric_rag):
        """Test successful search_metrics execution."""
        # Execute the method
        result = context_search_tools.search_metrics(
            query_text="revenue metrics",
            domain="Sales",
            layer1="Revenue",
            layer2="Monthly",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="test_schema",
            top_n=5,
        )

        # Verify the result
        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert len(result.result) == 1
        assert result.result[0]["name"] == "monthly_revenue"

        # Verify the metric RAG was called correctly
        mock_metric_rag.search_hybrid_metrics.assert_called_once_with(
            domain="Sales",
            layer1="Revenue",
            layer2="Monthly",
            query_text="revenue metrics",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="test_schema",
            top_n=5,
        )

    def test_search_metrics_failure(self, context_search_tools, mock_metric_rag):
        """Test search_metrics with exception."""
        mock_metric_rag.search_hybrid_metrics.side_effect = Exception("Metric search failed")

        result = context_search_tools.search_metrics(
            query_text="revenue metrics", domain="Sales", layer1="Revenue", layer2="Monthly"
        )

        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert "Metric search failed" in result.error
        assert result.result is None

    def test_search_external_knowledge_success(self, context_search_tools, mock_ext_knowledge_rag):
        """Test successful search_external_knowledge execution."""
        # Execute the method
        result = context_search_tools.search_external_knowledge(
            query_text="customer lifetime value", domain="Marketing", layer1="Analytics", layer2="Customer", top_n=5
        )

        # Verify the result
        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert len(result.result) == 1
        assert result.result[0]["terminology"] == "CLV"

        # Verify the external knowledge RAG was called correctly
        mock_ext_knowledge_rag.search_knowledge.assert_called_once_with(
            query_text="customer lifetime value", domain="Marketing", layer1="Analytics", layer2="Customer", top_n=5
        )

    def test_search_external_knowledge_failure(self, context_search_tools, mock_ext_knowledge_rag):
        """Test search_external_knowledge with exception."""
        mock_ext_knowledge_rag.search_knowledge.side_effect = Exception("Knowledge search failed")

        result = context_search_tools.search_external_knowledge(
            query_text="customer lifetime value", domain="Marketing", layer1="Analytics", layer2="Customer"
        )

        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert "Knowledge search failed" in result.error
        assert result.result is None

    def test_search_documents_success(self, context_search_tools, mock_doc_rag):
        """Test successful search_documents execution."""
        # Execute the method
        result = context_search_tools.search_documents(query_text="API documentation", top_n=5)

        # Verify the result
        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None
        assert len(result.result) == 1
        assert result.result[0]["title"] == "API Documentation"

        # Verify the document RAG was called correctly
        mock_doc_rag.search_similar_documents.assert_called_once_with(
            query_text="API documentation",
            top_n=5,
            select_fields=["title", "hierarchy", "keywords", "language", "chunk_text"],
        )

    def test_search_documents_failure(self, context_search_tools, mock_doc_rag):
        """Test search_documents with exception."""
        mock_doc_rag.search_similar_documents.side_effect = Exception("Document search failed")

        result = context_search_tools.search_documents(query_text="API documentation")

        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert "Document search failed" in result.error
        assert result.result is None

    def test_search_historical_sql(self, context_search_tools):
        result = context_search_tools.search_historical_sql(
            query_text="revenue query", domain="Sales", layer1="Revenue", layer2="Monthly"
        )

        # Should return FuncToolResult with success=0 since it's not implemented
        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.result is not None


class TestContextSearchToolsEdgeCases:
    """Test edge cases and error conditions for ContextSearchTools."""

    def test_empty_search_results(self, context_search_tools, mock_schema_rag):
        """Test search_table_metadata with empty results."""
        # Setup empty mock return values
        mock_metadata = Mock()
        mock_sample_values = Mock()

        mock_metadata.select.return_value.to_pylist.return_value = []
        mock_sample_values.select.return_value.to_pylist.return_value = []

        mock_schema_rag.search_similar.return_value = (mock_metadata, mock_sample_values)

        result = context_search_tools.search_table_metadata(query_text="nonexistent data")

        assert result.success == 1
        assert result.result["metadata"] == []
        assert result.result["sample_data"] == []

    def test_partial_search_results(self, context_search_tools, mock_schema_rag):
        """Test search_table_metadata with partial results (only metadata, no sample data)."""
        mock_metadata = Mock()
        mock_sample_values = Mock()

        mock_metadata.select.return_value.to_pylist.return_value = [{"table_name": "users", "table_type": "table"}]
        mock_sample_values.select.return_value.to_pylist.return_value = []

        mock_schema_rag.search_similar.return_value = (mock_metadata, mock_sample_values)

        result = context_search_tools.search_table_metadata(query_text="user data")

        assert result.success == 1
        assert len(result.result["metadata"]) == 1
        assert result.result["sample_data"] == []

    def test_error_handling_different_exceptions(self, context_search_tools, mock_schema_rag):
        """Test that different types of exceptions are handled properly."""
        test_cases = [
            (ValueError("Invalid parameter"), "Invalid parameter"),
            (RuntimeError("Connection failed"), "Connection failed"),
            (Exception("Generic error"), "Generic error"),
        ]

        for exception, expected_error in test_cases:
            mock_schema_rag.search_similar.side_effect = exception

            result = context_search_tools.search_table_metadata(query_text="test query")

            assert result.success == 0
            assert expected_error in result.error

    def test_default_parameter_values(self, context_search_tools, mock_schema_rag):
        """Test that methods use default parameter values correctly."""
        # Test search_table_metadata with defaults
        context_search_tools.search_table_metadata(query_text="test")

        mock_schema_rag.search_similar.assert_called_with(
            "test", catalog_name="", database_name="", schema_name="", table_type="full", top_n=5
        )

    def test_method_return_types(self, context_search_tools):
        """Test that all methods return FuncToolResult instances."""
        methods_to_test = [
            lambda: context_search_tools.search_table_metadata("test"),
            lambda: context_search_tools.search_metrics("test", "Sales", "Revenue", "Monthly"),
            lambda: context_search_tools.search_external_knowledge("test", "Marketing", "Analytics", "Customer"),
            lambda: context_search_tools.search_documents("test"),
        ]

        for method in methods_to_test:
            result = method()
            assert isinstance(result, FuncToolResult)


class TestContextSearchToolsIntegration:
    """Integration-style tests for ContextSearchTools."""

    def test_tool_transformation_integration(self, context_search_tools):
        """Test that tools can be transformed properly."""
        from datus.tools.tools import trans_to_function_tool

        # Test that the search_table_metadata method can be transformed
        tool = trans_to_function_tool(context_search_tools.search_table_metadata)

        assert tool is not None
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "params_json_schema")

        # Verify the schema doesn't contain 'self'
        schema = tool.params_json_schema
        if isinstance(schema, dict):
            assert "self" not in schema.get("properties", {})
            if "required" in schema:
                assert "self" not in schema["required"]

    def test_multiple_concurrent_searches(self, context_search_tools, mock_schema_rag):
        """Test that multiple search methods can be called concurrently."""
        import threading

        results = {}
        errors = {}

        def search_thread(thread_id):
            try:
                result = context_search_tools.search_table_metadata(query_text=f"test query {thread_id}")
                results[thread_id] = result
            except Exception as e:
                errors[thread_id] = str(e)

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=search_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all threads completed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify all results are FuncToolResult instances
        for result in results.values():
            assert isinstance(result, FuncToolResult)
