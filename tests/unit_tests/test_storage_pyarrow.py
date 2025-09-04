"""
Test cases for PyArrow-related storage modifications.
Tests the performance improvements and return type changes in storage modules.
"""

import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from datus.storage.document.store import DocumentStore
from datus.storage.embedding_models import get_db_embedding_model, get_metric_embedding_model
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.storage.metric.store import MetricStorage
from datus.storage.schema_metadata.store import SchemaStorage


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for testing storage operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_schema_data():
    """Sample schema data for testing."""
    return [
        {
            "identifier": "1",
            "catalog_name": "test_catalog",
            "database_name": "test_db",
            "schema_name": "test_schema",
            "table_name": "users",
            "table_type": "table",
            "definition": "CREATE TABLE users (id INT, name VARCHAR(50), email VARCHAR(100))",
        },
        {
            "identifier": "2",
            "catalog_name": "test_catalog",
            "database_name": "test_db",
            "schema_name": "test_schema",
            "table_name": "orders",
            "table_type": "table",
            "definition": "CREATE TABLE orders (order_id INT, user_id INT, amount DECIMAL(10,2))",
        },
        {
            "identifier": "3",
            "catalog_name": "test_catalog",
            "database_name": "test_db",
            "schema_name": "test_schema",
            "table_name": "products",
            "table_type": "table",
            "definition": "CREATE TABLE products (product_id INT, name VARCHAR(100), price DECIMAL(8,2))",
        },
    ]


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return [
        {
            "title": "Data Pipeline Best Practices",
            "hierarchy": "Engineering/Data",
            "keywords": ["pipeline", "ETL", "data"],
            "language": "en",
            "chunk_text": "Data pipelines should be designed for reliability and scalability.",
        },
        {
            "title": "SQL Optimization Guide",
            "hierarchy": "Engineering/Database",
            "keywords": ["SQL", "optimization", "performance"],
            "language": "en",
            "chunk_text": "Query optimization is crucial for database performance.",
        },
    ]


@pytest.fixture
def sample_ext_knowledge_data():
    """Sample external knowledge data for testing."""
    return [
        {
            "domain": "Finance",
            "layer1": "Banking",
            "layer2": "Retail",
            "terminology": "APR",
            "explanation": "Annual Percentage Rate - the yearly cost of a loan",
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "domain": "Finance",
            "layer1": "Investment",
            "layer2": "Stocks",
            "terminology": "P/E Ratio",
            "explanation": "Price-to-earnings ratio - a valuation metric",
            "created_at": "2023-01-02T00:00:00Z",
        },
    ]


@pytest.fixture
def sample_metric_data():
    """Sample metric data for testing."""
    return [
        {
            "domain": "Sales",
            "layer1": "Revenue",
            "layer2": "Monthly",
            "name": "monthly_revenue",
            "description": "Total monthly revenue across all channels",
            "constraint": "amount > 0",
            "sql_query": "SELECT SUM(amount) FROM sales WHERE month = CURRENT_MONTH",
            "semantic_model_name": "sales_model",
            "domain_layer1_layer2": "Sales_Revenue_Monthly",
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "domain": "Sales",
            "layer1": "Revenue",
            "layer2": "Daily",
            "name": "daily_revenue",
            "description": "Total daily revenue across all channels",
            "constraint": "amount > 0",
            "sql_query": "SELECT SUM(amount) FROM sales WHERE date = CURRENT_DATE",
            "semantic_model_name": "sales_model",
            "domain_layer1_layer2": "Sales_Revenue_Daily",
            "created_at": "2023-01-02T00:00:00Z",
        },
    ]


class TestSchemaStoragePyArrow:
    """Test PyArrow-related functionality in SchemaStorage."""

    def test_search_similar_returns_pyarrow_table(self, temp_db_path, sample_schema_data):
        """Test that search_similar returns PyArrow Table instead of List[Dict]."""
        storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_schema_data)

        result = storage.search_similar("user table", top_n=2)

        # Verify return type is PyArrow Table
        assert isinstance(result, pa.Table)
        assert result.num_rows <= 2
        assert result.num_rows > 0

        # Verify table has expected columns
        expected_columns = ["catalog_name", "database_name", "schema_name", "table_name", "definition"]
        for col in expected_columns:
            assert col in result.column_names

    def test_search_all_returns_pyarrow_table(self, temp_db_path, sample_schema_data):
        """Test that search_all returns PyArrow Table."""
        storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_schema_data)

        result = storage.search_all(catalog_name="test_catalog")

        assert isinstance(result, pa.Table)
        assert result.num_rows == 3

        # Test filtering capabilities
        catalog_names = result["catalog_name"].to_pylist()
        assert all(name == "test_catalog" for name in catalog_names)

    def test_get_schema_method(self, temp_db_path, sample_schema_data):
        """Test the new get_schema method."""
        storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_schema_data)

        result = storage.get_schema(
            table_name="users", catalog_name="test_catalog", database_name="test_db", schema_name="test_schema"
        )

        assert isinstance(result, pa.Table)
        assert result.num_rows == 1
        assert result["table_name"][0].as_py() == "users"

    def test_pyarrow_performance_with_large_dataset(self, temp_db_path):
        """Test performance with larger dataset using PyArrow operations."""
        storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())

        # Generate larger dataset
        large_dataset = []
        for i in range(100):
            large_dataset.append(
                {
                    "identifier": f"table_{i}",
                    "catalog_name": f"catalog_{i % 5}",
                    "database_name": f"database_{i % 10}",
                    "schema_name": f"schema_{i % 3}",
                    "table_name": f"table_{i}",
                    "table_type": "table",
                    "definition": f"CREATE TABLE table_{i} (id INT, value VARCHAR(100))",
                }
            )

        storage.store(large_dataset)

        # Test search with PyArrow table return
        result = storage.search_all(catalog_name="catalog_1")
        assert isinstance(result, pa.Table)

        # Test PyArrow filtering operations
        filtered = result.filter(pc.equal(result["database_name"], "database_1"))
        assert filtered.num_rows > 0

        # Test column selection
        selected = result.select(["table_name", "definition"])
        assert len(selected.column_names) == 2


class TestDocumentStorePyArrow:
    """Test PyArrow-related functionality in DocumentStore."""

    def test_search_similar_documents_returns_pyarrow_table(self, temp_db_path, sample_document_data):
        """Test that search_similar_documents returns PyArrow Table."""
        storage = DocumentStore(db_path=temp_db_path)
        storage.store(sample_document_data)

        result = storage.search_similar_documents(
            query_text="data pipeline optimization", select_fields=["title", "hierarchy", "chunk_text"], top_n=2
        )

        assert isinstance(result, pa.Table)
        assert result.num_rows <= 2

        # Verify selected fields
        expected_fields = ["title", "hierarchy", "chunk_text"]
        for field in expected_fields:
            assert field in result.column_names

    def test_document_search_with_pyarrow_compute(self, temp_db_path, sample_document_data):
        """Test document search with PyArrow compute operations."""
        storage = DocumentStore(db_path=temp_db_path)
        storage.store(sample_document_data)

        # Search all documents
        all_docs = storage._search_all()
        assert isinstance(all_docs, pa.Table)

        # Use PyArrow compute for filtering
        eng_docs = all_docs.filter(pc.match_substring(all_docs["hierarchy"], "Engineering"))
        assert eng_docs.num_rows > 0

        # Test string operations
        titles = eng_docs["title"]
        upper_titles = pc.utf8_upper(titles)
        assert all(title.isupper() for title in upper_titles.to_pylist())


class TestExtKnowledgeStorePyArrow:
    """Test PyArrow-related functionality in ExtKnowledgeStore."""

    def test_search_similar_knowledge_returns_pyarrow_table(self, temp_db_path, sample_ext_knowledge_data):
        """Test that search_similar_knowledge returns PyArrow Table."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_ext_knowledge_data)

        result = storage.search_knowledge(query_text="financial metrics", domain="Finance", top_n=2)

        assert isinstance(result, pa.Table)
        assert result.num_rows <= 2

    def test_get_all_knowledge_returns_pyarrow_table(self, temp_db_path, sample_ext_knowledge_data):
        """Test that get_all_knowledge returns PyArrow Table."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_ext_knowledge_data)

        result = storage.search_all_knowledge(domain="Finance")

        assert isinstance(result, pa.Table)
        assert result.num_rows == 2

        # Test domain filtering
        domains = result["domain"].to_pylist()
        assert all(domain == "Finance" for domain in domains)

    def test_knowledge_pyarrow_operations(self, temp_db_path, sample_ext_knowledge_data):
        """Test PyArrow operations on knowledge data."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_ext_knowledge_data)

        all_knowledge = storage._search_all()

        # Test grouping by domain
        domains = pc.unique(all_knowledge["domain"])
        assert len(domains) >= 1

        # Test concatenation operations (similar to those used in storage)
        from datus.utils.pyarrow_utils import concat_columns

        domain_layer_concat = concat_columns(all_knowledge, columns=["domain", "layer1", "layer2"], separator="_")

        expected_values = ["Finance_Banking_Retail", "Finance_Investment_Stocks"]
        assert domain_layer_concat.to_pylist() == expected_values


class TestMetricStoragePyArrow:
    """Test PyArrow-related functionality in MetricStorage."""

    def test_search_all_metrics_returns_pyarrow_table(self, temp_db_path, sample_metric_data):
        """Test that search_all_metrics returns PyArrow Table."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.store(sample_metric_data)

        # Simulate SemanticMetricsRAG usage
        class MockSemanticMetricsRAG:
            def __init__(self):
                self.metric_storage = storage

            def search_all_metrics(self, semantic_model_name="", select_fields=None):
                return self.metric_storage.search_all(semantic_model_name, select_fields=select_fields)

        rag = MockSemanticMetricsRAG()
        result = rag.search_all_metrics(select_fields=["name", "description"])

        assert isinstance(result, pa.Table)
        assert result.num_rows == 2
        assert "name" in result.column_names
        assert "description" in result.column_names

    def test_hybrid_metrics_search_with_pyarrow(self, temp_db_path, sample_metric_data):
        """Test hybrid metrics search using PyArrow operations."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.store(sample_metric_data)

        # Get all metrics as PyArrow table
        all_metrics = storage._search_all()

        # Test PyArrow filtering (simulating the filtering logic in search_hybrid_metrics)
        semantic_names_set = {"sales_model"}
        filtered_metrics = all_metrics.filter(
            pc.is_in(all_metrics["semantic_model_name"], pa.array(semantic_names_set))
        )

        assert isinstance(filtered_metrics, pa.Table)
        assert filtered_metrics.num_rows == 2

        # Test column selection
        selected_fields = filtered_metrics.select(["name", "description", "constraint", "sql_query"])
        assert len(selected_fields.column_names) == 4

    def test_metrics_detail_retrieval(self, temp_db_path, sample_metric_data):
        """Test metrics detail retrieval with PyArrow operations."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.store(sample_metric_data)

        # Test direct table querying
        result = storage._search_all(
            where="domain_layer1_layer2 = 'Sales_Revenue_Monthly' and name = 'monthly_revenue'",
            select_fields=["name", "description", "constraint", "sql_query"],
        )

        assert isinstance(result, pa.Table)
        assert result.num_rows == 1
        assert result["name"][0].as_py() == "monthly_revenue"


class TestPyArrowPerformance:
    """Test performance improvements with PyArrow operations."""

    def test_large_scale_concatenation_performance(self, temp_db_path):
        """Test performance of PyArrow concatenation with large datasets."""
        from datus.utils.pyarrow_utils import concat_columns_with_cleaning

        # Create large table
        size = 5000
        large_table = pa.table(
            {
                "domain": [f"Domain_{i % 100}" for i in range(size)],
                "layer1": [f"Layer1_{i % 50}" for i in range(size)],
                "layer2": [f"Layer2_{i % 25}" for i in range(size)],
                "name": [f"metric_{i}" for i in range(size)],
            }
        )

        # Test concatenation performance
        result = concat_columns_with_cleaning(
            large_table, columns=["domain", "layer1", "layer2"], separator="_", replacements={" ": "_", "/": "_"}
        )

        assert len(result) == size
        assert isinstance(result, (pa.Array, pa.ChunkedArray))

        # Verify first and last results
        result_list = result.to_pylist()
        assert result_list[0] == "Domain_0_Layer1_0_Layer2_0"
        assert result_list[-1] == f"Domain_{(size-1) % 100}_Layer1_{(size-1) % 50}_Layer2_{(size-1) % 25}"

    def test_memory_efficient_operations(self, temp_db_path):
        """Test memory-efficient PyArrow operations."""
        # Create schema storage with moderate dataset
        storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())

        datasets = []
        for i in range(500):
            datasets.append(
                {
                    "identifier": f"id_{i}",
                    "catalog_name": f"cat_{i % 10}",
                    "database_name": f"db_{i % 5}",
                    "schema_name": f"schema_{i % 3}",
                    "table_name": f"table_{i}",
                    "table_type": "table",
                    "definition": f"CREATE TABLE table_{i} (id INT, data VARCHAR(1000))",
                }
            )

        storage.store(datasets)

        # Test that operations return PyArrow tables (memory efficient)
        result = storage.search_all()
        assert isinstance(result, pa.Table)
        assert result.num_rows == 500

        # Test column-wise operations
        unique_catalogs = pc.unique(result["catalog_name"])
        assert len(unique_catalogs) == 10

        # Test filtering without converting to Python
        filtered = result.filter(pc.equal(result["catalog_name"], "cat_1"))
        assert filtered.num_rows == 50

    def test_pyarrow_compute_integration(self, temp_db_path, sample_schema_data):
        """Test integration with PyArrow compute functions."""
        storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_schema_data)

        result = storage.search_all()

        # Test various compute operations
        # String operations
        upper_table_names = pc.utf8_upper(result["table_name"])
        assert all(name.isupper() for name in upper_table_names.to_pylist())

        # Filtering operations
        user_tables = result.filter(pc.match_substring(result["table_name"], "user"))
        assert user_tables.num_rows >= 1

        # Aggregation operations
        unique_schemas = pc.unique(result["schema_name"])
        assert len(unique_schemas) >= 1

        # Sorting operations
        sorted_result = result.sort_by([("table_name", "ascending")])
        table_names = sorted_result["table_name"].to_pylist()
        assert table_names == sorted(table_names)


class TestReturnTypeConsistency:
    """Test consistency of return types across all storage modules."""

    def test_all_search_methods_return_pyarrow_tables(self, temp_db_path):
        """Test that all search methods consistently return PyArrow Tables."""
        # Schema storage
        schema_storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        schema_data = [
            {
                "identifier": "1",
                "catalog_name": "cat",
                "database_name": "db",
                "schema_name": "schema",
                "table_name": "table1",
                "table_type": "table",
                "definition": "CREATE TABLE table1 (id INT)",
            }
        ]
        schema_storage.store(schema_data)

        schema_result = schema_storage.search_all()
        assert isinstance(schema_result, pa.Table)

        # Document storage
        doc_storage = DocumentStore(db_path=temp_db_path + "_doc")
        doc_data = [
            {
                "title": "Test Doc",
                "hierarchy": "Test",
                "keywords": ["test"],
                "language": "en",
                "chunk_text": "Test content",
            }
        ]
        doc_storage.store(doc_data)

        doc_result = doc_storage.search_similar_documents("test", top_n=1)
        assert isinstance(doc_result, pa.Table)

        # External knowledge storage
        ext_storage = ExtKnowledgeStore(db_path=temp_db_path + "_ext", embedding_model=get_db_embedding_model())
        ext_data = [
            {
                "domain": "Test",
                "layer1": "L1",
                "layer2": "L2",
                "terminology": "term",
                "explanation": "explanation",
                "created_at": "2023-01-01T00:00:00Z",
            }
        ]
        ext_storage.store(ext_data)

        ext_result = ext_storage.search_all_knowledge()
        assert isinstance(ext_result, pa.Table)

    def test_backwards_compatibility_with_to_pylist(self, temp_db_path, sample_schema_data):
        """Test that PyArrow Tables can be easily converted to previous List[Dict] format."""
        storage = SchemaStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.store(sample_schema_data)

        result = storage.search_all()

        # Convert to old format for backwards compatibility
        old_format = result.to_pylist()
        assert isinstance(old_format, list)
        assert len(old_format) == 3
        assert isinstance(old_format[0], dict)

        # Verify all expected fields are present
        expected_fields = ["catalog_name", "database_name", "schema_name", "table_name"]
        for field in expected_fields:
            assert field in old_format[0]
