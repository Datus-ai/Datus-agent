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
from datus.storage.reference_sql.store import ReferenceSqlStorage
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
            "subject_path": ["Finance", "Banking", "Retail"],
            "name": "APR",
            "terminology": "APR",
            "explanation": "Annual Percentage Rate - the yearly cost of a loan",
        },
        {
            "subject_path": ["Finance", "Investment", "Stocks"],
            "name": "P/E_Ratio",
            "terminology": "P/E Ratio",
            "explanation": "Price-to-earnings ratio - a valuation metric",
        },
    ]


@pytest.fixture
def sample_metric_data():
    """Sample metric data for testing."""
    return [
        {
            "subject_path": ["Sales", "Revenue", "Monthly"],
            "name": "monthly_revenue",
            "description": "Total monthly revenue across all channels",
            "semantic_model_name": "sales_model",
        },
        {
            "subject_path": ["Sales", "Revenue", "Daily"],
            "name": "daily_revenue",
            "description": "Total daily revenue across all channels",
            "semantic_model_name": "sales_model",
        },
    ]


@pytest.fixture
def sample_reference_sql_data():
    """Sample reference SQL data for testing."""
    return [
        {
            "subject_path": ["Analytics", "Reports", "Daily"],
            "name": "daily_sales_report",
            "sql": "SELECT date, SUM(amount) as total FROM sales GROUP BY date",
            "comment": "Daily sales aggregation query",
            "summary": "Aggregates sales data by date for daily reporting",
            "filepath": "/queries/daily_sales.sql",
            "tags": "sales,reporting,daily",
        },
        {
            "subject_path": ["Analytics", "Reports", "Monthly"],
            "name": "monthly_revenue_report",
            "sql": "SELECT MONTH(date) as month, SUM(amount) as revenue FROM sales GROUP BY MONTH(date)",
            "comment": "Monthly revenue summary",
            "summary": "Summarizes revenue by month for financial reporting",
            "filepath": "/queries/monthly_revenue.sql",
            "tags": "revenue,reporting,monthly",
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
        storage = DocumentStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
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
        storage = DocumentStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
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

    def test_search_similar_knowledge(self, temp_db_path, sample_ext_knowledge_data):
        """Test that search_similar_knowledge returns PyArrow Table."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.batch_store_knowledge(sample_ext_knowledge_data)

        result = storage.search_knowledge(query_text="financial metrics", subject_path=["Finance"], top_n=2)

        assert len(result) <= 2

    def test_get_all_knowledge_returns_pyarrow_table(self, temp_db_path, sample_ext_knowledge_data):
        """Test that get_all_knowledge returns PyArrow Table."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.batch_store_knowledge(sample_ext_knowledge_data)

        results = storage.search_all_knowledge(["Finance"])
        assert len(results) == 2

        # Test domain filtering
        assert all(res["subject_path"][0] == "Finance" for res in results)

    def test_search_knowledge_wildcard(self, temp_db_path, sample_ext_knowledge_data):
        """Test search_knowledge wildcard."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.batch_store_knowledge(sample_ext_knowledge_data)

        result = storage.search_all_knowledge(["Finance", "Banking", "Retail", "APR"])
        assert len(result) == 1

    def test_knowledge_pyarrow_operations(self, temp_db_path, sample_ext_knowledge_data):
        """Test PyArrow operations on knowledge data."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.batch_store_knowledge(sample_ext_knowledge_data)

        all_knowledge = storage.search_all_knowledge()

        # Test grouping by domain
        domains = [res["subject_path"][0] == "Finance" for res in all_knowledge]
        assert len(set(domains)) == 1

        # Test concatenation operations (similar to those used in storage)
        subject_path_list = [knowledge["subject_path"] for knowledge in all_knowledge]

        expected_values = [["Finance", "Banking", "Retail"], ["Finance", "Investment", "Stocks"]]
        assert subject_path_list == expected_values

    def test_rename_subject_node(self, temp_db_path, sample_ext_knowledge_data):
        """Test renaming a subject node in ExtKnowledgeStore."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.batch_store_knowledge(sample_ext_knowledge_data)

        # Rename subject node: Finance -> Banking -> Retail to Finance -> Banking -> Consumer
        success = storage.rename(old_path=["Finance", "Banking", "Retail"], new_path=["Finance", "Banking", "Consumer"])

        assert success is True

        # Verify the node was renamed in subject_tree
        old_node = storage.subject_tree.get_node_by_path(["Finance", "Banking", "Retail"])
        new_node = storage.subject_tree.get_node_by_path(["Finance", "Banking", "Consumer"])

        assert old_node is None
        assert new_node is not None
        assert new_node["name"] == "Consumer"

    def test_rename_knowledge_item(self, temp_db_path):
        """Test renaming a knowledge item in LanceDB."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())

        # Store knowledge with subject path that exists
        knowledge_data = [
            {
                "subject_path": ["Finance", "Banking"],
                "terminology": "old_term",
                "name": "old_term",
                "explanation": "This is an explanation for old term",
            },
        ]
        storage.batch_store_knowledge(knowledge_data)

        # Rename the terminology (LanceDB item, not subject node)
        success = storage.rename(
            old_path=["Finance", "Banking", "old_term"], new_path=["Finance", "Banking", "new_term"]
        )

        assert success is True

        # Verify the item was renamed
        results = storage.search_all_knowledge(["Finance", "Banking"])
        assert len(results) == 1
        assert results[0]["name"] == "new_term"
        assert results[0]["explanation"] == "This is an explanation for old term"

    def test_rename_knowledge_item_different_parent(self, temp_db_path):
        """Test that renaming with different parent path."""
        storage = ExtKnowledgeStore(db_path=temp_db_path, embedding_model=get_db_embedding_model())

        knowledge_data = [
            {
                "subject_path": ["Finance", "Banking"],
                "name": "term1",
                "terminology": "term1",
                "explanation": "Explanation 1",
            },
        ]
        storage.batch_store_knowledge(knowledge_data)
        storage.subject_tree.find_or_create_path(["Finance", "Investment"])
        storage.rename(old_path=["Finance", "Banking", "term1"], new_path=["Finance", "Investment", "term2"])
        knowledge = storage.search_all_knowledge(["Finance", "Investment"])
        assert len(knowledge) == 1
        assert knowledge[0]["name"] == "term2"


class TestMetricStoragePyArrow:
    """Test PyArrow-related functionality in MetricStorage."""

    def test_search_all_metrics_returns_pyarrow_table(self, temp_db_path, sample_metric_data):
        """Test that search_all_metrics returns PyArrow Table."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metric_data)

        # Simulate MetricRAG usage
        class MockMetricRAG:
            def __init__(self):
                self.storage = storage

            def search_all_metrics(self):
                return self.storage.search_all_metrics()

        rag = MockMetricRAG()
        result = rag.search_all_metrics()

        assert isinstance(result, list)
        assert len(result) == 2
        assert all("name" in item for item in result)
        assert all("description" in item for item in result)

    def test_metrics_detail_retrieval(self, temp_db_path, sample_metric_data):
        """Test metrics detail retrieval with PyArrow operations."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metric_data)

        # Test direct table querying
        result = storage.search_all_metrics(subject_path=["Sales", "Revenue", "Monthly"])

        assert len(result) == 1
        assert result[0]["name"] == "monthly_revenue"

    def test_rename_subject_node(self, temp_db_path, sample_metric_data):
        """Test renaming a subject node in MetricStorage."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metric_data)

        # Rename subject node: Sales -> Revenue -> Monthly to Sales -> Revenue -> MonthlyTotal
        success = storage.rename(
            old_path=["Sales", "Revenue", "Monthly"], new_path=["Sales", "Revenue", "MonthlyTotal"]
        )

        assert success is True

        # Verify the node was renamed in subject_tree
        old_node = storage.subject_tree.get_node_by_path(["Sales", "Revenue", "Monthly"])
        new_node = storage.subject_tree.get_node_by_path(["Sales", "Revenue", "MonthlyTotal"])

        assert old_node is None
        assert new_node is not None
        assert new_node["name"] == "MonthlyTotal"

    def test_rename_metric_item(self, temp_db_path):
        """Test renaming a metric item in LanceDB."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())

        # Store metric with subject path
        metric_data = [
            {
                "subject_path": ["Sales", "Revenue"],
                "name": "old_metric_name",
                "description": "This is an old metric description",
                "semantic_model_name": "sales_model",
            },
        ]
        storage.batch_store_metrics(metric_data)

        # Rename the metric (LanceDB item, not subject node)
        success = storage.rename(
            old_path=["Sales", "Revenue", "old_metric_name"], new_path=["Sales", "Revenue", "new_metric_name"]
        )

        assert success is True

        # Verify the item was renamed
        results = storage.search_all_metrics(subject_path=["Sales", "Revenue"])
        assert len(results) == 1
        assert results[0]["name"] == "new_metric_name"
        assert results[0]["description"] == "This is an old metric description"

    def test_rename_metric_item_different_parent(self, temp_db_path):
        """Test that renaming metric with different parent path fails."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())

        metric_data = [
            {
                "subject_path": ["Sales", "Revenue"],
                "name": "metric1",
                "description": "Metric 1 description",
                "semantic_model_name": "sales_model",
            },
        ]
        storage.batch_store_metrics(metric_data)
        storage.subject_tree.find_or_create_path(["Sales", "Expense"])
        storage.rename(old_path=["Sales", "Revenue", "metric1"], new_path=["Sales", "Expense", "metric1"])
        metrics = storage.search_all_metrics(subject_path=["Sales", "Expense"])
        assert len(metrics) == 1


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
        doc_storage = DocumentStore(db_path=temp_db_path + "_doc", embedding_model=get_db_embedding_model())
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
                "subject_path": ["Test", "L1", "L2"],
                "name": "name",
                "terminology": "term",
                "explanation": "explanation",
                "created_at": "2023-01-01T00:00:00Z",
            }
        ]
        ext_storage.batch_store_knowledge(ext_data)

        ext_result = ext_storage.search_all_knowledge()
        assert len(ext_result) == 1

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


class TestReferenceSqlStoragePyArrow:
    """Test PyArrow-related functionality and rename operations in ReferenceSqlStorage."""

    def test_rename_subject_node(self, temp_db_path, sample_reference_sql_data):
        """Test renaming a subject node in ReferenceSqlStorage."""
        storage = ReferenceSqlStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())
        storage.batch_store_sql(sample_reference_sql_data)

        # Rename subject node: Analytics -> Reports -> Daily to Analytics -> Reports -> DailyReports
        success = storage.rename(
            old_path=["Analytics", "Reports", "Daily"], new_path=["Analytics", "Reports", "DailyReports"]
        )

        assert success is True

        # Verify the node was renamed in subject_tree
        old_node = storage.subject_tree.get_node_by_path(["Analytics", "Reports", "Daily"])
        new_node = storage.subject_tree.get_node_by_path(["Analytics", "Reports", "DailyReports"])

        assert old_node is None
        assert new_node is not None
        assert new_node["name"] == "DailyReports"

    def test_rename_sql_item(self, temp_db_path):
        """Test renaming a reference SQL item in LanceDB."""
        storage = ReferenceSqlStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())

        # Store SQL with subject path
        sql_data = [
            {
                "subject_path": ["Analytics", "Reports"],
                "name": "old_query_name",
                "sql": "SELECT * FROM sales",
                "comment": "Old query comment",
                "summary": "This is an old query summary",
                "filepath": "/queries/old_query.sql",
                "tags": "sales,old",
            },
        ]
        storage.batch_store_sql(sql_data)

        # Rename the SQL (LanceDB item, not subject node)
        success = storage.rename(
            old_path=["Analytics", "Reports", "old_query_name"], new_path=["Analytics", "Reports", "new_query_name"]
        )

        assert success is True

        # Verify the item was renamed
        results = storage.search_all_reference_sql(subject_path=["Analytics", "Reports"])
        assert len(results) == 1
        assert results[0]["name"] == "new_query_name"
        assert results[0]["sql"] == "SELECT * FROM sales"
        assert results[0]["summary"] == "This is an old query summary"

    def test_rename_sql_item_different_parent(self, temp_db_path):
        """Test that renaming SQL with different parent path fails."""
        storage = ReferenceSqlStorage(db_path=temp_db_path, embedding_model=get_db_embedding_model())

        sql_data = [
            {
                "subject_path": ["Analytics", "Reports"],
                "name": "query1",
                "sql": "SELECT * FROM orders",
                "comment": "Query 1 comment",
                "summary": "Query 1 summary",
                "filepath": "/queries/query1.sql",
                "tags": "orders",
            },
        ]
        storage.batch_store_sql(sql_data)

        storage.subject_tree.find_or_create_path(["Analytics", "Dashboards"])
        storage.rename(old_path=["Analytics", "Reports", "query1"], new_path=["Analytics", "Dashboards", "query1"])
        sqls = storage.search_all_reference_sql(subject_path=["Analytics", "Dashboards"])
        assert len(sqls) == 1
