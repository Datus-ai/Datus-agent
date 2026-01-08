"""
Integration tests for PyArrow modifications in storage modules.
Tests the integration between PyArrow utilities and storage operations.
"""

import tempfile
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from datus.configuration.agent_config import AgentConfig
from datus.storage.embedding_models import get_metric_embedding_model
from datus.storage.metric.store import MetricRAG, MetricStorage


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_agent_config(temp_db_path):
    """Create a mock agent configuration."""
    config = Mock(spec=AgentConfig)
    config.rag_storage_path.return_value = temp_db_path
    return config


@pytest.fixture
def sample_semantic_models():
    """Sample semantic model data."""
    return [
        {
            "semantic_model_name": "sales_model",
            "database_name": "main_db",
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "semantic_model_name": "user_model",
            "database_name": "main_db",
            "created_at": "2023-01-02T00:00:00Z",
        },
    ]


@pytest.fixture
def sample_metrics_with_domain_layers():
    """Sample metrics data with domain layer concatenation."""
    return [
        {
            "subject_path": ["Sales", "Revenue", "Monthly"],
            "name": "monthly_revenue",
            "description": "Monthly revenue across all channels",
            "semantic_model_name": "sales_model",
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "subject_path": ["Sales", "Revenue", "Daily"],
            "name": "daily_revenue",
            "description": "Daily revenue across all channels",
            "semantic_model_name": "sales_model",
            "created_at": "2023-01-02T00:00:00Z",
        },
        {
            "subject_path": ["Marketing", "Campaigns", "Performance"],
            "name": "campaign_ctr",
            "description": "Campaign click-through rate",
            "semantic_model_name": "user_model",
            "created_at": "2023-01-03T00:00:00Z",
        },
    ]


class TestMetricRAGPyArrow:
    """Test PyArrow integration in MetricRAG."""

    def test_search_all_metrics_returns_pyarrow_table(self, temp_db_path, sample_metrics_with_domain_layers):
        """Test that search_all_metrics returns PyArrow Table."""
        metric_storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        metric_storage.batch_store_metrics(sample_metrics_with_domain_layers)

        # Mock cache for MetricRAG
        rag = MetricRAG.__new__(MetricRAG)
        rag.storage = metric_storage

        result = rag.search_all_metrics()

        assert len(result) == 3
        assert all(element["name"] for element in result)
        assert all(element["description"] for element in result)

    def test_hybrid_search_with_pyarrow_filtering(
        self, temp_db_path, sample_metrics_with_domain_layers, sample_semantic_models
    ):
        """Test hybrid search using PyArrow filtering operations."""
        # Setup storages
        metric_storage = MetricStorage(db_path=temp_db_path + "_metrics", embedding_model=get_metric_embedding_model())
        metric_storage.batch_store_metrics(sample_metrics_with_domain_layers)

        semantic_storage = Mock()
        semantic_search_result = pa.table({"semantic_model_name": ["sales_model", "sales_model"]})
        semantic_storage.search.return_value = semantic_search_result

        # Setup RAG
        rag = MetricRAG.__new__(MetricRAG)
        rag.storage = metric_storage

        # Test the filtering logic that uses PyArrow compute
        all_metrics = pa.Table.from_pylist(metric_storage.search_all_metrics())

        # Simulate the filtering done in search_metrics
        semantic_names_set = semantic_search_result["semantic_model_name"].unique()

        filtered_metrics = all_metrics.select(["name", "description"]).filter(
            pc.is_in(all_metrics["semantic_model_name"], semantic_names_set)
        )

        assert isinstance(filtered_metrics, pa.Table)
        assert filtered_metrics.num_rows == 2  # Only metrics with sales_model

        # Verify the filtering worked correctly
        result_list = filtered_metrics.to_pylist()
        metric_names = [item["name"] for item in result_list]
        assert "monthly_revenue" in metric_names
        assert "daily_revenue" in metric_names
        assert "campaign_ctr" not in metric_names  # Different semantic model

    def test_get_metrics_detail_with_compound_where_clause(self, temp_db_path, sample_metrics_with_domain_layers):
        """Test metrics detail retrieval with compound WHERE clauses."""
        metric_storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        metric_storage.batch_store_metrics(sample_metrics_with_domain_layers)

        rag = MetricRAG.__new__(MetricRAG)
        rag.storage = metric_storage

        # Test the get_metrics_detail method functionality
        result = rag.get_metrics_detail(subject_path=["Sales", "Revenue", "Monthly"], name="monthly_revenue")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "monthly_revenue"
        assert "Monthly revenue across all channels" in result[0]["description"]

    def test_domain_layer_concatenation_consistency(self, temp_db_path):
        """Test that domain_layer concatenation is consistent with PyArrow utilities."""
        # Create metrics data that needs domain layer concatenation
        raw_metrics = [
            {
                "subject_path": ["Test Domain", "Test/Layer1", "Test Layer2"],
                "name": "test_metric",
                "description": "Test metric description",
                "semantic_model_name": "test_model",
                "created_at": "2023-01-01T00:00:00Z",
            }
        ]

        # Store and verify
        metric_storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        metric_storage.batch_store_metrics(raw_metrics)

        # Verify the stored data can be filtered correctly
        result = metric_storage.search_all_metrics(
            subject_path=["Test Domain", "Test/Layer1", "Test Layer2"],
        )

        assert len(result) == 1
        assert result[0]["subject_path"][0] == "Test Domain"


class TestPyArrowComputeIntegration:
    """Test integration with PyArrow compute functions."""

    def test_complex_filtering_operations(self, temp_db_path, sample_metrics_with_domain_layers):
        """Test complex filtering operations using PyArrow compute."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metrics_with_domain_layers)

        all_metrics_list = storage.search_all_metrics()

        # Test multiple filtering conditions
        all_metrics = pa.Table.from_pylist(all_metrics_list)
        subject_path_0 = pc.list_element(all_metrics["subject_path"], 0)
        subject_path_2 = pc.list_element(all_metrics["subject_path"], 2)
        sales_metrics = pc.filter(all_metrics, pc.equal(subject_path_0, "Sales"))
        assert sales_metrics.num_rows == 2

        # Test string pattern matching
        revenue_metrics = all_metrics.filter(pc.match_substring(all_metrics["name"], "revenue"))
        assert revenue_metrics.num_rows == 2

        # Test combining filters
        daily_sales = all_metrics.filter(
            pc.and_(pc.equal(subject_path_0, "Sales"), pc.match_substring(subject_path_2, "Daily"))
        )
        assert daily_sales.num_rows == 1
        assert daily_sales["name"][0].as_py() == "daily_revenue"

    def test_aggregation_operations(self, temp_db_path, sample_metrics_with_domain_layers):
        """Test aggregation operations on PyArrow tables."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metrics_with_domain_layers)

        all_metrics = pa.Table.from_pylist(storage.search_all_metrics())

        # Test unique values
        unique_domains = pc.unique(pc.list_element(all_metrics["subject_path"], 0))
        unique_domains_list = unique_domains.to_pylist()
        assert "Sales" in unique_domains_list
        assert "Marketing" in unique_domains_list
        assert len(unique_domains_list) == 2

        # Test counting
        domain_counts = pc.value_counts(pc.list_element(all_metrics["subject_path"], 0))
        values = pc.struct_field(domain_counts, [0])
        counts = pc.struct_field(domain_counts, [1])
        counts_dict = dict(zip(values.to_pylist(), counts.to_pylist()))
        assert counts_dict["Sales"] == 2
        assert counts_dict["Marketing"] == 1

    def test_string_operations_on_metadata(self, temp_db_path, sample_metrics_with_domain_layers):
        """Test string operations on metadata fields."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metrics_with_domain_layers)

        all_metrics = pa.Table.from_pylist(storage.search_all_metrics())

        # Test string transformations
        upper_names = pc.utf8_upper(all_metrics["name"])
        upper_names_list = upper_names.to_pylist()
        assert "MONTHLY_REVENUE" in upper_names_list
        assert "DAILY_REVENUE" in upper_names_list

        # Test string length
        name_lengths = pc.utf8_length(all_metrics["name"])
        lengths_list = name_lengths.to_pylist()
        assert all(length > 0 for length in lengths_list)

        # Test string replacement
        cleaned_descriptions = pc.replace_substring(all_metrics["description"], "revenue", "income")
        cleaned_list = cleaned_descriptions.to_pylist()
        assert any("income" in desc for desc in cleaned_list)

    def test_sorting_and_ordering(self, temp_db_path, sample_metrics_with_domain_layers):
        """Test sorting operations on PyArrow tables."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metrics_with_domain_layers)

        all_metrics = pa.Table.from_pylist(storage.search_all_metrics())

        # Test sorting by name
        sorted_by_name = all_metrics.sort_by([("name", "ascending")])
        sorted_names = sorted_by_name["name"].to_pylist()
        assert sorted_names == sorted(sorted_names)


class TestPerformanceOptimizations:
    """Test performance optimizations with PyArrow."""

    def test_large_dataset_operations(self, temp_db_path):
        """Test operations on larger datasets for performance."""
        # Generate larger dataset
        size = 1000
        large_dataset = []

        for i in range(size):
            large_dataset.append(
                {
                    "subject_path": [f"Domain_{i % 10}", f"Layer1_{i % 5}", f"Layer2_{i % 3}"],
                    "name": f"metric_{i}",
                    "description": f"Description for metric {i}",
                    "semantic_model_name": f"model_{i % 20}",
                    "created_at": "2023-01-01T00:00:00Z",
                }
            )

        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(large_dataset)

        # Test that operations complete efficiently
        import time

        start_time = time.time()
        all_metrics = pa.Table.from_pylist(storage.search_all_metrics())
        end_time = time.time()

        assert (end_time - start_time) < 2.0  # Should complete quickly
        assert all_metrics.num_rows == size

        # Test filtering performance
        start_time = time.time()
        filtered = all_metrics.filter(pc.equal(pc.list_element(all_metrics["subject_path"], 0), "Domain_1"))
        end_time = time.time()

        assert (end_time - start_time) < 1.0  # Filtering should be fast
        assert filtered.num_rows == size // 10  # Every 10th item

    def test_memory_efficient_operations(self, temp_db_path):
        """Test memory-efficient operations with PyArrow."""
        # Create dataset with larger text fields
        dataset = []
        for i in range(500):
            dataset.append(
                {
                    "subject_path": ["LargeDomain", "LargeLayer1", "LargeLayer2"],
                    "name": f"large_metric_{i}",
                    "description": f"This is a very long description for metric {i}. " * 20,
                    "semantic_model_name": f"large_model_{i % 10}",
                    "created_at": "2023-01-01T00:00:00Z",
                }
            )

        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(dataset)

        # Test that we can work with subsets without loading everything
        all_metrics = pa.Table.from_pylist(storage.search_all_metrics())

        # Test slicing for memory efficiency
        first_100 = all_metrics.slice(0, 100)
        assert first_100.num_rows == 100

        # Test column selection for memory efficiency
        minimal_cols = all_metrics.select(["name", "subject_path"])
        assert len(minimal_cols.column_names) == 2
        assert minimal_cols.num_rows == 500

    def test_concurrent_read_operations(self, temp_db_path, sample_metrics_with_domain_layers):
        """Test concurrent read operations on PyArrow tables."""
        storage = MetricStorage(db_path=temp_db_path, embedding_model=get_metric_embedding_model())
        storage.batch_store_metrics(sample_metrics_with_domain_layers)

        import concurrent.futures

        results = []
        errors = []

        def concurrent_reader(thread_id):
            try:
                all_metrics = pa.Table.from_pylist(storage.search_all_metrics())
                filtered = all_metrics.filter(pc.equal(pc.list_element(all_metrics["subject_path"], 0), "Sales"))
                results.append((thread_id, filtered.num_rows))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run multiple concurrent readers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_reader, f"thread_{i}") for i in range(10)]
            concurrent.futures.wait(futures)

        # Verify all operations completed successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # Verify all results are consistent
        for _, row_count in results:
            assert row_count == 2  # Should always be 2 Sales metrics
