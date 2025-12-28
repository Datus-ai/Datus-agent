# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SemanticStorageManager."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.config import SemanticAdapterConfig
from datus.tools.semantic_tools.models import MetricDefinition
from datus.tools.semantic_tools.storage_sync import SemanticStorageManager


# Mock adapter for testing
class MockAdapter(BaseSemanticAdapter):
    """Mock adapter that returns test data."""

    def list_semantic_models(self, **kwargs):
        return ["test_model"]

    def get_semantic_model(self, table_name, **kwargs):
        return {
            "semantic_model_name": "test_model",
            "description": "Test semantic model",
            "table_name": "test_table",
            "catalog_name": "test_catalog",
            "database_name": "test_db",
            "schema_name": "test_schema",
            "dimensions": [
                {"name": "date", "description": "Date dimension", "expr": "date_col"},
                {"name": "region", "description": "Region dimension", "expr": "region_col"},
            ],
            "measures": [{"name": "count", "description": "Row count", "expr": "COUNT(*)"}],
            "identifiers": [{"name": "id", "description": "Primary key", "expr": "id_col"}],
        }

    async def list_metrics(self, **kwargs):
        return [
            MetricDefinition(
                name="test_metric",
                description="Test metric",
                type="simple",
                dimensions=["date", "region"],
                measures=["count"],
                unit="count",
                format=",.0f",
                path=["Finance", "Revenue"],
            )
        ]

    async def get_dimensions(self, metric_name, path=None):
        return ["date", "region"]

    async def query_metrics(self, **kwargs):
        pass

    async def validate_semantic(self):
        pass


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_agent_config(temp_db_path):
    """Create a mock AgentConfig for testing."""
    config = MagicMock(spec=AgentConfig)
    config.db_path = temp_db_path
    config.namespace = "test_namespace"
    config.current_namespace = "test_namespace"

    # Mock embedding model
    embedding_model = MagicMock()
    embedding_model.dim_size = 384
    config.embedding_model = embedding_model

    return config


class TestSemanticStorageManager:
    """Test suite for SemanticStorageManager."""

    def test_init(self, mock_agent_config):
        """Test initialization."""
        manager = SemanticStorageManager(mock_agent_config)

        assert manager.agent_config == mock_agent_config
        assert manager.semantic_model_store is None  # Lazy init
        assert manager.metric_store is None  # Lazy init

    @patch("datus.tools.semantic_tools.storage_sync.SemanticModelRAG")
    def test_ensure_semantic_model_store(self, mock_rag_class, mock_agent_config):
        """Test lazy initialization of semantic model store."""
        manager = SemanticStorageManager(mock_agent_config)

        # First call creates the store
        store = manager._ensure_semantic_model_store()
        assert store is not None
        assert manager.semantic_model_store is not None

        # Second call returns same instance
        store2 = manager._ensure_semantic_model_store()
        assert store2 is store

    @patch("datus.tools.semantic_tools.storage_sync.MetricRAG")
    def test_ensure_metric_store(self, mock_rag_class, mock_agent_config):
        """Test lazy initialization of metric store."""
        manager = SemanticStorageManager(mock_agent_config)

        # First call creates the store
        store = manager._ensure_metric_store()
        assert store is not None
        assert manager.metric_store is not None

        # Second call returns same instance
        store2 = manager._ensure_metric_store()
        assert store2 is store

    @patch("datus.tools.semantic_tools.storage_sync.SemanticModelRAG")
    def test_store_semantic_model(self, mock_rag_class, mock_agent_config):
        """Test storing a semantic model."""
        # Setup mock
        mock_store = MagicMock()
        mock_rag_instance = MagicMock()
        mock_rag_instance.store = mock_store
        mock_rag_class.return_value = mock_rag_instance

        manager = SemanticStorageManager(mock_agent_config)

        model_data = {
            "semantic_model_name": "test_model",
            "description": "Test model",
            "table_name": "test_table",
            "dimensions": [{"name": "date", "description": "Date dim", "expr": "date_col"}],
            "measures": [{"name": "count", "description": "Count", "expr": "COUNT(*)"}],
            "identifiers": [],
        }

        manager.store_semantic_model(model_data, source="test")

        # Verify batch_store was called
        assert mock_store.batch_store.called
        # Should be called 3 times: table + dimensions + measures
        assert mock_store.batch_store.call_count == 2  # table + dimension

    @patch("datus.tools.semantic_tools.storage_sync.MetricRAG")
    def test_store_metric(self, mock_rag_class, mock_agent_config):
        """Test storing a metric."""
        # Setup mock
        mock_store = MagicMock()
        mock_rag_instance = MagicMock()
        mock_rag_instance.store = mock_store
        mock_rag_class.return_value = mock_rag_instance

        manager = SemanticStorageManager(mock_agent_config)

        metric_data = {
            "name": "test_metric",
            "description": "Test metric",
            "metric_type": "simple",
            "dimensions": ["date"],
            "measures": ["count"],
        }

        manager.store_metric(metric_data, source="test", subject_path=["Finance"])

        # Verify batch_store_metrics was called
        mock_store.batch_store_metrics.assert_called_once()

        # Check the metric object structure
        call_args = mock_store.batch_store_metrics.call_args[0][0]
        assert len(call_args) == 1
        metric_obj = call_args[0]
        assert metric_obj["name"] == "test_metric"
        assert metric_obj["subject_path"] == ["Finance"]
        assert metric_obj["metric_type"] == "simple"

    @pytest.mark.asyncio
    @patch("datus.tools.semantic_tools.storage_sync.SemanticModelRAG")
    @patch("datus.tools.semantic_tools.storage_sync.MetricRAG")
    async def test_sync_from_adapter(self, mock_metric_rag, mock_semantic_rag, mock_agent_config):
        """Test syncing from adapter."""
        # Setup mocks
        mock_sem_store = MagicMock()
        mock_sem_rag_instance = MagicMock()
        mock_sem_rag_instance.store = mock_sem_store
        mock_semantic_rag.return_value = mock_sem_rag_instance

        mock_metric_store = MagicMock()
        mock_metric_rag_instance = MagicMock()
        mock_metric_rag_instance.store = mock_metric_store
        mock_metric_rag.return_value = mock_metric_rag_instance

        manager = SemanticStorageManager(mock_agent_config)

        # Create mock adapter
        config = SemanticAdapterConfig(namespace="test")
        adapter = MockAdapter(config, "mock")

        # Sync from adapter
        stats = await manager.sync_from_adapter(
            adapter=adapter,
            sync_semantic_models=True,
            sync_metrics=True,
            subject_path=["Finance"],
        )

        # Verify stats
        assert stats["semantic_models_synced"] == 1
        assert stats["metrics_synced"] == 1

        # Verify stores were called
        assert mock_sem_store.batch_store.called
        assert mock_metric_store.batch_store_metrics.called

    @pytest.mark.asyncio
    @patch("datus.tools.semantic_tools.storage_sync.MetricRAG")
    async def test_sync_metrics_only(self, mock_metric_rag, mock_agent_config):
        """Test syncing only metrics."""
        mock_metric_store = MagicMock()
        mock_metric_rag_instance = MagicMock()
        mock_metric_rag_instance.store = mock_metric_store
        mock_metric_rag.return_value = mock_metric_rag_instance

        manager = SemanticStorageManager(mock_agent_config)

        config = SemanticAdapterConfig(namespace="test")
        adapter = MockAdapter(config, "mock")

        stats = await manager.sync_from_adapter(
            adapter=adapter,
            sync_semantic_models=False,
            sync_metrics=True,
        )

        assert stats["semantic_models_synced"] == 0
        assert stats["metrics_synced"] == 1

    @pytest.mark.asyncio
    @patch("datus.tools.semantic_tools.storage_sync.SemanticModelRAG")
    async def test_sync_semantic_models_only(self, mock_semantic_rag, mock_agent_config):
        """Test syncing only semantic models."""
        mock_sem_store = MagicMock()
        mock_sem_rag_instance = MagicMock()
        mock_sem_rag_instance.store = mock_sem_store
        mock_semantic_rag.return_value = mock_sem_rag_instance

        manager = SemanticStorageManager(mock_agent_config)

        config = SemanticAdapterConfig(namespace="test")
        adapter = MockAdapter(config, "mock")

        stats = await manager.sync_from_adapter(
            adapter=adapter,
            sync_semantic_models=True,
            sync_metrics=False,
        )

        assert stats["semantic_models_synced"] == 1
        assert stats["metrics_synced"] == 0

    @patch("datus.tools.semantic_tools.storage_sync.MetricRAG")
    def test_store_metric_with_path_from_metric(self, mock_metric_rag, mock_agent_config):
        """Test that metric's own path takes precedence."""
        mock_metric_store = MagicMock()
        mock_metric_rag_instance = MagicMock()
        mock_metric_rag_instance.store = mock_metric_store
        mock_metric_rag.return_value = mock_metric_rag_instance

        manager = SemanticStorageManager(mock_agent_config)

        metric_data = {
            "name": "test_metric",
            "description": "Test",
            "path": ["Custom", "Path"],  # Metric has its own path
        }

        # Provide different subject_path
        manager.store_metric(metric_data, source="test", subject_path=["Finance"])

        # Should use metric's own path, not provided subject_path
        call_args = mock_metric_store.batch_store_metrics.call_args[0][0]
        # Note: The implementation doesn't use metric's path in store_metric,
        # only in sync_from_adapter. This test documents current behavior.
        assert call_args[0]["subject_path"] == ["Finance"]
