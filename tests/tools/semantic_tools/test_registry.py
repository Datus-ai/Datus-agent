# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SemanticAdapterRegistry."""

import pytest

from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.config import SemanticAdapterConfig
from datus.tools.semantic_tools.models import MetricDefinition, QueryResult, ValidationResult
from datus.tools.semantic_tools.registry import AdapterMetadata, SemanticAdapterRegistry
from datus.utils.exceptions import DatusException, ErrorCode


# Mock adapter for testing
class MockSemanticAdapter(BaseSemanticAdapter):
    """Mock adapter for testing."""

    async def list_metrics(self, path=None, limit=100, offset=0):
        return [
            MetricDefinition(
                name="test_metric",
                description="Test metric",
                type="simple",
                dimensions=["date", "region"],
                measures=["count"],
            )
        ]

    async def get_dimensions(self, metric_name, path=None):
        return ["date", "region", "product"]

    async def query_metrics(self, metrics, dimensions=None, **kwargs):
        return QueryResult(
            columns=["date", "metric"],
            data=[{"date": "2024-01-01", "metric": 100}],
            metadata={"execution_time": "0.5s"},
        )

    async def validate_semantic(self):
        return ValidationResult(valid=True, issues=[])


class TestSemanticAdapterRegistry:
    """Test suite for SemanticAdapterRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        # Create a fresh registry instance for testing
        self.registry = SemanticAdapterRegistry()
        self.registry._adapters = {}
        self.registry._factories = {}
        self.registry._metadata = {}
        self.registry._initialized = False

    def test_register_adapter(self):
        """Test registering an adapter."""
        self.registry.register(
            service_type="mock",
            adapter_class=MockSemanticAdapter,
            config_class=SemanticAdapterConfig,
            display_name="Mock Adapter",
        )

        assert self.registry.is_registered("mock")
        assert "mock" in self.registry.list_adapters()
        assert self.registry._adapters["mock"] == MockSemanticAdapter

    def test_register_adapter_case_insensitive(self):
        """Test that adapter registration is case-insensitive."""
        self.registry.register(
            service_type="MockService",
            adapter_class=MockSemanticAdapter,
        )

        assert self.registry.is_registered("mockservice")
        assert self.registry.is_registered("MOCKSERVICE")
        assert self.registry.is_registered("MockService")

    def test_create_adapter(self):
        """Test creating an adapter instance."""
        self.registry.register(
            service_type="mock",
            adapter_class=MockSemanticAdapter,
        )

        config = SemanticAdapterConfig(namespace="test")
        adapter = self.registry.create_adapter("mock", config)

        assert isinstance(adapter, MockSemanticAdapter)
        assert adapter.service_type == "mock"
        assert adapter.config == config

    def test_create_adapter_not_found(self):
        """Test creating an adapter that doesn't exist."""
        config = SemanticAdapterConfig(namespace="test")

        with pytest.raises(DatusException) as exc_info:
            self.registry.create_adapter("nonexistent", config)

        assert exc_info.value.code == ErrorCode.SEMANTIC_ADAPTER_NOT_FOUND

    def test_get_metadata(self):
        """Test getting adapter metadata."""
        self.registry.register(
            service_type="mock",
            adapter_class=MockSemanticAdapter,
            config_class=SemanticAdapterConfig,
            display_name="Mock Adapter",
        )

        metadata = self.registry.get_metadata("mock")

        assert isinstance(metadata, AdapterMetadata)
        assert metadata.service_type == "mock"
        assert metadata.adapter_class == MockSemanticAdapter
        assert metadata.config_class == SemanticAdapterConfig
        assert metadata.display_name == "Mock Adapter"

    def test_get_metadata_not_found(self):
        """Test getting metadata for non-existent adapter."""
        metadata = self.registry.get_metadata("nonexistent")
        assert metadata is None

    def test_list_adapters(self):
        """Test listing all registered adapters."""
        self.registry.register("mock1", MockSemanticAdapter)
        self.registry.register("mock2", MockSemanticAdapter)

        adapters = self.registry.list_adapters()

        assert len(adapters) == 2
        assert "mock1" in adapters
        assert "mock2" in adapters

    def test_list_available_adapters(self):
        """Test listing available adapters with metadata."""
        self.registry.register(
            service_type="mock",
            adapter_class=MockSemanticAdapter,
            display_name="Mock Adapter",
        )

        available = self.registry.list_available_adapters()

        assert "mock" in available
        assert isinstance(available["mock"], AdapterMetadata)

    def test_adapter_metadata_get_config_fields(self):
        """Test extracting config fields from Pydantic model."""
        metadata = AdapterMetadata(
            service_type="mock",
            adapter_class=MockSemanticAdapter,
            config_class=SemanticAdapterConfig,
        )

        fields = metadata.get_config_fields()

        assert "namespace" in fields
        assert "timeout_seconds" in fields
        assert fields["namespace"]["required"] is False
        assert fields["timeout_seconds"]["default"] == 30

    def test_factory_method(self):
        """Test using custom factory method."""

        def custom_factory(config):
            adapter = MockSemanticAdapter(config)
            adapter.custom_flag = True
            return adapter

        self.registry.register(
            service_type="mock",
            adapter_class=MockSemanticAdapter,
            factory=custom_factory,
        )

        config = SemanticAdapterConfig(namespace="test")
        adapter = self.registry.create_adapter("mock", config)

        assert hasattr(adapter, "custom_flag")
        assert adapter.custom_flag is True


@pytest.mark.asyncio
class TestMockAdapter:
    """Test the mock adapter itself."""

    async def test_list_metrics(self):
        """Test listing metrics."""
        config = SemanticAdapterConfig(namespace="test")
        adapter = MockSemanticAdapter(config, "mock")

        metrics = await adapter.list_metrics()

        assert len(metrics) == 1
        assert metrics[0].name == "test_metric"
        assert "date" in metrics[0].dimensions

    async def test_get_dimensions(self):
        """Test getting dimensions."""
        config = SemanticAdapterConfig(namespace="test")
        adapter = MockSemanticAdapter(config, "mock")

        dimensions = await adapter.get_dimensions("test_metric")

        assert len(dimensions) == 3
        assert "date" in dimensions
        assert "region" in dimensions

    async def test_query_metrics(self):
        """Test querying metrics."""
        config = SemanticAdapterConfig(namespace="test")
        adapter = MockSemanticAdapter(config, "mock")

        result = await adapter.query_metrics(
            metrics=["test_metric"],
            dimensions=["date"],
        )

        assert len(result.columns) == 2
        assert len(result.data) == 1
        assert result.metadata["execution_time"] == "0.5s"

    async def test_validate_semantic(self):
        """Test semantic validation."""
        config = SemanticAdapterConfig(namespace="test")
        adapter = MockSemanticAdapter(config, "mock")

        result = await adapter.validate_semantic()

        assert result.valid is True
        assert len(result.issues) == 0
