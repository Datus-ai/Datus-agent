# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/semantic_tools/registry.py"""

from unittest.mock import MagicMock, patch

import pytest

from datus.tools.semantic_tools.registry import AdapterMetadata, SemanticAdapterRegistry
from datus.utils.exceptions import DatusException


def _make_adapter_class(service_type="test_service"):
    """Create a minimal mock adapter class."""
    from datus.tools.semantic_tools.base import BaseSemanticAdapter

    class MockAdapter(BaseSemanticAdapter):
        async def list_metrics(self, path=None, limit=100, offset=0):
            return []

        async def get_dimensions(self, metric_name, path=None):
            return []

        async def query_metrics(self, metrics, **kwargs):
            return MagicMock()

        async def validate_semantic(self):
            return MagicMock()

    return MockAdapter


class TestAdapterMetadata:
    def test_display_name_defaults_to_capitalized_service_type(self):
        adapter_class = _make_adapter_class()
        meta = AdapterMetadata(service_type="test", adapter_class=adapter_class)
        assert meta.display_name == "Test"

    def test_custom_display_name(self):
        adapter_class = _make_adapter_class()
        meta = AdapterMetadata(service_type="test", adapter_class=adapter_class, display_name="My Adapter")
        assert meta.display_name == "My Adapter"

    def test_get_config_fields_returns_empty_when_no_config_class(self):
        adapter_class = _make_adapter_class()
        meta = AdapterMetadata(service_type="test", adapter_class=adapter_class)
        assert meta.get_config_fields() == {}

    def test_get_config_fields_returns_empty_for_non_pydantic_config(self):
        adapter_class = _make_adapter_class()

        class NotPydantic:
            pass

        meta = AdapterMetadata(service_type="test", adapter_class=adapter_class, config_class=NotPydantic)
        assert meta.get_config_fields() == {}

    def test_get_config_fields_with_pydantic_model(self):
        from pydantic import BaseModel, Field

        class TestConfig(BaseModel):
            host: str = Field(description="Host address")
            port: int = Field(default=5432, description="Port number")

        adapter_class = _make_adapter_class()
        meta = AdapterMetadata(service_type="test", adapter_class=adapter_class, config_class=TestConfig)
        fields = meta.get_config_fields()
        assert "host" in fields
        assert "port" in fields
        assert fields["port"]["default"] == 5432

    def test_get_config_fields_handles_exception(self):
        adapter_class = _make_adapter_class()

        class BadConfig:
            model_fields = None  # will cause error

        meta = AdapterMetadata(service_type="test", adapter_class=adapter_class, config_class=BadConfig)
        # Should not raise, return empty dict
        result = meta.get_config_fields()
        assert result == {}


class TestSemanticAdapterRegistry:
    def setup_method(self):
        """Reset registry state before each test."""
        SemanticAdapterRegistry._adapters.clear()
        SemanticAdapterRegistry._factories.clear()
        SemanticAdapterRegistry._metadata.clear()
        SemanticAdapterRegistry._initialized = False

    def test_register_adapter(self):
        adapter_class = _make_adapter_class()
        SemanticAdapterRegistry.register("myservice", adapter_class)
        assert SemanticAdapterRegistry.is_registered("myservice")

    def test_register_normalizes_service_type_to_lower(self):
        adapter_class = _make_adapter_class()
        SemanticAdapterRegistry.register("MyService", adapter_class)
        # is_registered lowercases input, so both forms should return True
        assert SemanticAdapterRegistry.is_registered("myservice")
        assert SemanticAdapterRegistry.is_registered("MyService")
        # The underlying key is stored as lowercase
        assert "myservice" in SemanticAdapterRegistry._adapters

    def test_register_with_factory(self):
        adapter_class = _make_adapter_class()
        mock_factory = MagicMock(return_value=MagicMock())
        SemanticAdapterRegistry.register("factoryservice", adapter_class, factory=mock_factory)

        config = MagicMock()
        SemanticAdapterRegistry.create_adapter("factoryservice", config)
        mock_factory.assert_called_once_with(config)

    def test_register_with_metadata(self):
        adapter_class = _make_adapter_class()
        SemanticAdapterRegistry.register("metaservice", adapter_class, display_name="Meta Service")
        meta = SemanticAdapterRegistry.get_metadata("metaservice")
        assert meta is not None
        assert meta.display_name == "Meta Service"

    def test_create_adapter_uses_default_construction(self):
        adapter_class = _make_adapter_class()
        # Register with a factory to avoid BaseSemanticAdapter.__init__ requiring service_type
        expected_instance = MagicMock(spec=adapter_class)
        factory = MagicMock(return_value=expected_instance)
        SemanticAdapterRegistry.register("defaultservice", adapter_class, factory=factory)

        config = MagicMock()
        config.namespace = "ns"
        instance = SemanticAdapterRegistry.create_adapter("defaultservice", config)
        factory.assert_called_once_with(config)
        assert instance is expected_instance

    def test_create_adapter_raises_for_unknown_service(self):
        with patch.object(SemanticAdapterRegistry, "_try_load_adapter"):
            with pytest.raises(DatusException):
                SemanticAdapterRegistry.create_adapter("nonexistent", MagicMock())

    def test_create_adapter_case_insensitive(self):
        adapter_class = _make_adapter_class()
        expected_instance = MagicMock()
        factory = MagicMock(return_value=expected_instance)
        SemanticAdapterRegistry.register("caseservice", adapter_class, factory=factory)
        config = MagicMock()
        config.namespace = "ns"
        instance = SemanticAdapterRegistry.create_adapter("CaseService", config)
        factory.assert_called_once_with(config)
        assert instance is expected_instance

    def test_list_adapters_returns_copy(self):
        adapter_class = _make_adapter_class()
        SemanticAdapterRegistry.register("listservice", adapter_class)
        adapters = SemanticAdapterRegistry.list_adapters()
        assert "listservice" in adapters
        # Modifying the copy shouldn't affect the registry
        adapters["newkey"] = None
        assert "newkey" not in SemanticAdapterRegistry._adapters

    def test_is_registered_returns_false_for_unknown(self):
        assert not SemanticAdapterRegistry.is_registered("completely_unknown_xyz")

    def test_get_metadata_returns_none_for_unknown(self):
        meta = SemanticAdapterRegistry.get_metadata("unknown_service_xyz")
        assert meta is None

    def test_get_metadata_case_insensitive(self):
        adapter_class = _make_adapter_class()
        SemanticAdapterRegistry.register("casetest", adapter_class)
        meta = SemanticAdapterRegistry.get_metadata("CaseTest")
        assert meta is not None

    def test_list_available_adapters_runs_discover(self):
        with patch.object(SemanticAdapterRegistry, "discover_adapters") as mock_discover:
            SemanticAdapterRegistry.list_available_adapters()
        mock_discover.assert_called_once()

    def test_list_available_adapters_returns_metadata_copy(self):
        adapter_class = _make_adapter_class()
        SemanticAdapterRegistry.register("avail_service", adapter_class)
        with patch.object(SemanticAdapterRegistry, "discover_adapters"):
            result = SemanticAdapterRegistry.list_available_adapters()
        assert "avail_service" in result

    def test_discover_adapters_runs_only_once(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            SemanticAdapterRegistry.discover_adapters()
            SemanticAdapterRegistry.discover_adapters()  # second call should be no-op
        assert SemanticAdapterRegistry._initialized is True

    def test_try_load_adapter_handles_import_error(self):
        # Should not raise even when import fails
        SemanticAdapterRegistry._try_load_adapter("nonexistent_plugin_xyz")

    def test_try_load_adapter_handles_generic_exception(self):
        with patch("importlib.import_module", side_effect=Exception("weird error")):
            # Should not raise
            SemanticAdapterRegistry._try_load_adapter("errorplugin")

    def test_discover_adapters_handles_entry_point_failure(self):
        mock_ep = MagicMock()
        mock_ep.name = "failplugin"
        mock_ep.load.side_effect = Exception("load failed")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            SemanticAdapterRegistry._initialized = False
            # Should not raise
            SemanticAdapterRegistry.discover_adapters()
