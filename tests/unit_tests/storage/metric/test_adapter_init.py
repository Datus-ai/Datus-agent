# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.metric.adapter_init."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(datasource="test_ns", adapter_type_config=None):
    """Create a mock AgentConfig."""
    config = MagicMock()
    config.current_datasource = datasource
    config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: (
        adapter_type.lower().strip() if adapter_type else None
    )
    config.build_semantic_adapter_config.side_effect = lambda adapter_type=None: {"datasource": datasource}
    # By default, no adapter-specific config on agent_config
    if adapter_type_config is not None:
        for key, val in adapter_type_config.items():
            setattr(config, key, val)
    return config


# ---------------------------------------------------------------------------
# init_from_adapter
# ---------------------------------------------------------------------------


class TestInitFromAdapter:
    """Tests for init_from_adapter in metric/adapter_init.py."""

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_successful_sync(self, mock_registry, MockStorageManager):
        """Should return (True, '') when metrics are synced successfully."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata

        mock_adapter = MagicMock()
        mock_registry.create_adapter.return_value = mock_adapter

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 5})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        success, error = await init_from_adapter(config, "metricflow")

        assert success is True
        assert error == ""
        mock_manager.sync_from_adapter.assert_called_once()

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_zero_metrics_returns_failure(self, mock_registry, MockStorageManager):
        """Should return (False, ...) when no metrics found."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 0})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        success, error = await init_from_adapter(config, "dbt")

        assert success is False
        assert "No metrics found" in error

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_adapter_type_normalized(self, mock_registry, MockStorageManager):
        """Adapter type should be normalized to lowercase."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 3})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        await init_from_adapter(config, "  MetricFlow  ")

        mock_registry.get_metadata.assert_called_with("metricflow")

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_exception_returns_failure(self, mock_registry, MockStorageManager):
        """Exceptions should be caught and returned as failure."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_registry.get_metadata.side_effect = Exception("Registry broken")

        config = _make_agent_config()
        success, error = await init_from_adapter(config, "broken")

        assert success is False
        assert "Failed to sync metrics" in error

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_dict_adapter_config_sets_datasource(self, mock_registry, MockStorageManager):
        """When adapter_config is a dict, datasource should be set from base config."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config(datasource="my_ns")

        # Pass a dict config without datasource - it should be auto-set from base config
        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt", adapter_config={"timeout_seconds": 60})

            # Verify datasource was added to the dict before creating config
            MockConfig.assert_called_once()
            call_kwargs = MockConfig.call_args[1]
            assert call_kwargs["datasource"] == "my_ns"

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_dict_config_preserves_existing_datasource(self, mock_registry, MockStorageManager):
        """When adapter_config dict already has datasource, it should not be overwritten."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config(datasource="default_ns")

        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt", adapter_config={"datasource": "custom_ns"})

            call_kwargs = MockConfig.call_args[1]
            assert call_kwargs["datasource"] == "custom_ns"

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_none_config_with_metadata_config_class(self, mock_registry, MockStorageManager):
        """When no adapter_config and metadata has a config_class, should use it."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_metadata = MagicMock()
        mock_metadata.config_class = mock_config_class
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 2})
        MockStorageManager.return_value = mock_manager

        # agent_config must NOT have metricflow_config so code falls through
        # to the metadata.config_class branch. Use spec to restrict attributes.
        config = MagicMock(spec=["current_datasource", "datasource_configs", "path_manager"])
        config.current_datasource = "ns1"
        config.datasource_configs = {}
        config.path_manager.semantic_model_path.return_value = "/tmp/project/subject/semantic_models"

        await init_from_adapter(config, "metricflow")

        mock_config_class.assert_called_once_with(
            datasource="ns1", db_config=None, semantic_models_path="/tmp/project/subject/semantic_models"
        )

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_subject_path_passed_to_sync(self, mock_registry, MockStorageManager):
        """subject_path should be forwarded to sync_from_adapter."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        await init_from_adapter(config, "dbt", subject_path=["Finance", "Revenue"])

        call_kwargs = mock_manager.sync_from_adapter.call_args[1]
        assert call_kwargs["subject_path"] == ["Finance", "Revenue"]
        assert call_kwargs["sync_metrics"] is True
        assert call_kwargs["sync_semantic_models"] is False

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_datasource_from_current_datasource_fallback(self, mock_registry, MockStorageManager):
        """Should use current_datasource when building adapter config."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 1})
        MockStorageManager.return_value = mock_manager

        # Use spec to prevent auto-generated attributes like dbt_config
        config = MagicMock(spec=["current_datasource", "datasource_configs", "path_manager"])
        config.current_datasource = "fallback_ns"
        config.datasource_configs = {}
        config.path_manager.semantic_model_path.return_value = "/tmp/project/subject/semantic_models"

        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt")

            MockConfig.assert_called_once_with(datasource="fallback_ns")

    @pytest.mark.asyncio
    @patch("datus.storage.metric.adapter_init.SemanticStorageManager")
    @patch("datus.storage.metric.adapter_init.semantic_adapter_registry")
    async def test_none_config_extracts_db_config_from_datasource_configs(self, mock_registry, MockStorageManager):
        """When adapter_config is None and datasource_configs has data, should extract db_config."""
        from datus.storage.metric.adapter_init import init_from_adapter

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_metadata = MagicMock()
        mock_metadata.config_class = mock_config_class
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"metrics_synced": 2})
        MockStorageManager.return_value = mock_manager

        # Set up agent_config with datasource_configs containing a DbConfig
        mock_db_config = MagicMock()
        mock_db_config.to_dict.return_value = {
            "db_type": "mysql",
            "host": "localhost",
            "port": 3306,
            "username": "root",
            "password": "pass",
            "database": "testdb",
            "extra": "ignore_me",
            "logic_name": "ignore_me_too",
        }

        config = MagicMock(spec=["current_datasource", "datasource_configs", "path_manager"])
        config.current_datasource = "ns1"
        config.datasource_configs = {"ns1": {"default": mock_db_config}}
        config.path_manager.semantic_model_path.return_value = "/tmp/project/subject/semantic_models"

        await init_from_adapter(config, "metricflow")

        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["datasource"] == "ns1"
        assert call_kwargs["semantic_models_path"] == "/tmp/project/subject/semantic_models"
        # db_config should contain stringified values, excluding "extra" and "logic_name"
        db_config = call_kwargs["db_config"]
        assert db_config["db_type"] == "mysql"
        assert db_config["host"] == "localhost"
        assert db_config["port"] == "3306"
        assert "extra" not in db_config
        assert "logic_name" not in db_config
