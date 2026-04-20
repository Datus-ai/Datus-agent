# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.semantic_model.adapter_init."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(namespace="test_ns"):
    """Create a mock AgentConfig."""
    config = MagicMock()
    config.namespace = namespace
    config.current_database = namespace
    return config


# ---------------------------------------------------------------------------
# init_from_adapter
# ---------------------------------------------------------------------------


class TestInitFromAdapter:
    """Tests for init_from_adapter in semantic_model/adapter_init.py."""

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_successful_sync(self, mock_registry, MockStorageManager):
        """Should return (True, '') when models are synced successfully."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 3})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        success, error = await init_from_adapter(config, "metricflow")

        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_zero_models_returns_failure(self, mock_registry, MockStorageManager):
        """Should return (False, ...) when no models found."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 0})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        success, error = await init_from_adapter(config, "dbt")

        assert success is False
        assert "No semantic models found" in error

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_adapter_type_normalized(self, mock_registry, MockStorageManager):
        """Adapter type should be lowercased and stripped."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        await init_from_adapter(config, "  DBS  ")

        mock_registry.get_metadata.assert_called_with("dbs")

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_exception_returns_failure(self, mock_registry, MockStorageManager):
        """Exceptions should be caught and returned as failure."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_registry.get_metadata.side_effect = Exception("Registry broken")

        config = _make_agent_config()
        success, error = await init_from_adapter(config, "broken")

        assert success is False
        assert "Failed to sync semantic models" in error

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_sync_semantic_models_true_metrics_false(self, mock_registry, MockStorageManager):
        """Should sync semantic models only, not metrics."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config()
        await init_from_adapter(config, "dbt")

        call_kwargs = mock_manager.sync_from_adapter.call_args[1]
        assert call_kwargs["sync_semantic_models"] is True
        assert call_kwargs["sync_metrics"] is False

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_dict_adapter_config_namespace_defaulting(self, mock_registry, MockStorageManager):
        """When adapter_config is a dict without namespace, it should be added."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config(namespace="my_ns")

        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt", adapter_config={"timeout_seconds": 60})

            call_kwargs = MockConfig.call_args[1]
            assert call_kwargs["namespace"] == "my_ns"

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_dict_config_preserves_namespace(self, mock_registry, MockStorageManager):
        """Dict config with existing namespace should not be overwritten."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config(namespace="default_ns")

        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt", adapter_config={"namespace": "custom_ns"})

            call_kwargs = MockConfig.call_args[1]
            assert call_kwargs["namespace"] == "custom_ns"

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_none_config_uses_metadata_config_class(self, mock_registry, MockStorageManager):
        """When no adapter_config and metadata has config_class, should use it."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_metadata = MagicMock()
        mock_metadata.config_class = mock_config_class
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 2})
        MockStorageManager.return_value = mock_manager

        # agent_config must NOT have cube_config so code falls through
        # to the metadata.config_class branch. Use spec to restrict attributes.
        config = MagicMock(spec=["namespace", "current_database", "namespaces", "path_manager"])
        config.namespace = "ns1"
        config.current_database = "ns1"
        config.namespaces = {}
        config.path_manager.semantic_models_dir = "/tmp/project/subject/semantic_models"

        await init_from_adapter(config, "cube")

        mock_config_class.assert_called_once_with(
            namespace="ns1", db_config=None, semantic_models_path="/tmp/project/subject/semantic_models"
        )

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_namespace_falls_back_to_current_database(self, mock_registry, MockStorageManager):
        """Should use current_database when namespace attr is None."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        # Use spec to prevent auto-generated attributes like dbt_config
        config = MagicMock(spec=["namespace", "current_database", "namespaces", "path_manager"])
        config.namespace = None
        config.current_database = "fallback_ns"
        config.namespaces = {}
        config.path_manager.semantic_models_dir = "/tmp/project/subject/semantic_models"

        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt")

            MockConfig.assert_called_once_with(namespace="fallback_ns")

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_none_config_falls_back_to_agent_config_attr(self, mock_registry, MockStorageManager):
        """When adapter_config is None, should try agent_config.{adapter_type}_config."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        # agent_config has dbt_config attribute set to a config object
        existing_config = MagicMock()
        config = _make_agent_config()
        config.dbt_config = existing_config

        await init_from_adapter(config, "dbt")

        # The existing config object should be passed directly to create_adapter
        mock_registry.create_adapter.assert_called_once_with("dbt", existing_config)

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_dict_config_with_metadata_config_class(self, mock_registry, MockStorageManager):
        """Dict config should use metadata.config_class if available."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_metadata = MagicMock()
        mock_metadata.config_class = mock_config_class
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config(namespace="ns1")
        await init_from_adapter(config, "metricflow", adapter_config={"timeout_seconds": 60})

        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["namespace"] == "ns1"
        assert call_kwargs["timeout_seconds"] == 60

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_none_config_extracts_db_config_from_namespaces(self, mock_registry, MockStorageManager):
        """When adapter_config is None and namespaces has data, should extract db_config."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_metadata = MagicMock()
        mock_metadata.config_class = mock_config_class
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 2})
        MockStorageManager.return_value = mock_manager

        # Set up agent_config with namespaces containing a DbConfig
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

        config = MagicMock(spec=["namespace", "current_database", "namespaces", "path_manager"])
        config.namespace = "ns1"
        config.current_database = "ns1"
        config.namespaces = {"ns1": {"default": mock_db_config}}
        config.path_manager.semantic_models_dir = "/tmp/project/subject/semantic_models"

        await init_from_adapter(config, "metricflow")

        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["namespace"] == "ns1"
        assert call_kwargs["semantic_models_path"] == "/tmp/project/subject/semantic_models"
        # db_config should contain stringified values, excluding "extra" and "logic_name"
        db_config = call_kwargs["db_config"]
        assert db_config["db_type"] == "mysql"
        assert db_config["host"] == "localhost"
        assert db_config["port"] == "3306"
        assert "extra" not in db_config
        assert "logic_name" not in db_config
