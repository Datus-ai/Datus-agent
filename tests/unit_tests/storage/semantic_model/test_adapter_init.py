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


def _make_agent_config(datasource="test_ns"):
    """Create a mock AgentConfig."""
    config = MagicMock()
    config.current_datasource = datasource
    config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: (
        adapter_type.lower().strip() if adapter_type else None
    )
    config.build_semantic_adapter_config.side_effect = lambda adapter_type=None: {"datasource": datasource}
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
    async def test_dict_adapter_config_datasource_defaulting(self, mock_registry, MockStorageManager):
        """When adapter_config is a dict without datasource, it should be added from base config."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config(datasource="my_ns")

        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt", adapter_config={"timeout_seconds": 60})

            call_kwargs = MockConfig.call_args[1]
            assert call_kwargs["datasource"] == "my_ns"

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_dict_config_preserves_datasource(self, mock_registry, MockStorageManager):
        """Dict config with existing datasource should not be overwritten."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        config = _make_agent_config(datasource="default_ns")

        with patch("datus.tools.semantic_tools.config.SemanticAdapterConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            await init_from_adapter(config, "dbt", adapter_config={"datasource": "custom_ns"})

            call_kwargs = MockConfig.call_args[1]
            assert call_kwargs["datasource"] == "custom_ns"

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
        config = MagicMock(spec=["current_datasource", "datasource_configs", "path_manager"])
        config.current_datasource = "ns1"
        config.datasource_configs = {}
        config.path_manager.semantic_model_path.return_value = "/tmp/project/subject/semantic_models"

        await init_from_adapter(config, "cube")

        mock_config_class.assert_called_once_with(
            datasource="ns1", db_config=None, semantic_models_path="/tmp/project/subject/semantic_models"
        )

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_datasource_falls_back_to_current_datasource(self, mock_registry, MockStorageManager):
        """Should use current_datasource when building adapter config."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
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
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_none_config_uses_build_semantic_adapter_config(self, mock_registry, MockStorageManager):
        """When adapter_config is None, should use agent_config.build_semantic_adapter_config()."""
        from datus.storage.semantic_model.adapter_init import init_from_adapter

        mock_metadata = MagicMock()
        mock_metadata.config_class = None
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.create_adapter.return_value = MagicMock()

        mock_manager = MagicMock()
        mock_manager.sync_from_adapter = AsyncMock(return_value={"semantic_models_synced": 1})
        MockStorageManager.return_value = mock_manager

        existing_config = MagicMock()
        config = _make_agent_config()
        config.build_semantic_adapter_config.side_effect = None
        config.build_semantic_adapter_config.return_value = existing_config

        await init_from_adapter(config, "dbt")

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

        config = _make_agent_config(datasource="ns1")
        await init_from_adapter(config, "metricflow", adapter_config={"timeout_seconds": 60})

        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["datasource"] == "ns1"
        assert call_kwargs["timeout_seconds"] == 60

    @pytest.mark.asyncio
    @patch("datus.storage.semantic_model.adapter_init.SemanticStorageManager")
    @patch("datus.storage.semantic_model.adapter_init.semantic_adapter_registry")
    async def test_none_config_extracts_db_config_from_datasource_configs(self, mock_registry, MockStorageManager):
        """When adapter_config is None and datasource_configs has data, should extract db_config."""
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
