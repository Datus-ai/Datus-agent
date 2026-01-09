# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Initialize semantic models from semantic adapters.
"""

from typing import Optional

from datus.configuration.agent_config import AgentConfig
from datus.tools.semantic_tools.registry import semantic_adapter_registry
from datus.tools.semantic_tools.storage_sync import SemanticStorageManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def init_from_adapter(
    agent_config: AgentConfig,
    adapter_type: str,
    adapter_config: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Pull semantic models from adapter and sync to storage.

    Workflow:
    1. Get adapter from registry
    2. Call adapter.list_semantic_models()
    3. For each model: adapter.get_semantic_model()
    4. Sync to SemanticModelStorage via SemanticStorageManager

    Args:
        agent_config: Agent configuration
        adapter_type: Type of adapter (e.g., "metricflow", "dbt", "cube")
        adapter_config: Optional adapter-specific configuration dict

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Normalize adapter_type to lowercase for registry lookup
        adapter_type = adapter_type.lower().strip()

        # Get namespace from agent_config
        namespace = getattr(agent_config, "namespace", None) or agent_config.current_namespace

        # Get the registered config class for this adapter type
        metadata = semantic_adapter_registry.get_metadata(adapter_type)

        # Build adapter config
        if adapter_config is None:
            # Try to get config from agent_config if available
            adapter_config = getattr(agent_config, f"{adapter_type}_config", None)

        if adapter_config is None:
            # No config provided, create default config
            if metadata and metadata.config_class:
                adapter_config = metadata.config_class(namespace=namespace)
            else:
                from datus.tools.semantic_tools.config import SemanticAdapterConfig

                adapter_config = SemanticAdapterConfig(namespace=namespace)
        elif isinstance(adapter_config, dict):
            # Convert dict to adapter-specific config class
            # Ensure namespace is set
            if "namespace" not in adapter_config:
                adapter_config["namespace"] = namespace

            if metadata and metadata.config_class:
                adapter_config = metadata.config_class(**adapter_config)
            else:
                from datus.tools.semantic_tools.config import SemanticAdapterConfig

                adapter_config = SemanticAdapterConfig(**adapter_config)

        adapter = semantic_adapter_registry.create_adapter(adapter_type, adapter_config)

        # Create storage manager
        storage_manager = SemanticStorageManager(agent_config)

        # Sync semantic models only
        stats = await storage_manager.sync_from_adapter(
            adapter=adapter,
            sync_semantic_models=True,
            sync_metrics=False,
        )

        models_count = stats["semantic_models_synced"]
        if models_count == 0:
            return False, f"No semantic models found in {adapter_type} adapter"

        logger.info(f"Successfully synced {models_count} semantic models from {adapter_type}")
        return True, ""

    except Exception as e:
        error_msg = f"Failed to sync semantic models from {adapter_type}: {str(e)}"
        logger.exception(error_msg)
        return False, error_msg
