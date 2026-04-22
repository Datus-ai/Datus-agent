# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Initialize metrics from semantic adapters.
"""

from typing import List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.tools.semantic_tools.registry import semantic_adapter_registry
from datus.tools.semantic_tools.storage_sync import SemanticStorageManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def init_from_adapter(
    agent_config: AgentConfig,
    adapter_type: str,
    subject_path: Optional[List[str]] = None,
    adapter_config: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Pull metrics from adapter and sync to storage.

    Workflow:
    1. Get adapter from registry
    2. Call adapter.list_metrics()
    3. Sync to MetricStorage via SemanticStorageManager
    4. Optionally apply subject_tree categorization

    Args:
        agent_config: Agent configuration
        adapter_type: Type of adapter (e.g., "metricflow", "dbt", "cube")
        subject_path: Subject tree path for categorization (e.g., ["Finance", "Revenue"])
        adapter_config: Optional adapter-specific configuration dict

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Normalize adapter_type to lowercase for registry lookup
        adapter_type = adapter_type.lower().strip()
        resolver = getattr(agent_config, "resolve_semantic_adapter", None)
        if callable(resolver):
            adapter_type = resolver(adapter_type) or adapter_type

        namespace = getattr(agent_config, "namespace", None) or agent_config.current_datasource

        # Get the registered config class for this adapter type
        metadata = semantic_adapter_registry.get_metadata(adapter_type)
        builder = getattr(agent_config, "build_semantic_adapter_config", None)
        base_config = builder(adapter_type) if callable(builder) else None

        # Build adapter config
        if adapter_config is None:
            adapter_config = base_config
        elif isinstance(adapter_config, dict):
            # Convert dict to adapter-specific config class
            adapter_config = {**(base_config or {}), **adapter_config}

        if adapter_config is None:
            ns_configs = agent_config.namespaces.get(namespace)
            db_config = None
            if ns_configs:
                db_config_obj = list(ns_configs.values())[0]
                raw = db_config_obj.to_dict()
                db_config = {
                    k: str(v)
                    for k, v in raw.items()
                    if v is not None and v != "" and k not in ("extra", "logic_name", "path_pattern", "catalog")
                }
            semantic_models_path = str(agent_config.path_manager.semantic_models_dir)

            if metadata and metadata.config_class:
                adapter_config = metadata.config_class(
                    namespace=namespace,
                    db_config=db_config,
                    semantic_models_path=semantic_models_path,
                )
            else:
                from datus.tools.semantic_tools.config import SemanticAdapterConfig

                adapter_config = SemanticAdapterConfig(namespace=namespace)

        if isinstance(adapter_config, dict):
            if metadata and metadata.config_class:
                adapter_config = metadata.config_class(**adapter_config)
            else:
                from datus.tools.semantic_tools.config import SemanticAdapterConfig

                adapter_config = SemanticAdapterConfig(**adapter_config)

        adapter = semantic_adapter_registry.create_adapter(adapter_type, adapter_config)

        # Create storage manager
        storage_manager = SemanticStorageManager(agent_config)

        # Sync metrics only
        stats = await storage_manager.sync_from_adapter(
            adapter=adapter,
            sync_semantic_models=False,
            sync_metrics=True,
            subject_path=subject_path,
        )

        metrics_count = stats["metrics_synced"]
        if metrics_count == 0:
            return False, f"No metrics found in {adapter_type} adapter"

        logger.info(f"Successfully synced {metrics_count} metrics from {adapter_type}")
        return True, ""

    except Exception as e:
        error_msg = f"Failed to sync metrics from {adapter_type}: {str(e)}"
        logger.exception(error_msg)
        return False, error_msg
