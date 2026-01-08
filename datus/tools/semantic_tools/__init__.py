# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Tools Package

Provides unified abstraction layer for semantic layer services (MetricFlow, dbt, Cube, etc.)

Core Components:
- BaseSemanticAdapter: Abstract base class for semantic adapters
- SemanticAdapterRegistry: Registry for adapter registration and discovery
- SemanticStorageManager: Sync adapter data to unified storage
- Models: Data models (MetricDefinition, QueryResult, ValidationResult)
- Config: Configuration classes for different semantic services
"""

from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.config import SemanticAdapterConfig
from datus.tools.semantic_tools.models import MetricDefinition, QueryResult, ValidationResult
from datus.tools.semantic_tools.registry import AdapterMetadata, SemanticAdapterRegistry, semantic_adapter_registry
from datus.tools.semantic_tools.storage_sync import SemanticStorageManager

# Auto-discover adapters on import
semantic_adapter_registry.discover_adapters()

__all__ = [
    # Base class
    "BaseSemanticAdapter",
    # Registry
    "SemanticAdapterRegistry",
    "semantic_adapter_registry",
    "AdapterMetadata",
    # Config
    "SemanticAdapterConfig",
    # Storage
    "SemanticStorageManager",
    # Models
    "MetricDefinition",
    "QueryResult",
    "ValidationResult",
]
