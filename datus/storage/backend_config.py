# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Configuration dataclasses for storage backends."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RdbBackendConfig:
    """Configuration for the relational database backend."""

    type: str = "sqlite"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorBackendConfig:
    """Configuration for the vector database backend."""

    type: str = "lance"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageBackendConfig:
    """Unified configuration for all storage backends."""

    rdb: RdbBackendConfig = field(default_factory=RdbBackendConfig)
    vector: VectorBackendConfig = field(default_factory=VectorBackendConfig)

    @staticmethod
    def from_dict(storage_config: Dict[str, Any]) -> "StorageBackendConfig":
        """Parse storage backend configuration from a dict.

        Expected format:
            storage:
              rdb:
                type: sqlite  # or mysql, postgresql
                # ... backend-specific params
              vector:
                type: lance  # or pgvector, milvus
                # ... backend-specific params
        """
        rdb_section = dict(storage_config.get("rdb", {})) if isinstance(storage_config.get("rdb", {}), dict) else {}
        vector_section = (
            dict(storage_config.get("vector", {})) if isinstance(storage_config.get("vector", {}), dict) else {}
        )

        rdb_type = rdb_section.pop("type", "sqlite") if rdb_section else "sqlite"
        vector_type = vector_section.pop("type", "lance") if vector_section else "lance"

        return StorageBackendConfig(
            rdb=RdbBackendConfig(type=rdb_type, params=rdb_section),
            vector=VectorBackendConfig(type=vector_type, params=vector_section),
        )
