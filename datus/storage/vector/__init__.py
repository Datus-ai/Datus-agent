# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Vector DB abstraction layer with pluggable backends."""

from datus_storage_base.vector.base import BaseVectorBackend
from datus_storage_base.vector.registry import VectorRegistry

from datus.storage.vector.lance_backend import LanceVectorBackend

# Register built-in LanceDB backend
VectorRegistry.register("lance", LanceVectorBackend)

# Discover external adapters via entry points
VectorRegistry.discover_adapters()

__all__ = [
    "BaseVectorBackend",
    "LanceVectorBackend",
    "VectorRegistry",
]
