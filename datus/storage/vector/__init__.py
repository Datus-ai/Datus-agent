# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Vector DB abstraction layer with pluggable backends."""

from datus.storage.vector.base import BaseVectorBackend
from datus.storage.vector.lance_backend import LanceVectorBackend
from datus.storage.vector.registry import VectorRegistry

# Register built-in LanceDB backend
VectorRegistry.register("lance", LanceVectorBackend)

# Discover external adapters via entry points
VectorRegistry.discover_adapters()

__all__ = [
    "BaseVectorBackend",
    "LanceVectorBackend",
    "VectorRegistry",
]
