# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Vector DB abstraction layer with pluggable backends."""

from typing import Any

from datus_storage_base.vector.base import BaseVectorBackend
from datus_storage_base.vector.registry import VectorRegistry


class _LazyLanceVectorBackend(BaseVectorBackend):
    """Registry placeholder that defers ``import lancedb`` to first use.

    lancedb's native library is built for the x86-64-haswell baseline
    (AVX2/FMA/F16C): on a CPU without those instructions the *import* itself
    dies with SIGILL, which no try/except can catch because it is a signal,
    not an exception. Deployments running another vector backend (pgvector)
    must therefore never execute that import.

    ``VectorRegistry.create_backend`` instantiates the registered class and
    then calls ``initialize(config)`` on the result. Returning a foreign
    instance from ``__new__`` makes Python skip this proxy's ``__init__``,
    so the real backend is constructed and initialized exactly as before.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseVectorBackend:
        from datus.storage.vector.lance_backend import LanceVectorBackend

        return LanceVectorBackend(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Keep ``from datus.storage.vector import LanceVectorBackend`` lazy (PEP 562)."""
    if name == "LanceVectorBackend":
        from datus.storage.vector.lance_backend import LanceVectorBackend

        return LanceVectorBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register built-in LanceDB backend (constructing it imports lancedb, importing this module does not)
VectorRegistry.register("lance", _LazyLanceVectorBackend)

# Discover external adapters via entry points
VectorRegistry.discover_adapters()

__all__ = [
    "BaseVectorBackend",
    "LanceVectorBackend",
    "VectorRegistry",
]
