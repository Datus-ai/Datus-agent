# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Registry for vector database backends."""

from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type

from datus.storage.vector.base import BaseVectorBackend
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class VectorRegistry:
    """Central registry for vector DB backends."""

    _backends: Dict[str, Type[BaseVectorBackend]] = {}
    _factories: Dict[str, Callable[..., BaseVectorBackend]] = {}
    _initialized: bool = False
    _init_lock = Lock()

    @classmethod
    def register(
        cls,
        backend_type: str,
        backend_class: Type[BaseVectorBackend],
        factory: Optional[Callable[..., BaseVectorBackend]] = None,
    ) -> None:
        key = backend_type.lower()
        cls._backends[key] = backend_class
        if factory:
            cls._factories[key] = factory
        logger.debug(f"Registered vector backend: {key} -> {backend_class.__name__}")

    @classmethod
    def create_backend(cls, backend_type: str, config: Dict[str, Any]) -> BaseVectorBackend:
        cls.discover_adapters()
        key = backend_type.lower()

        if key not in cls._backends:
            cls._try_load_adapter(key)

        if key not in cls._backends:
            raise DatusException(
                ErrorCode.COMMON_UNSUPPORTED,
                message=f"Vector backend '{backend_type}' not found. "
                f"Available: {list(cls._backends.keys())}. "
                f"Install: pip install datus-storage-{key}",
            )

        if key in cls._factories:
            return cls._factories[key](config)

        backend = cls._backends[key]()
        backend.initialize(config)
        return backend

    @classmethod
    def _try_load_adapter(cls, backend_type: str) -> None:
        try:
            import importlib

            module_name = f"datus_storage_{backend_type}"
            module = importlib.import_module(module_name)
            if hasattr(module, "register"):
                module.register()
                logger.info(f"Dynamically loaded vector adapter: {backend_type}")
        except ImportError:
            logger.debug(f"No vector adapter found for: {backend_type}")
        except Exception as e:
            logger.warning(f"Failed to load vector adapter {backend_type}: {e}")

    @classmethod
    def discover_adapters(cls) -> None:
        if cls._initialized:
            return
        with cls._init_lock:
            if cls._initialized:
                return

            # Register built-in LanceDB backend
            from datus.storage.vector.lance_backend import LanceVectorBackend

            if not cls.is_registered("lance"):
                cls.register("lance", LanceVectorBackend)

            try:
                from importlib.metadata import entry_points

                eps = entry_points()
                group = (
                    eps.get("datus.storage.vector", [])
                    if isinstance(eps, dict)
                    else eps.select(group="datus.storage.vector")
                )
                for ep in group:
                    try:
                        register_fn = ep.load()
                        if callable(register_fn):
                            register_fn()
                        logger.debug(f"Discovered vector adapter via entry point: {ep.name}")
                    except Exception as e:
                        logger.debug(f"Failed to load vector entry point {ep.name}: {e}")
            except Exception as e:
                logger.debug(f"Entry point discovery not available: {e}")

            cls._initialized = True

    @classmethod
    def registered_types(cls) -> List[str]:
        """Return all registered backend type names (triggers discovery)."""
        cls.discover_adapters()
        return list(cls._backends.keys())

    @classmethod
    def get_backend_class(cls, backend_type: str) -> Optional[Type[BaseVectorBackend]]:
        """Return the registered backend class, or None."""
        cls.discover_adapters()
        return cls._backends.get(backend_type.lower())

    @classmethod
    def is_registered(cls, backend_type: str) -> bool:
        return backend_type.lower() in cls._backends

    @classmethod
    def reset(cls) -> None:
        cls._backends.clear()
        cls._factories.clear()
        cls._initialized = False
