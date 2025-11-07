# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Connector Registry

Responsibilities:
1. Register built-in connectors
2. Auto-discover plugins via Entry Points
3. Dynamically load adapters
4. Create connector instances
"""

from typing import Callable, Dict, Optional, Type

from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ConnectorRegistry:
    """Central registry for database connectors."""

    _connectors: Dict[str, Type[BaseSqlConnector]] = {}
    _factories: Dict[str, Callable] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls,
        db_type: str,
        connector_class: Type[BaseSqlConnector],
        factory: Optional[Callable] = None,
    ):
        """
        Register a database connector.

        Args:
            db_type: Database type (e.g., "sqlite", "mysql")
            connector_class: Connector class
            factory: Optional factory method for custom instantiation logic
        """
        db_type_lower = db_type.lower()
        cls._connectors[db_type_lower] = connector_class
        if factory:
            cls._factories[db_type_lower] = factory
        logger.debug(f"Registered connector: {db_type} -> {connector_class.__name__}")

    @classmethod
    def create_connector(cls, db_type: str, config) -> BaseSqlConnector:
        """
        Create a connector instance.

        Args:
            db_type: Database type
            config: Database configuration object

        Returns:
            Connector instance

        Raises:
            DatusException: If connector is not registered
        """
        db_type_lower = db_type.lower()

        # Try to dynamically load if not registered
        if db_type_lower not in cls._connectors:
            cls._try_load_adapter(db_type_lower)

        # Check again after attempting to load
        if db_type_lower not in cls._connectors:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message=f"Connector '{db_type}' not found. "
                f"Available connectors: {list(cls._connectors.keys())}. "
                f"For additional databases, install: pip install datus-{db_type_lower}",
            )

        # Prefer factory method if available
        if db_type_lower in cls._factories:
            return cls._factories[db_type_lower](config)

        # Use default construction
        connector_class = cls._connectors[db_type_lower]
        return connector_class(config)

    @classmethod
    def _try_load_adapter(cls, db_type: str):
        """
        Attempt to dynamically load a plugin adapter.

        Args:
            db_type: Database type
        """
        try:
            # Try to import the plugin package
            module_name = f"datus_{db_type}"
            import importlib

            module = importlib.import_module(module_name)
            if hasattr(module, "register"):
                module.register()
                logger.info(f"Dynamically loaded adapter: {db_type}")
        except ImportError:
            logger.debug(f"No adapter found for: {db_type}")
        except Exception as e:
            logger.warning(f"Failed to load adapter {db_type}: {e}")

    @classmethod
    def discover_adapters(cls):
        """Auto-discover plugins via Entry Points."""
        if cls._initialized:
            return
        cls._initialized = True

        try:
            from importlib.metadata import entry_points

            # Python 3.10+ uses select(), Python 3.9 uses dict access
            try:
                adapter_eps = entry_points(group="datus.adapters")
            except TypeError:
                # Python 3.9 fallback
                eps = entry_points()
                adapter_eps = eps.get("datus.adapters", [])

            for ep in adapter_eps:
                try:
                    register_func = ep.load()
                    register_func()
                    logger.info(f"Discovered adapter: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load adapter {ep.name}: {e}")
        except Exception as e:
            logger.warning(f"Entry points discovery failed: {e}")

    @classmethod
    def list_connectors(cls) -> Dict[str, Type[BaseSqlConnector]]:
        """
        List all registered connectors.

        Returns:
            Dictionary of connectors {db_type: connector_class}
        """
        return cls._connectors.copy()

    @classmethod
    def is_registered(cls, db_type: str) -> bool:
        """
        Check if a connector is registered.

        Args:
            db_type: Database type

        Returns:
            True if registered, False otherwise
        """
        return db_type.lower() in cls._connectors


# Global instance
connector_registry = ConnectorRegistry()
