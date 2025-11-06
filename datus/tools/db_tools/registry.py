# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Callable, Dict, Optional, Type

from datus.configuration.agent_config import DbConfig
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ConnectorRegistry:
    """Connector registry for dynamic plugin loading and management"""

    _connectors: Dict[str, Type[BaseSqlConnector]] = {}
    _factories: Dict[str, Callable] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls, db_type: str, connector_class: Type[BaseSqlConnector], factory: Optional[Callable] = None
    ) -> None:
        """Register a connector class, optionally with a factory method

        Args:
            db_type: Database type identifier (e.g., 'mysql', 'postgresql')
            connector_class: The connector class to register
            factory: Optional factory function to create connector instances
        """
        db_type_lower = db_type.lower()
        cls._connectors[db_type_lower] = connector_class
        if factory:
            cls._factories[db_type_lower] = factory
        logger.debug(f"Registered connector: {db_type} -> {connector_class.__name__}")

    @classmethod
    def create_connector(cls, db_type: str, db_config: DbConfig) -> BaseSqlConnector:
        """Create a connector instance for the given database type

        Args:
            db_type: Database type identifier
            db_config: Database configuration

        Returns:
            Initialized connector instance

        Raises:
            DatusException: If connector is not found
        """
        db_type_lower = db_type.lower()

        # If not registered, try to load plugin dynamically
        if db_type_lower not in cls._connectors:
            cls._try_load_adapter(db_type_lower)

        # Check again after loading attempt
        if db_type_lower not in cls._connectors:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message=f"Connector '{db_type}' not found. "
                f"Install it with: pip install datus-adapter-{db_type_lower}",
            )

        # Use factory method if available
        if db_type_lower in cls._factories:
            return cls._factories[db_type_lower](db_config)

        # Use default construction
        connector_class = cls._connectors[db_type_lower]
        params = cls._build_default_params(db_config)
        return connector_class(**params)

    @classmethod
    def _try_load_adapter(cls, db_type: str) -> None:
        """Try to dynamically import adapter package

        Args:
            db_type: Database type identifier
        """
        try:
            adapter_module = f"datus_adapter_{db_type}"
            import importlib

            module = importlib.import_module(adapter_module)
            # The adapter's __init__.py should call register()
            if hasattr(module, "register"):
                module.register()
                logger.info(f"Dynamically loaded adapter: {adapter_module}")
        except ImportError:
            logger.debug(f"Adapter {db_type} not installed")
        except Exception as e:
            logger.warning(f"Failed to load adapter {db_type}: {e}")

    @classmethod
    def _build_default_params(cls, db_config: DbConfig) -> dict:
        """Build default connection parameters from config

        Args:
            db_config: Database configuration

        Returns:
            Dictionary of connection parameters
        """
        params: Dict[str, Any] = {}

        # Handle URI-based configuration
        if db_config.uri:
            if db_config.type in ("sqlite", "duckdb"):
                # File-based databases
                params["db_path"] = db_config.uri.replace(f"{db_config.type}:///", "")
                if db_config.database:
                    params["database_name"] = db_config.database
            else:
                # SQLAlchemy-based databases
                params["connection_string"] = db_config.uri
                params["dialect"] = db_config.type
            return params

        # Build from individual fields
        if db_config.host:
            params["host"] = db_config.host
        if db_config.port:
            params["port"] = int(db_config.port)
        if db_config.username:
            params["user"] = db_config.username
        if db_config.password:
            params["password"] = db_config.password
        if db_config.database:
            params["database"] = db_config.database
        if db_config.catalog:
            params["catalog"] = db_config.catalog
        if db_config.schema:
            params["schema"] = db_config.schema
        if db_config.warehouse:
            params["warehouse"] = db_config.warehouse
        if db_config.account:
            params["account"] = db_config.account

        return params

    @classmethod
    def discover_adapters(cls) -> None:
        """Auto-discover plugins via Entry Points

        This method uses Python's entry points mechanism to automatically
        discover and load installed adapter packages.
        """
        if cls._initialized:
            return
        cls._initialized = True

        try:
            from importlib.metadata import entry_points

            adapter_eps = entry_points(group="datus.adapters")
            for ep in adapter_eps:
                try:
                    register_func = ep.load()
                    register_func()
                    logger.debug(f"Discovered adapter via entry point: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load adapter {ep.name}: {e}")
        except Exception as e:
            logger.warning(f"Entry points discovery failed: {e}")


# Global instance
connector_registry = ConnectorRegistry()
