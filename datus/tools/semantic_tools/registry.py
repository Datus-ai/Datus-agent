# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Adapter Registry

Responsibilities:
1. Register built-in semantic adapters
2. Auto-discover plugins via Entry Points
3. Dynamically load adapters
4. Create adapter instances
5. Provide adapter metadata for dynamic configuration
"""

from typing import Any, Callable, Dict, Optional, Type

from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class AdapterMetadata:
    """Metadata for a semantic adapter."""

    def __init__(
        self,
        service_type: str,
        adapter_class: Type[BaseSemanticAdapter],
        config_class: Optional[Type] = None,
        display_name: Optional[str] = None,
    ):
        self.service_type = service_type
        self.adapter_class = adapter_class
        self.config_class = config_class
        self.display_name = display_name or service_type.capitalize()

    def get_config_fields(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration fields from Pydantic config model."""
        if not self.config_class:
            return {}

        try:
            from pydantic import BaseModel

            if not issubclass(self.config_class, BaseModel):
                return {}

            fields_info = {}
            for field_name, field_info in self.config_class.model_fields.items():
                field_data = {
                    "required": field_info.is_required(),
                    "default": field_info.default if not field_info.is_required() else None,
                    "description": field_info.description or "",
                    "type": field_info.annotation.__name__
                    if hasattr(field_info.annotation, "__name__")
                    else str(field_info.annotation),
                }

                # Extract json_schema_extra metadata for special handling
                if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                    field_data.update(field_info.json_schema_extra)

                fields_info[field_name] = field_data
            return fields_info
        except Exception as e:
            logger.debug(f"Failed to extract config fields for {self.service_type}: {e}")
            return {}


class SemanticAdapterRegistry:
    """Central registry for semantic adapters."""

    _adapters: Dict[str, Type[BaseSemanticAdapter]] = {}
    _factories: Dict[str, Callable] = {}
    _metadata: Dict[str, AdapterMetadata] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls,
        service_type: str,
        adapter_class: Type[BaseSemanticAdapter],
        factory: Optional[Callable] = None,
        config_class: Optional[Type] = None,
        display_name: Optional[str] = None,
    ):
        """
        Register a semantic adapter.

        Args:
            service_type: Service type (e.g., "metricflow", "dbt", "cube")
            adapter_class: Adapter class
            factory: Optional factory method for custom instantiation logic
            config_class: Optional Pydantic config model for field metadata
            display_name: Optional display name for the adapter
        """
        service_type_lower = service_type.lower()
        cls._adapters[service_type_lower] = adapter_class
        if factory:
            cls._factories[service_type_lower] = factory

        # Store metadata
        cls._metadata[service_type_lower] = AdapterMetadata(
            service_type=service_type_lower,
            adapter_class=adapter_class,
            config_class=config_class,
            display_name=display_name,
        )

        logger.debug(f"Registered semantic adapter: {service_type} -> {adapter_class.__name__}")

    @classmethod
    def create_adapter(cls, service_type: str, config) -> BaseSemanticAdapter:
        """
        Create an adapter instance.

        Args:
            service_type: Service type (e.g., "metricflow", "dbt", "cube")
            config: Adapter configuration object

        Returns:
            Adapter instance

        Raises:
            DatusException: If adapter is not registered
        """
        service_type_lower = service_type.lower()

        # Try to dynamically load if not registered
        if service_type_lower not in cls._adapters:
            cls._try_load_adapter(service_type_lower)

        # Check again after attempting to load
        if service_type_lower not in cls._adapters:
            raise DatusException(
                ErrorCode.SEMANTIC_ADAPTER_NOT_FOUND,
                message=f"Semantic adapter '{service_type}' not found. "
                f"Available adapters: {list(cls._adapters.keys())}. "
                f"For additional semantic services, install: pip install datus-{service_type_lower}",
            )

        # Prefer factory method if available
        if service_type_lower in cls._factories:
            return cls._factories[service_type_lower](config)

        # Use default construction
        adapter_class = cls._adapters[service_type_lower]
        return adapter_class(config)

    @classmethod
    def _try_load_adapter(cls, service_type: str):
        """
        Attempt to dynamically load a plugin adapter.

        Args:
            service_type: Service type
        """
        try:
            # Try to import the plugin package
            module_name = f"datus_semantic_{service_type}"
            import importlib

            module = importlib.import_module(module_name)
            if hasattr(module, "register"):
                module.register()
                logger.info(f"Dynamically loaded semantic adapter: {service_type}")
        except ImportError:
            logger.debug(f"No semantic adapter found for: {service_type}")
        except Exception as e:
            logger.warning(f"Failed to load semantic adapter {service_type}: {e}")

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
                adapter_eps = entry_points(group="datus.semantic_adapters")
            except TypeError:
                # Python 3.9 fallback
                eps = entry_points()
                adapter_eps = eps.get("datus.semantic_adapters", [])

            for ep in adapter_eps:
                try:
                    register_func = ep.load()
                    register_func()
                    logger.info(f"Discovered semantic adapter: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load semantic adapter {ep.name}: {e}")
        except Exception as e:
            logger.warning(f"Entry points discovery failed: {e}")

    @classmethod
    def list_adapters(cls) -> Dict[str, Type[BaseSemanticAdapter]]:
        """
        List all registered adapters.

        Returns:
            Dictionary of adapters {service_type: adapter_class}
        """
        return cls._adapters.copy()

    @classmethod
    def is_registered(cls, service_type: str) -> bool:
        """
        Check if an adapter is registered.

        Args:
            service_type: Service type

        Returns:
            True if registered, False otherwise
        """
        return service_type.lower() in cls._adapters

    @classmethod
    def get_metadata(cls, service_type: str) -> Optional[AdapterMetadata]:
        """
        Get metadata for a specific adapter.

        Args:
            service_type: Service type

        Returns:
            AdapterMetadata if registered, None otherwise
        """
        return cls._metadata.get(service_type.lower())

    @classmethod
    def list_available_adapters(cls) -> Dict[str, AdapterMetadata]:
        """
        List all available adapters with their metadata.

        Returns:
            Dictionary of {service_type: AdapterMetadata}
        """
        # Ensure discovery has been run
        cls.discover_adapters()

        # Return copy of metadata dict
        return cls._metadata.copy()


# Global instance
semantic_adapter_registry = SemanticAdapterRegistry()
