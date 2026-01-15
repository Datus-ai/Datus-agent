# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
BI Adaptor Registry

Responsibilities:
1. Register built-in BI adaptors
2. Auto-discover plugins via Entry Points
3. Provide adaptor metadata for CLI selection
"""

from typing import ClassVar, Dict, Optional, Type

from datus.tools.bi_tools.base_adaptor import AuthType, BIAdaptorBase
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class AdaptorMetadata:
    """Metadata for a BI adaptor."""

    def __init__(
        self,
        platform: str,
        adaptor_class: Type[BIAdaptorBase],
        auth_type: AuthType,
        display_name: Optional[str] = None,
    ) -> None:
        self.platform = platform
        self.adaptor_class = adaptor_class
        self.auth_type = auth_type
        self.display_name = display_name or platform.capitalize()


class BIAdaptorRegistry:
    """Central registry for BI adaptors."""

    _adaptors: ClassVar[Dict[str, Type[BIAdaptorBase]]] = {}
    _metadata: ClassVar[Dict[str, AdaptorMetadata]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register(
        cls,
        platform: str,
        adaptor_class: Type[BIAdaptorBase],
        auth_type: AuthType,
        display_name: Optional[str] = None,
    ) -> None:
        """Register a BI adaptor."""
        key = (platform or "").strip().lower()
        if not key:
            logger.warning("Skipped registering BI adaptor with empty platform name.")
            return

        cls._adaptors[key] = adaptor_class
        cls._metadata[key] = AdaptorMetadata(
            platform=key,
            adaptor_class=adaptor_class,
            auth_type=auth_type,
            display_name=display_name,
        )
        logger.debug("Registered BI adaptor: %s -> %s", key, adaptor_class.__name__)

    @classmethod
    def list_adaptors(cls) -> Dict[str, Type[BIAdaptorBase]]:
        cls.discover_adaptors()
        return cls._adaptors.copy()

    @classmethod
    def get_metadata(cls, platform: str) -> Optional[AdaptorMetadata]:
        cls.discover_adaptors()
        return cls._metadata.get((platform or "").strip().lower())

    @classmethod
    def is_registered(cls, platform: str) -> bool:
        cls.discover_adaptors()
        return (platform or "").strip().lower() in cls._adaptors

    @classmethod
    def discover_adaptors(cls) -> None:
        """Load built-in adaptors and optional plugins."""
        if cls._initialized:
            return
        cls._initialized = True

        cls._load_builtin_adaptors()
        cls._discover_plugins()

    @classmethod
    def _load_builtin_adaptors(cls) -> None:
        try:
            import datus.tools.bi_tools.superset.superset_adaptor  # noqa: F401
        except Exception as exc:
            logger.debug("Failed to import built-in BI adaptor(s): %s", exc)

    @classmethod
    def _discover_plugins(cls) -> None:
        try:
            from importlib.metadata import entry_points

            try:
                adaptor_eps = entry_points(group="datus.bi_adaptors")
            except TypeError:
                eps = entry_points()
                adaptor_eps = eps.get("datus.bi_adaptors", [])

            for ep in adaptor_eps:
                try:
                    register_func = ep.load()
                    register_func()
                    logger.info("Discovered BI adaptor: %s", ep.name)
                except Exception as exc:
                    logger.warning("Failed to load BI adaptor %s: %s", ep.name, exc)
        except Exception as exc:
            logger.warning("BI adaptor entry point discovery failed: %s", exc)


adaptor_registry = BIAdaptorRegistry()
