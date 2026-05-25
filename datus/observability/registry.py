# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Registry for observability adapters."""

from __future__ import annotations

import threading
from typing import ClassVar

from datus.observability.adapters.base import ObservabilityAdapter
from datus.observability.adapters.langfuse import LangfuseAdapter
from datus.observability.adapters.otlp import OtlpAdapter
from datus.observability.adapters.platforms import BraintrustAdapter, DatadogAdapter, LangSmithAdapter
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ObservabilityAdapterRegistry:
    _adapters: ClassVar[dict[str, type[ObservabilityAdapter]]] = {}
    _initialized: ClassVar[bool] = False
    _lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def register(cls, adapter_type: str, adapter_cls: type[ObservabilityAdapter]) -> None:
        key = (adapter_type or "").strip().lower()
        if not key:
            logger.warning("Skipped registering observability adapter with empty type")
            return
        with cls._lock:
            cls._adapters[key] = adapter_cls

    @classmethod
    def get(cls, adapter_type: str) -> type[ObservabilityAdapter] | None:
        cls.discover_adapters()
        with cls._lock:
            return cls._adapters.get((adapter_type or "").strip().lower())

    @classmethod
    def discover_adapters(cls) -> None:
        with cls._lock:
            if cls._initialized:
                return
            cls._initialized = True
            cls.register("otlp", OtlpAdapter)
            cls.register("langfuse", LangfuseAdapter)
            cls.register("langsmith", LangSmithAdapter)
            cls.register("braintrust", BraintrustAdapter)
            cls.register("datadog", DatadogAdapter)
            cls._discover_plugins()

    @classmethod
    def _discover_plugins(cls) -> None:
        try:
            from importlib.metadata import entry_points

            try:
                adapter_eps = entry_points(group="datus.observability_adapters")
            except TypeError:
                eps = entry_points()
                adapter_eps = eps.get("datus.observability_adapters", [])

            for ep in adapter_eps:
                try:
                    register_func = ep.load()
                    register_func()
                    logger.info("Discovered observability adapter: %s", ep.name)
                except Exception as exc:
                    logger.warning("Failed to load observability adapter %s: %s", ep.name, exc)
        except Exception as exc:
            logger.warning("Observability adapter entry point discovery failed: %s", exc)

    @classmethod
    def list_adapters(cls) -> dict[str, type[ObservabilityAdapter]]:
        cls.discover_adapters()
        with cls._lock:
            return cls._adapters.copy()


adapter_registry = ObservabilityAdapterRegistry()
