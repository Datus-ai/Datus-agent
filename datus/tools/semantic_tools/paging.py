# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Shared semantic-adapter catalog paging configuration."""

from __future__ import annotations

from typing import Any

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

METRIC_CATALOG_PAGE_SIZE = 5000
METRIC_CATALOG_MAX_PAGES = 200


def metric_catalog_paging(agent_config: Any, adapter_type: str) -> tuple[int, int]:
    """Resolve metric catalog paging from the active adapter configuration."""

    config: dict[str, Any] = {}
    getter = getattr(agent_config, "get_semantic_layer_config", None)
    if callable(getter):
        try:
            config = getter(adapter_type) or {}
        except Exception as exc:  # noqa: BLE001 - invalid config falls back to safe defaults
            logger.warning("Could not read semantic layer config for metric paging: %s", exc)

    def _positive(key: str, default: int) -> int:
        try:
            value = int(config.get(key) or default)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; using %d", key, config.get(key), default)
            return default
        return value if value > 0 else default

    return (
        _positive("metric_catalog_page_size", METRIC_CATALOG_PAGE_SIZE),
        _positive("metric_catalog_max_pages", METRIC_CATALOG_MAX_PAGES),
    )


__all__ = ["METRIC_CATALOG_MAX_PAGES", "METRIC_CATALOG_PAGE_SIZE", "metric_catalog_paging"]
