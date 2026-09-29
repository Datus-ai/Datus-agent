# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Shared Dosi catalog paging bounds."""

from __future__ import annotations

METRIC_CATALOG_PAGE_SIZE = 5000
METRIC_CATALOG_MAX_PAGES = 200


def metric_catalog_paging() -> tuple[int, int]:
    """Return bounded native Dosi catalog scan settings."""
    return METRIC_CATALOG_PAGE_SIZE, METRIC_CATALOG_MAX_PAGES


__all__ = ["METRIC_CATALOG_MAX_PAGES", "METRIC_CATALOG_PAGE_SIZE", "metric_catalog_paging"]
