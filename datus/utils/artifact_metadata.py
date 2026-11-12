# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Derive artifact metadata from live query files or an immutable publication bundle."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Callable, Iterable


def collect_query_datasources(paths: Iterable[str], read_content: Callable[[str], str]) -> list[str]:
    """Distinct saved-query sources, in file order, without using a stale manifest.

    Only metadata paired with a SQL/template/metric query contributes. The reader
    can access disk at finalize time or an already-loaded publication snapshot.
    Read errors propagate so the caller cannot publish a partial disk scan as a
    complete manifest. Malformed/non-query JSON contributes no datasource.
    """
    paths = list(paths)
    available = set(paths)
    labels: list[str] = []
    for path in paths:
        if PurePosixPath(path).parent != PurePosixPath("queries") or not path.endswith(".json"):
            continue
        if path.endswith((".brief.json", ".metric.json")):
            continue
        stem = path.removesuffix(".params.json") if path.endswith(".params.json") else path.removesuffix(".json")
        if not any(stem + suffix in available for suffix in (".sql", ".sql.j2", ".metric.json")):
            continue
        try:
            payload = json.loads(read_content(path))
        except (ValueError, TypeError):
            continue
        label = payload.get("datasource") if isinstance(payload, dict) else None
        if isinstance(label, str) and label.strip() and label.strip() not in labels:
            labels.append(label.strip())
    return labels
