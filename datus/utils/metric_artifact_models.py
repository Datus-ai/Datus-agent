# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Resolve frozen metric models from an artifact, never from the live project."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from datus.schemas.metric_artifact_query import MetricQueryFile


def resolve_metric_query_snapshot(
    query: MetricQueryFile,
    *,
    files: Mapping[str, str] | None = None,
    artifact_dir: Path | None = None,
) -> MetricQueryFile:
    """Hydrate execution-only model bytes while retaining the saved reference.

    A reference is authoritative, even when inline bytes are also present. A
    broken reference must not fall back to an embedded or current Studio model.
    Legacy 1.0 recipes continue to execute their inline snapshot unchanged.
    """
    content = query.model_snapshot
    if query.model_snapshot_path is not None:
        relative = query.model_snapshot_path
        if files is not None:
            content = files.get(relative)
        elif artifact_dir is not None:
            root = artifact_dir.resolve()
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise ValueError("metric model snapshot escapes the artifact directory")
            try:
                content = path.read_bytes().decode("utf-8")
            except OSError as exc:
                raise ValueError(f"metric model snapshot missing: {relative}") from exc
        else:
            content = None
    if content is None:
        raise ValueError("metric model snapshot missing; resave the metric query before publishing")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != query.model_revision:
        raise ValueError("metric model snapshot must match model_revision")
    return query.model_copy(update={"model_snapshot": content})
